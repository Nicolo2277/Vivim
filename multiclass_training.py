from typing import Optional
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3' Only in case if u have multiple GPUs and want to use a specific one
import numpy as np
import copy
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import cfg
from torchmetrics.classification import MulticlassJaccardIndex, Dice

import pytorch_lightning as pl
import yaml
from easydict import EasyDict
import random
from pytorch_lightning import callbacks
from pytorch_lightning.accelerators import accelerator
from pytorch_lightning.core.hooks import CheckpointHooks
from pytorch_lightning.callbacks import ModelCheckpoint,DeviceStatsMonitor,EarlyStopping,LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from argparse import Namespace

from OTU_dataset import *
from torch.utils.data import DataLoader, Subset
from loss import *
# from models2.refinenet import RefineNet
from torchvision.utils import save_image
import wandb
import misc2
from create_train_data_multiclass import *

args = cfg.parse_args()

output_dir = 'logs'
version_name='multiclass segmentation'
wandb.init(
    project='Vivim_multiclass_segmentation',
    name='multiclass segmentation',
    config=vars(args)
)

logger = WandbLogger(save_dir='.',
                     name='Baseline',
                     project='Vivim_multiclass_segmentation',
                     log_model=True)


import matplotlib.pyplot as plt
# import tent
import math

from medpy import metric
# from misc import *
import torchmetrics
from modeling.vivim import Vivim

from poloy_metrics import *
from modeling.utils import JointEdgeSegLoss
# torch.set_float32_matmul_precision('high')
from Multiclass_Data import *


def multiclass_structure_loss_bce(logits, targets_onehot, eps=1e-6):
    """
    logits: [B, 3, H, W] raw scores
    targets_onehot: [B, 3, H, W] binary masks per class
    """
    losses = []
    B, C, H, W = logits.shape

    for c in range(C):
        pred_c = logits[:, c:c+1]           # [B,1,H,W]
        mask_c = targets_onehot[:, c:c+1]   # [B,1,H,W]

        # per-pixel weight map as before
        weit = 1 + 5 * torch.abs(
            F.avg_pool2d(mask_c, 31, 1, 15)
            - mask_c
        )

        # weighted BCE
        wbce = F.binary_cross_entropy_with_logits(
            pred_c, mask_c, reduction='none'
        )
        wbce = (weit * wbce).sum((2,3)) / weit.sum((2,3))

        # weighted IoU
        prob = torch.sigmoid(pred_c)
        inter = (prob * mask_c * weit).sum((2,3))
        union = ((prob + mask_c) * weit).sum((2,3))
        wiou  = 1 - (inter + eps) / (union - inter + eps)

        losses.append((wbce + wiou).mean())

    # average over classes
    return sum(losses) / C


def get_loader(args, mode, num_fold):
        input_root = 'Multiclass_Folds/fold_' + str(num_fold)

        train_path = os.path.join(input_root, 'train')
        train_output_root = 'Multiclass_TrainData_fold_' + str(num_fold)
        train_output_root = os.path.join(train_output_root, 'train')
        gather_multiclass_frames(Path(train_path), Path(train_output_root))
        full_train_dataset = MainDataset(root=train_output_root, trainsize=args.image_size, clip_len=args.clip_length)

        val_path = os.path.join(input_root, 'val')
        val_output_root = 'Multiclass_TrainData_fold_' + str(num_fold)
        val_output_root = os.path.join(val_output_root, 'val')
        gather_multiclass_frames(Path(val_path), Path(val_output_root))
        full_val_dataset = MainDataset(root=val_output_root, trainsize=args.image_size, clip_len=args.clip_length)

        if mode == 'training': 
            train_loader = DataLoader(full_train_dataset,  batch_size=args.train_bs, shuffle=True, num_workers=args.num_workers, pin_memory=True)
            return train_loader
        elif mode=='validation':
            val_loader = DataLoader(full_val_dataset, batch_size=args.val_bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            return val_loader


class CoolSystem(pl.LightningModule):
    
    def __init__(self, hparams, fold_number):
        super(CoolSystem, self).__init__()

        self.params = hparams
        self.save_path= self.params.save_path
        self.data_root=self.params.data_path

        self.train_batchsize = self.params.train_bs
        self.val_batchsize = self.params.val_bs

        self.fold_number = fold_number 
        
        #Train setting
        self.initlr = self.params.initlr #initial learning rate
        self.weight_decay = self.params.weight_decay #optimizers weight decay
        self.crop_size = self.params.crop_size #random crop size
        self.num_workers = self.params.num_workers
        self.epochs = self.params.epochs
        self.shift_length = self.params.shift_length
        self.val_aug = self.params.val_aug
        self.with_edge = self.params.with_edge

        self.num_classes = self.params.num_classes

        self.gts = []
        self.preds = []
        
        self.nFrames = self.params.clip_length
        self.upscale_factor = 1
        self.data_augmentation = True

        self.criterion = JointEdgeSegLoss(classes=self.params.num_classes) if self.with_edge else multiclass_structure_loss_bce

        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = Dice(num_classes=self.num_classes, average="macro")
        
        # Metrics
        self.train_jaccard = MulticlassJaccardIndex(num_classes=self.num_classes, average="macro")
        self.val_jaccard   = MulticlassJaccardIndex(num_classes=self.num_classes, average="macro")
        self.val_dice      = Dice(num_classes=self.num_classes, average="macro")

        self.model = Vivim(with_edge=self.with_edge, out_chans=self.params.num_classes)

        self.save_hyperparameters()

        self.val_losses = []
    
    def configure_optimizers(self):
        #We filter the parameters that require gradients, avoid updating frozen parts
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.initlr,betas=[0.9,0.999])#,weight_decay=self.weight_decay)
         
        # optimizer = Lion(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.initlr,betas=[0.9,0.99],weight_decay=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=self.initlr * 0.01)

        return [optimizer], [scheduler]
    
    def init_weight(self,ckpt_path=None): #Load pretrained weights
    
        if ckpt_path:
            checkpoint = torch.load(ckpt_path)
            print(checkpoint.keys())
            checkpoint_model = checkpoint
            state_dict = self.model.state_dict()
            # # 1. filter out unnecessary keys
            checkpoint_model = {k: v for k, v in checkpoint_model.items() if k in state_dict.keys()}
            print(checkpoint_model.keys())
            # 2. overwrite entries in the existing state dict
            state_dict.update(checkpoint_model)
            
            self.model.load_state_dict(checkpoint_model, strict=False) 

    def evaluate_one_img(self, pred, gt): #It computes useful metrics
        dice = misc2.dice(pred, gt)
        specificity = misc2.specificity(pred, gt)
        jaccard = misc2.jaccard(pred, gt)
        precision = misc2.precision(pred, gt)
        recall = misc2.recall(pred, gt)
        f_measure = misc2.fscore(pred, gt)
        return dice, specificity, precision, recall, f_measure, jaccard
    
    def training_step(self, batch, batch_idx):
        neighbor, target, _ = batch
        # neighbor: [B, T, 3, H, W] or [B, 3, H, W]
        # target:   [B, H, W] with values in {0,..,C-1}
        logits = self.model(neighbor)        # => [B*T, C, H, W] or [B, C, H, W]
        # flatten temporal dim if needed:
        if logits.ndim == 5:
            B, T, C, H, W = logits.shape
            logits = logits.view(B*T, C, H, W)
            target = target.unsqueeze(1).repeat(1, T, 1, 1).view(B*T, H, W)

        loss_ce = self.ce_loss(logits, target)
        # compute Dice loss (1 - Dice score)
        dice_score = self.dice_loss(logits.softmax(dim=1), target)
        loss = loss_ce + (1 - dice_score)

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/jaccard", self.train_jaccard(logits.softmax(dim=1), target), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        neighbor, target, _ = batch
        logits = self.model(neighbor)
        # same reshape logic...
        loss = self.ce_loss(logits, target)
        preds = logits.argmax(dim=1)  # [B*T, H, W]
        self.val_losses.append(loss)

        self.val_jaccard.update(preds, target)
        self.val_dice.update(preds, target)
        sample_pred = preds.view(-1, *preds.shape[-2:])[0].cpu().numpy()
        sample_gt   = target.view(-1, *target.shape[-2:])[0].cpu().numpy()
        # convert class indices into an RGB palette:
        pred_rgb = wandb.Image(sample_pred, masks={
            "prediction": {"mask_data": sample_pred, "class_labels": {i: f"class_{i}" for i in range(self.num_classes)}}
        })
        gt_rgb = wandb.Image(sample_gt, masks={
            "ground_truth": {"mask_data": sample_gt, "class_labels": {i: f"class_{i}" for i in range(self.num_classes)}}
        })
        self.logger.log_image(key="val_examples", images=[pred_rgb, gt_rgb])

        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_losses).mean()
        jaccard = self.val_jaccard.compute()
        dice    = self.val_dice.compute()

        self.log("val/loss", avg_loss)
        self.log("val/jaccard", jaccard)
        self.log("val/dice", dice)

        # reset
        self.val_losses.clear()
        self.val_jaccard.reset()
        self.val_dice.reset()
    
    def train_dataloader(self):
        train_loader = get_loader(self.params, mode='training', num_fold=self.fold_number)
        return train_loader
    
    def val_dataloader(self):
        val_loader = get_loader(self.params, mode='validation', num_fold=self.fold_number)
        return val_loader 

    def _compute_loss(self, neigbor, target, rest):
        # factor out training/validation loss computation
        # existing model forward + criterion
        if not self.with_edge:
            pred = self.model(neigbor.cuda())
            #print('Pred: ', pred.shape) #torch.Size([3, 1, 256, 256])
            #print('Target: ', target.shape) #torch.Size([1, 3, 1, 256, 256])
            target = target.cuda().reshape(pred.shape)
            return self.criterion(pred[self.nFrames//2::self.nFrames], target[self.nFrames//2::self.nFrames])
        else:
            pred, e0 = self.model(neigbor.cuda())
            target = target.cuda().reshape(pred.shape)
            edge_gt = rest[0].cuda().reshape(e0.shape)
            return self.criterion((pred[self.nFrames//2::self.nFrames], e0[self.nFrames//2::self.nFrames]),
                                   (target[self.nFrames//2::self.nFrames], edge_gt[self.nFrames//2::self.nFrames]))

def main():
   
    pl.seed_everything(args.seed, workers=True)
    
    for fold in range(args.num_folds):
        
        print('Start training for fold number ', fold)

        #resume_checkpoint_path = 'logs/vivim_OTU/version_35/checkpoints/ultra-epoch00-Dice-0.8606-Jaccard-0.7772.ckpt'
        resume_checkpoint_path = args.resume_path
      
        model = CoolSystem(args, fold_number=fold)

        checkpoint_callback = ModelCheckpoint(
        monitor='Dice',
        #dirpath='/mnt/data/yt/Documents/TSANet-underwater/snapshots',
        filename='ultra-epoch{epoch:02d}-Dice-{Dice:.4f}-Jaccard-{Jaccard:.4f}',
        auto_insert_metric_name=False,   
        every_n_epochs=1,
        save_top_k=1,
        mode = "max",
        save_last=True
        )

        lr_monitor_callback = LearningRateMonitor(logging_interval='step')
        trainer = pl.Trainer(
            check_val_every_n_epoch=args.val_freq,
            max_epochs=args.epochs,
            accelerator='gpu',
            devices=1,
            precision=16,
            logger=logger,
            strategy="auto",
            enable_progress_bar=True,
            log_every_n_steps=5,
            callbacks = [checkpoint_callback,lr_monitor_callback]
        ) 


        trainer.fit(model,ckpt_path=resume_checkpoint_path)
        # val_path=r'/home/yijun/project/ultra/logs/uentm_polyp/version_60/checkpoints/ultra-epoch.ckpt'
        # trainer.validate(model,ckpt_path=val_path)
    
if __name__ == '__main__':
    main()