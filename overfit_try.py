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


def multiclass_structure_loss(logits: torch.Tensor,
                              targets: torch.Tensor,
                              num_classes: int,
                              eps: float = 1e-6) -> torch.Tensor:
    """
    logits:  [N, C, H, W]    raw scores for C classes
    targets: [N, H, W]       integer labels in {0,1,…,C-1}
    returns: scalar loss
    """
    N, C, H, W = logits.shape
    # 1) convert targets to one–hot: [N, C, H, W]
    targets_onehot = F.one_hot(targets.long(), num_classes=C) \
                      .permute(0, 3, 1, 2).float()

    # 2) per-class weighted‐structure loss
    losses = []
    for c in range(C):
        pred_c = logits[:, c:c+1, ...]           # [N,1,H,W]
        mask_c = targets_onehot[:, c:c+1, ...]   # [N,1,H,W]

        # spatial weight map
        weit = 1 + 5 * torch.abs(
            F.avg_pool2d(mask_c, kernel_size=31, stride=1, padding=15)
            - mask_c
        )

        # weighted BCE
        wbce = F.binary_cross_entropy_with_logits(
            pred_c, mask_c, reduction='none'
        )
        wbce = (weit * wbce).sum(dim=(2,3)) / weit.sum(dim=(2,3))

        # weighted IoU
        prob  = torch.sigmoid(pred_c)
        inter = (prob * mask_c * weit).sum(dim=(2,3))
        union = ((prob + mask_c) * weit).sum(dim=(2,3))
        wiou  = 1 - (inter + eps) / (union - inter + eps)

        # average over batch
        losses.append((wbce + wiou).mean())

    # 3) average over classes
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
            train_loader = DataLoader(full_train_dataset,  batch_size=args.train_bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)
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

        self.criterion = JointEdgeSegLoss(classes=self.params.num_classes) if self.with_edge else multiclass_structure_loss

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
        self.model.train()
        neighbor, target, _ = batch
        logits = self.model(neighbor)  # → could be [B, C, H, W] or [B, T, C, H, W]
        #print(logits.shape) #?torch.Size([3, 3, 256, 256])
        # 1) If there's a time dimension, flatten it into the batch:

        if logits.ndim == 5:
            B, T, C, H, W = logits.shape
            # reshape logits → [B*T, C, H, W]
            logits = logits.view(B * T, C, H, W)
            # replicate target across time → [B*T, H, W]
            #target = target.unsqueeze(1).repeat(1, T, 1, 1).view(B * T, H, W)
        if target.ndim == 5:
            B, T, C, H, W = target.shape
            # assume one-hot: [B, T, C, H, W]
            # convert to class‐index mask: [B, T, H, W]
            target = target.argmax(dim=2)
        #print(target.shape) #[1, 3, 256, 256]
        target = target.view(B*T, H, W)
        #print(target.shape) torch.Size([3, 256, 256])
        
        #print(logits.shape) #torch.Size([3, 3, 256, 256])
        #print(target.shape) #torch.Size([1, 3, 3, 256, 256])
        loss_ce = self.criterion(logits, target, num_classes=self.num_classes)
        # compute Dice loss (1 - Dice score)
        dice_score = self.dice_loss(logits.softmax(dim=1), target)
        loss = loss_ce + (1 - dice_score)

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/jaccard", self.train_jaccard(logits.softmax(dim=1), target), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        neighbor, target, _ = batch
        logits = self.model(neighbor)  # → could be [B, C, H, W] or [B, T, C, H, W]
        #print(logits.shape) #?torch.Size([3, 3, 256, 256])
        # 1) If there's a time dimension, flatten it into the batch:

        if logits.ndim == 5:
            B, T, C, H, W = logits.shape
            # reshape logits → [B*T, C, H, W]
            logits = logits.view(B * T, C, H, W)
            # replicate target across time → [B*T, H, W]
            #target = target.unsqueeze(1).repeat(1, T, 1, 1).view(B * T, H, W)
        if target.ndim == 5:
            B, T, C, H, W = target.shape
            # assume one-hot: [B, T, C, H, W]
            # convert to class‐index mask: [B, T, H, W]
            target = target.argmax(dim=2)
        #print(target.shape) #[1, 3, 256, 256]
        target = target.view(B*T, H, W)

        #print(target.shape) torch.Size([3, 256, 256])
        #print(logits.shape) #torch.Size([3, 3, 256, 256])

        loss = self.criterion(logits, target, num_classes=self.num_classes)
        preds = logits.argmax(dim=1)  # [B*T or B, H, W]

        # log image examples (unchanged)
        sample_pred = preds[0].cpu().numpy()
        sample_gt   = target[0].cpu().numpy()
        pred_rgb = wandb.Image(
            sample_pred,
            masks={"prediction": {
                "mask_data": sample_pred,
                "class_labels": {i: f"class_{i}" for i in range(self.num_classes)}
            }}
        )
        gt_rgb = wandb.Image(
            sample_gt,
            masks={"ground_truth": {
                "mask_data": sample_gt,
                "class_labels": {i: f"class_{i}" for i in range(self.num_classes)}
            }}
        )
        self.logger.experiment.log(
        {
            "val_examples": [pred_rgb, gt_rgb],
            "global_step": self.global_step  # so it shows up at the right step
        }
        )
        # accumulate for epoch‐end
        self.preds.append(preds.cpu())
        self.gts.append(target.cpu())
        self.val_losses.append(loss.detach)

        return loss


    def on_validation_epoch_end(self):
        import numpy as np

    def on_validation_epoch_end(self):
        #Shape [total_samples, H, W]
        all_preds = torch.cat(self.preds, dim=0).numpy()
        all_gts   = torch.cat(self.gts,   dim=0).numpy()
        C = self.num_classes

        #For each class, we'll collect one S, one E, one MAE, etc.
        sm_class = [Smeasure() for _ in range(C)]
        em_class = [Emeasure() for _ in range(C)]
        mae_class= [MAE()      for _ in range(C)]

        # If you still want threshold‐based metrics like you did for binary:
        thresholds = np.linspace(0, 1, 256)
        dice_per_class      = [[] for _ in range(C)]
        specificity_per_class = [[] for _ in range(C)]
        precision_per_class   = [[] for _ in range(C)]
        recall_per_class      = [[] for _ in range(C)]
        fmeasure_per_class    = [[] for _ in range(C)]
        jaccard_per_class     = [[] for _ in range(C)]

        #Iterate over each sample
        for pred_mask, gt_mask in zip(all_preds, all_gts):
            # For each class c, build a binary mask
            for c in range(C):
                pred_c = (pred_mask == c).astype(np.uint8)
                gt_c   = (gt_mask   == c).astype(np.uint8)

                # Update structural metrics
                sm_class[c].step(pred_c, gt_c)
                em_class[c].step(pred_c, gt_c)
                mae_class[c].step(pred_c, gt_c)

                # Threshold‐sweep metrics
                d_lst, s_lst, p_lst, r_lst, f_lst, j_lst = [], [], [], [], [], []
                for t in thresholds:
                    bin_pred = (pred_mask == c).astype(np.uint8)  # same as pred_c, but you could simulate uncertainty
                    # If you had soft‐preds, you'd threshold them here
                    dice, spec, prec, rec, fmeas, jacc = \
                        self.evaluate_one_img(bin_pred, gt_c)
                    d_lst.append(dice)
                    s_lst.append(spec)
                    p_lst.append(prec)
                    r_lst.append(rec)
                    f_lst.append(fmeas)
                    j_lst.append(jacc)

                # average over thresholds
                dice_per_class[c].append(np.mean(d_lst))
                specificity_per_class[c].append(np.mean(s_lst))
                precision_per_class[c].append(np.mean(p_lst))
                recall_per_class[c].append(np.mean(r_lst))
                fmeasure_per_class[c].append(np.mean(f_lst))
                jaccard_per_class[c].append(np.mean(j_lst))

        #Compute per‐class and macro‐averages
        logs = {}
        for c in range(C):
            logs[f"class_{c}/Smeasure"]     = sm_class[c].get_results()["Smeasure"]
            logs[f"class_{c}/Emeasure"]     = em_class[c].get_results()["meanEm"]
            logs[f"class_{c}/MAE"]          = mae_class[c].get_results()["MAE"]
            logs[f"class_{c}/Dice"]         = np.mean(dice_per_class[c])
            logs[f"class_{c}/Jaccard"]      = np.mean(jaccard_per_class[c])
            logs[f"class_{c}/Precision"]    = np.mean(precision_per_class[c])
            logs[f"class_{c}/Recall"]       = np.mean(recall_per_class[c])
            logs[f"class_{c}/Fmeasure"]     = np.mean(fmeasure_per_class[c])
            logs[f"class_{c}/Specificity"]  = np.mean(specificity_per_class[c])

        # Macro‐average across classes
        macro = lambda name: np.mean([logs[f"class_{c}/{name}"] for c in range(C)])
        logs["macro/Smeasure"]    = macro("Smeasure")
        logs["macro/Emeasure"]    = macro("Emeasure")
        logs["macro/MAE"]         = macro("MAE")
        logs["macro/Dice"]        = macro("Dice")
        logs["macro/Jaccard"]     = macro("Jaccard")
        logs["macro/Precision"]   = macro("Precision")
        logs["macro/Recall"]      = macro("Recall")
        logs["macro/Fmeasure"]    = macro("Fmeasure")
        logs["macro/Specificity"] = macro("Specificity")

        #Log everything
        for k, v in logs.items():
            # if you want them in TensorBoard / W&B via Lightning:
            self.log(k, v, prog_bar=(k.startswith("macro/")), sync_dist=True)

        #cleanup
        self.preds.clear()
        self.gts.clear()
        self.val_losses.clear()

    
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
   
    
    for fold in range(args.num_folds):
        wandb.init(
        project='Vivim_multiclass_segmentation',
        name='Vivim multiclass overfit',
        config=vars(args)
       )

        logger = WandbLogger(save_dir='.',
                        name='Vivim multiclass overfit',
                        project='Vivim_multiclass_segmentation',
                        log_model=True)
        pl.seed_everything(args.seed, workers=True)
        
        print('Start training for fold number ', fold)

        #resume_checkpoint_path = 'logs/vivim_OTU/version_35/checkpoints/ultra-epoch00-Dice-0.8606-Jaccard-0.7772.ckpt'
        resume_checkpoint_path = args.resume_path

        model = CoolSystem(args, fold_number=fold)

        checkpoint_callback = ModelCheckpoint(
        monitor='macro/Dice',
        #dirpath='/mnt/data/yt/Documents/TSANet-underwater/snapshots',
        filename='ultra-epoch{epoch:02d}-macroDice-{macro/Dice:.4f}',
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
            overfit_batches=5,
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
        wandb.finish()
    
if __name__ == '__main__':
    main()