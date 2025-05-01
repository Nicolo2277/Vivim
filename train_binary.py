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
from create_train_set import *

args = cfg.parse_args()

output_dir = 'logs'
version_name='binary segmentation'
wandb.init(
    project='Vivim_binary_segmentation',
    name='binary segmentation',
    config=vars(args)
)

logger = WandbLogger(save_dir='.',
                     name='Baseline',
                     project='Vivim_binary_segmentation',
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
from main_dataset import *


def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

def get_loader(args, mode, num_fold):
        input_root = 'Folds/fold_' + str(num_fold)

        train_path = os.path.join(input_root, 'train')
        train_output_root = 'TrainData_fold_' + str(num_fold)
        train_output_root = os.path.join(train_output_root, 'train')
        gather_annotated_frames(Path(train_path), Path(train_output_root))
        full_train_dataset = MainDataset(root=train_output_root, trainsize=args.image_size, clip_len=args.clip_length)

        val_path = os.path.join(input_root, 'val')
        val_output_root = 'TrainData_fold_' + str(num_fold)
        val_output_root = os.path.join(val_output_root, 'val')
        gather_annotated_frames(Path(val_path), Path(val_output_root))
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

        self.gts = []
        self.preds = []
        
        self.nFrames = self.params.clip_length
        self.upscale_factor = 1
        self.data_augmentation = True

        self.criterion = JointEdgeSegLoss(classes=self.params.num_classes) if self.with_edge else structure_loss

        self.model = Vivim(with_edge=self.with_edge)

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
        
        neigbor, target, edge_gt = batch
        target = target.cuda()
        neigbor = neigbor.cuda()

        #print(neigbor.shape) #torch.Size([2, 3, 3, 256, 256])
        #print(target.shape) #torch.Size([2, 3, 1, 256, 256])
        
        if len(neigbor.shape) == 5:
            bz, nf, nc, h, w = target.shape
        elif len(neigbor.shape) == 4:
            bz, nc, h, w = neigbor.shape
            nf = 1
            neigbor = neigbor.unsqueeze(1)
        else:
            raise ValueError('Unexpected tensor shape for neigbor')
        
        if not self.with_edge:
            pred = self.model(neigbor)
            #print(pred.shape) 
            target = target.reshape(bz*nf, nc, h, w)
            loss = self.criterion(pred[self.nFrames//2::self.nFrames], target[self.nFrames//2::self.nFrames])

        else:
            pred, e0 = self.model(neigbor)
            target = target.reshape(bz*nf, nc, h, w)
            edge_gt = edge_gt.reshape(bz*nf, 1, h, w)
            loss = self.criterion((pred[self.nFrames//2::self.nFrames], e0[self.nFrames//2::self.nFrames]), 
                                   (target[self.nFrames//2::self.nFrames], edge_gt[self.nFrames//2::self.nFrames]))
        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}

    def on_validation_epoch_end(self):

        self.sm = Smeasure()
        self.em = Emeasure()
        self.mae = MAE()

        dice_lst, specificity_lst, precision_lst, recall_lst, f_measure_lst, jaccard_lst = [], [], [], [], [], []
        Thresholds = np.linspace(1, 0, 256)
        # print(Thresholds)

        for pred, gt in zip(self.preds,self.gts):
            #pred = torch.sigmoid(pred)
            # gt = gt.to(int)
            #print(pred.shape) #shape = [1, 1, 256, 256]
            #print(gt.shape) #shape = [3, 1, 256, 256]
            self.sm.step(pred.squeeze(0).squeeze(0).detach().cpu().numpy(),gt.squeeze(0).squeeze(0).detach().cpu().numpy())
            self.em.step(pred.squeeze(0).squeeze(0).detach().cpu().numpy(),gt.squeeze(0).squeeze(0).detach().cpu().numpy())
            self.mae.step(pred.squeeze(0).squeeze(0).detach().cpu().numpy(),gt.squeeze(0).squeeze(0).detach().cpu().numpy())
            gt = (gt>0.5).to(int)
            dice_l, specificity_l, precision_l, recall_l, f_measure_l, jaccard_l = [], [], [], [], [], []
            for j, threshold in enumerate(Thresholds):
                # print(threshold)
                pred_one_hot = (pred>threshold).to(int)
                
                dice, specificity, precision, recall, f_measure, jaccard = self.evaluate_one_img(pred_one_hot.detach().cpu().numpy(), gt.detach().cpu().numpy())
                # print(dice)
                dice_l.append(dice)
                specificity_l.append(specificity)
                precision_l.append(precision)
                recall_l.append(recall)
                f_measure_l.append(f_measure)
                jaccard_l.append(jaccard)
            dice_lst.append(sum(dice_l) / len(dice_l))
            specificity_lst.append(sum(specificity_l) / len(specificity_l))
            precision_lst.append(sum(precision_l) / len(precision_l))
            recall_lst.append(sum(recall_l) / len(recall_l))
            f_measure_lst.append(sum(f_measure_l) / len(f_measure_l))
            jaccard_lst.append(sum(jaccard_l) / len(jaccard_l))


            # print(sum(dice_l) / len(dice_l))

        # mean
        dice = sum(dice_lst) / len(dice_lst)
        acc = sum(specificity_lst) / len(specificity_lst)
        precision = sum(precision_lst) / len(precision_lst)
        recall = sum(recall_lst) / len(recall_lst)
        f_measure = sum(f_measure_lst) / len(f_measure_lst)
        jac = sum(jaccard_lst) / len(jaccard_lst)

        sm = self.sm.get_results()['Smeasure']
        em = self.em.get_results()['meanEm']
        mae = self.mae.get_results()['MAE']

        #print(len(self.gts))
        #print(len(self.preds))
        
        self.log('Dice',dice)
        self.log('Jaccard',jac)
        self.log('Precision',precision)
        self.log('Recall',recall)
        self.log('Fmeasure',f_measure)
        self.log('specificity',acc)
        self.log('Smeasure',sm)
        self.log('Emeasure',em)
        self.log('MAE',mae)

        self.gts = []
        self.preds = []

        if self.trainer.current_epoch == 0:
            # skip first epoch
            pass
        else:
            print(f"Epoch {self.current_epoch}: Avg val loss = {torch.stack(self.val_losses).mean():.4f}")
        self.val_losses.clear()

        print("Val: Dice {0}, Jaccard {1}, Precision {2}, Recall {3}, Fmeasure {4}, specificity: {5}, Smeasure {6}, Emeasure {7}, MAE: {8}".format(dice,jac,precision,recall,f_measure,acc,sm,em,mae))

    def validation_step(self, batch, batch_idx):
        # torch.set_grad_enabled(True)
        self.model.eval()
        
        neigbor, target, _= batch

        bz, nf, nc, h, w = neigbor.shape

        if not self.with_edge:
            samples = self.model(neigbor)
        else:
            samples,_ = self.model(neigbor)

        #print('Samples shape: ',samples.shape) #torch.Size([3, 1, 256, 256])

        samples = samples[self.nFrames//2::self.nFrames]

        #print('Samples shape: ',samples.shape) #torch.Size([1, 1, 256, 256])

        target = target.squeeze(0)
        target = target[self.nFrames//2::self.nFrames]
        #print('Target shape: ', target.shape) #torch.Size([1, 1, 256, 256])
        loss = self.criterion(samples, target)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.val_losses.append(loss.detach())

        #ONLY FOR Binary class !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        samples = torch.sigmoid(samples)
        samples = (samples > 0.5).float()

        filename = "sample_{}.png".format(batch_idx)
        save_image(samples,os.path.join(self.save_path, filename))      
        filename = "target_{}.png".format(batch_idx)
        save_image(target,os.path.join(self.save_path, filename))

        imgs = [samples.cpu(), target.cpu()]
        captions = ["prediction", "ground_truth"]

        # Log both images under one key
        self.logger.log_image(
            key="val_examples", 
            images=imgs, 
            caption=captions
        )

        self.preds.append(samples)
        self.gts.append(target)
        
        return loss

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