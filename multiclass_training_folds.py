'''Script for training Vivim multiclass on a k-fold Cross Validation'''
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
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.segmentation import DiceScore
import seaborn as sns
from sklearn.metrics import confusion_matrix

from torchmetrics.segmentation import DiceScore


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

from complements.OTU_dataset import *
from torch.utils.data import DataLoader, Subset
from loss import *
# from models2.refinenet import RefineNet
from torchvision.utils import save_image
import wandb
import misc2
from complements.create_train_data_multiclass import *

args = cfg.parse_args()

output_dir = 'logs_multiclass_final_training'
version_name='multiclass segmentation final training'


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


class MulticlassMetricsTracker:

    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.reset()
       
    def reset(self):
       
        self.dice_per_class = [[] for _ in range(self.num_classes)]
        self.jaccard_per_class = [[] for _ in range(self.num_classes)]
        self.precision_per_class = [[] for _ in range(self.num_classes)]
        self.recall_per_class = [[] for _ in range(self.num_classes)]
        self.f_measure_per_class = [[] for _ in range(self.num_classes)]
        self.specificity_per_class = [[] for _ in range(self.num_classes)]
       
        
        self.class_counts = [0 for _ in range(self.num_classes)]
       
    def update(self, pred, gt):

        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
        if torch.is_tensor(gt):
            gt = gt.detach().cpu().numpy()
       
        
        if pred.ndim == 4:  # [B, C, H, W]
            B, C, H, W = pred.shape
            pred = pred.reshape(-1, C, H, W)
            gt = gt.reshape(-1, H, W)
           
        
        for sample_idx in range(len(pred)):
            sample_pred = pred[sample_idx]  # [C, H, W]
            sample_gt = gt[sample_idx]      # [H, W]
           
            
            for class_idx in range(self.num_classes):
                
                if np.any(sample_gt == class_idx):
                    self.class_counts[class_idx] += 1
                   
                    
                    pred_binary = (sample_pred.argmax(axis=0) == class_idx).astype(np.int32)
                    gt_binary = (sample_gt == class_idx).astype(np.int32)
                   
                    
                    dice = misc2.dice(pred_binary, gt_binary)
                    jaccard = misc2.jaccard(pred_binary, gt_binary)
                    precision = misc2.precision(pred_binary, gt_binary)
                    recall = misc2.recall(pred_binary, gt_binary)
                    f_measure = misc2.fscore(pred_binary, gt_binary)
                    specificity = misc2.specificity(pred_binary, gt_binary)
                   
                    
                    self.dice_per_class[class_idx].append(dice)
                    self.jaccard_per_class[class_idx].append(jaccard)
                    self.precision_per_class[class_idx].append(precision)
                    self.recall_per_class[class_idx].append(recall)
                    self.f_measure_per_class[class_idx].append(f_measure)
                    self.specificity_per_class[class_idx].append(specificity)
   
    def get_results(self):
        
        dice_values = [np.mean(self.dice_per_class[i]) if self.class_counts[i] > 0 else None
                     for i in range(self.num_classes)]
       
        jaccard_values = [np.mean(self.jaccard_per_class[i]) if self.class_counts[i] > 0 else None
                        for i in range(self.num_classes)]
       
        precision_values = [np.mean(self.precision_per_class[i]) if self.class_counts[i] > 0 else None
                          for i in range(self.num_classes)]
       
        recall_values = [np.mean(self.recall_per_class[i]) if self.class_counts[i] > 0 else None
                       for i in range(self.num_classes)]
       
        f_measure_values = [np.mean(self.f_measure_per_class[i]) if self.class_counts[i] > 0 else None
                          for i in range(self.num_classes)]
       
        specificity_values = [np.mean(self.specificity_per_class[i]) if self.class_counts[i] > 0 else None
                            for i in range(self.num_classes)]
       
        
        def safe_mean(values):
            valid_values = [v for v in values if v is not None]
            return np.mean(valid_values) if valid_values else 0.0
       
        
        results = {
            'dice': {
                'per_class': dice_values,
                'mean': safe_mean(dice_values)
            },
            'jaccard': {
                'per_class': jaccard_values,
                'mean': safe_mean(jaccard_values)
            },
            'precision': {
                'per_class': precision_values,
                'mean': safe_mean(precision_values)
            },
            'recall': {
                'per_class': recall_values,
                'mean': safe_mean(recall_values)
            },
            'f_measure': {
                'per_class': f_measure_values,
                'mean': safe_mean(f_measure_values)
            },
            'specificity': {
                'per_class': specificity_values,
                'mean': safe_mean(specificity_values)
            },
            'class_counts': self.class_counts
        }
       
        return results

def dice_loss(logits, targets, num_classes, smooth=1e-6):
    """
    Dice loss for multiclass segmentation
    
    Args:
        logits: [N, C, H, W] raw scores for C classes
        targets: [N, H, W] integer labels in {0,1,...,C-1}
        num_classes: Number of classes
        smooth: Smoothing factor
        
    Returns: scalar loss
    """
    N, C, H, W = logits.shape
    
    # Convert predictions to probabilities
    probs = F.softmax(logits, dim=1)
    
    # One-hot encode targets
    targets_onehot = F.one_hot(targets.long(), num_classes=C).permute(0, 3, 1, 2).float()
    
    # Calculate Dice for each class
    dice_scores = []
    for c in range(C):
        pred_c = probs[:, c, ...]
        targ_c = targets_onehot[:, c, ...]
        
        # Calculate intersection and union
        intersection = (pred_c * targ_c).sum(dim=(1, 2))
        union = pred_c.sum(dim=(1, 2)) + targ_c.sum(dim=(1, 2))
        
        # Dice formula
        dice_c = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(1 - dice_c.mean())  # Convert to loss
    
    return sum(dice_scores) / C  # Average across classes

def tversky_loss(logits, targets, num_classes, alpha=0.3, beta=0.7, smooth=1e-6):
    """
    Tversky loss with beta > alpha to prioritize recall
    
    Args:
        logits: [N, C, H, W] raw scores for C classes
        targets: [N, H, W] integer labels in {0,1,...,C-1}
        num_classes: Number of classes
        alpha: Weight for false positives
        beta: Weight for false negatives (higher value prioritizes recall)
        smooth: Smoothing factor
        
    Returns: scalar loss
    """
    N, C, H, W = logits.shape
    
    # Convert predictions to probabilities
    probs = F.softmax(logits, dim=1)
    
    # One-hot encode targets
    targets_onehot = F.one_hot(targets.long(), num_classes=C).permute(0, 3, 1, 2).float()
    
    # Calculate Tversky for each class
    tversky_scores = []
    for c in range(C):
        pred_c = probs[:, c, ...]
        targ_c = targets_onehot[:, c, ...]
        
        # Calculate true positives, false positives, false negatives
        tp = (pred_c * targ_c).sum(dim=(1, 2))
        fp = (pred_c * (1 - targ_c)).sum(dim=(1, 2))
        fn = ((1 - pred_c) * targ_c).sum(dim=(1, 2))
        
        # Tversky formula
        tversky_c = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        tversky_scores.append(1 - tversky_c.mean())  # Convert to loss
    
    return sum(tversky_scores) / C  # Average across classes

def boundary_aware_loss(logits, targets, num_classes, weight=0.5):
    """
    Boundary-aware loss that gives more weight to edge pixels
    
    Args:
        logits: [N, C, H, W] raw scores for C classes
        targets: [N, H, W] integer labels in {0,1,...,C-1}
        num_classes: Number of classes
        weight: Weight for boundary loss
        
    Returns: scalar loss
    """
    import torch.nn.functional as F
    
    N, C, H, W = logits.shape
    
    # Convert targets to one-hot
    targets_onehot = F.one_hot(targets.long(), num_classes=C).permute(0, 3, 1, 2).float()
    
    # Calculate boundary masks using simple gradient operation
    boundary_masks = []
    for c in range(C):
        targ_c = targets_onehot[:, c, ...]
        
        # Calculate gradient in X and Y directions
        grad_x = torch.abs(targ_c[:, :, 1:] - targ_c[:, :, :-1])
        grad_y = torch.abs(targ_c[:, 1:, :] - targ_c[:, :-1, :])
        
        # Pad gradients to match original size
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        
        # Combine to get boundary mask
        boundary = torch.clamp(grad_x + grad_y, 0, 1)
        boundary_masks.append(boundary)
    
    # Stack boundary masks
    boundary_masks = torch.stack(boundary_masks, dim=1)
    
    # Calculate weighted BCE loss with boundary emphasis
    probs = F.softmax(logits, dim=1)
    interior_loss = F.cross_entropy(logits, targets)
    
    # For boundary pixels, calculate weighted BCE
    boundary_loss = 0
    for c in range(C):
        pred_c = probs[:, c, ...]
        targ_c = targets_onehot[:, c, ...]
        bound_c = boundary_masks[:, c, ...]
        
        # Calculate BCE weighted by boundary mask
        bce = -targ_c * torch.log(pred_c + 1e-6) - (1 - targ_c) * torch.log(1 - pred_c + 1e-6)
        boundary_loss += (bound_c * bce).mean()
    
    boundary_loss /= C
    
    # Combined loss
    return interior_loss + weight * boundary_loss

def combined_focal_dice_loss(logits, targets, num_classes, gamma=3.0, alpha=None, dice_weight=0.5):
    """
    Combined focal and dice loss for improved segmentation
    
    Args:
        logits: [N, C, H, W] raw scores for C classes
        targets: [N, H, W] integer labels in {0,1,...,C-1}
        num_classes: Number of classes
        gamma: Focusing parameter for focal loss
        alpha: Class weights for focal loss (None for automatic)
        dice_weight: Weight for dice loss component
        
    Returns: scalar loss
    """
    # Calculate focal loss with explicit alpha (if provided)
    focal = class_balanced_focal_loss(logits, targets, num_classes, gamma=gamma, alpha=alpha)
    
    # Calculate dice loss
    dice = dice_loss(logits, targets, num_classes)
    
    # Combined weighted loss (adjust weights as needed)
    return (1.0 - dice_weight) * focal + dice_weight * dice

def recall_focused_loss(logits, targets, num_classes, gamma=2.0):
    """
    Loss function specifically designed to improve recall
    
    Args:
        logits: [N, C, H, W] raw scores for C classes
        targets: [N, H, W] integer labels in {0,1,...,C-1}
        num_classes: Number of classes
        gamma: Focusing parameter
        
    Returns: scalar loss
    """
    # Strong class weighting to prioritize minority classes
    alpha = [0.05, 0.475, 0.475]  # [background, solid, non-solid]
    
    # Use Tversky loss with beta>alpha to prioritize recall
    tversky = tversky_loss(logits, targets, num_classes, alpha=0.3, beta=0.7)
    
    # Calculate focal loss with explicit alpha
    focal = class_balanced_focal_loss(logits, targets, num_classes, gamma=gamma, alpha=alpha)
    
    # Weighted combination prioritizing tversky loss for better recall
    return 0.4 * focal + 0.6 * tversky

def class_balanced_focal_loss(logits, targets, num_classes, gamma=2.0, alpha=None):
    """
    Class-balanced focal loss
    
    Args:
        logits: [N, C, H, W] raw scores for C classes
        targets: [N, H, W] integer labels in {0,1,…,C-1}
        num_classes: Number of classes
        gamma: Focusing parameter
        alpha: Optional class weights
        
    Returns: scalar loss
    """
    N, C, H, W = logits.shape

    #print(logits.shape) #[16, 3, 256, 256]
    #print(targets.shape) #[16, 256, 256]
    
    
    # If no specific class weights are provided, calculate them based on inverse frequency
    if alpha is None:
        # Count class frequencies
        class_counts = []
        for c in range(num_classes):
            class_counts.append((targets == c).sum().float() + 1e-6)  # Add small eps to avoid div by 0
        
            #print("Class counts:", [count.item() for count in class_counts])

        # Inverse frequency as weight
        total_pixels = N * H * W
        class_weights = total_pixels / (num_classes * torch.tensor(class_counts).to(logits.device))
        
        # Normalize weights to sum to 1
        alpha = class_weights / class_weights.sum()

        #print("Automatically calculated alpha values:", [weight.item() for weight in alpha])
    
    else:
        alpha = torch.tensor(alpha).to(logits.device)
    
    # Compute focal loss with class balancing
    probs = F.softmax(logits, dim=1)
    targets_onehot = F.one_hot(targets.long(), num_classes=C).permute(0, 3, 1, 2).float()
    
    focal_losses = []
    for c in range(C):
        # Extract class probability and target
        p_c = probs[:, c, ...]
        t_c = targets_onehot[:, c, ...]
        
        # Focal weight: (1-p)^gamma for positive examples, p^gamma for negative examples
        focal_weight = t_c * (1 - p_c)**gamma + (1 - t_c) * p_c**gamma
        
        # Binary cross entropy with focus weight
        bce = -t_c * torch.log(p_c + 1e-6) - (1 - t_c) * torch.log(1 - p_c + 1e-6)
        
        # Apply focal weight and class balancing weight
        class_loss = (alpha[c] * focal_weight * bce).mean()
        focal_losses.append(class_loss)
    
    return sum(focal_losses)


def get_loader(args, mode, num_fold):
        input_root = 'Multiclass_Folds/fold_' + str(num_fold)

        train_path = os.path.join(input_root, 'train')
        train_output_root = 'Multiclass_TrainData_fold_' + str(num_fold)
        train_output_root = os.path.join(train_output_root, 'train')
        gather_multiclass_frames(Path(train_path), Path(train_output_root))
        full_train_dataset = MainDataset(root=train_output_root, trainsize=args.image_size, clip_len=args.clip_length, max_num=args.max_numerosity)

        val_path = os.path.join(input_root, 'val')
        val_output_root = 'Multiclass_TrainData_fold_' + str(num_fold)
        val_output_root = os.path.join(val_output_root, 'val')
        gather_multiclass_frames(Path(val_path), Path(val_output_root))
        full_val_dataset = MainDataset(root=val_output_root, trainsize=args.image_size, clip_len=args.clip_length, max_num=args.max_numerosity)

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

        self.all_targets = []
        self.all_preds = []
        
        self.nFrames = self.params.clip_length
        self.upscale_factor = 1
        self.data_augmentation = True

        self.criterion = JointEdgeSegLoss(classes=self.params.num_classes) if self.with_edge else recall_focused_loss

        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceScore(num_classes=self.num_classes, average='macro')
        
        self.train_jaccard = MulticlassJaccardIndex(num_classes=self.num_classes, average="micro")
        self.val_jaccard   = MulticlassJaccardIndex(num_classes=self.num_classes, average="micro")
        self.val_dice      = DiceScore(num_classes=self.num_classes, average="macro", )

        self.model = Vivim(with_edge=self.with_edge, out_chans=self.params.num_classes)

        self.multiclass_metrics = MulticlassMetricsTracker(num_classes=self.num_classes)


        self.save_hyperparameters()

        self.val_losses = []
    
    def configure_optimizers(self):
        #We filter the parameters that require gradients, avoid updating frozen parts
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.initlr, betas=[0.9,0.999], weight_decay=self.weight_decay)#,weight_decay=self.weight_decay)

        #added gradient clipping to help with convergence:
        for param_groups in optimizer.param_groups:
            param_groups['clip_grad_norm'] = 1.0

        # optimizer = Lion(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.initlr,betas=[0.9,0.99],weight_decay=0)

        #Added a scheduler to improve the model convergence  

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
        dice = misc2.dice(pred, gt)#, nan_for_nonexisting=True)
        specificity = misc2.specificity(pred, gt)
        jaccard = misc2.jaccard(pred, gt)
        precision = misc2.precision(pred, gt)
        recall = misc2.recall(pred, gt)
        f_measure = misc2.fscore(pred, gt)
        return dice, specificity, precision, recall, f_measure, jaccard
    
    def training_step(self, batch, batch_idx):
        self.model.train()
        neighbor, target, _ = batch
        logits = self.model(neighbor)  #  could be [B, C, H, W] or [B, T, C, H, W]
        #print(logits.shape) #?torch.Size([3, 3, 256, 256])

        if logits.ndim == 5:
            B, T, C, H, W = logits.shape
            # reshape logits [B*T, C, H, W]
            logits = logits.view(B * T, C, H, W)
            # replicate target across time  [B*T, H, W]
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
        loss_combo = recall_focused_loss(logits, target, num_classes=self.num_classes)
        # compute Dice loss (1 - Dice score)
        #dice_score = self.dice_loss(logits.softmax(dim=1), target)
        loss = loss_combo #+ (1 - dice_score)

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/jaccard", self.train_jaccard(logits.softmax(dim=1), target), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        neighbor, target, _ = batch
        logits = self.model(neighbor)  #  could be [B, C, H, W] or [B, T, C, H, W]
        #print(logits.shape) #?torch.Size([3, 3, 256, 256])

        if logits.ndim == 5:
            B, T, C, H, W = logits.shape
            # reshape logits  [B*T, C, H, W]
            logits = logits.view(B * T, C, H, W)
            # replicate target across time  [B*T, H, W]
            #target = target.unsqueeze(1).repeat(1, T, 1, 1).view(B * T, H, W)
        if target.ndim == 5:
            B, T, C, H, W = target.shape
            # assume one-hot: [B, T, C, H, W]
            # convert to class‐index mask: [B, T, H, W]
            target = target.argmax(dim=2)
        #print(target.shape) #[1, 3, 256, 256]
        target = target.view(B*T, H, W)

        #print(target.shape) #torch.Size([5, 256, 256])
        #print(logits.shape) #torch.Size([5, 3, 256, 256])

        loss = recall_focused_loss(logits, target, num_classes=self.num_classes)
        self.val_losses.append(loss)

        pred_classes = logits.detach().cpu().argmax(dim=1).numpy().flatten()
        target_classes = target.detach().cpu().numpy().flatten()


        self.all_preds.extend(pred_classes)
        self.all_targets.extend(target_classes)

        preds = logits.argmax(dim=1)  # [B*T or B, H, W]
        '''
        # If you want to log samples during validation:
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
        '''
        
        self.val_jaccard.update(preds, target)
        self.val_dice.update(preds, target)
        self.multiclass_metrics.update(logits, target)

        
        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_losses).mean()
        jaccard = self.val_jaccard.compute()
        dice = self.val_dice.compute()

        self.log('val/loss', avg_loss)
        self.log('val/jacc', jaccard)
        self.log('val/dice', dice)

        results = self.multiclass_metrics.get_results()

        self.log("Dice", results['dice']['mean'])
        self.log("Jaccard", results['jaccard']['mean'])
        self.log("Precision", results['precision']['mean'])
        self.log("Recall", results['recall']['mean'])
        self.log("Fmeasure", results['f_measure']['mean'])
        self.log("Specificity", results['specificity']['mean'])
       
       
        class_names = ["background", "solid", "non_solid"]
        for i in range(self.num_classes):
            
            if results['class_counts'][i] > 0:
                self.log(f"Dice_class_{class_names[i]}", results['dice']['per_class'][i])
                self.log(f"Jaccard_class_{class_names[i]}", results['jaccard']['per_class'][i])
                self.log(f"Precision_class_{class_names[i]}", results['precision']['per_class'][i])
                self.log(f"Recall_class_{class_names[i]}", results['recall']['per_class'][i])
       
        
        print(f"Class counts during validation: {results['class_counts']}")
       
        
        print(f"Val: Dice {results['dice']['mean']:.4f}, "
              f"Jaccard {results['jaccard']['mean']:.4f}, "
              f"Precision {results['precision']['mean']:.4f}, "
              f"Recall {results['recall']['mean']:.4f}, "
              f"Fmeasure {results['f_measure']['mean']:.4f}, "
              f"Specificity {results['specificity']['mean']:.4f}")
       
        
        for i, name in enumerate(class_names):
            if results['class_counts'][i] > 0:
                print(f"  Class {name}: Dice {results['dice']['per_class'][i]:.4f}, "
                      f"Jaccard {results['jaccard']['per_class'][i]:.4f}")
        
        conf_matrix = confusion_matrix(self.all_targets, self.all_preds, labels=range(self.num_classes))

        
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        class_names = ["background", "solid", "non-solid"]

        
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[0])
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        axes[0].set_title(f'Raw Confusion Matrix')

        row_sums = conf_matrix.sum(axis=1)
        row_norm_conf = conf_matrix / row_sums[:, np.newaxis]
        sns.heatmap(row_norm_conf, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[1])
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
        axes[1].set_title(f'Row-Normalized Confusion Matrix (Recall)')

        col_sums = conf_matrix.sum(axis=0)
        col_norm_conf = conf_matrix / col_sums[np.newaxis, :]
        sns.heatmap(col_norm_conf, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[2])
        axes[2].set_ylabel('True Label')
        axes[2].set_xlabel('Predicted Label')
        axes[2].set_title(f'Column-Normalized Confusion Matrix (Precision)')

        plt.tight_layout()

        wandb.log({
            'val/confusion_matrix_raw': wandb.Image(axes[0].figure),
            'val/confusion_matrix_row_norm': wandb.Image(axes[1].figure),
            'val/confusion_matrix_col_norm': wandb.Image(axes[2].figure),
        })
       
        self.val_losses.clear()
        self.val_jaccard.reset()
        self.val_dice.reset()
        self.multiclass_metrics.reset()
        self.preds.clear()
        self.gts.clear()
        self.all_targets.clear()
        self.all_preds.clear()
        

    def on_train_epoch_end(self):
        '''Log learning rate after each epoch'''
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=False, on_epoch=True)


    def train_dataloader(self):
        train_loader = get_loader(self.params, mode='training', num_fold=self.fold_number)
        return train_loader
    
    def val_dataloader(self):
        val_loader = get_loader(self.params, mode='validation', num_fold=self.fold_number)
        return val_loader 

    def _compute_loss(self, neigbor, target, rest):   #Not used, at the moment we average over each clip frames (currently the best performing heuristc)
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
        name=f'final multiclass segmentation fold {fold}',
        config=vars(args)
       )

        logger = WandbLogger(save_dir='.',
                        name=f'final multiclass segmentation fold {fold}',
                        project='Vivim_multiclass_segmentation',
                        #log_model=True
                        )
        pl.seed_everything(args.seed, workers=True)
        
        print('Start training for fold number ', fold)

        #resume_checkpoint_path = 'Logs/ultra-epoch14-fold-0.ckpt'
        resume_checkpoint_path = args.resume_path

        model = CoolSystem(args, fold_number=fold)

        checkpoint_callback = ModelCheckpoint(
        monitor='val/dice',
        dirpath=output_dir,
        filename='ultra-epoch{epoch:02d}-macroDice-{macro/Dice:.4f}-fold{fold}',
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
        #val_path=r'logs_multiclass/ultra-epoch54.fold1.ckpt'
        #trainer.validate(model,ckpt_path=val_path)
        wandb.finish()
    
if __name__ == '__main__':
    main()