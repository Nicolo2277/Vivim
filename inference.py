'Script for Vivim Multiclass Inference'
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import cv2
import wandb
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time

from modeling.vivim import Vivim
from Multiclass_Data import TestDataset
from misc2 import dice, jaccard, precision, recall, fscore, specificity
from complements.create_train_data_multiclass import *

# For visualization
def visualize_prediction(image, pred, gt=None, save_path=None):

    colors = {
        0: [0, 0, 0],       # Background: black
        1: [0, 0, 255],     # Solid: red
        2: [0, 255, 255]    # Non-solid: yellow
    }
    
    # Convert prediction to RGB visualization
    pred_vis = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for cls_id, color in colors.items():
        pred_vis[pred == cls_id] = color
    
    # Create figure
    if gt is not None:
        # Convert ground truth to RGB visualization
        gt_vis = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
        for cls_id, color in colors.items():
            gt_vis[gt == cls_id] = color
            
        # Create a figure with input image, prediction and ground truth
        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(image)
        plt.title('Input Image')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(pred_vis)
        plt.title('Prediction')
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(gt_vis)
        plt.title('Ground Truth')
        plt.axis('off')
    else:
        # Create a figure with input image and prediction
        plt.figure(figsize=(8, 4))
        plt.subplot(121)
        plt.imshow(image)
        plt.title('Input Image')
        plt.axis('off')
        
        plt.subplot(122)
        plt.imshow(pred_vis)
        plt.title('Prediction')
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(conf_matrix, class_names, save_path=None):
    
    '''Plot confusion matrix'''
    
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix,
        display_labels=class_names
    )
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def calculate_metrics(pred, gt):

    num_classes = 3
    class_names = ["background", "solid", "non_solid"]
    metrics = {
        "dice": {},
        "jaccard": {},
        "precision": {},
        "recall": {},
        "f_measure": {},
        "specificity": {}
    }
    
    for cls_id in range(num_classes):
        pred_binary = (pred == cls_id).astype(np.int32)
        gt_binary = (gt == cls_id).astype(np.int32)
        
        # Calculate all metrics using the helper functions
        metrics["dice"][class_names[cls_id]] = dice(pred_binary, gt_binary)
        metrics["jaccard"][class_names[cls_id]] = jaccard(pred_binary, gt_binary)
        metrics["precision"][class_names[cls_id]] = precision(pred_binary, gt_binary)
        metrics["recall"][class_names[cls_id]] = recall(pred_binary, gt_binary)
        metrics["f_measure"][class_names[cls_id]] = fscore(pred_binary, gt_binary)
        metrics["specificity"][class_names[cls_id]] = specificity(pred_binary, gt_binary)
    
    # Calculate mean metrics (excluding background - classes 1 and 2 only)
    for metric_name in metrics:
        metrics[metric_name]["mean"] = np.mean([metrics[metric_name][class_names[i]] for i in range(1, num_classes)])
    
    return metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Vivim Inference Script')
    
    # Model parameters
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--with_edge', type=bool, default=False, help='Use edge detection')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='/shared/tesi-signani-data/dataset-segmentation/raw_dataset/test', help='Path to test data directory')
    parser.add_argument('--image_size', type=int, default=256, help='Image size for testing')
    parser.add_argument('--clip_length', type=int, default=5, help='Clip length for temporal models')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results_multiclass', help='Output directory for results')
    parser.add_argument('--save_vis', type=bool, default=False, help='Save visualization results')
    parser.add_argument('--vis_count', type=int, default=20, help='Number of examples to visualize')
    
    # Device parameters
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (use -1 for CPU)')
    
    # Batch size
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    
    # Wandb parameters
    parser.add_argument('--wandb_project', type=str, default='Vivim_multiclass_segmentation', help='Wandb project name')
    parser.add_argument('--wandb_name', type=str, default='vivim_inference', help='Wandb run name')
    parser.add_argument('--wandb', type=bool, default=True, help='Log to wandb')
    parser.add_argument('-cv_group', type=str, default='Vivim_Inference')
    
    return parser.parse_args()

def prepare_test_data(data_dir, image_size, clip_length, batch_size, max_num=None):
    """
    Prepare test data loader
    
    Args:
        data_dir: Path to test data directory
        image_size: Image size for testing
        clip_length: Clip length for temporal models
        batch_size: Batch size for inference
    
    Returns:
        Test data loader
    """
    # Create test dataset

    output_data_root = 'TestData_multiclass'
    gather_multiclass_frames(Path(data_dir), Path(output_data_root))

    test_dataset = TestDataset(
        root=output_data_root,
        testsize=image_size,
        clip_len=clip_length,
    )
    
    # Create test data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True
    )
    
    return test_loader

def load_model(ckpt_path, with_edge, num_classes, device):
    """
    Load the Vivim model from checkpoint
    
    Args:
        ckpt_path: Path to checkpoint file
        with_edge: Whether to use edge detection
        num_classes: Number of classes
        device: Device to load the model on
    
    Returns:
        Loaded model
    """
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Initialize model
    model = Vivim(with_edge=with_edge, out_chans=num_classes)
    
    # Load model weights
    if 'state_dict' in checkpoint:
        # Lightning checkpoint
        model_state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix from keys if present
        model_state_dict = {k.replace('model.', ''): v for k, v in model_state_dict.items()}
        model.load_state_dict(model_state_dict)
    else:
        # Direct model checkpoint
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model

def run_inference(model, data_loader, device, output_dir, save_vis, vis_count=20, use_wandb=False):
    """
    Run inference on test data
    
    Args:
        model: Loaded model
        data_loader: Test data loader
        device: Device to run inference on
        output_dir: Output directory for results
        save_vis: Whether to save visualization results
        vis_count: Number of examples to visualize
        use_wandb: Whether to log to wandb
    
    Returns:
        Dictionary with metrics if ground truth is available
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_metrics = {
        "dice": {
            "background": [], "solid": [], "non_solid": [], "mean": []
        },
        "jaccard": {
            "background": [], "solid": [], "non_solid": [], "mean": []
        },
        "precision": {
            "background": [], "solid": [], "non_solid": [], "mean": []
        },
        "recall": {
            "background": [], "solid": [], "non_solid": [], "mean": []
        },
        "f_measure": {
            "background": [], "solid": [], "non_solid": [], "mean": []
        },
        "specificity": {
            "background": [], "solid": [], "non_solid": [], "mean": []
        }
    }

    dep_metrics = {
        'inference_times' : [],
    }
    
    
    class_names = ["background", "solid", "non_solid"]
    num_classes = len(class_names)
    
    has_ground_truth = False
    
    
    vis_samples = {'images': [],
                   'preds' : [],
                   'targets': []}
    vis_filenames = []
    
   
    all_preds = []
    all_gts = []

    total_frames = 0
    total_time = 0.0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc="Running inference")):
           
            if len(batch) == 3:  # Image, target, filename
                images, targets, filenames = batch
                has_ground_truth = True
            else:  # Just image and filename
                images, filenames = batch
                has_ground_truth = False
            
          
            images = images.to(device)
            
            batch_size = images.shape[0]
            if images.ndim == 5:
                frames_per_batch = batch_size * images.shape[1]
            else:
                frames_per_batch = batch_size

            start_time = time.time()
           
            if model.with_edge:
                logits, _ = model(images)
            else:
                logits = model(images)
            
            inference_time = time.time() - start_time

            dep_metrics['inference_times'].append(inference_time)

            total_frames += frames_per_batch
            total_time += inference_time
            
            # Handle temporal dimension if present
            if logits.ndim == 5:  # [B, T, C, H, W]
                B, T, C, H, W = logits.shape
                logits = logits.view(B*T, C, H, W)
            
            # Get predictions 
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            
            
            for j in range(preds.shape[0]):
                # Get prediction for this image
                pred = preds[j]
                
                # Get filename for this image
                if isinstance(filenames, list):
                    filename = filenames[j]
                else:
                    filename = filenames[0]
                
                #print(images.shape) #torch.Size([5, 3, 256, 256])
                #print(preds.shape) #torch.Size([5, 256, 256])
                images = images.squeeze()
                img_tensor = images[j].cpu()
                
                # Check tensor dimensions and handle appropriately
                if img_tensor.ndim == 4:  # [T, C, H, W] - video clip
                    # Use the middle frame for visualization
                    middle_frame_idx = img_tensor.shape[0] // 2
                    img = img_tensor[middle_frame_idx].permute(1, 2, 0).numpy()
                elif img_tensor.ndim == 3:  # [C, H, W] - single image
                    img = img_tensor.permute(1, 2, 0).numpy()
                else:
                    # Fallback for unexpected dimensions
                    print(f"Warning: Unexpected tensor dimensions: {img_tensor.shape}")
                    # Create a blank image with correct shape for visualization
                    img = np.zeros((pred.shape[0], pred.shape[1], 3))
                
                # Normalize image for visualization
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # Added small epsilon to avoid division by zero
                
                # If ground truth is available
                if has_ground_truth:
                    # Get ground truth for this image
                    if targets.ndim == 5:  # [B, T, C, H, W]
                        B, T, C, H, W = targets.shape
                        targets = targets.argmax(dim=2)  # Convert one-hot to indices
                        targets = targets.view(B*T, H, W)
                    
                    gt = targets[j].cpu().numpy()
                    
                    all_preds.append(pred.flatten())
                    all_gts.append(gt.flatten())
                    
                    metrics = calculate_metrics(pred, gt)
                    
                    for metric_type in all_metrics:
                        for class_name in metrics[metric_type]:
                            all_metrics[metric_type][class_name].append(metrics[metric_type][class_name])
                    
                    if len(vis_samples['images']) < vis_count:
                        vis_samples['images'].append(img)
                        vis_samples['preds'].append(pred)
                        vis_samples['targets'].append(gt)
                        print(f'Image visualization: {filename}')
                    
                    
                    if save_vis:
                        vis_path = os.path.join(output_dir, f"{Path(filename).stem}_vis.png")
                        visualize_prediction(img, pred, gt, save_path=vis_path)
                else:
                    
                    if len(vis_samples['images']) < vis_count:
                        vis_samples.append((img, pred, None))
                        vis_filenames.append(filename)
                        
                    
                    
                    if save_vis:
                        vis_path = os.path.join(output_dir, f"{Path(filename).stem}_vis.png")
                        visualize_prediction(img, pred, save_path=vis_path)
                
                
                if save_vis:
                    pred_path = os.path.join(output_dir, f"{Path(filename[0]).stem}_pred.png")
                    cv2.imwrite(pred_path, pred.astype(np.uint8))
                    
    fps = total_frames / total_time if total_time > 0 else 0
    
    batch_inference_time = np.array(dep_metrics['inference_times'])

    dep_summary = {
        'fps': fps,
        'avg_inference_time_per_batch': np.mean(batch_inference_time),
        'min_inference_time': np.min(batch_inference_time),
        'max_inference_time': np.max(batch_inference_time),
        'total_frames_processed': total_frames,
        'total_inference_time': inference_time
    }

    print('\nDep Metrics: ')
    print(f'FPS: {fps:.2f}')
    print(f"Average inference time per batch: {dep_summary['avg_inference_time_per_batch'] * 1000:.2f} ms")

    if use_wandb:
        wandb.log({
            'fps': fps,
            'avg_inference_time_ms': dep_summary['avg_inference_time_per_batch'] * 1000,
            'total_inference_time': total_time
        })

    if has_ground_truth and len(all_metrics["dice"]["mean"]) > 0:
        final_metrics = {}

        
        for metric_type in all_metrics:
            final_metrics[metric_type] = {
                "background": np.mean(all_metrics[metric_type]["background"]),
                "solid": np.mean(all_metrics[metric_type]["solid"]),
                "non_solid": np.mean(all_metrics[metric_type]["non_solid"]),
                "mean": np.mean(all_metrics[metric_type]["mean"])
            }
        
        
        all_preds_flat = np.concatenate(all_preds)
        all_gts_flat = np.concatenate(all_gts)
        conf_matrix = confusion_matrix(all_gts_flat, all_preds_flat, labels=list(range(num_classes)))
        
        
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plot_confusion_matrix(conf_matrix, class_names, save_path=cm_path)

        
        print("\nOverall Metrics:")
        print(f"Mean Dice: {final_metrics['dice']['mean']:.4f}")
        print(f"Mean Jaccard: {final_metrics['jaccard']['mean']:.4f}")
        print(f"Mean Precision: {final_metrics['precision']['mean']:.4f}")
        print(f"Mean Recall: {final_metrics['recall']['mean']:.4f}")
        print(f"Mean F-measure: {final_metrics['f_measure']['mean']:.4f}")
        print(f"Mean Specificity: {final_metrics['specificity']['mean']:.4f}")
        
        print("\nPer-class Metrics:")
        for i, class_name in enumerate(class_names):
            print(f"{class_name.capitalize()}:")
            print(f"  Dice: {final_metrics['dice'][class_name]:.4f}")
            print(f"  Jaccard: {final_metrics['jaccard'][class_name]:.4f}")
            print(f"  Precision: {final_metrics['precision'][class_name]:.4f}")
            print(f"  Recall: {final_metrics['recall'][class_name]:.4f}")
            print(f"  F-measure: {final_metrics['f_measure'][class_name]:.4f}")
            print(f"  Specificity: {final_metrics['specificity'][class_name]:.4f}")
        
        
        if use_wandb:
            
            wandb_metrics = {}
            for metric_type in final_metrics:
                for class_name in final_metrics[metric_type]:
                    wandb_metrics[f"{metric_type}_{class_name}"] = final_metrics[metric_type][class_name]
            
            
            wandb.log(wandb_metrics)
            
            cm_fig = plt.figure(figsize=(10, 8))
            disp = ConfusionMatrixDisplay(
                confusion_matrix=conf_matrix,
                display_labels=class_names
            )
            disp.plot(cmap=plt.cm.Blues, values_format='d', ax=cm_fig.gca())
            plt.title('Confusion Matrix')
            plt.tight_layout()
            
            wandb.log({"confusion_matrix": wandb.Image(cm_fig)})
            plt.close(cm_fig)
            
            row_sums = conf_matrix.sum(axis=1)
            # Avoid division by zero
            #row_sums[row_sums == 0] = 1
            
            norm_conf_matrix = conf_matrix.astype('float') / row_sums[:, np.newaxis]
            
            norm_cm_fig = plt.figure(figsize=(10, 8))
            disp = ConfusionMatrixDisplay(
                confusion_matrix=norm_conf_matrix,
                display_labels=class_names
            )
            disp.plot(cmap=plt.cm.Blues, values_format='.2f', ax=norm_cm_fig.gca())
            plt.title('Normalized Row Confusion Matrix')
            plt.tight_layout()
            
            wandb.log({"normalized_confusion_matrix_row": wandb.Image(norm_cm_fig)})
            plt.close(norm_cm_fig)
             
            col_sums = conf_matrix.sum(axis=0)
            norm_conf_matrix = conf_matrix.astype('float') / col_sums[np.newaxis, :]
            
            norm_cm_fig = plt.figure(figsize=(10, 8))
            disp = ConfusionMatrixDisplay(
                confusion_matrix=norm_conf_matrix,
                display_labels=class_names
            )
            disp.plot(cmap=plt.cm.Blues, values_format='.2f', ax=norm_cm_fig.gca())
            plt.title('Normalized Confusion Matrix Columns')
            plt.tight_layout()
            
            wandb.log({"normalized_confusion_matrix_col": wandb.Image(norm_cm_fig)})
            plt.close(norm_cm_fig)

            if len(vis_samples['images']) > 0:

                def visualize_samples(images, preds, targets):
                        """Create grid of images with predictions and ground truth"""
                        
                        num_samples = len(images)
                        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
                        
                        if num_samples == 1:
                            axes = axes.reshape(1, -1)
                            
                        class_colors = [
                            [0, 0, 0],       # background - black
                            [255, 0, 0],     # solid - red
                            [255, 255, 0]    # non-solid - yellow
                        ]
                        
                        for i in range(num_samples):
                            # Original image - images[i] is already a numpy array with shape (H, W, 3)
                            axes[i, 0].imshow(images[i])
                            axes[i, 0].set_title("Original Image")
                            axes[i, 0].axis('off')
                            
                            # Prediction visualization
                            pred_vis = np.zeros((preds[i].shape[0], preds[i].shape[1], 3), dtype=np.uint8)
                            for c in range(len(class_colors)):
                                pred_vis[preds[i] == c] = class_colors[c]
                            axes[i, 1].imshow(pred_vis)
                            axes[i, 1].set_title("Prediction")
                            axes[i, 1].axis('off')
                            
                            # Ground truth
                            gt_vis = np.zeros((targets[i].shape[0], targets[i].shape[1], 3), dtype=np.uint8)
                            for c in range(len(class_colors)):
                                gt_vis[targets[i] == c] = class_colors[c]
                            axes[i, 2].imshow(gt_vis)
                            axes[i, 2].set_title("Ground Truth")
                            axes[i, 2].axis('off')
                        
                        plt.tight_layout()
                        return fig
                
                vis_fig = visualize_samples(
                    vis_samples['images'], 
                    vis_samples['preds'], 
                    vis_samples['targets']
                )
                
                
                wandb.log({'val/sample_predictions': wandb.Image(vis_fig)})
                plt.close(vis_fig)
                
       
        final_metrics["confusion_matrix"] = conf_matrix.tolist()

        final_metrics['DEP'] = dep_summary

        return final_metrics
    
    return None

def main():
    args = parse_args()
    
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")
    

    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args)
        )
    
   
    test_loader = prepare_test_data(
        args.data_dir, 
        args.image_size, 
        args.clip_length, 
        args.batch_size,
    )
    
    
    model = load_model(
        args.ckpt, 
        args.with_edge, 
        args.num_classes, 
        device
    )
    
    metrics = run_inference(
        model, 
        test_loader, 
        device, 
        args.output_dir, 
        args.save_vis,
        args.vis_count,
        args.wandb
    )
    
    
    if metrics:
        import json
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {metrics_path}")
    
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()