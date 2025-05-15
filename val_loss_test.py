import torch
import matplotlib.pyplot as plt
import numpy as np
import wandb
from collections import defaultdict
from pathlib import Path
import seaborn as sns

class ValidationLossDiagnostics:
    """Helper class to diagnose validation loss issues"""
    
    def __init__(self, model, val_loader, output_dir='diagnostics'):
        """
            model: Your PyTorch Lightning model
            val_loader: Validation data loader
            output_dir: Directory to save diagnostic plots
        """
        self.model = model
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def analyze_predictions(self, num_batches=5):
        """Analyze prediction distributions and compare to targets"""
        self.model.eval()
        pred_stats = defaultdict(list)
        target_stats = defaultdict(list)
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= num_batches:
                    break
                    
                neigbor, target, _ = batch
                
                if not self.model.with_edge:
                    samples = self.model(neigbor.cuda())
                else:
                    samples, _ = self.model(neigbor.cuda())
                
                # Get center frame
                samples = samples[self.model.nFrames//2::self.model.nFrames]
                target = target.squeeze(0)
                target = target[self.model.nFrames//2::self.model.nFrames].cuda()
                
                # Calculate sigmoid of predictions
                pred_probs = torch.sigmoid(samples)
                
                # Collect statistics
                pred_stats['mean'].append(pred_probs.mean().item())
                pred_stats['std'].append(pred_probs.std().item())
                pred_stats['min'].append(pred_probs.min().item())
                pred_stats['max'].append(pred_probs.max().item())
                
                target_stats['mean'].append(target.mean().item())
                target_stats['std'].append(target.std().item())
                target_stats['foreground_ratio'].append((target > 0.5).float().mean().item())
                
                # Analyze specific batch further
                if i == 0:
                    self._plot_histogram(pred_probs.flatten().cpu().numpy(), 
                                        target.flatten().cpu().numpy(),
                                        "prediction_distribution.png")
                    
                    self._plot_calibration(pred_probs.cpu().numpy(), 
                                          target.cpu().numpy(), 
                                          "calibration_curve.png")
                    
                    # Save example images
                    self._save_example_predictions(pred_probs, target, samples, i)
        
        # Print statistics
        print("\n=== Prediction Statistics ===")
        for key, values in pred_stats.items():
            print(f"Pred {key}: {np.mean(values):.4f} ± {np.std(values):.4f}")
            
        print("\n=== Target Statistics ===")
        for key, values in target_stats.items():
            print(f"Target {key}: {np.mean(values):.4f} ± {np.std(values):.4f}")
            
        return pred_stats, target_stats
    
    def analyze_loss_landscape(self):
        """Analyze loss landscape by slightly perturbing model parameters"""
        self.model.eval()
        original_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Get baseline loss
        baseline_loss = self._compute_validation_loss()
        print(f"Baseline validation loss: {baseline_loss:.4f}")
        
        # Analyze sensitivity to parameter perturbations
        perturbation_results = []
        
        for eps in [0.001, 0.005, 0.01, 0.05]:
            # Perturb parameters
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    noise = torch.randn_like(param) * eps * torch.abs(param)
                    param.data.add_(noise)
            
            # Compute loss with perturbed parameters
            perturbed_loss = self._compute_validation_loss()
            perturbation_results.append((eps, perturbed_loss))
            print(f"Loss with {eps:.3f} perturbation: {perturbed_loss:.4f}")
            
            # Restore original parameters
            for name, param in self.model.named_parameters():
                if name in original_params:
                    param.data.copy_(original_params[name])
        
        # Plot loss sensitivity
        self._plot_perturbation_sensitivity(perturbation_results)
        
        return perturbation_results
    
    def analyze_gradient_flow(self):
        """Analyze gradient magnitudes during backpropagation"""
        self.model.train()  # Need to be in train mode for gradients
        
        # Get a single batch
        batch = next(iter(self.val_loader))
        neigbor, target, _ = batch
        
        # Forward pass
        if not self.model.with_edge:
            pred = self.model(neigbor.cuda())
        else:
            pred, e0 = self.model(neigbor.cuda())
        
        # Center frame
        samples = pred[self.model.nFrames//2::self.model.nFrames]
        target = target.squeeze(0)
        target = target[self.model.nFrames//2::self.model.nFrames].cuda()
        
        # Compute loss
        loss = self.model.criterion(samples, target)
        
        # Backward
        loss.backward()
        
        # Collect gradient info
        grad_info = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_info[name] = {
                    'mean': param.grad.abs().mean().item(),
                    'std': param.grad.abs().std().item(),
                    'max': param.grad.abs().max().item()
                }
        
        # Plot gradient flow
        self._plot_gradient_flow(grad_info)
        
        return grad_info
    
    def _compute_validation_loss(self):
        """Compute average validation loss"""
        self.model.eval()
        total_loss = 0
        count = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                neigbor, target, edge_gt = batch
                
                if not self.model.with_edge:
                    samples = self.model(neigbor.cuda())
                    target = target.squeeze(0)
                    target = target[self.model.nFrames//2::self.model.nFrames].cuda()
                    loss = self.model.criterion(samples[self.model.nFrames//2::self.model.nFrames], target)
                else:
                    samples, e0 = self.model(neigbor.cuda())
                    target = target.squeeze(0).cuda()
                    edge_gt = edge_gt.squeeze(0).cuda()
                    target = target[self.model.nFrames//2::self.model.nFrames]
                    edge_gt = edge_gt[self.model.nFrames//2::self.model.nFrames]
                    loss = self.model.criterion(
                        (samples[self.model.nFrames//2::self.model.nFrames], e0[self.model.nFrames//2::self.model.nFrames]),
                        (target, edge_gt)
                    )
                
                total_loss += loss.item()
                count += 1
                
        return total_loss / (count if count > 0 else 1)
    
    def _plot_histogram(self, predictions, targets, filename):
        """Plot histogram of predictions vs targets"""
        plt.figure(figsize=(10, 6))
        
        plt.hist(predictions, bins=50, alpha=0.5, label='Predictions')
        plt.hist(targets, bins=2, alpha=0.5, label='Targets')
        
        plt.title('Distribution of Predictions vs Targets')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / filename)
        plt.close()
    
    def _plot_calibration(self, pred_probs, targets, filename):
        """Plot calibration curve (reliability diagram)"""
        # Flatten arrays
        pred_probs = pred_probs.flatten()
        targets = targets.flatten()
        
        # Create bins and compute calibration
        bins = np.linspace(0, 1, 11)  # 10 bins
        bin_indices = np.digitize(pred_probs, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
        
        bin_sums = np.bincount(bin_indices, weights=targets, minlength=len(bins) - 1)
        bin_counts = np.bincount(bin_indices, minlength=len(bins) - 1)
        bin_means = np.zeros(len(bins) - 1)
        
        nonzero_mask = bin_counts > 0
        bin_means[nonzero_mask] = bin_sums[nonzero_mask] / bin_counts[nonzero_mask]
        
        # Plot
        plt.figure(figsize=(8, 8))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        
        bin_centers = (bins[:-1] + bins[1:]) / 2
        plt.plot(bin_centers, bin_means, 'o-', label='Model calibration')
        
        plt.title('Calibration Curve')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / filename)
        plt.close()
    
    def _save_example_predictions(self, pred_probs, targets, raw_preds, batch_idx):
        """Save example prediction images"""
        # Convert to numpy for easier handling
        pred_np = pred_probs.cpu().numpy()
        targets_np = targets.cpu().numpy()
        raw_preds_np = raw_preds.cpu().numpy()
        
        # Save up to 3 examples
        for i in range(min(3, pred_np.shape[0])):
            plt.figure(figsize=(15, 5))
            
            # Original prediction logits
            plt.subplot(1, 4, 1)
            plt.imshow(raw_preds_np[i, 0], cmap='viridis')
            plt.title(f'Raw logits (min={raw_preds_np[i, 0].min():.2f}, max={raw_preds_np[i, 0].max():.2f})')
            plt.colorbar()
            
            # Prediction probabilities
            plt.subplot(1, 4, 2)
            plt.imshow(pred_np[i, 0], cmap='viridis', vmin=0, vmax=1)
            plt.title(f'Prediction prob (mean={pred_np[i, 0].mean():.2f})')
            plt.colorbar()
            
            # Binary prediction
            plt.subplot(1, 4, 3)
            plt.imshow((pred_np[i, 0] > 0.5).astype(np.float32), cmap='gray')
            plt.title('Binary prediction (>0.5)')
            
            # Target
            plt.subplot(1, 4, 4)
            plt.imshow(targets_np[i, 0], cmap='gray')
            plt.title(f'Target (positive: {(targets_np[i, 0] > 0.5).mean():.2%})')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'example_{batch_idx}_{i}.png')
            plt.close()
    
    def _plot_perturbation_sensitivity(self, perturbation_results):
        """Plot sensitivity to parameter perturbations"""
        eps_values, loss_values = zip(*perturbation_results)
        
        plt.figure(figsize=(8, 6))
        plt.plot(eps_values, loss_values, 'o-')
        plt.axhline(y=loss_values[0], color='r', linestyle='--', label='Baseline loss')
        
        plt.title('Loss Sensitivity to Parameter Perturbations')
        plt.xlabel('Perturbation magnitude')
        plt.ylabel('Validation loss')
        plt.grid(True)
        plt.legend()
        plt.savefig(self.output_dir / 'perturbation_sensitivity.png')
        plt.close()
    
    def _plot_gradient_flow(self, grad_info):
        """Plot gradient flow information"""
        # Sort layers by mean gradient magnitude
        sorted_layers = sorted(grad_info.items(), key=lambda x: x[1]['mean'])
        layer_names = [name.split('.')[-1][:10] + '...' if len(name) > 20 else name for name, _ in sorted_layers]
        mean_grads = [info['mean'] for _, info in sorted_layers]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(layer_names)), mean_grads, color='b', alpha=0.6)
        plt.yticks(range(len(layer_names)), layer_names)
        plt.title('Mean Gradient Magnitudes Across Layers')
        plt.xlabel('Mean Gradient Magnitude')
        plt.ylabel('Layers')
        plt.grid(True, axis='x')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'gradient_flow.png')
        plt.close()


# Example usage:
# diagnostics = ValidationLossDiagnostics(model, val_loader)
# pred_stats, target_stats = diagnostics.analyze_predictions()
# perturbation_results = diagnostics.analyze_loss_landscape()
# grad_info = diagnostics.analyze_gradient_flow()


def add_diagnostics_to_trainer(trainer, model, val_loader):
    """Add diagnostics hooks to a PyTorch Lightning Trainer"""
    
    # Store original validation_epoch_end
    original_val_epoch_end = model.on_validation_epoch_end
    
    def enhanced_val_epoch_end():
        # Call the original method first
        original_val_epoch_end()
        
        # Run diagnostics every 5 epochs
        if trainer.current_epoch % 5 == 0:
            print("\n=== Running validation loss diagnostics ===")
            diagnostics = ValidationLossDiagnostics(
                model, 
                val_loader,
                output_dir=f"{model.save_path}/diagnostics/epoch_{trainer.current_epoch}"
            )
            pred_stats, target_stats = diagnostics.analyze_predictions(num_batches=2)
            
            # Log to wandb if available
            if hasattr(trainer, 'logger') and hasattr(trainer.logger, 'experiment'):
                trainer.logger.experiment.log({
                    f"diagnostics/pred_mean": np.mean(pred_stats['mean']),
                    f"diagnostics/pred_std": np.mean(pred_stats['std']),
                    f"diagnostics/target_mean": np.mean(target_stats['mean']),
                    f"diagnostics/target_fg_ratio": np.mean(target_stats['foreground_ratio']),
                })
    
    # Replace the method
    model.on_validation_epoch_end = enhanced_val_epoch_end
    
    return trainer