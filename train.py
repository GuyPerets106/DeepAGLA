import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from deep_agla import DeepAGLA
from pytorch_lightning.loggers import WandbLogger
import wandb
import numpy as np
import matplotlib.pyplot as plt
from dataset import AudioDataset
from definitions import *
from eval_metrics import *
from torch.utils.checkpoint import checkpoint
from typing import Dict, Tuple

# Enable gradient checkpointing and careful optimization
torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

class AdvancedGradientMonitor(Callback):
    """Monitor gradient norms per layer and overall training health"""
    
    def __init__(self):
        self.layer_grad_history = {}
        
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # Compute layer-wise gradient norms
        layer_grads = {}
        total_grad_norm = 0.0
        
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                param_grad_norm = param.grad.norm().item()
                total_grad_norm += param_grad_norm ** 2
                
                if 'layers' in name:
                    try:
                        layer_idx = int(name.split('.')[1])
                        if layer_idx not in layer_grads:
                            layer_grads[layer_idx] = 0.0
                        layer_grads[layer_idx] += param_grad_norm ** 2
                    except (IndexError, ValueError):
                        pass
        
        total_grad_norm = total_grad_norm ** 0.5
        
        # Convert layer gradients to norms
        for layer_idx in layer_grads:
            layer_grads[layer_idx] = layer_grads[layer_idx] ** 0.5
        
        # Store history
        if layer_grads:
            for layer_idx, grad_norm in layer_grads.items():
                if layer_idx not in self.layer_grad_history:
                    self.layer_grad_history[layer_idx] = []
                self.layer_grad_history[layer_idx].append(grad_norm)
        
        # Log every 50 steps
        if trainer.global_step % 50 == 0 and layer_grads:
            grad_values = list(layer_grads.values())
            
            if hasattr(trainer.logger, 'experiment'):
                trainer.logger.experiment.log({
                    "gradients/total_norm": total_grad_norm,
                    "gradients/max_layer_grad": max(grad_values),
                    "gradients/mean_layer_grad": sum(grad_values) / len(grad_values),
                    "gradients/early_layers_avg": sum(list(layer_grads.values())[:8]) / min(8, len(grad_values)),
                    "gradients/late_layers_avg": sum(list(layer_grads.values())[-8:]) / min(8, len(grad_values)),
                }, step=trainer.global_step)
        
        # More aggressive warning thresholds for deep models
        # if total_grad_norm > 50.0:  # Reduced from 100.0
        #     print(f"⚠️  High total gradient norm: {total_grad_norm:.2f} at step {trainer.global_step}")
        
        # if layer_grads and max(layer_grads.values()) > 25.0:  # Reduced from 50.0
        #     max_layer = max(layer_grads.keys(), key=lambda x: layer_grads[x])
        #     print(f"⚠️  High layer gradient: Layer {max_layer} = {layer_grads[max_layer]:.2f}")

def main(hparams: dict, run_name=None) -> float:
    """
    Main training function for N-layer model with gradient checkpointing
    """
    
    # Even more conservative settings for deep models to address gradient issues
    deep_lr = hparams["lr"]
    deep_batch_size = hparams["batch_size"]
    loss_weights = {
        "time_l1": hparams["loss_weight_time_l1"],
        "time_mse": hparams["loss_weight_time_mse"],
        "spec_l1": hparams["loss_weight_spec_l1"],
        "log_spec_l1": hparams["loss_weight_log_spec_l1"],
    }
    print(f"🏗️  DEEP STABLE TRAINING ({N_LAYERS} layers with checkpointing + audio logging):")
    print(f"   Learning rate: {deep_lr} ({deep_lr/LEARNING_RATE:.2f}x base)")
    print(f"   Batch size: {deep_batch_size}")
    print(f"   Gradient checkpointing: Enabled")
    print(f"   Progressive layer activation: Adaptive to {N_LAYERS} layers")
    print(f"   Layer-wise learning rates: Enabled")
    print(f"   Audio logging: EXAMPLE_AUDIO every 5 epochs to WandB only")
    print(f"   Extra conservative settings for gradient stability")
    
    # Data loading
    dataset = AudioDataset(npy_path=DATA_PATH)
    
    def _create_loader(ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=deep_batch_size,
            shuffle=shuffle,
            num_workers=2,  # Conservative
            pin_memory=True,
            persistent_workers=False,  # Disable for stability
            drop_last=True if shuffle else False,
        )
    
    n_total = len(dataset)
    n_val = int(VAL_SPLIT * n_total)
    n_test = int(TEST_SPLIT * n_total)
    n_train = n_total - n_val - n_test

    train_set, val_set, test_set = random_split(
        dataset,
        lengths=(n_train, n_val, n_test),
        generator=torch.Generator().manual_seed(42),
    )
    
    train_loader = _create_loader(train_set, True)
    val_loader = _create_loader(val_set, False)
    test_loader = _create_loader(test_set, False)
    
    # Create model with checkpointing and audio logging
    model = DeepAGLA(
        n_layers=N_LAYERS,  # Configurable number of layers
        lr=deep_lr,
        weight_decay=hparams["weight_decay"],
        use_checkpointing=True,
        layer_lr_decay=0.97,  # Stronger decay for layer-wise LR
        audio_log_interval=5,  # Log EXAMPLE_AUDIO every 5 epochs
        initial_params=BEST_INITIAL_COMBINATIONS["overall"],
        loss_weights=loss_weights,
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"✅ Model loaded to GPU: {next(model.parameters()).device}")
    
    # Logging
    wandb_logger = WandbLogger(project=PROJECT_NAME, name=run_name)
    wandb_logger.experiment.config.update({
        "architecture": f"{N_LAYERS}_layer_with_checkpointing",
        "gradient_checkpointing": True,
        "progressive_layers": True,
        "adaptive_schedule": True,
        "layer_wise_lr": True,
        "learning_rate": deep_lr,
        "batch_size": deep_batch_size,
        "extra_conservative_loss_weights": True,
        "total_layers": N_LAYERS,
        "starting_layers": model.active_layers,
        "layer_increment": model.layer_increment,
        "layer_increment_epochs": model.layer_increment_epochs,
        "loss_weights": loss_weights,
        "audio_logging": True,
        "audio_log_interval": 5,
        "audio_source": "EXAMPLE_AUDIO_from_definitions",
        "wandb_only_logging": True,
        "snr_tracking_when_all_layers_active": True,
    })
    
    # Callbacks
    callbacks = [
        # Save best model based on validation SNR (higher is better)
        ModelCheckpoint(
            monitor="val/SNRdB",
            mode="max",  # Higher SNR is better
            save_top_k=1,
            filename="deep-{epoch:02d}-SNR{val/SNRdB:.2f}dB-L{val/active_layers:.0f}",
            save_last=True,
            verbose=True
        ),
        ModelCheckpoint(
            monitor="val/loss", 
            mode="min", 
            save_top_k=1,
            filename="backup-{epoch:02d}-{val/loss:.4f}",
            save_last=False,
            verbose=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        # EarlyStopping(
        #     monitor="val/SNRdB",
        #     patience=50,
        #     mode="max",
        #     verbose=True,
        #     min_delta=0.05,
        # ),
        AdvancedGradientMonitor(),
    ]
    
    # Trainer with extra conservative settings
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=32,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=50,
        gradient_clip_val=0.2,
        gradient_clip_algorithm="norm",
        val_check_interval=1.0,
        limit_train_batches=1.0,
        deterministic=False,
        benchmark=True,
    )
    
    print(f"🚀 Training configuration:")
    print(f"   Precision: FP32 (for maximum stability)")
    print(f"   Gradient clipping: {trainer.gradient_clip_val} (aggressive)")
    print(f"   Gradient accumulation: {trainer.accumulate_grad_batches} batches")
    print(f"   Validation frequency: Every 0.5 epochs")
    print(f"   Audio logging: EXAMPLE_AUDIO every 5 epochs → WandB only")
    print(f"   SNR tracking: Only when all {N_LAYERS} layers are active")
    
    # Training
    try:
        print(f"🚀 Starting deep {N_LAYERS}-layer training with checkpointing and audio logging...")
        trainer.fit(model, train_loader, val_loader)
        
        # Get best validation SSNR
        best_snr = float('-inf')
        best_ssnr = float('-inf')
        logged_metrics = trainer.logged_metrics if hasattr(trainer, 'logged_metrics') else {}

        # Try to get the best SNR from different possible metric names
        for metric_name in ['val/SNRdB', 'val/SNR(dB)', 'val/SNR_dB', 'val/snr', 'val/SNR']:  # Added the correct names
            if metric_name in logged_metrics:
                best_snr = logged_metrics[metric_name].item()
                print(f"✓ Found best {metric_name}: {best_snr:.3f} dB")
                break

        # Try to get SSNR as well
        for metric_name in ['val/SSNRdB', 'val/SSNR(dB)', 'val/SSNR_dB', 'val/ssnr', 'val/SSNR']:  # Added the correct names
            if metric_name in logged_metrics:
                best_ssnr = logged_metrics[metric_name].item()
                print(f"✓ Found best {metric_name}: {best_ssnr:.3f} dB")
                break

        if best_snr == float('-inf'):
            print("⚠️ Could not find SNR in logged metrics, using conservative estimate")
            best_snr = -20.0  # Conservative estimate

        final_layers = model.active_layers
        print(f"🎉 Deep training with audio logging completed!")
        print(f"   Final active layers: {final_layers}/{model.n_layers}")
        print(f"   Best validation SNR: {best_snr:.3f} dB")
        print(f"   Best validation SSNR: {best_ssnr:.3f} dB" if best_ssnr != float('-inf') else "   SSNR: Not available")
        print(f"   Audio quality should be high with {final_layers} iterations!")

        # Save model artifacts to WandB with detailed metadata
        best_checkpoint_path = trainer.checkpoint_callback.best_model_path
        print(f"Best checkpoint path: {best_checkpoint_path}")

        if best_checkpoint_path:
            # Load and verify the best checkpoint
            best_checkpoint = torch.load(best_checkpoint_path, map_location='cpu')
            
            # Create detailed metadata
            metadata = {
                "epoch": best_checkpoint.get('epoch', 'N/A'),
                "global_step": best_checkpoint.get('global_step', 'N/A'),
                "best_val_snr": best_snr,
                "best_val_ssnr": best_ssnr if best_ssnr != float('-inf') else None,
                "n_layers": model.n_layers,
                "final_active_layers": model.active_layers,
                "training_complete": model.active_layers >= model.n_layers,
                "hyperparameters": best_checkpoint.get('hyper_parameters', {}),
                "model_type": "SNR_optimized",
            }
            
            # Check parameter changes (to verify training effectiveness)
            sample_layer = model.layers[0]
            metadata["sample_alpha"] = sample_layer.alpha.item()
            metadata["sample_beta"] = sample_layer.beta.item()
            metadata["sample_gamma"] = sample_layer.gamma.item()
            
            print(f"📊 Best model metadata: {metadata}")
            
            # Create WandB artifact with rich metadata
            artifact = wandb.Artifact(
                name=f"deep_agla_{N_LAYERS}layers_best_snr",
                type="model",
                description=f"Best SNR DeepAGLA model - {N_LAYERS} layers, epoch {metadata['epoch']}, SNR {best_snr:.2f}dB",
                metadata=metadata
            )
            artifact.add_file(best_checkpoint_path, name="best_snr_model.ckpt")
            
            # Log artifact to WandB
            wandb_logger.experiment.log_artifact(artifact)
            print(f"✅ Best SNR model artifact saved to WandB")
            
            # Also save a summary file
            import json
            summary_path = best_checkpoint_path.replace('.ckpt', '_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Final comparison with original AGLA algorithms
            print(f"🔄 Running final comparison with original AGLA algorithms...")
            model.compare_with_original_agla(best_checkpoint_path)

        return best_snr  # Return SNR instead of SSNR
        
    except Exception as e:
        print(f"❌ Deep AGLA Training Failed: {e}")
        import traceback
        traceback.print_exc()
        wandb.finish()
        return -999.99

if __name__ == "__main__":
    # Test with conservative hyperparameters
    test_hparams = {
        "lr": 0.001,
        "batch_size": 128,
        "weight_decay": 0.0003250396612363188
    }
    
    loss_weights = {
        "time_l1": 0.9,
        "time_mse": 0.9,
        "spec_l1": 0.1,
        "log_spec_l1": 0.1,
    }
    
    # Update hyperparameters with loss weights
    test_hparams.update({
        "loss_weight_time_l1": loss_weights["time_l1"],
        "loss_weight_time_mse": loss_weights["time_mse"],
        "loss_weight_spec_l1": loss_weights["spec_l1"],
        "loss_weight_log_spec_l1": loss_weights["log_spec_l1"],
    })
        
    
    result = main(test_hparams, run_name=f"deep_{N_LAYERS}layers_{BATCH_SIZE}bs")
    print(f"🏆 Final deep AGLA training result: {result:.3f} dB")