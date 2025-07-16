import argparse
from re import S
import time
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
)
import librosa
from pytorch_lightning.loggers import WandbLogger
import wandb  # required, not optional
from dataset import AudioDataset
from definitions import *
from eval_metrics import *

torch.set_float32_matmul_precision("medium")

# --------------------------------------------------------------------------- #
# Monitoring callbacks
class CoefficientMonitor(Callback):
    """Logs α/β/γ values and their gradient L2 norms every training step."""

    def on_train_batch_end(self, trainer, pl_module, *_):
        # Safety check for logger
        if not hasattr(trainer, 'logger') or trainer.logger is None:
            return
        if not hasattr(trainer.logger, 'experiment'):
            return
            
        try:
            run = trainer.logger.experiment

            # Histograms once per logging step
            for name in ("alpha", "beta", "gamma"):
                vals = torch.stack([getattr(l, name).detach().cpu() for l in pl_module.layers])
                run.log({f"hist/{name}": wandb.Histogram(vals.numpy())},
                        step=trainer.global_step)
        except Exception as e:
            print(f"Warning: CoefficientMonitor logging failed: {e}")
            pass  # Don't crash training for logging issues


# class GlobalGradNorm(Callback):
#     """Logs total gradient L2 norm after each backward pass."""

#     def on_after_backward(self, trainer, pl_module):
#         total_sq = 0.0
#         for p in pl_module.parameters():
#             if p.grad is not None:
#                 total_sq += p.grad.data.norm(2).pow(2).item()
#         trainer.logger.experiment.log(  # type: ignore[attr-defined]
#             {"grad/total_norm": total_sq ** 0.5},
#             step=trainer.global_step,
#         )


def _make_loaders(
    dataset: PreprocessedAudioDataset,
    batch_size: int = 32,
    val_split: float = 0.05,
    test_split: float = 0.05,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Deterministic  train / val / test split."""
    n_total = len(dataset)
    n_val   = int(val_split  * n_total)
    n_test  = int(test_split * n_total)
    n_train = n_total - n_val - n_test

    train_set, val_set, test_set = random_split(
        dataset,
        lengths=(n_train, n_val, n_test),
        generator=torch.Generator().manual_seed(42),
    )

    def _loader(ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    return _loader(train_set, True), _loader(val_set, False), _loader(test_set, False)

# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(hparams: dict, run_name=None) -> float:
    """
    Train a DeepAGLA model.
    
    Args:
        hparams: Dictionary with lr, batch_size, weight_decay
        run_name: Optional name for WandB run
        
    Returns:
        SSNR score of the best model validation set
    """
    from deep_agla_new import DeepAGLA  # local import avoids circular deps

    # ---------------- data ---------------
    dataset = PreprocessedAudioDataset(npy_path=DATA_PATH)
    train_loader, val_loader, test_loader = _make_loaders(
        dataset,
        batch_size=hparams["batch_size"],
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        num_workers=NUM_WORKERS,
    )
    # ---------------- model --------------
    model  = DeepAGLA(
        n_layers      = N_LAYERS,
        n_fft         = N_FFT,
        hop           = HOP,
        win_length    = WIN_LEN,
        lr            = hparams["lr"],
        weight_decay  = hparams["weight_decay"],
        initial_params= BEST_INITIAL_COMBINATIONS["overall"],
    )
    
    # Manually add batch_size to the model's hyperparameters for logging
    model.hparams.batch_size = hparams["batch_size"]
    # ---------------- logging ------------
    wandb_logger = WandbLogger(project=PROJECT_NAME, name=run_name)
    
    # Log hyperparameters including batch size
    wandb_logger.experiment.config.update({
        "batch_size": hparams["batch_size"],
        "learning_rate": hparams["lr"], 
        "weight_decay": hparams["weight_decay"],
        "n_layers": N_LAYERS,
        "n_fft": N_FFT,
        "hop_length": HOP,
        "win_length": WIN_LEN,
        "num_epochs": NUM_EPOCHS,
        "val_split": VAL_SPLIT,
        "test_split": TEST_SPLIT
    })
    
    checkpoint_cb = ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=3)
    lr_cb = LearningRateMonitor(logging_interval="step")
    # ---------------- callbacks ----------
    callbacks = [
        checkpoint_cb,
        lr_cb,
        CoefficientMonitor(),
    ]
    # ---------------- trainer ------------
    trainer = pl.Trainer(
        max_epochs        = NUM_EPOCHS,
        accelerator       = "auto",  # Automatically uses GPU if available
        logger            = wandb_logger,
        callbacks         = callbacks,
        num_sanity_val_steps=0
    )

    # ---------------- fit / test ---------
    try:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
        # Only finish WandB after successfully getting the best model path
        if hasattr(checkpoint_cb, 'best_model_path') and checkpoint_cb.best_model_path is not None:
            best_model_path = checkpoint_cb.best_model_path
            print(f"Training completed. Best model: {best_model_path}")
        else:
            print("Training completed but no best model checkpoint found")
            wandb.finish()
            return float('-inf')
            
    except Exception as e:
        print(f"Training failed: {e}")
        wandb.finish()
        return float('-inf')  # Return worst possible score for failed runs
    
    # Clean up trainer and model references to free memory
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load the best model and compute validation SSNR
    try:
        print(f"Loading best model from: {best_model_path}")
        best_model = DeepAGLA.load_from_checkpoint(best_model_path)
        best_model.eval()  # Set to evaluation mode
        best_model.freeze()  # Freeze the model parameters
        
        # Get the device from the model or use auto-detection
        device = next(best_model.parameters()).device
        if device.type == 'cpu' and torch.cuda.is_available():
            device = torch.device('cuda')
            best_model = best_model.to(device)
        
        # Compute validation SSNR for hyperparameter optimization
        print("Computing validation SSNR...")
        val_ssnr = compute_validation_ssnr(best_model, val_loader, device)
        print(f"Validation SSNR: {val_ssnr:.3f} dB")
        
        # Now finish WandB after successful completion
        wandb.finish()
        
    except Exception as e:
        print(f"Failed to load best model or compute validation SSNR: {e}")
        wandb.finish()
        return float('-inf')
    
    # Clean up all references to avoid memory leaks in hyperparameter optimization
    del best_model
    import gc
    gc.collect()
    
    return val_ssnr  # Return SSNR for hyperparameter optimization