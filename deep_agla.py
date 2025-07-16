from typing import Dict, Tuple, Optional
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.profiler import profile, record_function, ProfilerActivity
import pytorch_lightning as pl
import soundfile as sf
import wandb

from torch import Tensor
import torch.nn.functional as F
from eval_metrics import *
from definitions import *


def preprocess_loaded_audio(
    x: np.ndarray,
    sr: int,
    target_rms: float = 0.15,
    hp_cutoff: float = 30.0,
    n_fft: int = N_FFT,
    hop_length: int = HOP,
) -> tuple[np.ndarray, dict[str, int | str | bool]]:
    """
    Pre‑process a time‑domain signal for Griffin‑Lim (or FGLA).
    """
    # 1) High‑pass filter (2nd‑order Butterworth)
    if hp_cutoff is not None and hp_cutoff > 0:
        b, a = scipy.signal.butter(2, hp_cutoff, "hp", fs=sr)
        x = scipy.signal.lfilter(b, a, x)

    # 2) RMS normalisation
    if target_rms is not None and target_rms > 0:
        rms = np.sqrt(np.mean(x ** 2)) + 1e-12  # avoid /0 on silence
        x = x * (target_rms / rms)

    # 3) Pad to a multiple of hop_length
    hop_length = hop_length or (n_fft // 4)
    pad = (-len(x)) % hop_length
    if pad:
        x = np.pad(x, (0, pad))

    # 4) Cast to float32
    x = x.astype(np.float32, copy=False)
    return x


def _to_tensor(x: np.ndarray, *, device: torch.device | None = None) -> torch.Tensor:
    """Utility: convert numpy array to torch tensor on the desired device."""
    return torch.as_tensor(x, dtype=torch.float32, device=device or "cpu")


# -----------------------------------------------------------------------------
#                          STFT / ISTFT BACKED BY TORCH
# -----------------------------------------------------------------------------

class STFT(nn.Module):
    """
    STFT that yields bit-wise identical results to `librosa.stft`.
    Output shape: (B, 1 + n_fft//2, n_frames) = (B, F, T)
    """
    def __init__(
        self,
        n_fft: int = N_FFT,
        hop: int = HOP,
        win_length: int = WIN_LEN,
        center: bool = True,
        pad_mode: str = "reflect",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.win_length = win_length
        self.center = center
        self.pad_mode = pad_mode

        win = torch.hann_window(win_length, periodic=True, dtype=dtype)
        self.register_buffer("win", win, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T) real-valued
        → X: (B, F, T) complex64/128
        """
        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win_length,
            window=self.win,
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        return X


class ISTFT(nn.Module):
    """
    Inverse STFT numerically aligned with `librosa.istft`.
    Accepts (B, F, T) complex input.
    """
    def __init__(
        self,
        n_fft: int = N_FFT,
        hop: int = HOP,
        win_length: int = WIN_LEN,
        center: bool = True,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.win_length = win_length
        self.center = center

        win = torch.hann_window(win_length, periodic=True, dtype=dtype)
        self.register_buffer("win", win, persistent=False)

    def forward(self, X: torch.Tensor, *, length: int | None = None) -> torch.Tensor:
        """
        X: (B, F, T) complex
        length: original time-domain length (required when center=True)
        → x: (B, length) real
        """
        # torch.istft expects (..., F, T)
        x = torch.istft(
            X,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win_length,
            window=self.win,
            center=self.center,
            normalized=False,
            onesided=True,
            length=length,
        )
        return x

# -------------------------------------------------------------------------
# Learnable AGLA iteration
# -------------------------------------------------------------------------
class AGLALayer(nn.Module):
    """
    One *Accelerated Griffin-Lim* iteration with learnable α, β, γ.
    Formulas follow https://arxiv.org/pdf/2306.12504 paper.
    """
    def __init__(self, 
                 init_alpha: float = 0.1,
                 init_beta:  float = 1.1,
                 init_gamma: float = 0.2) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))
        self.beta  = nn.Parameter(torch.tensor(init_beta,  dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32))

    def forward(
        self,
        c: Tensor,
        t_prev: Tensor,
        d_prev: Tensor,
        s: Tensor,
        proj_pc1: callable,
        proj_pc2: callable
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        c_prev, t_prev, d_prev : complex Tensor
            Spectra from the previous iteration.
        target_mag : Tensor
            Fixed target magnitude |STFT|.
        Returns
        -------
        c, t, d : Tensor
            Updated spectra for the next layer.
        """
        t = (1 - self.gamma) * d_prev + self.gamma * proj_pc1(proj_pc2(c, s))
        c = t + self.alpha * (t - t_prev)
        d = t + self.beta * (t - t_prev)
        return c, t, d


# -------------------------------------------------------------------------
# DeepAGLA model
# -------------------------------------------------------------------------
class DeepAGLA(pl.LightningModule):
    """
    Stack of *L* AGLA layers, trained with a composite
    waveform/spectrogram loss.
    """
    def __init__(
        self,
        n_layers:     int   = N_LAYERS,
        n_fft:        int   = N_FFT,
        hop:          int   = HOP,
        win_length:   int   = WIN_LEN,
        lr:           float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        initial_params: Dict[str, float] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # STFT settings ----------------------------------------------------
        self.n_fft      = int(n_fft)
        self.hop        = int(hop)
        self.win_length = int(win_length)
        self.stft  = STFT()
        self.istft = ISTFT()
        self.n_layers = int(n_layers)
        # Learnable iteration stack
        init_gamma = initial_params.get("gamma", 1.25) if initial_params else 1.25
        self.layers = nn.ModuleList(
            AGLALayer((n - 1) / (n + 2), n / (n + 3), init_gamma)
            for n in range(1, n_layers + 1)
        )
        
        # Use L1 Loss for training
        self.criterion = nn.L1Loss()
        
    def proj_pc1(self, c):
        return self.stft(self.istft(c))

    @staticmethod
    def proj_pc2(c, s):
        return s * torch.exp(1j * torch.angle(c))

    def forward(self, sig: Tensor) -> Tensor:  # noqa: D401
        s = torch.abs(self.stft(sig))  # Constant throughout the iterations
        c0 = s.to(torch.complex64)  # Initialize with the magnitude spectrogram as complex numbers, no phase information
        
        # Match librosa exactly: start with the initial projection
        t_prev = self.proj_pc1(self.proj_pc2(c0, s))  # Initial projection
        d_prev = t_prev.clone()  # Initialize d_prev with the first projection  
        c = t_prev.clone()       # Initialize c with the first projection
        
        # Now iterate through layers, passing previous values correctly
        for layer in self.layers:
            c, t, d = layer(c, t_prev, d_prev, s, self.proj_pc1, self.proj_pc2)
            # Update for next iteration
            t_prev = t.clone()
            d_prev = d.clone()

        predicted_signals = self.istft(c, length=sig.size(1))
        return predicted_signals

    # ---------------------------------------------------------------------
    # Lightning hooks
    # ---------------------------------------------------------------------
    
    def on_train_start(self):
        """Log initial parameter values and device information before training starts."""
        # Log device information
        model_device = next(self.parameters()).device
        self.log("device/model_device", hash(str(model_device)), on_epoch=True, prog_bar=False)
        
        # Log device as string in hyperparameters if we have a logger
        if hasattr(self.logger, 'experiment') and self.logger.experiment is not None:
            self.logger.experiment.config.update({
                "device": str(model_device),
                "device_type": model_device.type,
                "device_index": model_device.index if model_device.index is not None else 0
            })
        
        # Assert all modules are on the same device (CUDA)
        all_devices = set()
        for name, param in self.named_parameters():
            all_devices.add(param.device)
        
        # Check that all parameters are on the same device
        assert len(all_devices) == 1, f"Model parameters are on different devices: {all_devices}"
        
        # Assert that the device is CUDA (if CUDA is available)
        if torch.cuda.is_available():
            assert model_device.type == "cuda", f"Expected model to be on CUDA, but found on {model_device}"
        
        print(f"✓ All model parameters confirmed on device: {model_device}")
        
        # Log initial parameter values (before any training/backward passes)
        self.log("init/alpha_first_layer", self.layers[0].alpha.detach(), on_epoch=True, prog_bar=False)
        self.log("init/beta_first_layer",  self.layers[0].beta.detach(), on_epoch=True, prog_bar=False)
        self.log("init/gamma_first_layer", self.layers[0].gamma.detach(), on_epoch=True, prog_bar=False)
        self.log("init/alpha_last_layer", self.layers[-1].alpha.detach(), on_epoch=True, prog_bar=False)
        self.log("init/beta_last_layer",  self.layers[-1].beta.detach(), on_epoch=True, prog_bar=False)
        self.log("init/gamma_last_layer", self.layers[-1].gamma.detach(), on_epoch=True, prog_bar=False)
        
        print(f"✓ Initial parameters logged:")
        print(f"  First layer - α: {self.layers[0].alpha.item():.6f}, β: {self.layers[0].beta.item():.6f}, γ: {self.layers[0].gamma.item():.6f}")
        print(f"  Last layer  - α: {self.layers[-1].alpha.item():.6f}, β: {self.layers[-1].beta.item():.6f}, γ: {self.layers[-1].gamma.item():.6f}")
        
    def training_step(self, batch, batch_idx):
        preprocessed_signals = batch.to(self.device, non_blocking=True)
        
        # Assert data and model are on the same device
        model_device = next(self.parameters()).device
        assert preprocessed_signals.device == model_device, f"Data on {preprocessed_signals.device}, model on {model_device}"
        if torch.cuda.is_available():
            assert model_device.type == "cuda", f"Expected CUDA but model is on {model_device}"
        
        pred_signals = self(preprocessed_signals)
        
        # Assert output is on the correct device
        assert pred_signals.device == model_device, f"Output on {pred_signals.device}, expected {model_device}"
        
        # Match signals
        matched_orig, matched_pred = match_signals(preprocessed_signals, pred_signals)
        loss = self.criterion(matched_pred, matched_orig)
        
        # Log loss - L1Loss returns a scalar
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    

    def validation_step(self, batch, batch_idx):
        preprocessed_signals = batch.to(self.device, non_blocking=True)
        
        # Assert data and model are on the same device
        model_device = next(self.parameters()).device
        assert preprocessed_signals.device == model_device, f"Data on {preprocessed_signals.device}, model on {model_device}"
        if torch.cuda.is_available():
            assert model_device.type == "cuda", f"Expected CUDA but model is on {model_device}"
        
        pred_signals = self(preprocessed_signals)
        
        # Assert output is on the correct device
        assert pred_signals.device == model_device, f"Output on {pred_signals.device}, expected {model_device}"
        
        # Match signals
        matched_orig, matched_pred = match_signals(preprocessed_signals, pred_signals)
        metrics = compute_all_metrics(matched_orig, matched_pred)
        loss = self.criterion(matched_pred, matched_orig)
        
        # Log with on_epoch=True so Lightning automatically averages across batches
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Log evaluation metrics (averaged across epoch)
        for k, v in metrics.items():
            self.log(f"val/{k.replace(' ', '_')}", v, on_step=False, on_epoch=True, 
                    prog_bar=False, sync_dist=True)
            
        return loss

    def on_validation_epoch_end(self):
        # Log layer parameters once per epoch
        self.log("val/alpha_first_layer", self.layers[0].alpha, on_epoch=True, prog_bar=False)
        self.log("val/beta_first_layer",  self.layers[0].beta, on_epoch=True, prog_bar=False)
        self.log("val/gamma_first_layer", self.layers[0].gamma, on_epoch=True, prog_bar=False)
        self.log("val/alpha_last_layer", self.layers[-1].alpha, on_epoch=True, prog_bar=False)
        self.log("val/beta_last_layer",  self.layers[-1].beta, on_epoch=True, prog_bar=False)
        self.log("val/gamma_last_layer", self.layers[-1].gamma, on_epoch=True, prog_bar=False)
        
        # Only log audio every 5 epochs to avoid overwhelming WandB
        if self.current_epoch % 5 == 0:
            try:
                preprocessed_signal = torch.from_numpy(EXAMPLE_AUDIO).float().to(self.device)
                with torch.no_grad():
                    output_signal = self(preprocessed_signal.unsqueeze(0))
                    matched_orig, matched_pred = match_signals(preprocessed_signal.unsqueeze(0), output_signal)
                
                # Convert to numpy and ensure proper format
                input_audio = matched_orig.squeeze(0).detach().cpu().numpy()
                output_audio = matched_pred.squeeze(0).detach().cpu().numpy()
                
                # Ensure audio is 1D
                if input_audio.ndim > 1:
                    input_audio = input_audio.flatten()
                if output_audio.ndim > 1:
                    output_audio = output_audio.flatten()
                
                # Normalize audio to prevent clipping (but preserve relative scale)
                max_val = max(np.abs(input_audio).max(), np.abs(output_audio).max()) + 1e-8
                input_audio = input_audio / max_val
                output_audio = output_audio / max_val
                
                # Use Lightning's logger to log audio samples
                if hasattr(self.logger, 'experiment') and self.logger.experiment is not None:
                    try:
                        self.logger.experiment.log({
                            "audio/sample_output": wandb.Audio(
                                output_audio,
                                sample_rate=SAMPLE_RATE,
                                caption=f"Output signal - Epoch {self.current_epoch}"
                            )
                        }, step=self.global_step, commit=False)
                        print(f"✓ Logged audio samples for epoch {self.current_epoch}")
                    except Exception as audio_log_error:
                        print(f"Failed to log audio via Lightning logger: {audio_log_error}")
                else:
                    print("No Lightning logger available for audio logging")
                    
            except Exception as e:
                print(f"Error logging audio: {e}")
                import traceback
                traceback.print_exc()

    def test_step(self, batch, batch_idx):
        preprocessed_signals = batch.to(self.device, non_blocking=True)
        
        pred_signals = self(preprocessed_signals)
        
        # Match signals
        matched_orig, matched_pred = match_signals(preprocessed_signals, pred_signals)
        metrics = compute_all_metrics(matched_orig, matched_pred)
        loss = self.criterion(matched_pred, matched_orig)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for k, v in metrics.items():
            self.log(f"test/{k.replace(' ', '_')}", v,
                     on_step=False, on_epoch=True, prog_bar=False)
        return loss

    # ---------------------------------------------------------------------
    # Optimiser / LR scheduler
    # ---------------------------------------------------------------------
    def configure_optimizers(self):
        # --------------------------- optimiser --------------------------------
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        return optimizer
        