from typing import Dict, Tuple, Optional

import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb

from torch import Tensor
from loss import CompositeLoss
from eval_metrics import evaluate_batch
from definitions import *          # SAMPLE_RATE, N_FFT, HOP, etc.
import audio_utilities


# -------------------------------------------------------------------------
# STFT helpers
# -------------------------------------------------------------------------
def _align(pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """Crop both tensors to the shorter length so shapes match."""
    L = min(pred.size(-1), target.size(-1))
    return pred[..., :L], target[..., :L]


def _stft(
    x: Tensor,
    n_fft:       int  = N_FFT,
    hop_length:  Optional[int] = HOP,
    win_length:  Optional[int] = N_FFT,
    window:      Optional[Tensor] = None,
    center: bool = True,
    pad_mode: str = "constant"
) -> Tensor:
    """
    Torch wrapper that mirrors **librosa.stft** defaults:
      • Hann window (`np.hanning`) of length *win_length* (default = n_fft)
      • hop_length = n_fft // 4 if omitted
      • center = True  (pads ⌊n_fft/2⌋ samples at start/end)
      • pad_mode = "reflect"
      • output shape:  (..., n_fft//2 + 1, n_frames) — freq × time
    """
    if window is None:
        window = torch.hann_window(win_length, periodic=False, dtype=x.dtype, device=x.device)

    return torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        normalized=False,
        onesided=True,
        return_complex=True
    )


def _istft(
    Z: Tensor,
    n_fft:       int  = N_FFT,
    hop_length:  Optional[int] = HOP,
    win_length:  Optional[int] = N_FFT,
    window:      Optional[Tensor] = None,
    center: bool = True,
    length: Optional[int] = None
) -> Tensor:
    """
    Torch wrapper that mirrors **librosa.istft** defaults:
      • Uses the same Hann window as analysis
      • hop_length = n_fft // 4 if omitted
      • center = True  (assumes padding was added in `_stft`)
      • `length` can be provided to force an exact number of output samples,
        otherwise Torch computes the natural inverse length.
    """
    if window is None:
        window = torch.hann_window(win_length, periodic=False, dtype=Z.dtype, device=Z.device)
        
    if length is None:
        n_frames = Z.shape[-1]
        length = hop_length * (n_frames - 1)
        
        
    return torch.istft(
        Z,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        normalized=False,
        onesided=True,
        length=length
    )

# -------------------------------------------------------------------------
# Projection operators
# -------------------------------------------------------------------------
def proj_C1(z: Tensor, n_fft: int = N_FFT, hop: int = HOP,
            win_length: int | None = None, window: Tensor | None = None,
            length: Optional[int] = None) -> Tensor:
    """Projection onto the set of **consistent** spectra (C₁)."""
    return _stft(_istft(z, n_fft, hop, win_length, window), n_fft, hop, win_length, window)


def proj_C2(z: Tensor, target_mag: Tensor) -> Tensor:
    """Projection onto spectra with the **desired magnitude** (C₂)."""
    phase = z / (z.abs() + 1e-8)
    return target_mag * phase


# -------------------------------------------------------------------------
# Learnable AGLA iteration
# -------------------------------------------------------------------------
class AGLALayer(nn.Module):
    """
    One *Accelerated Griffin-Lim* iteration with learnable α, β, γ.
    Formulas follow https://arxiv.org/pdf/2306.12504 paper.
    """
    def __init__(self, init_alpha: float = 0.1,
                 init_beta:  float = 1.1,
                 init_gamma: float = 0.2) -> None:
        super().__init__()
        # Unconstrained parameters (stored)
        self.u_alpha = nn.Parameter(torch.atanh(torch.tensor(2*init_alpha - 1)))
        self.u_beta  = nn.Parameter(torch.log(torch.tensor(init_beta)))
        self.u_gamma = nn.Parameter(torch.atanh(torch.tensor(init_gamma - 1)))

    # --- helper -----------------------------------------------------------
    @staticmethod
    def _max_alpha(beta: Tensor, gamma: Tensor) -> Tensor:
        """Eq. (10) upper bound for α > 0."""
        mask = gamma <= 1.0
        max_a = torch.where(
            mask,
            (1 - 1/gamma) * beta + 1/gamma - 0.5,
            1.0 / (2 * beta * (gamma - 1) + gamma) - 0.5
        )
        return max_a.clamp_min(1e-4)

    def _max_beta(self, gamma: Tensor) -> Tensor:
        """Eq. (9) upper bound for β > 0."""
        mask = gamma <= 1.0
        out = torch.where(
            mask,
            (2.0 - gamma) / (2.0 * (1.0 - gamma + 1e-6)),
            (2.0 - gamma) / (2.0 * (gamma - 1.0 + 1e-6))
        )
        return out.clamp_min(1e-4)

    # --- constrained parameters ------------------------------------------
    def _constrained_params(self) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Map the unconstrained u-parameters to (α, β, γ) satisfying the
        AGLA stability conditions.
        """
        gamma = 1 + 0.5 * (1 + torch.tanh(self.u_gamma))      # γ ∈ (1, 2)
        beta_max = self._max_beta(gamma)
        beta  = beta_max * torch.sigmoid(self.u_beta)         # β ∈ (0, β_max)
        alpha_max = self._max_alpha(beta, gamma)
        alpha = alpha_max * torch.sigmoid(self.u_alpha)       # α ∈ (0, α_max)
        return alpha, beta, gamma

    def forward(
        self,
        c_prev: Tensor, t_prev: Tensor, d_prev: Tensor,
        target_mag: Tensor,
        *, n_fft: int, hop: int, win_length: int, window: Tensor,
        length: Optional[int] = None
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
        alpha, beta, gamma = self._constrained_params()

        y = proj_C1(proj_C2(c_prev, target_mag), n_fft, hop, win_length, window, length=length)
        t = (1 - gamma) * d_prev + gamma * y
        c = t + alpha * (t - t_prev)
        d = t +  beta  * (t - t_prev)
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
        win_length:   int   | None = None,
        sample_rate:  int   = SAMPLE_RATE,
        lr:           float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        loss_weights: Dict[str, float] | None = None,
        initial_params: Dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # STFT settings ----------------------------------------------------
        self.n_fft      = int(n_fft)
        self.hop        = int(hop)
        self.win_length = self.n_fft if win_length is None else int(win_length)
        self.register_buffer("window",
                             torch.hann_window(self.win_length, periodic=False),
                             persistent=False)

        # Learnable iteration stack ---------------------------------------
        init_alpha = initial_params.get("alpha", 0.1)
        init_beta  = initial_params.get("beta",  1.1)
        init_gamma = initial_params.get("gamma", 0.2)
        self.layers = nn.ModuleList(
            AGLALayer(init_alpha, init_beta, init_gamma)
            for _ in range(n_layers)
        )

        # Loss -------------------------------------------------------------
        default_w = dict(mrstft=1.0, mag_l2=0.5, sc=0.5, l1=0.05)
        self.criterion = CompositeLoss(weights=loss_weights or default_w)

    # ---------------------------------------------------------------------
    # Forward:  magnitude → audio
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def _zero_phase(self, mag: Tensor) -> Tensor:
        """Initial complex STFT with zero phase."""
        return mag * torch.exp(1j * torch.zeros_like(mag))

    def forward(self, mag: Tensor, *, length: int | None = None) -> Tensor:  # noqa: D401
        """
        Parameters
        ----------
        mag : Tensor (B, F, T)
            Linear-scale magnitude spectrogram (sqrt power).
        length : int, optional
            Desired output length in samples; if omitted, ISTFT's default
            length (hop*(T-1) + win_len) is used.
        Returns
        -------
        audio : Tensor (B, N)
            Reconstructed waveform in [-1, 1].
        """
        # Shape check ------------------------------------------------------
        if mag.dim() != 3:
            raise ValueError("`mag` must have shape (B, F, T)")

        c = self._zero_phase(mag)                          # z₀
        t = d = proj_C1(c, self.n_fft, self.hop,
                        self.win_length, self.window)      # warm-up

        for layer in self.layers:
            c, t, d = layer(
                c, t, d, mag,
                n_fft=self.n_fft, hop=self.hop,
                win_length=self.win_length, window=self.window,
                length=length
            )

        audio = _istft(c, self.n_fft, self.hop, self.win_length,
                       self.window)
        return audio.clamp(-1.0, 1.0)

    # ---------------------------------------------------------------------
    # Lightning hooks
    # ---------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        mag, target_audio = batch                     # (B, F, T), (B, N)
        pred_audio = self(mag, length=target_audio.size(-1))
        pred_audio, target_audio = _align(pred_audio, target_audio)

        loss = self.criterion(pred_audio, target_audio)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mag, target_audio = batch
        pred_audio = self(mag, length=target_audio.size(-1))
        pred_audio, target_audio = _align(pred_audio, target_audio)
        # ------ metrics & loss -------------------------------------------
        metrics = evaluate_batch(pred_audio, target_audio, fs=SAMPLE_RATE)
        loss    = self.criterion(pred_audio, target_audio)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for k, v in metrics.items():
            self.log(f"val/{k.replace(' ', '_')}", v,
                     on_step=False, on_epoch=True, prog_bar=False)

        # Store sample for WandB logging (first batch only) ---------------
        if batch_idx == 0 and self.trainer.is_global_zero:
            clip = pred_audio[0].detach().cpu()
            clip = clip / clip.abs().max().clamp(min=1e-8)
            five_sec = SAMPLE_RATE * 5
            clip = (clip[:five_sec] if clip.numel() >= five_sec else
                    torch.nn.functional.pad(clip, (0, five_sec - clip.numel())))
            self._val_clip = clip
        return loss

    def on_validation_epoch_end(self):
        if isinstance(self.logger, pl.loggers.WandbLogger) and hasattr(self, "_val_clip"):
            self.logger.experiment.log(
                {"val/reconstructed_audio":
                    wandb.Audio(self._val_clip.numpy(),
                                sample_rate=SAMPLE_RATE,
                                caption=f"Epoch {self.current_epoch}")},
                commit=True, step=self.global_step)
            del self._val_clip

    def test_step(self, batch, batch_idx):
        mag, target_audio = batch
        pred_audio = self(mag, length=target_audio.size(-1))
        pred_audio, target_audio = _align(pred_audio, target_audio)
        metrics = evaluate_batch(pred_audio, target_audio, fs=SAMPLE_RATE)
        loss    = self.criterion(pred_audio, target_audio)

        self.log("test/loss", loss, on_epoch=True)
        for k, v in metrics.items():
            self.log(f"test/{k}", v, on_epoch=True)
        return {"loss": loss, "pred_audio": pred_audio}

    # ---------------------------------------------------------------------
    # Optimiser / LR scheduler
    # ---------------------------------------------------------------------
    def configure_optimizers(self):
        agla_params  = [p for m in self.layers for p in m.parameters()]
        other_params = [p for p in self.parameters() if p not in agla_params]

        optimizer = torch.optim.Adam(
            [{"params": agla_params,  "weight_decay": 0.0},
             {"params": other_params, "weight_decay": self.hparams.weight_decay}],
            lr=self.hparams.lr,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5,
            patience=3, min_lr=1e-5, verbose=True
        )
        return {
            "optimizer":  optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/SSNRdB",
                "interval": "epoch",
            },
        }