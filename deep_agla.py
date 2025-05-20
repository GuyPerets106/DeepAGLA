from typing import Dict, Tuple

import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import Tensor
from loss import CompositeLoss
from torchaudio.transforms import MelSpectrogram
from definitions import *


def _stft(x: Tensor, n_fft: int = N_FFT, hop: int = HOP, win_length: int | None = None,
          window: Tensor | None = None) -> Tensor:
    if win_length is None:
        win_length = n_fft
    if window is None:
        window = torch.hann_window(win_length, device=x.device, dtype=x.dtype)
    return torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win_length,
                      window=window, return_complex=True, center=True, pad_mode="reflect")


def _istft(z: Tensor, n_fft: int = N_FFT, hop: int = HOP, win_length: int | None = None,
           window: Tensor | None = None, length: int | None = None) -> Tensor:
    if win_length is None:
        win_length = n_fft
    if window is None:
        window = torch.hann_window(win_length, device=z.device, dtype=z.dtype)
    return torch.istft(z, n_fft=n_fft, hop_length=hop, win_length=win_length,
                       window=window, center=True, length=length)


def proj_C1(z: Tensor, n_fft: int = N_FFT, hop: int = HOP, win_length: int | None = None,
            window: Tensor | None = None) -> Tensor:
    """
    Projection onto consistent spectra (C1).  STFT → ISTFT → STFT
    """
    x = _istft(z, n_fft, hop, win_length, window)   # time domain
    return _stft(x, n_fft, hop, win_length, window)  # back to complex STFT


def proj_C2(z: Tensor, target_mag: Tensor) -> Tensor:
    """
    Projection onto spectra with target magnitude (C2): preserve phase
    """
    phase = z / (torch.abs(z) + 1e-8)
    return target_mag * phase

# Learnable AGLA layer
class AGLALayer(nn.Module):
    """
    Single Accelerated Griffin-Lim iteration with learnable parameters
    """
    def __init__(self, init_alpha: float = 0.1, init_beta: float = 1.1, init_gamma: float = 0.2):
        super().__init__()
        self.u_alpha = nn.Parameter(torch.tensor(math.atanh(2 * init_alpha - 1), dtype=torch.float32))
        self.u_beta  = nn.Parameter(torch.tensor(math.log(init_beta), dtype=torch.float32))
        self.u_gamma = nn.Parameter(torch.tensor(math.atanh(init_gamma - 1), dtype=torch.float32))

    @staticmethod
    def _max_alpha(beta: Tensor, gamma: Tensor) -> Tensor:
        """
        Compute maximal alpha in the range (0, alpha_max) that satisfies Eq. (10) in the paper
        """
        mask = gamma <= 1.0
        max_a = torch.where(mask, (1 - 1/gamma) * beta + 1/gamma - 0.5, 1.0 / (2 * beta * (gamma - 1) + gamma) - 0.5)
        # Numerical floor to keep >0
        return torch.clamp(max_a, min=1e-4)

    def _constrained_params(self) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Map unconstrained params (alpha, beta, gamma) in the admissible domain
        """
        gamma = torch.tanh(self.u_gamma) + 1 # (0, 2)
        beta  = torch.exp(self.u_beta) # (0, inf)
        alpha_raw = (torch.tanh(self.u_alpha) + 1) * 0.5
        alpha = alpha_raw * self._max_alpha(beta, gamma)
        return alpha, beta, gamma

    def forward(self, c_prev: Tensor, t_prev: Tensor, d_prev: Tensor, target_mag: Tensor,
                *, n_fft: int = N_FFT, hop: int = HOP, win_length: int | None = None,
                window: Tensor | None = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Perform one accelerated iteration

        Parameters
        ----------
        c_prev, t_prev, d_prev
            Complex STFT tensors from the previous layer.
        target_mag
            Magnitude spectrogram of the *input* (fixed during forward).
        Returns
        -------
        c, t, d : Tensor
            Updated complex spectra ready for the next layer.
        """
        alpha, beta, gamma = self._constrained_params()

        y = proj_C1(proj_C2(c_prev, target_mag), n_fft, hop, win_length, window)
        t = (1.0 - gamma) * d_prev + gamma * y
        c = t + alpha * (t - t_prev)
        d = t + beta * (t - t_prev)
        return c, t, d

class DeepAGLA(pl.LightningModule):
    """
    Stack of L AGLA layers trained with a composite spectrogram + waveform loss
    """

    def __init__(
        self,
        n_layers: int = 6,
        n_fft: int = N_FFT,
        hop: int = HOP,
        win_length: int | None = None,
        sample_rate: int = SAMPLE_RATE,
        lr: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        loss_weights: Dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.win_length = int(win_length) if win_length is not None else self.n_fft
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)

        self.mel = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop,
            n_mels=N_MELS,
            power=1.0,
            center=True,
        )
        fb = self.mel.mel_scale.fb
        self.register_buffer("mel_fb", fb, persistent=False)
        self._mel_inv: Tensor | None = None
        
        # Learnable iteration stack
        self.layers = nn.ModuleList([AGLALayer() for _ in range(n_layers)])

        # Loss
        self.criterion = CompositeLoss(loss_weights) # ! Consider use other loss functions

    def forward(self, mag: Tensor, length: int = None) -> Tensor:
        """
        Run L accelerated iterations and return the reconstructed waveform
        """
        # Initialise with zero‑phase
        c = mag.clone()  # real‑valued mag (magnitude) promoted to complex when multiplied
        c = c * torch.exp(torch.zeros_like(c) * 1j)
        t = d = proj_C1(c, self.n_fft, self.hop, self.win_length, self.window)

        for layer in self.layers:
            c, t, d = layer(c, t, d, mag, n_fft=self.n_fft, hop=self.hop, win_length=self.win_length, window=self.window)

        # Final ISTFT
        audio = _istft(c, self.n_fft, self.hop, self.win_length, self.window, length=length)
        # Clamp to [-1, 1] for stable training
        return torch.clamp(audio, -1.0, 1.0)

    def training_step(self, batch, batch_idx):
        mel, target_audio = batch
        target_mag = self._mel_to_mag(mel)
        pred_audio = self(target_mag, length=target_audio.shape[-1])
        loss = self.criterion(pred_audio, target_audio)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mel, target_audio = batch
        target_mag = self._mel_to_mag(mel)
        pred_audio = self(target_mag, length=target_audio.shape[-1])
        loss = self.criterion(pred_audio, target_audio)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        mel, target_audio = batch
        target_mag = self._mel_to_mag(mel)
        pred_audio = self(target_mag, length=target_audio.shape[-1])
        loss = self.criterion(pred_audio, target_audio)
        self.log("test/loss", loss, prog_bar=False)
        return {"loss" : loss, "pred_audio" : pred_audio}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def _mel_to_mag(self, mel: Tensor) -> Tensor:
        mel = mel.transpose(-2, -1)

        if self._mel_inv is None:
            self._mel_inv = torch.linalg.pinv(self.mel_fb).to(self.mel_fb)

        mag = mel @ self._mel_inv
        mag = mag.transpose(-2, -1).contiguous()

        return torch.clamp(mag, min=0.0)