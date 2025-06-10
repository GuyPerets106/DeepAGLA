from typing import Dict, Tuple

import math
from numpy import fmin
import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
import librosa
from torch import Tensor
from loss import CompositeLoss, WaveformL1, PhaseOnlyLoss
from eval_metrics import evaluate_batch
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
    def __init__(self, init_alpha: float = 0.1, init_beta: float = 1.1, init_gamma: float = 0.2) -> None:
        super().__init__()
        self.u_alpha = nn.Parameter(torch.tensor(math.atanh(2 * init_alpha - 1), dtype=torch.float32))
        self.u_beta  = nn.Parameter(torch.tensor(math.log(init_beta), dtype=torch.float32))
        self.u_gamma = nn.Parameter(torch.tensor(math.atanh(init_gamma - 1), dtype=torch.float32))

        # self.reset_parameters(init_alpha, init_beta, init_gamma)
        
    @staticmethod
    def _max_alpha(beta: Tensor, gamma: Tensor) -> Tensor:
        """
        Compute maximal alpha in the range (0, alpha_max) that satisfies Eq. (10) in the paper
        """
        mask = gamma <= 1.0
        max_a = torch.where(mask, (1 - 1/gamma) * beta + 1/gamma - 0.5, 1.0 / (2 * beta * (gamma - 1) + gamma) - 0.5)
        # Numerical floor to keep >0
        return torch.clamp(max_a, min=1e-4)
    
    def _max_beta(self, gamma: Tensor) -> Tensor:
        """ Max β allowed by Eq. (9): 2β|1-γ| < 2-γ """
        mask = gamma <= 1.0
        out = torch.where(
            mask,
            (2.0 - gamma) / (2.0 * (1.0 - gamma + 1e-6)),
            (2.0 - gamma) / (2.0 * (gamma - 1.0 + 1e-6))
        )
        return torch.clamp(out, min=1e-4)
    
    def reset_parameters(self, init_alpha, init_beta, init_gamma):
        with torch.no_grad():
            device = self.u_gamma.device
            dtype  = self.u_gamma.dtype          # keep fp32 or fp16 consistent

            # ---- γ ----------------------------------------------------------
            # forward map: γ = 1 + 0.5 * (1 + tanh uγ)
            # inverse:     uγ = atanh(2γ - 3)
            gamma = torch.as_tensor(init_gamma, dtype=dtype, device=device)
            self.u_gamma.copy_(torch.atanh(2 * gamma - 3))   # tensor → OK

            # ---- β ----------------------------------------------------------
            beta_max = self._max_beta(gamma)
            beta = torch.as_tensor(init_beta, dtype=dtype, device=device)
            if beta >= beta_max:
                beta = 0.999 * beta_max
            p_beta = beta / beta_max
            self.u_beta.copy_(torch.logit(p_beta))

            # ---- α ----------------------------------------------------------
            alpha_max = self._max_alpha(beta, gamma)
            alpha = torch.as_tensor(init_alpha, dtype=dtype, device=device)
            if alpha >= alpha_max:
                alpha = 0.999 * alpha_max
            p_alpha = alpha / alpha_max
            self.u_alpha.copy_(torch.logit(p_alpha))
            
    def _constrained_params(self):
        gamma = 1 + 0.5 * (1 + torch.tanh(self.u_gamma))

        # smooth upper bound for β
        beta_max = self._max_beta(gamma)
        beta = beta_max * torch.sigmoid(self.u_beta)

        # smooth upper bound for α
        alpha_max = self._max_alpha(beta, gamma)
        alpha = alpha_max * torch.sigmoid(self.u_alpha)

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
        n_layers: int = N_LAYERS,
        n_fft: int = N_FFT,
        hop: int = HOP,
        win_length: int | None = None,
        sample_rate: int = SAMPLE_RATE,
        lr: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        loss_weights: Dict[str, float] | None = None,
        initial_params: Dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.n_fft = int(n_fft)
        self.sample_rate = int(sample_rate)
        self.hop = int(hop)
        self.win_length = int(win_length) if win_length is not None else self.n_fft
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)

        mel_fb_np = librosa.filters.mel(
            sr=SAMPLE_RATE, 
            n_fft=N_FFT, 
            n_mels=N_MELS,
            fmin=FMIN, fmax=FMAX, 
            htk=False)
        mel_fb = torch.from_numpy(mel_fb_np).float()
        self.register_buffer("mel_fb", mel_fb, persistent=False)
        self._mel_inv: Tensor | None = None
        
        # Learnable iteration stack
        init_alpha = initial_params["alpha"]
        init_beta = initial_params["beta"]
        init_gamma = initial_params["gamma"]
        self.layers = nn.ModuleList([AGLALayer(init_alpha, init_beta, init_gamma) for _ in range(n_layers)])

        weights = dict(mrstft=1.0, mag_l2=0.5, sc=0.5, l1=0.05)
        self.criterion = CompositeLoss(weights=weights)

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
        return torch.clamp(audio, -1.0, 1.0)
    
    def on_fit_start(self):
        # Take one batch from the val loader
        batch = next(iter(self.trainer.datamodule.val_dataloader()))
        mel, target = batch
        mel = mel.to(self.device)
        target = target.to(self.device)

        # Linear-mag inversion check
        mag_hat = self._mel_to_mag(mel)
        mel_hat = torch.matmul(self.mel_fb, mag_hat)             # F x T
        l2_rel  = torch.mean((mel - mel_hat) ** 2) / torch.mean(mel ** 2)

        # STFT perfect-reconstruction check on the first waveform
        recon = _istft(_stft(target[0], self.n_fft, self.hop, self.win_length,
                             self.window), self.n_fft, self.hop, self.win_length,
                       self.window, length=target.shape[-1])
        pcm_err = torch.mean(torch.abs(target[0] - recon))

        # Log and assert
        self.log_dict(
            {"sanity/mel_L2rel":  l2_rel,
             "sanity/stft_L1":    pcm_err},
            prog_bar=False, on_step=False, on_epoch=False
        )
        if l2_rel > 1e-2 or pcm_err > 1e-5:
            raise RuntimeError("Sanity check failed: dataset or STFT config inconsistent.")
        
    def training_step(self, batch, batch_idx):
        mel, target_audio = batch
        target_mag = self._mel_to_mag(mel)
        pred_audio = self(target_mag, length=target_audio.shape[-1])
        loss = self.criterion(pred_audio, target_audio)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mel, target_audio = batch
        pred_audio = self(self._mel_to_mag(mel), length=target_audio.size(-1))

        metrics = evaluate_batch(pred_audio, target_audio, fs=SAMPLE_RATE)

        # --------------- log per epoch ------------------
        for k, v in metrics.items():
            tag = k.replace("(", "").replace(")", "").replace("-", "_")
            self.log(f"val/{tag}", v,
                    on_step=False, on_epoch=True,
                    prog_bar=False, sync_dist=True)
        
        loss    = self.criterion(pred_audio, target_audio)

        # log batch-wise so PL aggregates mean for us
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # keep first sample of first batch
        if batch_idx == 1 and self.trainer.is_global_zero:
            clip = pred_audio[0].detach().cpu()
            peak = clip.abs().max()
            if peak > 0:
                clip = clip / peak
            five_sec = SAMPLE_RATE * 5
            clip = clip[:five_sec] if clip.numel() >= five_sec else \
                torch.nn.functional.pad(clip, (0, five_sec - clip.numel()), mode="constant", value=0.0)
            self._val_clip = clip            # store for epoch_end hook

        return loss

    def on_validation_epoch_end(self):
        if not isinstance(self.logger, pl.loggers.WandbLogger):
            return
        if getattr(self, "_val_clip", None) is None:
            return

        self.logger.experiment.log(
            {
                "val/reconstructed_audio": wandb.Audio(
                    self._val_clip.numpy(), sample_rate=SAMPLE_RATE,
                    caption=f"Reconstructed – epoch {self.current_epoch}"
                )
            },
            commit=True,
            step=self.global_step,
        )
        del self._val_clip

    def test_step(self, batch, batch_idx):
        mel, target_audio = batch
        target_mag  = self._mel_to_mag(mel)
        pred_audio  = self(target_mag, length=target_audio.shape[-1])

        metrics = evaluate_batch(pred_audio, target_audio, fs=SAMPLE_RATE)
        loss    = self.criterion(pred_audio, target_audio)

        self.log("test/loss", loss,
                 prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        for name, value in metrics.items():
            self.log(f"test/avg_{name}", torch.as_tensor(value, device=self.device),
                     prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        return {"loss": loss, "pred_audio": pred_audio}
        
    def configure_optimizers(self):
        agla_params  = [p for m in self.layers for p in m.parameters()]
        other_params = [p for p in self.parameters() if p not in agla_params]

        optimizer = torch.optim.Adam(
            [
                {"params": agla_params,  "weight_decay": 0.0},
                {"params": other_params, "weight_decay": self.hparams.weight_decay},
            ],
            lr= self.hparams.lr,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=True,
        )

        return {
            "optimizer":  optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/SSNRdB",
                "interval":  "epoch",
                "frequency": 1,
            },
        }
    
    def _mel_to_mag(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Mel-magnitude (B, 1024, T) ➜ STFT-magnitude (B, 2049, T)
        using the exact pseudo-inverse of the librosa filterbank.
        """
        mel = mel.transpose(-2, -1)

        if self._mel_inv is None:
            self._mel_inv = torch.linalg.pinv(self.mel_fb).mT

        mag_hat = torch.sqrt(mel @ self._mel_inv)
        mag = mag_hat.transpose(-2, -1).clamp(min=1e-8)

        return mag