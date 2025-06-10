from typing import Sequence, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Helper functions
def _stft(x: Tensor, n_fft: int, hop_length: int, win_length: int, window: Tensor | None = None) -> Tensor:
    """
    Return complex STFT with symmetric padding for perfect reconstruction
    """
    if window is None:
        window = torch.hann_window(win_length, device=x.device, dtype=x.dtype)
    return torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True, center=True, pad_mode="reflect")


def _complex_mag(z: Tensor) -> Tensor:
    """
    Magnitude of a complex tensor
    """
    return torch.abs(z)

# Spectrogram losses
class SpectralMagnitudeL2(nn.Module):
    """
    Corresponds to d(C2) in the Accelerated Griffin‑Lim paper (https://arxiv.org/pdf/2306.12504)
    """
    def forward(self, pred_mag: Tensor, target_mag: Tensor) -> Tensor:
        return F.mse_loss(pred_mag, target_mag)


class SpectralConvergence(nn.Module):
    """
    Spectral convergence loss introduced in (https://arxiv.org/pdf/1808.06719)
    """
    def forward(self, pred_mag: Tensor, target_mag: Tensor) -> Tensor:
        diff = torch.linalg.vector_norm(pred_mag - target_mag, ord=2, dim=(-2, -1))
        ref = torch.linalg.vector_norm(target_mag, ord=2, dim=(-2, -1))
        return torch.mean(diff / (ref + 1e-8))


class MultiResolutionSTFTLoss(nn.Module):
    """
    Sum of magnitude-L2 and spectral convergence over multiple FFT sizes
    """
    def __init__(
        self,
        n_ffts: Sequence[int] = (1024, 2048, 512),
        hop_scales: Sequence[float] = (0.25, 0.25, 0.125),
        win_scales: Sequence[float] | None = None,
        mag_weight: float = 1.0,
        sc_weight: float = 1.0,
    ):
        super().__init__()
        assert len(n_ffts) == len(hop_scales) and (win_scales is None or len(win_scales) == len(n_ffts)), "Mismatch in STFT resolution lists"
        self.resolutions = [(n, int(n * hop_scales[i]), int(n * (win_scales[i] if win_scales else 1.0))) for i, n in enumerate(n_ffts)]
        self.mag = SpectralMagnitudeL2()
        self.sc = SpectralConvergence()
        self.mag_weight = float(mag_weight)
        self.sc_weight = float(sc_weight)

    def forward(self, pred_audio: Tensor, target_audio: Tensor) -> Tensor:  # type: ignore[override]
        losses = []
        for n_fft, hop, win in self.resolutions:
            pred_stft = _stft(pred_audio, n_fft, hop, win)
            tgt_stft = _stft(target_audio, n_fft, hop, win)
            pred_mag, tgt_mag = _complex_mag(pred_stft), _complex_mag(tgt_stft)
            loss = (self.mag_weight * self.mag(pred_mag, tgt_mag) + self.sc_weight * self.sc(pred_mag, tgt_mag))
            losses.append(loss)
        return torch.mean(torch.stack(losses))


# Waveform‑domain losses
class WaveformL1(nn.Module):
    """
    L1 loss in the time domain
    """
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(pred, target)


class PhaseOnlyLoss(nn.Module):
    """
    Phase-only loss from:
      S. Takaki et al., STFT Spectral Loss for Training a Neural Speech Waveform Model,
      ICASSP 2019, Eq. (10)/(15) . https://arxiv.org/pdf/1810.11945
    """
    def __init__(self, n_fft=1024, hop_length=256, voiced_weighting=False):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop_length
        self.win = torch.hann_window(n_fft)          # Hann window (paper default)
        self.voiced_weighting = voiced_weighting     # if True, expect mask in forward

    def stft_phase(self, x):
        """
        x : (B, T) waveform in -1…1 range
        returns phase tensor (B, F, Frames)
        """
        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.win.to(x.device),
            center=True,
            return_complex=True
        )
        return torch.angle(X)                        # phase ∈ (-π, π]

    def forward(self, target, pred, voiced_mask=None):
        """
        target, pred : (B, T) tensors
        voiced_mask  : (B, Frames) bool/float, 1 = voiced; used only if voiced_weighting
        """
        phi_t = self.stft_phase(target)
        phi_p = self.stft_phase(pred)

        delta = torch.atan2(torch.sin(phi_p - phi_t), torch.cos(phi_p - phi_t))

        phase_err = 1.0 - torch.cos(delta)

        if self.voiced_weighting and voiced_mask is not None:
            phase_err = phase_err * voiced_mask.unsqueeze(1)

        return phase_err.mean()

# Combined spectral and waveform losses
class CompositeLoss(nn.Module):
    """
    Weighted sum of arbitrary sub losses

    Example
    -------
    >>> criterion = CompositeLoss({"mrstft": 1.0, "l1": 10.0})
    >>> loss = criterion(pred_audio, target_audio)
    """

    loss_options = {
        "mrstft": MultiResolutionSTFTLoss(),
        "l1": WaveformL1(),
        "mag_l2": SpectralMagnitudeL2(),
        "sc": SpectralConvergence(),
        "phase": PhaseOnlyLoss()
    }

    def __init__(self, weights: Dict[str, float] | None = None):
        super().__init__()
        if weights is None:
            # Default weights
            weights = {"mrstft": 1.0, "l1": 10.0}
        self._weights = {k: float(v) for k, v in weights.items() if v != 0.0 and k in self.loss_options}
        self.loss_fns = nn.ModuleDict({k: self.loss_options[k] for k in self._weights})

    def forward(self, pred_audio: Tensor, target_audio: Tensor) -> Tensor:
        total = 0.0
        details = {}
        for name, weight in self._weights.items():
            loss_val = self.loss_fns[name](pred_audio, target_audio)
            total += weight * loss_val
            details[f"loss/{name}"] = loss_val

        # Optional Lightning logging
        if torch.is_grad_enabled() and hasattr(self, "log_dict"):
            self.log_dict(details, prog_bar=False, on_step=True, on_epoch=True)
        return total    
    
# Example usage
if __name__ == "__main__":
    pred = torch.randn(1, 44100)
    target = torch.randn(1, 44100)
    
    # Test all loss functions
    mrstft_loss = MultiResolutionSTFTLoss()
    l1_loss = WaveformL1()
    composite_loss = CompositeLoss({"mrstft": 1.0, "l1": 10.0})
    print("MRSTFT Loss:", mrstft_loss(pred, target).item())
    print("L1 Loss:", l1_loss(pred, target).item())
    print("Composite Loss:", composite_loss(pred, target).item())
    
    