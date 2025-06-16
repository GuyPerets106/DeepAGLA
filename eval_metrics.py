from typing import Dict
import torch
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio, SignalDistortionRatio
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torch_pesq import PesqLoss

def ssnr_db(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int = 44100,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = None,
    eps: float = 1e-8,
) -> float:
    """
    Compute Spectral-SNR in dB on STFT magnitudes. \\
    Perfect reconstruction → +∞ dB; higher is better.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape: got Pred {pred.shape} != Target {target.shape}")

    window = torch.hann_window(win_length or n_fft, device=pred.device)

    def _mag(x: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        ).abs()

    mag_pred = _mag(pred)
    mag_true = _mag(target)

    num = torch.sum((mag_pred - mag_true) ** 2)
    den = torch.sum(mag_true ** 2) + eps
    ssnr = -10.0 * torch.log10(num / den + eps)
    return float(ssnr.item())



_sisdr_metric = ScaleInvariantSignalDistortionRatio().cpu()
def si_sdr_db(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Scale-Invariant Signal-to-Distortion Ratio (dB)
    """
    pred   = pred.detach().float().reshape(1, -1)    # <-- contiguous-safe
    target = target.detach().float().reshape(1, -1)
    _sisdr_metric.reset()
    _sisdr_metric(pred.cpu(), target.cpu())
    return float(_sisdr_metric.compute().item())


_sdr_metric = SignalDistortionRatio().cpu()
def sdr_db(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Signal-to-Distortion Ratio (dB)
    """
    pred   = pred.detach().float().reshape(1, -1)    # <-- contiguous-safe
    target = target.detach().float().reshape(1, -1)
    _sdr_metric.reset()
    _sdr_metric(pred.cpu(), target.cpu())
    return float(_sdr_metric.compute().item())


# -----------------------------------------------------------------------------
# PESQ – ITU‑T P.862 Perceptual Evaluation of Speech Quality
# -----------------------------------------------------------------------------

def pesq_mos(pred: torch.Tensor, target: torch.Tensor, fs: int) -> float:
    """
    Compute PESQ MOS
    """

    pred = pred.detach().cpu()
    target = target.detach().cpu()

    loss = PesqLoss(factor=0.0, sample_rate=fs)
    mos = loss.mos(target, pred)
    return mos.mean().item()



_stoi_metric = ShortTimeObjectiveIntelligibility(fs=44100, extended=False).cpu()
def stoi(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Short-Time Objective Intelligibility (range 0-1).  Resamples if necessary
    """
    _stoi_metric.reset()
    _stoi_metric(pred.cpu(), target.cpu())
    return float(_stoi_metric.compute().item())



def evaluate_batch(pred: torch.Tensor, target: torch.Tensor, fs: int = 44_100) -> Dict[str, float]:
    """Return a dict with all metrics for *one batch*.

    Example
    -------
    >>> out = model(mag)
    >>> metrics = evaluate_batch(out, target, fs=44_100)
    >>> wandb.log(metrics)
    """
    return {
        "SSNR(dB)": ssnr_db(pred, target),
        "SI-SDR(dB)": si_sdr_db(pred, target),
        "SDR(dB)": sdr_db(pred, target),
        "PESQ": pesq_mos(pred, target, fs=fs),
        "STOI": stoi(pred, target),
    }
    

# Example usage
if __name__ == "__main__":
    # Dummy tensors for testing
    pred = torch.randn(1, 44100)
    target = torch.randn(1, 44100)

    metrics = evaluate_batch(pred, target)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")