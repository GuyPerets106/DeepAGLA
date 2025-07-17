import math
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import wandb
import scipy.signal
from typing import Dict, Tuple
from torch import Tensor
from definitions import N_FFT, HOP, WIN_LEN, SAMPLE_RATE, EXAMPLE_AUDIO

# Constants
EPS = 1e-12  # Small constant to avoid division by zero

# Pre-create window to reuse across metrics (will be moved to appropriate device when needed)
_METRIC_WINDOW = torch.hann_window(WIN_LEN, dtype=torch.float32)

# -----------------------------------------------------------------------------
#                                HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def _as_tensor(x, *, device: torch.device | None = None) -> torch.Tensor:
    """Utility: convert numpy array to torch tensor on the desired device."""
    if torch.is_tensor(x):
        return x.to(device) if device is not None else x
    return torch.as_tensor(x, dtype=torch.float32, device=device)

def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    """(L,) -> (1, L); (B, L) left untouched."""
    return x.unsqueeze(0) if x.ndim == 1 else x

def match_length(original: Tensor, estimated: Tensor) -> Tuple[Tensor, Tensor]:
    """Trim signals to the same length."""
    min_len = min(original.shape[-1], estimated.shape[-1])
    return original[..., :min_len], estimated[..., :min_len]

def align_signals_fft(ref: torch.Tensor, est: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable circular alignment using FFT-based correlation.
    Args:
        ref, est: (B, L) float32 tensors
    Returns:
        (ref, shifted_est): aligned signals
    """
    if ref.ndim == 1:
        ref = ref.unsqueeze(0)
        est = est.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    B, L = ref.shape
    n_fft = 2 ** math.ceil(math.log2(2 * L - 1))  # next power of 2

    REF = torch.fft.rfft(ref, n=n_fft)            # (B, F)
    EST = torch.fft.rfft(est, n=n_fft)

    corr = torch.fft.irfft(EST * REF.conj(), n=n_fft)  # (B, n_fft)
    corr = torch.roll(corr, shifts=L-1, dims=-1)       # centre lag 0
    lag  = corr.argmax(dim=-1) - (L - 1)               # (B,)

    shifted = torch.stack([
        torch.roll(est[i], -int(lag_i))
        for i, lag_i in enumerate(lag)
    ])

    if squeeze_output:
        ref = ref.squeeze(0)
        shifted = shifted.squeeze(0)
    
    return ref, shifted

def prepare_signals(original: Tensor, estimated: Tensor) -> Tuple[Tensor, Tensor]:
    """Prepare signals for metric computation: align, match length, ensure same device."""
    # Preserve device of original signal
    target_device = original.device if torch.is_tensor(original) else None
    
    original = _as_tensor(original, device=target_device)
    estimated = _as_tensor(estimated, device=target_device)
    
    # Ensure both are on the same device
    if original.device != estimated.device:
        estimated = estimated.to(original.device)
    
    # Align signals
    original, estimated = align_signals_fft(original, estimated)
    
    # Match lengths
    original, estimated = match_length(original, estimated)
    
    # Ensure 2D for batched computation
    original = _ensure_2d(original)
    estimated = _ensure_2d(estimated)
    
    return original, estimated

# -----------------------------------------------------------------------------
#                              CORE METRICS
# -----------------------------------------------------------------------------

def compute_ssnr_db(orig: Tensor, est: Tensor) -> float:
    """
    Spectral Signal-to-Noise Ratio in dB.
    This is the primary metric for DeepAGLA optimization.
    """
    orig, est = prepare_signals(orig, est)
    
    window = _METRIC_WINDOW.to(orig.device)
    S_orig = torch.stft(orig, n_fft=N_FFT, hop_length=HOP, win_length=WIN_LEN,
                        window=window, center=True, return_complex=True)
    S_est  = torch.stft(est,  n_fft=N_FFT, hop_length=HOP, win_length=WIN_LEN,
                        window=window, center=True, return_complex=True)
    
    signal_power = torch.sum(torch.abs(S_orig) ** 2, dim=(1,2))
    noise_power = torch.sum(torch.abs(S_orig - S_est) ** 2, dim=(1,2)) + EPS
    ssnr = 10.0 * torch.log10(signal_power / noise_power)
    return ssnr.mean().item()

def compute_si_sdr_db(orig: Tensor, est: Tensor) -> float:
    """
    Scale-Invariant Signal-to-Distortion Ratio in dB.
    Important for evaluating perceptual quality.
    """
    orig, est = prepare_signals(orig, est)

    # Zero-mean signals
    orig_zm = orig - orig.mean(dim=1, keepdim=True)
    est_zm = est - est.mean(dim=1, keepdim=True)

    # Optimal scaling factor
    scale = torch.sum(orig_zm * est_zm, dim=1) / (torch.sum(est_zm ** 2, dim=1) + EPS)
    est_scaled = scale.unsqueeze(1) * est_zm

    # SI-SDR
    signal_power = torch.sum(orig_zm ** 2, dim=1)
    distortion_power = torch.sum((orig_zm - est_scaled) ** 2, dim=1) + EPS
    si_sdr = 10.0 * torch.log10(signal_power / distortion_power)
    return si_sdr.mean().item()

def compute_snr_db(orig: Tensor, est: Tensor) -> float:
    """
    Signal-to-Noise Ratio in dB.
    Basic but important time-domain metric.
    """
    orig, est = prepare_signals(orig, est)
    
    signal_power = torch.sum(orig ** 2, dim=1)
    noise_power = torch.sum((orig - est) ** 2, dim=1) + EPS
    snr = 10.0 * torch.log10(signal_power / noise_power)
    return snr.mean().item()

def compute_lsd_db(orig: Tensor, est: Tensor) -> float:
    """
    Log-Spectral Distance in dB.
    Measures spectral accuracy, important for audio reconstruction.
    """
    orig, est = prepare_signals(orig, est)
    
    # Use FFT for spectral analysis
    REF = torch.fft.rfft(orig, n=N_FFT, dim=-1)
    EST = torch.fft.rfft(est, n=N_FFT, dim=-1)
    
    mag_ref = torch.abs(REF) + EPS
    mag_est = torch.abs(EST) + EPS
    
    log_diff = 10.0 * torch.log10(mag_ref) - 10.0 * torch.log10(mag_est)
    lsd = torch.mean(torch.abs(log_diff), dim=1)
    return lsd.mean().item()

def compute_spectral_convergence(orig: Tensor, est: Tensor) -> float:
    """
    Spectral Convergence metric.
    Measures how well the spectral magnitudes match.
    """
    orig, est = prepare_signals(orig, est)
    
    window = _METRIC_WINDOW.to(orig.device)
    S_orig = torch.stft(orig, n_fft=N_FFT, hop_length=HOP, win_length=WIN_LEN,
                        window=window, center=True, return_complex=True)
    S_est  = torch.stft(est,  n_fft=N_FFT, hop_length=HOP, win_length=WIN_LEN,
                        window=window, center=True, return_complex=True)
    
    mag_orig = torch.abs(S_orig)
    mag_est = torch.abs(S_est)
    
    numerator = torch.linalg.norm(mag_est - mag_orig, dim=(1,2))
    denominator = torch.linalg.norm(mag_orig, dim=(1,2)) + EPS
    return (numerator / denominator).mean().item()

def compute_mse(orig: Tensor, est: Tensor) -> float:
    """Mean Squared Error in time domain."""
    orig, est = prepare_signals(orig, est)
    return torch.mean((orig - est) ** 2, dim=1).mean().item()

# -----------------------------------------------------------------------------
#                              AGGREGATE METRICS
# -----------------------------------------------------------------------------

def evaluate_batch(pred_audio: Tensor, target_audio: Tensor) -> Dict[str, float]:
    """
    Compute all relevant metrics for a batch of audio predictions.
    
    This function is called by the DeepAGLA model during validation.
    Focuses on metrics most relevant for audio reconstruction quality.
    
    Args:
        pred_audio: Predicted audio (B, N) or (N,)
        target_audio: Target audio (B, N) or (N,)
        
    Returns:
        Dictionary with metric names and values
    """
    return {
        "SSNR(dB)": compute_ssnr_db(target_audio, pred_audio),
        "SI-SDR(dB)": compute_si_sdr_db(target_audio, pred_audio),
        "SNR(dB)": compute_snr_db(target_audio, pred_audio),
        "LSD(dB)": compute_lsd_db(target_audio, pred_audio),
        "SpectralConv": compute_spectral_convergence(target_audio, pred_audio),
        "MSE": compute_mse(target_audio, pred_audio),
    }

def compute_all_metrics(original, estimated) -> Dict[str, float]:
    """
    Legacy function name for compatibility with deep_agla.py.
    Redirects to evaluate_batch with appropriate argument order.
    """
    return evaluate_batch(estimated, original)

def match_signals(original, estimated):
    """
    Legacy function name for compatibility with deep_agla.py.
    Aligns and matches signals for metric computation.
    """
    return prepare_signals(original, estimated)

def compute_validation_ssnr(model, val_loader, device: torch.device) -> float:
    """
    Compute average SSNR on validation set for hyperparameter optimization.
    
    Args:
        model: Trained DeepAGLA model
        val_loader: Validation DataLoader
        device: Torch device
        
    Returns:
        Average SSNR in dB across validation set
    """
    model.eval()
    total_ssnr = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device, non_blocking=True)
            pred = model(batch)
            
            # Compute SSNR for this batch
            ssnr = compute_ssnr_db(batch, pred)
            total_ssnr += ssnr
            num_batches += 1
            
    return total_ssnr / num_batches if num_batches > 0 else 0.0

# -----------------------------------------------------------------------------
#                              TESTING & VALIDATION
# -----------------------------------------------------------------------------

def test_metrics_on_example():
    """Test metrics on example audio to verify correctness."""
    print("Testing evaluation metrics...")
    
    # Create test tensor from example audio
    if EXAMPLE_AUDIO is None:
        print("Warning: EXAMPLE_AUDIO not available, creating synthetic test signal")
        example_tensor = torch.randn(22050) * 0.1  # 1 second of synthetic audio
    else:
        example_tensor = torch.from_numpy(EXAMPLE_AUDIO).float()
    
    print(f"Example audio shape: {example_tensor.shape}")
    print(f"Example audio range: [{example_tensor.min():.3f}, {example_tensor.max():.3f}]")
    print(f"Example audio RMS: {torch.sqrt(torch.mean(example_tensor**2)):.3f}")
    
    # Test perfect reconstruction (should give excellent scores)
    print("\n--- Perfect Reconstruction Test ---")
    perfect_metrics = evaluate_batch(example_tensor, example_tensor)
    for metric, value in perfect_metrics.items():
        print(f"{metric}: {value:.3f}")
    
    # Test with slight noise (should give good but not perfect scores)
    print("\n--- Slight Noise Test (SNR ≈ 40dB) ---")
    noise_level = torch.sqrt(torch.mean(example_tensor**2)) * 0.01  # ~40dB SNR
    noisy_example = example_tensor + noise_level * torch.randn_like(example_tensor)
    noisy_metrics = evaluate_batch(noisy_example, example_tensor)
    for metric, value in noisy_metrics.items():
        print(f"{metric}: {value:.3f}")
    
    # Test with random signal (should give poor scores)
    print("\n--- Random Signal Test (should be poor) ---")
    random_signal = torch.randn_like(example_tensor) * torch.sqrt(torch.mean(example_tensor**2))
    random_metrics = evaluate_batch(random_signal, example_tensor)
    for metric, value in random_metrics.items():
        print(f"{metric}: {value:.3f}")
    
    print("\nMetrics test completed successfully!")
    
    # Validate that SSNR is reasonable for high-quality audio
    assert perfect_metrics["SSNR(dB)"] > 50, "Perfect reconstruction should have very high SSNR"
    assert noisy_metrics["SSNR(dB)"] > 20, "Slightly noisy signal should have decent SSNR"
    assert random_metrics["SSNR(dB)"] < 10, "Random signal should have poor SSNR"
    
    print("All metric validation checks passed!")

if __name__ == "__main__":
    test_metrics_on_example()