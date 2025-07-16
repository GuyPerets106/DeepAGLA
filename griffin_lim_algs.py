import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import soundfile as sf

EXAMPLE_AUDIO = "/Users/guyperets/Documents/DeepAGLA/data/saspeech_gold_standard/wavs/gold_000_line_000.wav"
OUT_DIR = "/Users/guyperets/Downloads/MetricsCheck_GL_Algs"
N_FFT = 512
HOP = N_FFT // 4
WIN_LEN = N_FFT
EPS = 1e-12 # Small constant to avoid division by zero

# HELPER FUNCTIONS
def preprocess_loaded_audio(
    x: np.ndarray,
    sr: int,
    target_rms: float = 0.15,
    hp_cutoff: float = 30.0,
    n_fft: int = N_FFT,
    hop_length: int = HOP
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
        rms = np.sqrt(np.mean(x**2)) + 1e-12  # avoid /0 on silence
        x = x * (target_rms / rms)

    # 3) Pad to a multiple of hop_length
    hop_length = hop_length or (n_fft // 4)
    pad = (-len(x)) % hop_length
    if pad:
        x = np.pad(x, (0, pad))

    # 4) Cast to float32
    x = x.astype(np.float32, copy=False)
    return x

def stft(signal, n_fft=N_FFT, hop_length=HOP, win_length=WIN_LEN):
    """Compute the Short-Time Fourier Transform (STFT) of a signal."""
    return librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann')

def istft(spectrum, hop_length=HOP, win_length=WIN_LEN):
    """Compute the Inverse Short-Time Fourier Transform (ISTFT) of a spectrum."""
    return librosa.istft(spectrum, hop_length=hop_length, win_length=win_length, window='hann')

def proj_pc1(c):
    """
    Use the AGLA paper definition of PC1 which is stft(istft(spectrum=c))
    """
    return stft(istft(c))

def proj_pc2_old(c, s):
    """
    Use the AGLA paper definition of PC2 which is (spectrum=c)*(orig_mag_spec=s)/(np.abs(spectrum=c)) if spectrum=c != 0 else orig_mag_spec=s
    """
    res = np.zeros_like(c, dtype=np.complex64)
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if c[i, j] != 0:
                res[i, j] = c[i, j] * s[i, j] / np.abs(c[i, j])
            else:
                res[i, j] = s[i, j]
    return res

def proj_pc2(c, s):
    """
    Use the AGLA paper definition of PC2 which is (spectrum=c)*(orig_mag_spec=s)/(np.abs(spectrum=c)) if spectrum=c != 0 else orig_mag_spec=s
    """
    return s * np.exp(1j * np.angle(c))

# METRIC FUNCTIONS (Pass original and estimated time domain signals)
def match_length(original, estimated):
    """Match the length of the original and estimated signals."""
    min_length = min(len(original), len(estimated))
    return original[:min_length], estimated[:min_length]

def match_dtype(original, estimated):
    """Match the data type of the original and estimated signals, using np.float64."""
    original = np.asarray(original, dtype=np.float32)
    estimated = np.asarray(estimated, dtype=np.float32)
    return original, estimated

def align_signals(original, estimated):
    lag = np.argmax(np.correlate(estimated, original, mode='full')) - len(original) + 1
    estimated_aligned = np.roll(estimated, -lag)
    return original, estimated_aligned

def match_signals(original, estimated):
    """Match the original and estimated signals by aligning them."""
    original, estimated = align_signals(original, estimated)
    original, estimated = match_length(original, estimated)
    original, estimated = match_dtype(original, estimated)
    return original, estimated


def compute_ssnr_db(original, estimated):
    """Compute the Spectral Signal-to-Noise Ratio (SSNR) between original and estimated signals."""
    original_stft = stft(original)
    estimated_stft = stft(estimated)
    noise = np.abs(original_stft - estimated_stft)
    ssnr = 10 * np.log10(np.sum(np.abs(original_stft)**2) / (np.sum(noise**2) + EPS))
    return ssnr

def compute_psnr_db(original, estimated):
    """Compute the Peak Signal-to-Noise Ratio (PSNR) between original and estimated signals."""
    mse = np.mean((original - estimated)**2)
    if mse == 0:
        return float('inf')  # No noise
    max_pixel = np.max(np.abs(original))
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def compute_mse(original, estimated):
    """Compute the Mean Squared Error (MSE) between original and estimated signals."""
    return np.mean((original - estimated)**2)

def compute_mae(original, estimated):
    """Compute the Mean Absolute Error (MAE) between original and estimated signals."""
    return np.mean(np.abs(original - estimated))

def compute_ser_db(original, estimated):
    """Compute the Signal-to-Error Ratio (SER) in the STFT domain."""
    X = stft(original)
    Y = stft(estimated)
    signal_energy = np.sum(np.abs(X) ** 2)
    error_energy  = np.sum(np.abs(X - Y) ** 2) + EPS

    ser = 10.0 * np.log10(signal_energy / error_energy)
    return ser


def compute_snr_db(original, estimated):
    """Compute the (non-scale-invariant) Signal-to-Noise Ratio (SNR)."""
    noise = original - estimated
    snr = 10.0 * np.log10(np.sum(original ** 2) / (np.sum(noise ** 2) + EPS))
    return snr

def compute_sisdr_db(original, estimated):
    """Compute the Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) between."""
    x = original- np.mean(original)
    y = estimated - np.mean(estimated)

    scale = np.dot(x, y) / (np.dot(y, y) + EPS)
    y_scaled = scale * y

    sisdr = 10.0 * np.log10(np.sum(x ** 2) / (np.sum((x - y_scaled) ** 2) + EPS))
    return sisdr


def compute_sisnr_db(original, estimated):
    """Compute the Scale-Invariant Signal-to-Noise Ratio (SI-SNR) between."""
    x = original.astype(np.float64) - np.mean(original)
    y = estimated.astype(np.float64) - np.mean(estimated)

    scale = np.dot(x, y) / (np.dot(x, x) + EPS)
    x_scaled = scale * x

    sisnr = 10.0 * np.log10(np.sum(x_scaled ** 2) / (np.sum((y - x_scaled) ** 2) + EPS))
    return sisnr

def compute_stoi(original, estimated, sr=22050):
    """Compute the Short-Time Objective Intelligibility (STOI) score."""
    try:
        from pystoi import stoi
    except ImportError as e:
        raise ImportError("Please `pip install pystoi` to use compute_stoi") from e

    return stoi(original, estimated, sr)


def compute_lsd_db(original, estimated):
    """Compute the Log-Spectral Distance (LSD) in dB."""
    S_ref = np.abs(stft(original)) + EPS
    S_est = np.abs(stft(estimated)) + EPS

    log_diff = 10.0 * np.log10(S_ref) - 10.0 * np.log10(S_est)
    return np.mean(np.abs(log_diff))


def compute_spectral_convergence(original, estimated):
    """Compute spectral convergence = ‖|S_est| - |S_ref|‖₂ / ‖S_ref‖₂."""
    S_ref = np.abs(stft(original))
    S_est = np.abs(stft(estimated))
    return np.linalg.norm(S_est - S_ref) / (np.linalg.norm(S_ref) + EPS)


def compute_all_metrics(original, estimated):
    """Compute all metrics between original and estimated signals."""
    return {
        'SNR (dB)': compute_snr_db(original, estimated),
        'SSNR (dB)': compute_ssnr_db(original, estimated),
        'PSNR (dB)': compute_psnr_db(original, estimated),
        'MSE': compute_mse(original, estimated),
        'MAE': compute_mae(original, estimated),
        'SER (dB)': compute_ser_db(original, estimated),
        'SISDR (dB)': compute_sisdr_db(original, estimated),
        'SISNR (dB)': compute_sisnr_db(original, estimated),
        'STOI': compute_stoi(original, estimated),
        'LSD (dB)': compute_lsd_db(original, estimated),
        'Spectral Convergence': compute_spectral_convergence(original, estimated)
    }
    
def init_metrics_dict():
    return {
        'SNR (dB)': [],
        'SSNR (dB)': [],
        'PSNR (dB)': [],
        'MSE': [],
        'MAE': [],
        'SER (dB)': [],
        'SISDR (dB)': [],
        'SISNR (dB)': [],
        'STOI': [],
        'LSD (dB)': [],
        'Spectral Convergence': []
    }
    
# GL Algorithms

def naive_griffin_lim(sig, n_iter=64):
    """
    Naive Griffin-Lim algorithm as described in AGLA paper.
    Reconstructs a signal from its magnitude spectrogram.
    Calculates metrics at each iterarion.
    Returns the reconstructed signal and a list of metrics.
    """
    s = np.abs(stft(sig)) # Constant throughout the iterations
    c = s.astype(np.complex64)  # Initialize with the magnitude spectrogram as complex numbers, no phase information
    metrics = init_metrics_dict()
    for n in range(1, n_iter + 1):
        c = proj_pc1(proj_pc2(c, s))
        if n > 2: # Start measure metrics
            curr_est = istft(c)
            # Calculate metrics
            matched_length_sig, matched_length_est = match_signals(sig, curr_est)
            curr_metrics = compute_all_metrics(matched_length_sig, matched_length_est)
            for key, value in curr_metrics.items():
                metrics[key].append(value)
        
        
    reconstructed_signal = istft(c)
    return reconstructed_signal, metrics

def fast_griffin_lim(sig, n_iter=64):
    """
    Fast Griffin-Lim algorithm with momentum.
    Reconstructs a signal from its magnitude spectrogram.
    Calculates metrics at each iteration.
    Returns the reconstructed signal and a list of metrics.
    """
    s = np.abs(stft(sig))  # Constant throughout the iterations
    c0 = s.astype(np.complex64)  # Initialize with the magnitude spectrogram as complex numbers, no phase information
    t_prev = proj_pc1(proj_pc2(c0, s))  # Initial projection
    c = t_prev.copy()  # Initialize c with the first projection
    metrics = init_metrics_dict()
    
    for n in range(1, n_iter + 1):
        momentum = (n - 1) / (n + 2) # Like the paper suggested - momentum decreases over iterations to assure monotonic convergence
        t = proj_pc1(proj_pc2(c, s))
        c = t + momentum * (t - t_prev)
        t_prev = t.copy()
        curr_est = istft(t_prev)
        # Calculate metrics
        matched_length_sig, matched_length_est = match_signals(sig, curr_est)
        curr_metrics = compute_all_metrics(matched_length_sig, matched_length_est)
        for key, value in curr_metrics.items():
            metrics[key].append(value)
            
    reconstructed_signal = istft(c)
    return reconstructed_signal, metrics

def accelerated_griffin_lim(sig, n_iter=64, alpha=0.09, beta=1.1, gamma=1.25):
    """
    Accelerated Griffin-Lim algorithm with inertial extrapolation.
    Reconstructs a signal from its magnitude spectrogram.
    Calculates metrics at each iteration.
    Returns the reconstructed signal and a list of metrics.
    """
    s = np.abs(stft(sig))  # Constant throughout the iterations
    c0 = s.astype(np.complex64)  # Initialize with the magnitude spectrogram as complex numbers, no phase information
    t_prev = proj_pc1(proj_pc2(c0, s))  # Initial projection
    d_prev = t_prev.copy()  # Initialize d with the first projection
    c = t_prev.copy()  # Initialize c with the first projection
    metrics = init_metrics_dict()
    
    for n in range(1, n_iter + 1):
        alpha = (n - 1) / (n + 2)  # Decrease alpha over iterations
        beta = n / (n + 3)  # Decrease beta over iterations
        t = (1 - gamma) * d_prev + gamma * proj_pc1(proj_pc2(c, s))
        c = t + alpha * (t - t_prev)
        d = t + beta * (t - t_prev)
        t_prev = t.copy()
        d_prev = d.copy()
        curr_est = istft(t_prev)
        # Calculate metrics
        matched_length_sig, matched_length_est = match_signals(sig, curr_est)
        curr_metrics = compute_all_metrics(matched_length_sig, matched_length_est)
        for key, value in curr_metrics.items():
            metrics[key].append(value)
            
    reconstructed_signal = istft(c)
    return reconstructed_signal, metrics
     
def audio_stats(x, name):
    print(f"{name:10s}  min={x.min():.2e}  max={x.max():.2e}  "
          f"nan={np.isnan(x).any()}  inf={np.isinf(x).any()}")
  

if __name__ == "__main__":
    n_iter = 64
    sr = 22050
    original_signal = np.load("/gpfs0/bgu-benshimo/users/guyperet/DeepAGLA/data/preprocessed_audio_gold.npy")[0]
    print(f"Original signal length: {len(original_signal)} samples, Sample rate: {sr} Hz")
    
    print("----- Baseline SNR and Librosa Griffin-Lim sanity check -----")
    x_ref = original_signal
    x_chk = istft(stft(original_signal))
    x_ref, x_chk = match_signals(x_ref, x_chk)
    print(f"STFT⇄ISTFT baseline SNR: {10*np.log10(np.sum(x_ref**2)/np.sum((x_ref-x_chk)**2))} [dB]")
    # Try to use librosa.griffin_lim as a sanity check
    x_librosa = librosa.griffinlim(np.abs(stft(x_ref)), n_iter=n_iter, init=None)
    x_ref, x_librosa = match_signals(x_ref, x_librosa)
    print(f"Librosa Griffin-Lim SNR: {10*np.log10(np.sum(x_ref**2)/np.sum((x_ref-x_librosa)**2))} [dB]")
    
    # original_signal = preprocess_loaded_audio(original_signal, sr)
    # Run the naive Griffin-Lim algorithm
    print("\nRunning Naive Griffin-Lim algorithm...")
    reconstructed_signal_naive, metrics_naive = naive_griffin_lim(original_signal, n_iter=n_iter)
    print(f"Reconstructed signal length: {len(reconstructed_signal_naive)} samples")
    print(f"Metrics after {n_iter} iterations:")
    for metric, values in metrics_naive.items():
        print(f"{metric}: {values[-1]:.5f}")
        
    # Plot the metrics progress for Naive Griffin-Lim
    # sf.write(f"{OUT_DIR}/recon_naive.wav", reconstructed_signal_naive, sr)
    
    print("\nRunning Fast Griffin-Lim algorithm...")
    reconstructed_signal_fast, metrics_fast = fast_griffin_lim(original_signal, n_iter=n_iter)
    print(f"Reconstructed signal length: {len(reconstructed_signal_fast)} samples")
    print(f"Metrics after {n_iter} iterations:")
    for metric, values in metrics_fast.items():
        print(f"{metric}: {values[-1]:.5f}")
    # Save the reconstructed signal for Fast Griffin-Lim
    # sf.write(f"{OUT_DIR}/recon_fast.wav", reconstructed_signal_fast, sr)
    
    print("\nRunning Accelerated Griffin-Lim algorithm...")
    reconstructed_signal_accel, metrics_accel = accelerated_griffin_lim(original_signal, n_iter=n_iter)
    print(f"Reconstructed signal length: {len(reconstructed_signal_accel)} samples")
    print(f"Metrics after {n_iter} iterations:")
    for metric, values in metrics_accel.items():
        print(f"{metric}: {values[-1]:.5f}")

    # Save the reconstructed signal for Accelerated Griffin-Lim
    # sf.write(f"{OUT_DIR}/recon_accel.wav", reconstructed_signal_accel, sr)
    
    # Plot the metrics progress for all three algorithms
    # for metric in metrics_naive.keys():
    #     plt.figure(figsize=(12, 8))
    #     plt.plot(metrics_naive[metric], label=f'Naive {metric}')
    #     plt.plot(metrics_fast[metric], label=f'Fast {metric}')
    #     plt.plot(metrics_accel[metric], label=f'Accelerated {metric}')
    #     plt.xlabel('Iteration')
    #     plt.ylabel(f'{metric} Value')
    #     plt.title(f'{metric} Progress Over Iterations (All Algorithms)')
    #     plt.legend()
    #     plt.grid()
    #     # plt.savefig(f"{OUT_DIR}/{metric}_progress.png")
    #     plt.close()
        
    audio_stats(original_signal, "orig")
    audio_stats(reconstructed_signal_naive, "naive")
    audio_stats(reconstructed_signal_fast, "fast")
    audio_stats(reconstructed_signal_accel, "accel")