import numpy as np
import librosa

def stft_for_reconstruction(x, fft_size, hopsamp):
    """Compute and return the STFT of the supplied time domain signal x.

    Args:
        x (1-dim Numpy array): A time domain signal.
        fft_size (int): FFT size. Should be a power of 2, otherwise DFT will be used.
        hopsamp (int):

    Returns:
        The STFT. The rows are the time slices and columns are the frequency bins.
    """
    window = np.hanning(fft_size)
    fft_size = int(fft_size)
    hopsamp = int(hopsamp)
    return np.array([np.fft.rfft(window*x[i:i+fft_size])
                     for i in range(0, len(x)-fft_size, hopsamp)])

def istft_for_reconstruction(X, fft_size, hopsamp):
    """Invert a STFT into a time domain signal.

    Args:
        X (2-dim Numpy array): Input spectrogram. The rows are the time slices and columns are the frequency bins.
        fft_size (int):
        hopsamp (int): The hop size, in samples.

    Returns:
        The inverse STFT.
    """
    fft_size = int(fft_size)
    hopsamp = int(hopsamp)
    window = np.hanning(fft_size)
    time_slices = X.shape[0]
    len_samples = int(time_slices*hopsamp + fft_size)
    x = np.zeros(len_samples)
    for n,i in enumerate(range(0, len(x)-fft_size, hopsamp)):
        x[i:i+fft_size] += window*np.real(np.fft.irfft(X[n]))
    return x


def griffin_lim_naive(magnitude_spectrogram, fft_size, hopsamp, iterations):
    """Reconstruct an audio signal from a magnitude spectrogram.

    Given a magnitude spectrogram as input, reconstruct
    the audio signal and return it using the Griffin-Lim algorithm from the paper:
    "Signal estimation from modified short-time fourier transform" by Griffin and Lim,
    in IEEE transactions on Acoustics, Speech, and Signal Processing. Vol ASSP-32, No. 2, April 1984.

    Args:
        magnitude_spectrogram (2-dim Numpy array): The magnitude spectrogram. The rows correspond to the time slices
            and the columns correspond to frequency bins.
        fft_size (int): The FFT size, which should be a power of 2.
        hopsamp (int): The hope size in samples.
        iterations (int): Number of iterations for the Griffin-Lim algorithm. Typically a few hundred
            is sufficient.

    Returns:
        The reconstructed time domain signal as a 1-dim Numpy array.
    """
    time_slices = magnitude_spectrogram.shape[0]
    len_samples = int(time_slices*hopsamp + fft_size)

    # Initialize the reconstructed signal to noise.
    x_reconstruct = np.random.randn(len_samples)
    n = iterations # number of iterations of Griffin-Lim algorithm.
    while n > 0:
        n -= 1
        reconstruction_spectrogram = stft_for_reconstruction(x_reconstruct, fft_size, hopsamp)
        reconstruction_angle = np.angle(reconstruction_spectrogram)

        # Discard magnitude part of the reconstruction and use the supplied magnitude spectrogram instead.
        proposal_spectrogram = magnitude_spectrogram*np.exp(1.0j*reconstruction_angle)
        prev_x = x_reconstruct
        x_reconstruct = istft_for_reconstruction(proposal_spectrogram, fft_size, hopsamp)
    return x_reconstruct

def fast_griffin_lim_librosa(
    magnitude_spectrogram,
    n_iter=60,
    hop_length=512,
    win_length=2048,
    window='hann',
    momentum=0.99,
    init='random',
    length=None
):
    """
    Reconstruct a time-domain signal from a magnitude spectrogram
    using the Fast Griffin-Lim Algorithm (FGLA) as implemented in librosa.

    Parameters
    ----------
    magnitude_spectrogram : np.ndarray [shape=(n_fft/2+1, t)]
        Magnitude spectrogram (typically obtained from STFT)

    n_iter : int
        Number of iterations

    hop_length : int or None
        Number of audio samples between adjacent STFT columns

    win_length : int or None
        Each frame of audio is windowed by `window()` of length `win_length`

    window : str
        Window function used in STFT and ISTFT

    momentum : float [0.0–1.0]
        Momentum parameter; use 0.99 for Fast Griffin-Lim, 0.0 for classic GLA

    init : str ['random', 'zeros']
        Phase initialization method

    length : int or None
        If provided, sets the output signal length (in samples)

    Returns
    -------
    y : np.ndarray
        Time-domain signal reconstructed from magnitude_spectrogram
    """
    return librosa.griffinlim(
        S=magnitude_spectrogram,
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        momentum=momentum,
        init=init,
        length=length
    )
def accelerated_griffin_lim(
    S_mag,
    n_iter=60,
    alpha=0.2,
    beta=0.8,
    gamma=0.99,
    hop_length=None,
    win_length=None,
    window="hann",
    init="random",
    length=None,
    random_state=None,
):
    """
    Implements Algorithm 3: Accelerated Griffin-Lim from Perraudin et al. (2013).

    Parameters
    ----------
    S_mag : np.ndarray
        Magnitude spectrogram (|STFT(x)|)
    n_iter : int
        Number of iterations
    alpha, beta, gamma : float
        Acceleration parameters (α, β, γ > 0)
    hop_length : int
        Hop length for STFT
    win_length : int
        Window length for STFT
    window : str
        Window type for STFT/ISTFT
    init : str
        'random' or 'zeros'
    length : int or None
        Target length of the reconstructed signal
    random_state : int or None
        For reproducibility if init='random'

    Returns
    -------
    y : np.ndarray
        Time-domain reconstructed signal
    """

    rng = np.random.default_rng(random_state)

    # Step 1: Initialize c0, t0, d0
    if init == "random":
        phase = np.exp(2j * np.pi * rng.random(S_mag.shape))
    elif init == "zeros":
        phase = np.ones_like(S_mag, dtype=np.complex64)
    else:
        raise ValueError("init must be 'random' or 'zeros'")

    c = S_mag * phase
    t = c.copy()
    d = c.copy()

    for n in range(n_iter):
        # t_n = (1 - γ) * d_{n-1} + γ * PC1(PC2(c_{n-1}))
        # PC2: project to magnitude (replace magnitude)
        # PC1: project to consistent spectrogram (via STFT of ISTFT)
        x_temp = librosa.istft(c, hop_length=hop_length, win_length=win_length, window=window, length=length)
        c_temp = librosa.stft(x_temp, hop_length=hop_length, win_length=win_length, window=window)

        phase_proj = c_temp / np.maximum(1e-8, np.abs(c_temp))
        c_proj = S_mag * phase_proj  # PC2 projection
        x_consistent = librosa.istft(c_proj, hop_length=hop_length, win_length=win_length, window=window, length=length)
        c_consistent = librosa.stft(x_consistent, hop_length=hop_length, win_length=win_length, window=window)

        t_new = (1 - gamma) * d + gamma * c_consistent

        # c_n = t_n + α (t_n - t_{n-1})
        c_new = t_new + alpha * (t_new - t)

        # d_n = t_n + β (t_n - t_{n-1})
        d_new = t_new + beta * (t_new - t)

        # Shift variables
        t = t_new
        c = c_new
        d = d_new

    # Final inverse transform: T†(c_N)
    y = librosa.istft(c, hop_length=hop_length, win_length=win_length, window=window, length=length)
    return y

def accelerated_griffin_lim_from_mel(
    M,
    *,
    sr: int,
    n_fft: int,
    hop_length: int | None = None,
    win_length: int | None = None,
    fmin: float = 0.0,
    fmax: float | None = None,
    power: float = 1.0,          # 1.0 → magnitude-mel, 2.0 → power-mel
    n_iter: int = 60,
    alpha: float = 0.2,
    beta: float = 0.8,
    gamma: float = 0.99,
    init: str = "random",
    length: int | None = None,
    random_state: int | None = None,
):
    """
    Reconstructs a time-domain signal from a **mel spectrogram** using the
    Accelerated Griffin-Lim algorithm (Perraudin et al., 2013).

    Parameters
    ----------
    M : np.ndarray [shape=(n_mels, n_frames)]
        Mel-spectrogram (magnitude or power, controlled by ``power``).
    sr, n_fft, hop_length, win_length : int
        STFT parameters (must match those used to create ``M``).
    fmin, fmax : float
        Lower / upper frequency limits of the mel filter bank.
    power : {1.0, 2.0}
        If 1.0 ``M`` is magnitude-mel, if 2.0 ``M`` is power-mel.
    n_iter, alpha, beta, gamma, init, length, random_state
        See ``accelerated_griffin_lim`` documentation.

    Returns
    -------
    y : np.ndarray
        Reconstructed time-domain signal.
    """

    # ------------------------------------------------------------------
    # 1) MEL → linear-frequency magnitude
    # ------------------------------------------------------------------
    #
    # librosa provides a numerically robust pseudo-inverse:
    #   librosa.feature.inverse.mel_to_stft
    #
    # power controls the (.)**(1/power) conversion so that
    # power=2.0 treats M as power-mel and returns magnitude STFT.
    #
    S_mag = librosa.feature.inverse.mel_to_stft(
        M,
        sr=sr,
        n_fft=n_fft,
        power=power,
        fmin=fmin,
        fmax=fmax,
    )

    # ------------------------------------------------------------------
    # 2) Run accelerated Griffin-Lim on the linear spectrogram
    # ------------------------------------------------------------------
    rng = np.random.default_rng(random_state)

    if init == "random":
        phase = np.exp(2j * np.pi * rng.random(S_mag.shape))
    elif init == "zeros":
        phase = np.ones_like(S_mag, dtype=np.complex64)
    else:
        raise ValueError("init must be 'random' or 'zeros'")

    c = S_mag * phase
    t = c.copy()
    d = c.copy()

    for _ in range(n_iter):
        x = librosa.istft(c, hop_length=hop_length, win_length=win_length, length=length)
        C = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

        phase_proj = C / np.maximum(1e-8, np.abs(C))
        C_proj = S_mag * phase_proj                        # PC₂
        x_consistent = librosa.istft(C_proj, hop_length=hop_length,
                                     win_length=win_length, length=length)
        C_consistent = librosa.stft(x_consistent, n_fft=n_fft,
                                    hop_length=hop_length, win_length=win_length)

        t_new = (1 - gamma) * d + gamma * C_consistent     # tₙ
        c_new = t_new + alpha * (t_new - t)                # cₙ
        d_new = t_new + beta * (t_new - t)                 # dₙ

        t, c, d = t_new, c_new, d_new

    return librosa.istft(c, hop_length=hop_length, win_length=win_length, length=length)