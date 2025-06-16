import argparse
import tarfile
import urllib.request
from pathlib import Path
from random import randint
from typing import List, Literal

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
import audio_utilities
from definitions import SAMPLE_RATE, N_FFT, HOP, N_MELS, FMIN, FMAX, N_LIN

SLR_URL_GOLD = "https://openslr.org/resources/134/saspeech_gold_standard_v1.0.tar.gz"
SLR_URL_AUTO = "https://openslr.org/resources/134/saspeech_automatic_data_v1.0.tar.gz"
Subset = Literal["gold", "auto"]

# -----------------------------------------------------------------------------
# Download helper
# -----------------------------------------------------------------------------

def download_data(dest: Path, subset: Subset = "auto") -> Path:
    url, archive_name, wav_glob = {
        "gold": (SLR_URL_GOLD, "saspeech_gold_standard_v1.0.tar.gz", "**/saspeech_gold_standard/wavs"),
        "auto": (SLR_URL_AUTO, "saspeech_automatic_data_v1.0.tar.gz", "**/saspeech_automatic_data/wavs"),
    }[subset]

    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / archive_name

    if not archive.exists():
        print("▸ Downloading", "Gold" if subset == "gold" else "Automatic", "subset …")
        with tqdm(unit="B", unit_scale=True, unit_divisor=1024, desc="  Progress") as t:
            def hook(b, bs, ts):
                if ts > 0:
                    t.total = ts
                t.update(b * bs - t.n)
            urllib.request.urlretrieve(url, archive, hook)
    else:
        print("▸ Archive already present – skipping download")

    print("▸ Extracting …")
    with tarfile.open(archive) as tf:
        tf.extractall(dest)

    [wav_root] = dest.glob(wav_glob)
    print("▸ WAVs extracted to", wav_root)
    return wav_root

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def median_length(vals: List[int]) -> int:
    vals = sorted(vals)
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) // 2


def force_length(wav: np.ndarray, target: int) -> np.ndarray:
    cur = wav.shape[-1]
    if cur == target:
        return wav
    if cur > target:
        start = randint(0, cur - target)
        return wav[start:start + target]
    reps = target // cur + 1
    return np.tile(wav, reps)[:target]

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--wav_dir", type=Path, metavar="DIR", help="Existing directory of .wav files")
    g.add_argument("--download", nargs="?", const="auto", choices=("auto", "gold"), metavar="{auto,gold}",
                   help="Download subset (defaults to 'auto')")
    p.add_argument("--data_dir", type=Path, required=True, metavar="DIR", help="Where to write .npy arrays")
    return p

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    wav_root = download_data(args.data_dir, subset=args.download) if args.download else args.wav_dir
    if not wav_root.is_dir():
        raise SystemExit("Provided --wav_dir is not a directory: " + str(wav_root))

    wav_paths = sorted(wav_root.rglob("*.wav"))
    if not wav_paths:
        raise RuntimeError("No .wav files found under " + str(wav_root))

    print("▸ Scanning durations …")
    lengths = []
    for p in tqdm(wav_paths, unit="file", desc="  Durations"):
        info = sf.info(str(p))
        frames = int(info.frames * SAMPLE_RATE / info.samplerate) if info.samplerate != SAMPLE_RATE else info.frames
        lengths.append(frames)

    med_len = median_length(lengths)
    print(f"• Median length = {med_len / SAMPLE_RATE:.2f} s  ({med_len} samples)")

    N          = len(wav_paths)
    T = 1 + (med_len - N_FFT) // HOP

    audio_buf  = np.empty((N, med_len), dtype=np.float32)
    mag_buf = np.empty((N, T, N_LIN), dtype=np.float32)

    print("▸ Processing …")
    for i, p in enumerate(tqdm(wav_paths, unit="file")):
        wav, sr = sf.read(p)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if sr != SAMPLE_RATE:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)

        wav = force_length(wav / max(np.abs(wav).max(), 1e-8), med_len)
        audio_buf[i] = wav

        stft = audio_utilities.stft_for_reconstruction(wav, N_FFT, HOP)
        power_spec = np.abs(stft)**2.0
        scale = 1.0 / np.amax(power_spec)
        power_spec *= scale
        mag_modified = power_spec/scale
        mag_modified = mag_modified**0.5 # abs(STFT)
        mag_buf[i] = mag_modified
        
    example_audio = audio_buf[0]
    example_mel = mag_buf[0]
    print(f"Data Example Shaes:")
    print(f"Audio Shape: {example_audio.shape}")
    print(f"Magnitude Spectrogram Shape: {example_mel.shape}")

    args.data_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.data_dir / f"audio_{args.download}.npy", audio_buf)
    np.save(args.data_dir / f"mag_{args.download}.npy", mag_buf)
    print("✔  Saved arrays in", args.data_dir)

if __name__ == "__main__":
    main()