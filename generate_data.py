import argparse
import tarfile
import urllib.request
from pathlib import Path
from random import randint
from typing import List
from definitions import *
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

MEL_TRANSFORM = torchaudio.transforms.MelSpectrogram(SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS).to("cpu")

SLR_URL1 = "https://openslr.org/resources/134/saspeech_gold_standard_v1.0.tar.gz" # Roboshaul data (Gold Standard Only)
SLR_URL2 = "https://openslr.org/resources/134/saspeech_automatic_data_v1.0.tar.gz" # TODO Add Roboshaul data (Automatic Data Only)


def download_data(dest: Path) -> Path:
    """
    Download and extract SLR-134 to *dest/wavs*; return that path
    """
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "saspeech_gold_standard_v1.0.tar.gz"

    if not archive.exists():
        print("▸ Downloading Roboshaul Gold Standard Dataset (967MB)")
        with tqdm(unit="B", unit_scale=True, unit_divisor=1024,
                  desc="  Progress", miniters=1) as t:

            def hook(blocks: int, block_size: int, total_size: int):
                if total_size > 0:
                    t.total = total_size
                t.update(blocks * block_size - t.n)

            urllib.request.urlretrieve(SLR_URL1, archive, reporthook=hook)

    print("▸ Extracting …")
    with tarfile.open(archive) as tf:
        tf.extractall(dest)

    wav_root = next(dest.glob("**/saspeech_gold_standard/wavs"))
    print(f"▸ WAVs extracted to {wav_root}")
    return wav_root

def median_length(samples_list: List[int]) -> int:
    """
    Return the median of a list of ints
    """
    samples_sorted = sorted(samples_list)
    mid = len(samples_sorted) // 2
    return (
        samples_sorted[mid]
        if len(samples_sorted) % 2
        else (samples_sorted[mid - 1] + samples_sorted[mid]) // 2
    )


def force_length(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Crop (random) or loop a waveform to exactly target_len samples
    """
    cur = wav.size(-1)
    if cur == target_len:
        return wav
    if cur > target_len:  # random crop
        start = randint(0, cur - target_len)
        return wav[start : start + target_len]
    # loop
    reps = target_len // cur + 1
    wav = wav.repeat(reps)[:target_len]
    return wav


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", type=Path, required=False, help="folder of *.wav files")
    ap.add_argument("--data_dir", type=Path, required=True, help="output folder")
    ap.add_argument("--download", action="store_true", help=f"fetch from {SLR_URL} automatically")
    args = ap.parse_args()

    if args.download:
        if args.wav_dir is not None:
            ap.error("Use --download OR --wav_dir, not both.")
        wav_root = download_data(args.data_dir)
    elif args.wav_dir is None:
        ap.error("Must supply --wav_dir or --download")
    else:
        wav_root = args.wav_dir

    wav_paths = sorted(p for p in wav_root.rglob("*.wav"))
    if not wav_paths:
        raise RuntimeError("No .wav files found")

    print("▸ Scanning durations …")
    lengths: List[int] = []
    for p in tqdm(wav_paths, unit="file"):
        info = torchaudio.info(str(p))
        sr = info.sample_rate
        dur = info.num_frames
        if sr != SAMPLE_RATE:
            dur = int(dur * SAMPLE_RATE / sr)
        lengths.append(dur)

    med_len = median_length(lengths)
    print(f"• Median Length = {med_len / SAMPLE_RATE:.2f} s  ({med_len} samples)")

    # Pre-allocate output arrays
    N = len(wav_paths)
    mel_frames = (med_len // HOP) + 1
    audio_buf = np.empty((N, med_len), dtype=np.float32)
    mel_buf   = np.empty((N, N_MELS, mel_frames), dtype=np.float32)

    print("▸ Processing & buffering arrays")
    for i, p in enumerate(tqdm(wav_paths, unit="file")):
        wav, sr = torchaudio.load(str(p))
        wav = wav.mean(0)  # 1-channel

        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

        wav = wav / torch.max(torch.abs(wav)).clamp_min(1e-8)  # normalize to [-1,1]
        wav = force_length(wav, med_len)

        audio_buf[i] = wav.numpy()

        mel = MEL_TRANSFORM(wav)[:mel_frames]
        mel_db = 10 * np.log10(mel.numpy() + np.finfo(np.float32).eps)
        mel_db = mel_db - mel_db.max() # 0db peak
        mel_buf[i] = mel_db.astype(np.float32)

    # ------------------------------------------------------------------ #
    # One-shot save → valid .npy with header
    args.data_dir.mkdir(parents=True, exist_ok=True)
    np.save(f"{args.data_dir}/audio.npy", audio_buf)
    np.save(f"{args.data_dir}/mel.npy", mel_buf)

    print("✔ Saved arrays to", args.data_dir)

if __name__ == "__main__":
    main()