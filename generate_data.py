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
import scipy.signal
from definitions import SAMPLE_RATE, N_FFT, HOP

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
    g.add_argument("--download", nargs="?", const="both", choices=("auto", "gold", "both"), metavar="{auto,gold,both}",
                   help="Download subset (defaults to 'both' for combined dataset)")
    p.add_argument("--data_dir", type=Path, required=True, metavar="DIR", help="Where to write .npy arrays")
    return p

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    # Handle different download options
    if args.download:
        if args.download == "both":
            # Download both datasets and combine them
            print("▸ Downloading and combining both GOLD and AUTO datasets...")
            gold_root = download_data(args.data_dir, subset="gold")
            auto_root = download_data(args.data_dir, subset="auto")
            
            # Collect WAV paths from both datasets
            gold_paths = sorted(gold_root.rglob("*.wav"))
            auto_paths = sorted(auto_root.rglob("*.wav"))
            wav_paths = gold_paths + auto_paths
            
            print(f"▸ Found {len(gold_paths)} GOLD files and {len(auto_paths)} AUTO files")
            print(f"▸ Total: {len(wav_paths)} WAV files")
            
        else:
            # Download single dataset
            wav_root = download_data(args.data_dir, subset=args.download)
            wav_paths = sorted(wav_root.rglob("*.wav"))
    else:
        # Use existing WAV directory
        wav_root = args.wav_dir
        if not wav_root.is_dir():
            raise SystemExit("Provided --wav_dir is not a directory: " + str(wav_root))
        wav_paths = sorted(wav_root.rglob("*.wav"))

    if not wav_paths:
        raise RuntimeError("No .wav files found")

    print("▸ Scanning durations …")
    lengths = []
    for p in tqdm(wav_paths, unit="file", desc="  Durations"):
        info = sf.info(str(p))
        frames = int(info.frames * SAMPLE_RATE / info.samplerate) if info.samplerate != SAMPLE_RATE else info.frames
        lengths.append(frames)

    med_len = median_length(lengths)
    print(f"• Median length = {med_len / SAMPLE_RATE:.2f} s  ({med_len} samples)")

    N = len(wav_paths)
    audio_buf = np.empty((N, med_len), dtype=np.float32)

    print("▸ Processing …")
    for i, p in enumerate(tqdm(wav_paths, unit="file")):
        audio, sr = librosa.load(p)
        audio_buf[i] = force_length(audio, med_len)
        
    example_audio = audio_buf[0]
    print(f"Data Example Shapes:")
    print(f"Audio Shape: {example_audio.shape}")
    print(f"Total Dataset Shape: {audio_buf.shape}")

    args.data_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output filename based on what was downloaded/processed
    if args.download == "both":
        output_name = "audio_gold_auto.npy"
        print(f"✔ Combining {len(gold_paths)} GOLD + {len(auto_paths)} AUTO files")
    elif args.download:
        output_name = f"audio_{args.download}.npy"
    
    output_path = args.data_dir / output_name
    np.save(output_path, audio_buf)
    print(f"✔ Saved combined dataset to {output_path}")
    print(f"✔ Final shape: {audio_buf.shape}")

if __name__ == "__main__":
    main()