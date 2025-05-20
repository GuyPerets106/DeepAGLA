import argparse
import csv
from pathlib import Path

import soundfile as sf
import torch
from tqdm import tqdm

import pytorch_lightning as pl

from dataset import MelAudioDataset
from deep_agla import DeepAGLA
from definitions import SAMPLE_RATE
from eval_metrics import evaluate_batch


# ------------------------------------------------------------------ #
def save_wave(path: Path, audio: torch.Tensor) -> None:
    audio = torch.clamp(audio, -1.0, 1.0).cpu().numpy()
    sf.write(path, audio, SAMPLE_RATE, subtype="FLOAT")


# ------------------------------------------------------------------ #
def cli_main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data_dir", type=Path, required=True, help="folder with test/mel.npy & audio.npy")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gpus", type=int, default=1)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Model (Lightning handles device mapping)
    model = DeepAGLA.load_from_checkpoint(str(args.checkpoint))
    model.eval()
    trainer = pl.Trainer(accelerator="auto", devices=args.gpus)

    # Data loader
    test_set = NpyMel2AudioDataset(Path(args.data_dir) / "test")
    loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    metrics_sum, n_items = {}, 0
    for i, (mel, ref) in enumerate(tqdm(loader, desc="Testing")):
        with torch.no_grad():
            pred = trainer.lightning_module(mel.to(model.device))

        # save wavs
        for b in range(pred.size(0)):
            idx = i * args.batch_size + b
            save_wave(args.out_dir / f"pred_{idx:05d}.wav", pred[b])
            save_wave(args.out_dir / f"ref_{idx:05d}.wav", ref[b])

        batch_metrics = evaluate_batch(pred.cpu(), ref.cpu(), fs=SAMPLE_RATE)
        for k, v in batch_metrics.items():
            metrics_sum[k] = metrics_sum.get(k, 0.0) + v * pred.size(0)
        n_items += pred.size(0)

    # Aggregate
    mean_metrics = {k: v / n_items for k, v in metrics_sum.items()}
    csv_path = args.out_dir / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in mean_metrics.items():
            writer.writerow([k, f"{v:.4f}"])
    print("Saved metrics →", csv_path)


if __name__ == "__main__":
    cli_main()