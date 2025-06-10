import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

from dataset import MelAudioDataset
from deep_agla import DeepAGLA
from definitions import *

import torch
torch.set_float32_matmul_precision("high")

class MelDataModule(pl.LightningDataModule):
    """
    Loads a single (mel.npy, audio.npy) bundle and splits it deterministically
    into train/val/test according to TRAIN_RATIO / VAL_RATIO constants.
    """
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.bs = batch_size
        self.nw = num_workers
        self.seed = seed

    def setup(self, stage: str | None = None) -> None:
        full_dataset = MelAudioDataset(self.data_dir)
        n = len(full_dataset)

        n_train = int(TRAIN_RATIO * n)
        n_val = int(VAL_RATIO * n)
        n_test = n - n_train - n_val

        lengths = [n_train, n_val, n_test]

        # Deterministic split for every call (fit, test, etc.)
        train_set, val_set, test_set = random_split(full_dataset, lengths, generator=torch.Generator().manual_seed(self.seed))

        if stage in (None, "fit"):
            self.train_set, self.val_set = train_set, val_set
        if stage in (None, "test"):
            self.test_set = test_set

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.bs,
            shuffle=True,
            num_workers=self.nw,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.bs,
            shuffle=False,
            num_workers=self.nw,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.bs,
            shuffle=False,
            num_workers=self.nw,
            pin_memory=True,
        )

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--n_layers", type=int, default=N_LAYERS)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--max_epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--project", type=str, default="deep-agla")
    p.add_argument("--mode", type=str, default="convergence", choices=["convergence", "overall"])
    args = p.parse_args()


    loss_weights = {
        "mrstft": 1.0,
        "l1":     10.0,
        "phase":  0.2
        }

    model = DeepAGLA(n_layers=args.n_layers, lr=args.lr, loss_weights=loss_weights, initial_params=BEST_INITIAL_COMBINATIONS[args.mode])
    dm = MelDataModule(args.data_dir, args.batch_size)

    logger = WandbLogger(project=args.project)
    ckpt_cb = ModelCheckpoint(monitor="val/SSNRdB", mode="max", save_top_k=1)
    lr_cb = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        logger=logger,
        callbacks=[ckpt_cb, lr_cb],
        gradient_clip_val=1.0
    )
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()