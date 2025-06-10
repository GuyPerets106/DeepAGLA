from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class MelAudioDataset(Dataset):
    def __init__(self, data_dir: str | Path):
        data_dir = Path(data_dir)
        self.mel: np.ndarray = np.load(f'{data_dir}/mel.npy', mmap_mode="r")  # (N, N_MELS, T)
        self.audio: np.ndarray = np.load(f'{data_dir}/audio.npy', mmap_mode="r")  # (N, FIXED_LEN)
        assert self.mel.shape[0] == self.audio.shape[0], "Number of samples in mel and audio must match"

    def __len__(self) -> int:
        return self.mel.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        mel = torch.tensor(self.mel[idx], dtype=torch.float32)
        wav = torch.tensor(self.audio[idx], dtype=torch.float32)
        return mel, wav