from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class MagAudioDataset(Dataset):
    """
    Dataset of paired items:
        • `mag`  : |STFT| magnitude  — shape (F, T)  float32
        • `audio`: waveform segment — shape (N,)   float32
    The data are stored in two `.npy` files created by the
    preprocessing script:

        data_dir/
            ├── mag.npy   : (num_items, F, T)
            └── audio.npy : (num_items, fixed_len)

    Memory-mapped loading keeps RAM usage low even for large sets.
    """
    def __init__(self, data_dir: str | Path) -> None:
        data_dir = Path(data_dir)
        self.mag:   np.ndarray = np.load(data_dir / "mag_gold.npy",   mmap_mode="r")
        self.audio: np.ndarray = np.load(data_dir / "audio_gold.npy", mmap_mode="r")

        if self.mag.shape[0] != self.audio.shape[0]:
            raise ValueError(
                f"#items mismatch: mag={self.mag.shape[0]}  "
                f"audio={self.audio.shape[0]}"
            )

    # ---- PyTorch Dataset interface -------------------------------------
    def __len__(self) -> int:
        return self.mag.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        mag   = torch.as_tensor(self.mag[idx],   dtype=torch.float32)  # (F, T)
        audio = torch.as_tensor(self.audio[idx], dtype=torch.float32)  # (N,)
        return mag.T, audio