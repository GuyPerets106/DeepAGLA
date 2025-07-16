
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    """
    Lazily-loaded 2-D NumPy memmap wrapped as a PyTorch ``Dataset``.

    Parameters
    ----------
    npy_path : str | Path
        Path to the **pre-processed** ``.npy`` file (shape = ``(N, T)``).
    dtype : np.dtype, default np.float32
        Dtype of the waveforms *inside* the file.  The returned tensors are
        always ``float32``.
    """

    def __init__(self, npy_path: str | Path, dtype: np.dtype = np.float32) -> None:
        npy_path = Path(npy_path).expanduser()
        if not npy_path.is_file():
            raise FileNotFoundError(npy_path)

        # Memory-map the file --> no data read yet
        self._data = np.load(npy_path, mmap_mode='r')
        if self._data.ndim != 2:
            raise ValueError(
                f"Expected 2-D array (N, T); got shape {self._data.shape}"
            )
        if self._data.dtype != dtype:
            raise ValueError(
                f"Dtype mismatch: file has {self._data.dtype}, expected {dtype}"
            )

    # ------------------------------------------------------------------ #
    # Required Dataset interface                                         #
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self._data.shape[0]

    def __getitem__(self, idx: int) -> Tuple[int, Tensor]:
        """
        Returns
        -------
        tuple
            ``(idx, waveform)`` where ``waveform`` is ``float32`` tensor
            of shape ``(T,)`` on the *CPU*.
        """
        sample: np.ndarray = self._data[idx]
        wav = torch.tensor(sample, dtype=torch.float32)
        return wav