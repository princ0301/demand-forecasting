import numpy as np
import torch
from torch.utils.data import Dataset

class TimeSeriesWindowDataset(Dataset):

    def __init__(
        self,
        series_arrays: dict[str, np.ndarray],
        series_static: dict[str, np.ndarray],
        lookback: int,
        horizon: int,
        stride: int,
    ) -> None:

        self.series_arrays = series_arrays
        self.series_static = series_static
        self.lookback = lookback
        self.horizon = horizon

        self.samples: list[tuple[str, int]] = []

        for series_id, arr in series_arrays.items():
            max_start = arr.shape[0] - lookback - horizon

            if max_start < 0:
                continue

            for start in range(0, max_start + 1, stride):
                self.samples.append((series_id, start))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        series_id, start = self.samples[idx]
        arr = self.series_arrays[series_id]

        encoder_window = arr[start: start + self.lookback]
        target = arr[start + self.lookback: start + self.lookback + self.horizon, 0]
        static = self.series_static[series_id]

        return (
            torch.from_numpy(encoder_window).float(),
            torch.from_numpy(static).long(),
            torch.from_numpy(target).float(),
        )