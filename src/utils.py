# Copyright 2026 Muhammad Nizwa
# SPDX-License-Identifier: Apache-2.0

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def set_random_seed(seed: int):
    """
    Set random seed for reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DeviceDataLoader:
    """
    DeviceDataLoader is a wrapper around torch DataLoader that moves data to the specified device.

    Args:
        dl:
            torch DataLoader to wrap
        device:
            torch device to move data to
    """

    def __init__(
        self,
        dl: DataLoader,
        device: torch.device,
        non_blocking: bool,
    ):
        self.dl = dl
        self.device = device
        self.non_blocking = non_blocking

    def _to_device(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch.to(device=self.device, non_blocking=self.non_blocking)
        elif isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        else:
            return batch  # leave other types untouched

    def __iter__(self):
        for batch in self.dl:
            yield self._to_device(batch)

    def __len__(self):
        return len(self.dl)


def time_formatter(sec_elapsed: float) -> str:
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m}:{round(s, 1)}"
