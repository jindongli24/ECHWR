import os
import random

import numpy as np
import torch

__all__ = ['seed_everything', 'seed_worker', 'sec2time']


def seed_everything(seed: int = 42) -> None:
    '''Seeds all random number generators for reproducibility.

    This covers PyTorch (CPU & GPU), NumPy, Python's random module, and sets
    the PYTHONHASHSEED environment variable. It also forces CuDNN to be
    deterministic to ensure consistent results across runs.

    Args:
        seed: The seed value to use. Defaults to 42.
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    '''Seeds a specific DataLoader worker.

    Intended to be used as the `worker_init_fn` in
    `torch.utils.data.DataLoader` to ensure that data loading processes are
    deterministic.

    Args:
        worker_id: The unique identifier for the worker.
    '''
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def sec2time(time_sec: float) -> str:
    '''Converts seconds into a formatted time string (H:MM:SS).

    Args:
        time_sec: The duration in seconds.

    Returns:
        The formatted time string.
    '''
    second = str(int(time_sec % 60)).zfill(2)
    minute = str(int(time_sec // 60) % 60).zfill(2)
    hour = int(time_sec // 3600)
    time = f'{hour}:{minute}:{second}'

    return time
