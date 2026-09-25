"""Deterministic seeding utility for reproducible experiments across runtimes."""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Sets global random seeds for Python, NumPy, and PyTorch (CPU and CUDA).
    
    Enforces deterministic algorithm execution where supported per Constitution Principle II.
    
    Args:
        seed: The integer random seed to apply globally.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
