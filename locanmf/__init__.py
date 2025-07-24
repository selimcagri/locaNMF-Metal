name = "demix"
__all__ = ["LocaNMF", "DEFAULT_DEVICE"]

import torch

# Default device preferring CUDA, then Apple Silicon (MPS), then CPU
DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else
    "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else
    "cpu"
)

from .LocaNMF import LocaNMF
