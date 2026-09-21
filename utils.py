"""Small shared helpers; experiment metadata stays inside PyTorch checkpoints."""
import csv
import inspect
import random
from pathlib import Path

import numpy as np
import torch

CHECKPOINT_FORMAT = "rhythmmamba_standalone_v1"


def new_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError("Use a new or empty output directory: " + str(path))
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty CSV: " + str(path))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cuda_device(name):
    device = torch.device(name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("RhythmMamba requires CUDA. Use Linux/WSL with the documented Mamba environment.")
    return device


def load_checkpoint(path):
    # Our checkpoints contain tensors plus primitive metadata, never model objects.
    options = {"weights_only": True} if "weights_only" in inspect.signature(torch.load).parameters else {}
    checkpoint = torch.load(path, map_location="cpu", **options)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Expected a checkpoint from this folder; use train.py --init-weights for upstream weights")
    return checkpoint


def restore_model(checkpoint, device):
    from model import RhythmMamba
    model = RhythmMamba(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model
