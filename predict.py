"""Predict a pulse waveform from one preprocessed RGB NumPy clip, without BVP."""
import argparse
from pathlib import Path

import numpy as np
import torch

from losses import normalize_waveform
from utils import cuda_device, load_checkpoint, restore_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="RGB array [T,H,W,3]")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New waveform .npy path")
    parser.add_argument("--representation", choices=("raw", "standardized"), required=True)
    parser.add_argument("--input-mean", type=float, help="Recording RGB mean from manifest.csv (raw only)")
    parser.add_argument("--input-std", type=float, help="Recording RGB std from manifest.csv (raw only)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.out.suffix.lower() != ".npy" or args.out.exists():
        parser.error("--out must be a new .npy file")
    if (args.input_mean is None) != (args.input_std is None):
        parser.error("Supply --input-mean and --input-std together")
    if args.representation == "standardized" and args.input_mean is not None:
        parser.error("Already standardized inputs do not use mean/std overrides")
    checkpoint = load_checkpoint(args.checkpoint)
    raw = np.load(args.input, allow_pickle=False)
    if raw.ndim != 4 or raw.shape[-1] != 3 or not np.isfinite(raw).all():
        raise ValueError("Expected finite RGB input [T,H,W,3]")
    config = checkpoint["data_config"]
    if raw.shape[1:3] != (config["height"], config["width"]):
        raise ValueError("Use the same spatial preprocessing size as training")
    if raw.shape[0] != config["frames"]:
        raise ValueError("Use the same clip length as training for this baseline")
    video = raw.astype(np.float32)
    if args.representation == "raw":
        if raw.dtype != np.uint8:
            raise ValueError("raw representation requires uint8 RGB")
        mean = float(video.mean()) if args.input_mean is None else args.input_mean
        std = float(video.std()) if args.input_std is None else args.input_std
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 1e-8:
            raise ValueError("RGB mean/std must be finite and std must be positive")
        if args.input_mean is None:
            print("Using this clip's RGB statistics; pass manifest mean/std to match recording normalization.")
        video = (video - mean) / std
    elif not np.issubdtype(raw.dtype, np.floating):
        raise ValueError("standardized representation requires floating-point RGB")
    device = cuda_device(args.device)
    model = restore_model(checkpoint, device)
    inputs = torch.from_numpy(video.transpose(0, 3, 1, 2).copy()).unsqueeze(0).to(device)
    with torch.no_grad():
        pulse = normalize_waveform(model(inputs))[0].cpu().numpy()
    if not np.isfinite(pulse).all():
        raise FloatingPointError("Nonfinite predicted waveform")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, pulse, allow_pickle=False)
    print(f"Saved {len(pulse)} normalized waveform samples at the expected {config['fps']} Hz: {args.out}")


if __name__ == "__main__":
    main()
