"""Evaluate a saved partition using chronological, nonoverlapping rPPG windows."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import PulseDataset, manifest_hash, read_manifest
from losses import normalize_waveform
from metrics import summary_metrics, waveform_metrics
from utils import cuda_device, load_checkpoint, new_directory, restore_model, write_csv


def evaluate_recording(clips, fs, window_seconds):
    """Join consecutive clips only; return window metrics, signals, and tail count."""
    clips = sorted(clips, key=lambda clip: clip[0]["start_frame"])
    expected = clips[0][0]["start_frame"]
    origin = expected
    for row, prediction, target in clips:
        if row["start_frame"] != expected:
            raise ValueError("Missing or overlapping clips in " + row["recording"])
        expected += len(prediction)
    prediction = np.concatenate([clip[1] for clip in clips])
    target = np.concatenate([clip[2] for clip in clips])
    length = int(round(window_seconds * fs))
    if length > len(prediction):
        raise ValueError("Recording shorter than the evaluation window: " + clips[0][0]["recording"])
    rows = []
    identity = clips[0][0]
    for start in range(0, len(prediction) - length + 1, length):
        scores = waveform_metrics(prediction[start:start + length], target[start:start + length], fs)
        rows.append(dict(subject=identity["subject"], recording=identity["recording"],
                         task=identity["task"], start_frame=origin + start,
                         frames=length, fps=fs, **scores))
    return rows, prediction, target, len(prediction) % length


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--save-waveforms", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.workers < 0 or not np.isfinite(args.window_seconds) or args.window_seconds < 5:
        parser.error("Use positive batch-size, nonnegative workers, and windows >=5 seconds")
    device = cuda_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint)
    if manifest_hash(args.manifest) != checkpoint["manifest_sha256"]:
        raise ValueError("Manifest differs from training. Use the original CSV to evaluate its saved partition.")
    rows = read_manifest(args.manifest)
    rows = [r for r in rows if checkpoint["splits"][r["subject"]] == args.split]
    loader = DataLoader(PulseDataset(rows), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers, pin_memory=True)
    model = restore_model(checkpoint, device)
    out = new_directory(args.out)
    # Hold only 1D waveforms, never full recording videos, for chronological evaluation.
    recordings = defaultdict(list)
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predict " + args.split):
            prediction = normalize_waveform(model(batch["video"].to(device))).cpu().numpy()
            if not np.isfinite(prediction).all():
                raise FloatingPointError("Nonfinite predicted waveform")
            for index, pulse, target in zip(batch["index"].tolist(), prediction, batch["target"].numpy()):
                row = rows[index]
                recordings[row["recording"]].append((row, pulse, target))
    windows, audit = [], []
    for name, clips in sorted(recordings.items()):
        scores, prediction, target, tail = evaluate_recording(clips, rows[0]["fps"], args.window_seconds)
        windows.extend(scores)
        audit.append(dict(recording=name, retained_frames=len(prediction), evaluated_windows=len(scores),
                          dropped_evaluation_tail_frames=tail))
        if args.save_waveforms:
            np.save(out / (name + "_prediction.npy"), prediction, allow_pickle=False)
            np.save(out / (name + "_target.npy"), target, allow_pickle=False)
    summaries = [dict(scope="all_windows", **summary_metrics(windows))]
    for task in sorted({r["task"] for r in windows}):
        summaries.append(dict(scope=task, **summary_metrics([r for r in windows if r["task"] == task])))
    write_csv(out / "windows.csv", windows)
    write_csv(out / "metrics.csv", summaries)
    write_csv(out / "recordings.csv", audit)
    for key, value in summaries[0].items():
        print(f"{key}: {value}")
    print("Results:", out)


if __name__ == "__main__":
    main()
