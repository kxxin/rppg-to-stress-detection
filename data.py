"""CSV manifests, subject-exclusive partitions, and lazy NumPy loading."""
import csv
import hashlib
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def manifest_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_manifest(path):
    path = Path(path).resolve()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"input_path", "label_path", "subject", "recording", "task",
                "clip_index", "start_frame", "frames", "fps", "height", "width",
                "input_mean", "input_std", "input_representation", "label_representation"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Empty or incompatible manifest; generate it with preprocess.py")
    keys, inputs, targets = set(), set(), set()
    for row in rows:
        if not re.fullmatch(r"s[1-9]\d*", row["subject"]) or row["task"] not in ("T1", "T2", "T3"):
            raise ValueError("Expected a canonical UBFC-PHYS subject (s1, s2, ...) and task T1/T2/T3")
        for name in ("clip_index", "start_frame", "frames", "height", "width"):
            row[name] = int(row[name])
        for name in ("fps", "input_mean", "input_std"):
            row[name] = float(row[name])
            if not np.isfinite(row[name]):
                raise ValueError("Nonfinite manifest value: " + name)
        if (row["frames"] < 16 or row["frames"] % 8 or
                min(row["height"], row["width"]) < 16 or
                row["height"] % 8 or row["width"] % 8):
            raise ValueError("Use frames >=16 divisible by 8 and image sides >=16 divisible by 8")
        if row["fps"] <= 5 or row["input_std"] <= 0:
            raise ValueError("FPS must exceed 5 Hz and input_std must be positive")
        if row["clip_index"] < 0 or row["start_frame"] < 0:
            raise ValueError("Negative clip/frame index")
        if row["start_frame"] != row["clip_index"] * row["frames"]:
            raise ValueError("This cache contract requires consecutive nonoverlapping clips from frame zero")
        if row["input_representation"] not in ("raw", "standardized"):
            raise ValueError("Only raw RGB and standardized RGB are supported")
        if row["label_representation"] != "standardized":
            raise ValueError("Targets must be standardized BVP, not differences or stress labels")
        if row["recording"] != row["subject"] + "_" + row["task"]:
            raise ValueError("Recording identity does not match subject/task")
        key = (row["recording"], row["clip_index"])
        if key in keys:
            raise ValueError("Duplicate recording/clip: " + str(key))
        keys.add(key)
        for name, seen in (("input_path", inputs), ("label_path", targets)):
            file = Path(row[name])
            file = (path.parent / file).resolve() if not file.is_absolute() else file.resolve()
            if file in seen or not file.is_file():
                raise ValueError("Missing or duplicated array: " + str(file))
            seen.add(file)
            row[name] = str(file)
        video = np.load(row["input_path"], mmap_mode="r", allow_pickle=False)
        target = np.load(row["label_path"], mmap_mode="r", allow_pickle=False)
        if video.shape != (row["frames"], row["height"], row["width"], 3):
            raise ValueError("Manifest/video shape mismatch: " + row["input_path"])
        if target.shape != (row["frames"],) or not np.issubdtype(target.dtype, np.floating):
            raise ValueError("Expected floating BVP array [T]: " + row["label_path"])
        expected_raw = row["input_representation"] == "raw"
        if (expected_raw and video.dtype != np.uint8) or (
                not expected_raw and not np.issubdtype(video.dtype, np.floating)):
            raise ValueError("Input dtype disagrees with representation: " + row["input_path"])
    signatures = {(r["frames"], r["height"], r["width"], r["fps"]) for r in rows}
    if len(signatures) != 1:
        raise ValueError("One run requires a common frame count, image size, and FPS")
    return sorted(rows, key=lambda r: (r["recording"], r["clip_index"]))


def subject_splits(rows, seed=100, splits_csv=None):
    subjects = sorted({r["subject"] for r in rows})
    if splits_csv:
        with Path(splits_csv).open(newline="", encoding="utf-8-sig") as handle:
            entries = list(csv.DictReader(handle))
        if any(set(r) != {"subject", "split"} for r in entries):
            raise ValueError("Split CSV requires exactly subject,split columns")
        assignments = {r["subject"]: r["split"] for r in entries}
        if len(assignments) != len(entries) or set(assignments) != set(subjects):
            raise ValueError("Split CSV must assign every subject exactly once")
    else:
        if len(subjects) < 5:
            raise ValueError("At least five subjects required for the default 60/20/20 split")
        shuffled = subjects.copy()
        random.Random(seed).shuffle(shuffled)
        n = max(1, round(len(shuffled) * 0.2))
        assignments = {s: "test" if i < n else "valid" if i < 2 * n else "train"
                       for i, s in enumerate(shuffled)}
    if set(assignments.values()) != {"train", "valid", "test"}:
        raise ValueError("Splits must contain nonempty train, valid, and test partitions")
    return assignments


class PulseDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
        if not rows:
            raise ValueError("Empty dataset partition")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        video = np.load(row["input_path"], allow_pickle=False).astype(np.float32)
        if row["input_representation"] == "raw":
            video = (video - row["input_mean"]) / row["input_std"]
        target = np.load(row["label_path"], allow_pickle=False).astype(np.float32)
        if not np.isfinite(video).all() or not np.isfinite(target).all():
            raise ValueError("Nonfinite data: " + row["input_path"])
        if target.std() < 1e-8:
            raise ValueError("Constant BVP target: " + row["label_path"])
        return {"video": torch.from_numpy(video.transpose(0, 3, 1, 2).copy()),
                "target": torch.from_numpy(target), "index": index}
