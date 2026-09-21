"""Exercise the pipeline with temporary synthetic data; requires CUDA, no real dataset."""
import csv
from pathlib import Path
import tempfile

import cv2
import numpy as np
import torch

from data import PulseDataset, read_manifest, subject_splits
from metrics import estimate_hr, waveform_metrics
from preprocess import main as preprocess
from train import main as train
from test import main as evaluate
from predict import main as predict
from utils import cuda_device, load_checkpoint


def synthetic_recording(root, subject, task, frames=500, fs=35):
    """Write a small moving-color AVI and a known pulse; no face detection needed."""
    folder = root / subject
    folder.mkdir(parents=True, exist_ok=True)
    recording = subject + "_" + task
    frequency = 1.0 + int(subject[1:]) * 0.07 + int(task[1:]) * 0.03
    video = folder / ("vid_" + recording + ".avi")
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), fs, (16, 16))
    if not writer.isOpened():
        raise RuntimeError("OpenCV needs an MJPG encoder for this synthetic check")
    try:
        spatial = np.indices((16, 16)).sum(axis=0)
        for frame in range(frames):
            pulse = 12 * np.sin(2 * np.pi * frequency * frame / fs)
            bgr = np.stack((40 + spatial, 100 + spatial + pulse, 160 + spatial), axis=-1)
            writer.write(np.clip(bgr, 0, 255).astype(np.uint8))
    finally:
        writer.release()
    signal_time = np.arange(int(frames / fs * 64)) / 64
    bvp = np.sin(2 * np.pi * frequency * signal_time)
    np.savetxt(folder / ("bvp_" + recording + ".csv"), bvp, delimiter=",")


def main():
    cuda_device("cuda:0")
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="rhythmmamba_check_") as name:
        root = Path(name)
        raw, cache, run, scores = [root / part for part in ("raw", "cache", "train", "test")]
        for subject in range(1, 6):
            for task in ("T1", "T2", "T3"):
                synthetic_recording(raw, "s" + str(subject), task)
        preprocess(["--raw-dir", str(raw), "--out", str(cache), "--frames", "160",
                    "--size", "16", "--crop", "full-frame"])
        manifest = cache / "manifest.csv"
        rows = read_manifest(manifest)
        assert len(rows) == 45
        assert {row["frames"] for row in rows} == {160}
        with (cache / "recording.csv").open(newline="") as handle:
            assert all(int(row["dropped_tail_frames"]) == 20 for row in csv.DictReader(handle))
        splits = subject_splits(rows)
        partition_subjects = [{r["subject"] for r in rows if splits[r["subject"]] == name}
                              for name in ("train", "valid", "test")]
        assert all(not partition_subjects[i] & partition_subjects[j]
                   for i in range(3) for j in range(i))
        sample = PulseDataset(rows)[0]
        expected = np.load(rows[0]["input_path"]).astype(np.float32)
        expected = (expected - rows[0]["input_mean"]) / rows[0]["input_std"]
        np.testing.assert_allclose(sample["video"].numpy(), expected.transpose(0, 3, 1, 2))
        preprocess(["--cache-dir", str(cache), "--out", str(root / "index"), "--fps", "35",
                    "--input-representation", "raw", "--label-representation", "standardized"])
        assert len(read_manifest(root / "index" / "manifest.csv")) == 45
        time = np.arange(350) / 35
        known = np.sin(2 * np.pi * 1.2 * time)
        assert abs(estimate_hr(known, 35) - 72) < 1
        assert waveform_metrics(known, known, 35)["waveform_pearson"] > 0.999
        assert np.isnan(waveform_metrics(np.zeros_like(known), known, 35)["hr_pred_bpm"])
        train(["--manifest", str(manifest), "--out", str(run), "--epochs", "2",
               "--batch-size", "2", "--depth", "1", "--embed-dim", "16"])
        checkpoint = load_checkpoint(run / "best.pt")
        assert checkpoint["splits"] == splits
        evaluate(["--manifest", str(manifest), "--checkpoint", str(run / "best.pt"),
                  "--out", str(scores), "--window-seconds", "10", "--save-waveforms"])
        selected = next(r for r in rows if splits[r["subject"]] == "test")
        output = root / "single.npy"
        predict(["--input", selected["input_path"], "--checkpoint", str(run / "best.pt"),
                 "--representation", "raw", "--input-mean", str(selected["input_mean"]),
                 "--input-std", str(selected["input_std"]), "--out", str(output)])
        whole = np.load(scores / (selected["recording"] + "_prediction.npy"))
        np.testing.assert_allclose(np.load(output), whole[:160], atol=3e-5, rtol=3e-5)
        with (scores / "windows.csv").open(newline="") as handle:
            windows = list(csv.DictReader(handle))
        assert len(windows) == 3
        assert {r["subject"] for r in windows} == partition_subjects[2]
        assert not list(root.rglob("*.json"))
    print("PASS: preprocessing, cache indexing, subject isolation, metrics, CUDA training, checkpoint reload, test, prediction.")
    print("Synthetic checks verify execution only; no UBFC-PHYS accuracy has been measured.")


if __name__ == "__main__":
    main()
