"""Exercise the pipeline with temporary synthetic data; requires CUDA, no real dataset."""
import argparse
import csv
import importlib.util
from pathlib import Path
import tempfile

import cv2
import numpy as np
import torch

from data import PulseDataset, read_manifest, subject_splits
from metrics import estimate_hr, waveform_metrics
from preprocess import main as preprocess, convert_recording
import face_crop
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


def compare_reference(reference_path, root):
    """Verify array equality against user-supplied source without running its CLI."""
    spec = importlib.util.spec_from_file_location("reference_preprocess", reference_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    raw = root / "parity_raw" / "s1"
    raw.mkdir(parents=True)
    video = raw / "vid_s1_T1.avi"
    bvp_path = raw / "bvp_s1_T1.csv"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 35, (32, 24))
    if not writer.isOpened():
        raise RuntimeError("MJPG encoder unavailable")
    try:
        yy, xx = np.indices((24, 32))
        for frame in range(40):
            rgb = np.stack((20 + xx + frame, 100 + yy, 170 + xx - yy), axis=-1)
            writer.write(rgb.astype(np.uint8))
    finally:
        writer.release()
    bvp = np.sin(np.arange(80) * 0.15) + 0.2 * np.cos(np.arange(80) * 0.31)
    with bvp_path.open("w") as handle:
        handle.write("BVP\n\n")
        np.savetxt(handle, bvp)
    for box in ([0, 0, 9, 12], [20, 10, 5, 8], [1, 2, 15, 11]):
        np.testing.assert_array_equal(face_crop._enlarge(box, 1.5), reference._enlarge(box, 1.5))
    class HaarFixture:
        def detectMultiScale(self, image):
            assert image.shape[-1] == 3  # RGB, with the same default detector arguments.
            return np.array([[1, 1, 4, 10], [2, 2, 8, 3]])
    fixture_rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    np.testing.assert_array_equal(face_crop._largest_face(HaarFixture(), fixture_rgb),
                                  reference._largest_face(HaarFixture(), fixture_rgb))
    class YoloFixture:
        def detect_face(self, image):
            return [2, 3, 17, 12]
    np.testing.assert_array_equal(face_crop._yolo_box(YoloFixture(), fixture_rgb),
                                  reference._yolo_box(YoloFixture(), fixture_rgb))
    no_face = [("none", lambda image: None)]
    found = [("fixture", lambda image: [1, 2, 15, 11])]
    for case, primary, fallback in (("detected", found, None), ("fallback", no_face, found),
                                     ("no_face", no_face, None)):
        expected_box, _ = reference.determine_face_box(str(video), primary, 5, 1.5, fallback)
        actual_box, _ = face_crop.determine_face_box(video, primary, 5, 1.5, fallback)
        np.testing.assert_array_equal(actual_box, expected_box)
        expected_rgb, corrupt = reference.stream_crop_resize(str(video), expected_box, 16, 16)
        assert corrupt == 0
        expected_bvp = reference.transform_label(reference.resample_signal(reference.read_signal_csv(str(bvp_path)),
                                                len(expected_rgb)), "Standardized")
        for uint8 in (True, False):
            out = root / (case + ("_raw" if uint8 else "_float"))
            out.mkdir()
            rows, audit = convert_recording(video, bvp_path, out, 16, 16, "auto",
                                             (primary, fallback), 5, 1.5, store_uint8=uint8)
            expected_input = expected_rgb.astype(np.uint8) if uint8 else reference.standardized_data(expected_rgb)
            assert audit["dropped_tail_frames"] == 8
            for row in rows:
                start = row["start_frame"]
                actual_input = np.load(out / row["input_path"])
                actual_bvp = np.load(out / row["label_path"])
                assert actual_input.dtype == expected_input.dtype
                assert actual_bvp.dtype == expected_bvp.dtype == np.float64
                np.testing.assert_array_equal(actual_input, expected_input[start:start + 16])
                np.testing.assert_array_equal(actual_bvp, expected_bvp[start:start + 16])
    corrected, _ = face_crop.determine_face_box(video, no_face, fix_full_frame_box=True)
    np.testing.assert_array_equal(corrected, [0, 0, 32, 24])
    print("PASS: reference crop paths and uint8/float32 RGB plus float64 BVP match exactly.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-preprocess", type=Path, help="Optional original Python source for parity checks")
    args = parser.parse_args()
    cuda_device("cuda:0")
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="rhythmmamba_check_") as name:
        root = Path(name)
        if args.reference_preprocess:
            compare_reference(args.reference_preprocess, root)
        raw, cache, run, scores = [root / part for part in ("raw", "cache", "train", "test")]
        for subject in range(1, 6):
            for task in ("T1", "T2", "T3"):
                synthetic_recording(raw, "s" + str(subject), task)
        preprocess(["--data_path", str(raw), "--cached_path", str(cache), "--chunk_length", "160",
                    "--size", "16", "--crop", "full-frame", "--data_type", "Standardized",
                    "--label_type", "Standardized", "--store_uint8"])
        manifest = cache / "manifest.csv"
        rows = read_manifest(manifest)
        assert len(rows) == 45
        assert {row["frames"] for row in rows} == {160}
        assert np.load(rows[0]["label_path"]).dtype == np.float64
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
