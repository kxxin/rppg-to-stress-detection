"""Detector/crop conventions from the user's standalone UBFC-PHYS preprocessor.

Boxes use [x, y, width, height]. Auto scans with Haar, then tries YOLO only on
the first frame. Preserve that ordering for compatibility with existing caches.
"""
from pathlib import Path
import sys

import cv2
import numpy as np


def _largest_face(detector, rgb):
    zones = detector.detectMultiScale(rgb[:, :, :3].astype(np.uint8))
    if len(zones) < 1:
        return None
    return list(zones[int(np.argmax(zones[:, 2]))]) if len(zones) >= 2 else list(zones[0])


def _yolo_box(yolo, rgb):
    result = yolo.detect_face(rgb[:, :, :3].astype(np.uint8))
    if result is None:
        return None
    x_min, y_min, x_max, y_max = result
    width, height = x_max - x_min, y_max - y_min
    cx, cy = x_min + width // 2, y_min + height // 2
    side = max(width, height)
    return [cx - side // 2, cy - side // 2, side, side]


def _enlarge(box, coef):
    box = list(box)
    box[0] = max(0, box[0] - (coef - 1.0) / 2 * box[2])
    box[1] = max(0, box[1] - (coef - 1.0) / 2 * box[3])
    box[2] = coef * box[2]
    box[3] = coef * box[3]
    return box


def build_haar_detectors(primary_cascade=None):
    # Prefer the original working-directory cascade when available, then OpenCV's
    # equivalent default cascade. The tested WSL copies have identical SHA256.
    local = Path("dataset/haarcascade_frontalface_default.xml")
    if primary_cascade is not None:
        primary = Path(primary_cascade)
        if not primary.is_file():
            raise FileNotFoundError("Missing explicitly requested cascade: " + str(primary))
    else:
        primary = local if local.is_file() else Path(cv2.data.haarcascades) / local.name
    paths = [primary]
    for name in ("haarcascade_frontalface_alt2.xml", "haarcascade_frontalface_alt.xml"):
        path = Path(cv2.data.haarcascades) / name
        if path.is_file() and path not in paths:
            paths.append(path)
    detectors = []
    for path in paths:
        if path.is_file():
            detector = cv2.CascadeClassifier(str(path))
            if detector.empty():
                raise ValueError("Invalid Haar cascade: " + str(path))
            detectors.append(("haar:" + path.name, lambda rgb, d=detector: _largest_face(d, rgb)))
    if not detectors:
        raise RuntimeError("No Haar detector available")
    return detectors


def load_yolo(device="cpu", toolbox_dir=None):
    """Optional adapter to an existing Toolbox installation; no weights bundled."""
    if toolbox_dir is not None:
        directory = Path(toolbox_dir).resolve()
        if not (directory / "dataset/data_loader/face_detector/YOLO5Face.py").is_file():
            raise ValueError("--toolbox-dir must contain dataset/data_loader/face_detector/YOLO5Face.py")
        sys.path.insert(0, str(directory))
    from dataset.data_loader.face_detector.YOLO5Face import YOLO5Face
    yolo = YOLO5Face("Y5F", device)
    return [("yolo5face", lambda rgb: _yolo_box(yolo, rgb))]


def build_detectors(mode="auto", cascade=None, yolo_device="cpu", toolbox_dir=None):
    haar = build_haar_detectors(cascade) if mode in ("haar", "auto") else []
    yolo = None
    if mode in ("yolo", "auto"):
        try:
            yolo = load_yolo(yolo_device, toolbox_dir)
            print("YOLO5Face loaded on " + yolo_device)
        except Exception as error:
            if mode == "yolo":
                raise RuntimeError("YOLO requested but unavailable: " + str(error)) from error
            print("YOLO unavailable; auto uses Haar only: " + str(error))
    if mode == "haar":
        return haar, None
    if mode == "yolo":
        return yolo, None
    if mode != "auto":
        raise ValueError("Detector must be auto, haar, or yolo")
    return haar, yolo


def determine_face_box(video, primary, search_frames=60, larger_box_coef=1.5,
                       fallback=None, fix_full_frame_box=False):
    """Match the reference's detector ordering, expansion, rounding, and fallback.

    Its historical no-face box swaps width and height. Preserve it by default
    for cache parity; fix_full_frame_box opts into a correctly sized full frame.
    """
    capture = cv2.VideoCapture(str(video))
    first_rgb, found, found_name, searched = None, None, None, 0
    try:
        success, frame = capture.read()
        while success and searched < search_frames:
            if frame is not None:
                rgb = cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB)
                if first_rgb is None:
                    first_rgb = rgb
                for name, detector in primary:
                    box = detector(rgb)
                    if box is not None:
                        found, found_name = box, name
                        break
                if found is not None:
                    break
                searched += 1
            success, frame = capture.read()
    finally:
        capture.release()
    if found is not None:
        return np.asarray(_enlarge(found, larger_box_coef), dtype=int), "face via {} (scanned {} frame[s])".format(found_name, searched + 1)
    if fallback and first_rgb is not None:
        for name, detector in fallback:
            box = detector(first_rgb)
            if box is not None:
                return np.asarray(_enlarge(box, larger_box_coef), dtype=int), "face via {} (fallback)".format(name)
    if first_rgb is None:
        raise ValueError("No readable frames: " + str(video))
    height, width = first_rgb.shape[:2]
    box = [0, 0, width, height] if fix_full_frame_box else [0, 0, height, width]
    status = "NO FACE FOUND -> " + ("corrected full frame" if fix_full_frame_box else "legacy full-frame box (height,width)")
    return np.asarray(box, dtype=int), status
