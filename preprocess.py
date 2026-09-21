"""UBFC-PHYS preprocessing compatible with the user's Standardized/uint8 command.

RGB is stored as uint8 THWC. The manifest records one RGB mean/std per full
recording so the dataset can reproduce recording-level z-score normalization.
BVP is interpolated to the decoded video length and standardized before
chunking, matching the upstream loader's endpoint-alignment assumption. This
is not timestamp-based synchronization. Existing caches require an explicit
declaration of their input/label representations and actual sampling rate.
"""

import argparse
import csv
from pathlib import Path
import re

import cv2
import numpy as np

from face_crop import build_detectors, determine_face_box


MANIFEST_FIELDS = [
    "input_path", "label_path", "subject", "recording", "task", "clip_index",
    "start_frame", "frames", "fps", "height", "width", "input_mean", "input_std",
    "input_representation", "label_representation",
]
AUDIT_FIELDS = [
    "subject", "recording", "task", "source", "bvp_source", "fps",
    "reported_frames", "decoded_frames", "retained_frames", "dropped_tail_frames",
    "clips", "crop", "crop_x1", "crop_y1", "crop_x2", "crop_y2",
    "input_mean", "input_std", "statistics_scope", "alignment", "detector_status",
]


def identity(path):
    """Return the canonical subject/task and reject folder/name disagreement."""
    path = Path(path)
    match = re.search(r"(?:^|_)s(\d+)_(T[123])(?:_|\.)", path.name, re.IGNORECASE)
    if match is None:
        raise ValueError("Cannot identify a UBFC-PHYS recording: {}".format(path))
    subject, task = "s{}".format(int(match[1])), match[2].upper()
    if re.fullmatch(r"s\d+", path.parent.name, re.IGNORECASE):
        parent_subject = "s{}".format(int(path.parent.name[1:]))
        if parent_subject != subject:
            raise ValueError("Subject folder/filename mismatch: {}".format(path))
    return subject, task, "{}_{}".format(subject, task)


def read_bvp(path):
    """Match the reference CSV parser (first column, skipping headers/empty rows)."""
    values = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            try:
                values.append(float(row[0]))
            except ValueError:
                continue
    signal = np.asarray(values, dtype=np.float64)
    if signal.size < 2 or not np.isfinite(signal).all() or np.std(signal) <= 1e-8:
        raise ValueError("BVP must contain finite, nonconstant samples: {}".format(path))
    return signal


def _stats(total, squared_total, count, source):
    mean = total / count
    std = np.sqrt(max(0.0, squared_total / count - mean * mean))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 1e-8:
        raise ValueError("RGB recording is nonfinite or constant: {}".format(source))
    return float(mean), float(std)


def validate_dimensions(frames, height, width):
    """Reject cache/model-incompatible shapes before writing a manifest."""
    if (frames < 16 or frames % 8 or min(height, width) < 16
            or height % 8 or width % 8):
        raise ValueError(
            "Use frames >=16 divisible by 8 and image sides >=16 divisible by 8"
        )


def convert_recording(video, bvp_path, out, frames=160, size=128, crop="auto",
                      detectors=None, search_frames=60, margin=1.5,
                      fix_full_frame_box=False, store_uint8=True):
    """Stream one video to disk; return manifest rows and recording audit data."""
    validate_dimensions(frames, size, size)
    if crop not in ("auto", "haar", "yolo", "full-frame"):
        raise ValueError("crop must be auto, haar, yolo, or full-frame")
    video, out = Path(video), Path(out)
    subject, task, recording = identity(video)
    bvp = read_bvp(bvp_path)
    box, status = None, "explicit corrected full frame"
    if crop != "full-frame":
        primary, fallback = detectors if detectors is not None else build_detectors(crop)
        box, status = determine_face_box(video, primary, search_frames, margin, fallback,
                                         fix_full_frame_box)
    capture = cv2.VideoCapture(str(video))
    rows, buffer, float_frames = [], [], []
    decoded, pixel_count = 0, 0
    total, squared_total = 0.0, 0.0
    original_shape = None
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open video: {}".format(video))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        reported = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or fps <= 5:
            raise ValueError("Video FPS must be finite and exceed 5 Hz: {}".format(video))
        if not np.isfinite(reported) or reported < 0:
            raise ValueError("Invalid reported video frame count: {}".format(video))
        reported_frames = int(round(reported))
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("Invalid decoded frame {}: {}".format(decoded, video))
            if original_shape is None:
                original_shape = frame.shape
                if box is None:
                    box = (0, 0, frame.shape[1], frame.shape[0])
            elif frame.shape != original_shape:
                raise ValueError("Video dimensions changed during decoding: {}".format(video))
            x, y, width, height = box
            rgb = cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB)
            cropped = rgb[max(y, 0):min(y + height, rgb.shape[0]),
                          max(x, 0):min(x + width, rgb.shape[1])]
            if not cropped.size:
                raise ValueError("Empty crop: {}".format(video))
            rgb = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
            if not store_uint8:
                float_frames.append(rgb.astype(np.float32))
            values = rgb.astype(np.float64)
            total += float(np.sum(values))
            squared_total += float(np.sum(values * values))
            pixel_count += rgb.size
            buffer.append(rgb)
            decoded += 1
            if len(buffer) == frames:
                index = len(rows)
                input_name = "{}_input{}.npy".format(recording, index)
                label_name = "{}_label{}.npy".format(recording, index)
                if store_uint8:
                    np.save(out / input_name, np.stack(buffer), allow_pickle=False)
                rows.append(dict(
                    input_path=input_name, label_path=label_name, subject=subject,
                    recording=recording, task=task, clip_index=index,
                    start_frame=decoded - frames, frames=frames, fps=fps,
                    height=size, width=size,
                    input_representation="raw" if store_uint8 else "standardized",
                    label_representation="standardized",
                ))
                buffer.clear()
    finally:
        capture.release()
    if decoded < frames:
        raise ValueError("Only {} frames decoded; need {}: {}".format(decoded, frames, video))
    if reported_frames > 0 and decoded != reported_frames:
        raise ValueError(
            "Decoded {} of {} reported frames; cannot assume BVP alignment: {}".format(
                decoded, reported_frames, video
            )
        )
    mean, std = _stats(total, squared_total, pixel_count, video)
    # This exactly follows upstream BaseLoader.resample_ppg's endpoint convention.
    aligned = np.interp(
        np.linspace(1, len(bvp), decoded), np.linspace(1, len(bvp), len(bvp)), bvp
    )
    centered_bvp = aligned - np.mean(aligned)
    bvp_std = float(np.std(centered_bvp))
    if not np.isfinite(bvp_std) or bvp_std <= 1e-8:
        raise ValueError("BVP became constant after resampling: {}".format(bvp_path))
    # Keep float64, as produced by the reference's read_signal_csv/transform_label.
    standardized = centered_bvp / bvp_std
    if not store_uint8:
        # Exact reference float32 reduction order; this optional mode holds the
        # complete resized recording in RAM, unlike the default uint8 path.
        transformed = np.asarray(float_frames, dtype=np.float32)
        transformed = transformed - np.mean(transformed)
        transformed = transformed / np.std(transformed)
    for row in rows:
        start = row["start_frame"]
        label = standardized[start:start + frames]
        if np.std(label) <= 1e-8:
            raise ValueError("Constant BVP in {} clip {}".format(recording, row["clip_index"]))
        np.save(out / row["label_path"], label, allow_pickle=False)
        if not store_uint8:
            np.save(out / row["input_path"], transformed[start:start + frames], allow_pickle=False)
        row.update(input_mean=mean if store_uint8 else 0.0, input_std=std if store_uint8 else 1.0)
    audit = dict(
        subject=subject, recording=recording, task=task, source=str(video.resolve()),
        bvp_source=str(Path(bvp_path).resolve()), fps=fps, reported_frames=reported_frames,
        decoded_frames=decoded, retained_frames=len(rows) * frames,
        dropped_tail_frames=len(buffer), clips=len(rows), crop=crop,
        crop_x1=box[0], crop_y1=box[1], crop_x2=box[0] + box[2], crop_y2=box[1] + box[3],
        input_mean=mean, input_std=std, statistics_scope="full_recording_including_tail",
        alignment="BVP_and_video_recording_endpoints_assumed_aligned",
        detector_status=status,
    )
    return rows, audit


def discover_recordings(raw_dir):
    """Validate all recording identities/pairs before producing output clips."""
    pairs, seen = [], set()
    for video in sorted(Path(raw_dir).rglob("*.avi")):
        if not re.fullmatch(r"vid_s\d+_T[123]\.avi", video.name, re.IGNORECASE):
            raise ValueError("Unexpected video filename: {}".format(video))
        subject, task, recording = identity(video)
        if recording in seen:
            raise ValueError("Duplicate recording: {}".format(recording))
        seen.add(recording)
        bvp_path = video.with_name("bvp_{}.csv".format(video.stem[4:]))
        if not bvp_path.is_file():
            raise FileNotFoundError("Missing paired BVP: {}".format(bvp_path))
        if identity(bvp_path) != (subject, task, recording):
            raise ValueError("BVP/video identities differ: {}".format(video))
        pairs.append((video, bvp_path))
    if not pairs:
        raise ValueError("No vid_sN_Tk.avi recordings found: {}".format(raw_dir))
    return pairs


def index_cache(cache_dir, fps, input_representation, label_representation):
    """Index existing nonoverlapping clips without copying or modifying arrays.

    The original dropped tail is unavailable. Raw-cache normalization therefore
    uses all available clips of a recording; already standardized input is kept
    unchanged. The representation declarations are assertions by the caller:
    finite array values alone cannot distinguish standardization from another
    preprocessing transform.
    """
    if not np.isfinite(fps) or fps <= 5:
        raise ValueError("Cache FPS must be finite and exceed 5 Hz")
    if input_representation not in ("raw", "standardized") or label_representation != "standardized":
        raise ValueError("Declare raw/standardized RGB and standardized BVP labels")
    rows, groups, keys = [], {}, set()
    for path in sorted(Path(cache_dir).rglob("*_input*.npy")):
        match = re.fullmatch(r"(.+)_input(\d+)\.npy", path.name)
        if match is None:
            raise ValueError("Unexpected cache filename: {}".format(path))
        subject, task, recording = identity(path)
        index = int(match[2])
        if (recording, index) in keys:
            raise ValueError("Duplicate cached recording/clip: {} {}".format(recording, index))
        keys.add((recording, index))
        label_path = path.with_name("{}_label{}.npy".format(match[1], match[2]))
        if not label_path.is_file():
            raise FileNotFoundError("Missing paired BVP clip: {}".format(label_path))
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        label = np.load(label_path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 4 or array.shape[-1] != 3 or min(array.shape[:3]) < 1:
            raise ValueError("Expected nonempty THWC RGB array: {}".format(path))
        validate_dimensions(*array.shape[:3])
        if input_representation == "raw" and array.dtype != np.uint8:
            raise ValueError("raw input requires uint8 RGB: {}".format(path))
        if input_representation == "standardized" and not np.issubdtype(array.dtype, np.floating):
            raise ValueError("standardized input requires floating point: {}".format(path))
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite video array: {}".format(path))
        if label.ndim != 1 or label.shape[0] != array.shape[0]:
            raise ValueError("Expected a length-T BVP vector: {}".format(label_path))
        if (not np.issubdtype(label.dtype, np.floating) or not np.isfinite(label).all()
                or np.std(label) <= 1e-8):
            raise ValueError("BVP clip must be finite, nonconstant floating point: {}".format(label_path))
        row = dict(
            input_path=str(path.resolve()), label_path=str(label_path.resolve()),
            subject=subject, recording=recording, task=task, clip_index=index,
            frames=int(array.shape[0]), fps=fps, height=int(array.shape[1]),
            width=int(array.shape[2]), input_representation=input_representation,
            label_representation=label_representation,
        )
        rows.append(row)
        group = groups.setdefault(recording, dict(rows=[], count=0, total=0.0, squared_total=0.0))
        group["rows"].append(row)
        if input_representation == "raw":
            # One frame at a time bounds additional memory even for a large cache clip.
            for frame in array:
                values = np.asarray(frame, dtype=np.float64)
                group["total"] += float(np.sum(values))
                group["squared_total"] += float(np.sum(values * values))
                group["count"] += values.size
    if not rows:
        raise ValueError("No *_inputN.npy clips found: {}".format(cache_dir))
    if len({(row["frames"], row["height"], row["width"]) for row in rows}) != 1:
        raise ValueError("One cache requires a common frame count and image size")
    audit = []
    for recording, group in sorted(groups.items()):
        clips = sorted(group["rows"], key=lambda row: row["clip_index"])
        shapes = {(row["frames"], row["height"], row["width"]) for row in clips}
        if len(shapes) != 1:
            raise ValueError("Inconsistent cached clip dimensions: {}".format(recording))
        if [row["clip_index"] for row in clips] != list(range(len(clips))):
            raise ValueError("Cache clip indices must be contiguous from zero: {}".format(recording))
        mean, std = (0.0, 1.0)
        if input_representation == "raw":
            mean, std = _stats(group["total"], group["squared_total"], group["count"], recording)
        for row in clips:
            row.update(input_mean=mean, input_std=std,
                       start_frame=row["clip_index"] * row["frames"])
        first = clips[0]
        retained = sum(row["frames"] for row in clips)
        audit.append(dict(
            subject=first["subject"], recording=recording, task=first["task"],
            source=str(Path(cache_dir).resolve()), bvp_source="paired_cached_labels",
            fps=fps, reported_frames="unknown", decoded_frames="unknown",
            retained_frames=retained, dropped_tail_frames="unknown", clips=len(clips),
            crop="existing_cache_unknown", input_mean=mean, input_std=std,
            statistics_scope=("available_cached_clips" if input_representation == "raw"
                              else "already_standardized_input_unchanged"),
            alignment="existing_cache_assumed_nonoverlapping_and_synchronized",
        ))
    rows.sort(key=lambda row: (int(row["subject"][1:]), row["task"], row["clip_index"]))
    return rows, audit


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--raw-dir", "--data_path", dest="raw_dir", type=Path, help="UBFC-PHYS root containing sN folders")
    source.add_argument("--cache-dir", type=Path, help="Existing paired *_inputN/*_labelN.npy cache")
    parser.add_argument("--out", "--cached_path", dest="out", type=Path, required=True, help="New or empty output directory")
    parser.add_argument("--frames", "--chunk_length", dest="frames", type=int, default=160, help="Frames per new raw-data clip")
    parser.add_argument("--size", type=int, default=128, help="New square RGB crop size")
    parser.add_argument("--crop", "--detector", dest="crop", choices=("auto", "haar", "yolo", "full-frame"), default="auto")
    parser.add_argument("--cascade", type=Path, help="Original Haar XML for detector parity")
    parser.add_argument("--toolbox-dir", type=Path, help="Optional existing rPPG-Toolbox root for YOLO5Face")
    parser.add_argument("--yolo_device", "--yolo-device", default="cpu")
    parser.add_argument("--detect_search_frames", "--detect-search-frames", type=int, default=60)
    parser.add_argument("--large_box_coef", "--large-box-coef", type=float, default=1.5)
    parser.add_argument("--fix-full-frame-box", action="store_true", help="Correct the reference's swapped no-face dimensions (changes cache pixels)")
    parser.add_argument("--data_type", choices=("Standardized",), default="Standardized")
    parser.add_argument("--label_type", choices=("Standardized",), default="Standardized")
    storage = parser.add_mutually_exclusive_group()
    storage.add_argument("--store_uint8", dest="store_uint8", action="store_true", help="Default: raw uint8 RGB; standardize during loading")
    storage.add_argument("--store-float", dest="store_uint8", action="store_false", help="Save recording-standardized float32 RGB (uses more RAM/disk)")
    parser.set_defaults(store_uint8=True)
    parser.add_argument("--tasks", nargs="+", choices=("T1", "T2", "T3"), default=None)
    parser.add_argument("--fps", type=float, help="Actual cache FPS; required with --cache-dir")
    parser.add_argument("--input-representation", choices=("raw", "standardized"),
                        help="Required declaration for existing cache")
    parser.add_argument("--label-representation", choices=("standardized",),
                        help="Required declaration for existing cache; not inferred from values")
    args = parser.parse_args(argv)
    if args.detect_search_frames < 1 or not np.isfinite(args.large_box_coef) or args.large_box_coef <= 0:
        parser.error("Detection search length and box coefficient must be positive")
    try:
        validate_dimensions(args.frames, args.size, args.size)
    except ValueError as error:
        parser.error(str(error))
    if args.cache_dir:
        if args.fps is None or not np.isfinite(args.fps) or args.fps <= 5:
            parser.error("--cache-dir requires finite --fps greater than 5 Hz")
        if args.input_representation is None or args.label_representation is None:
            parser.error("--cache-dir requires --input-representation and --label-representation")
        if not args.cache_dir.is_dir():
            parser.error("--cache-dir is not a directory")
    else:
        if args.fps is not None or args.input_representation is not None or args.label_representation is not None:
            parser.error("FPS/representation overrides are only valid for --cache-dir")
        if not args.raw_dir.is_dir():
            parser.error("--raw-dir is not a directory")
    out = args.out.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error("--out must be a new or empty directory; existing files are never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    rows, audit = [], []
    if args.raw_dir:
        pairs = discover_recordings(args.raw_dir)
        if args.tasks:
            pairs = [(v, b) for v, b in pairs if identity(v)[1] in args.tasks]
        if not pairs:
            raise ValueError("No recordings for the selected tasks")
        detectors = None if args.crop == "full-frame" else build_detectors(
            args.crop, args.cascade, args.yolo_device, args.toolbox_dir)
        print("Storage:", "raw uint8 RGB; --data_type is bypassed" if args.store_uint8 else "standardized float32 RGB")
        for video, bvp_path in pairs:
            clips, info = convert_recording(video, bvp_path, out, args.frames, args.size, args.crop,
                                            detectors, args.detect_search_frames, args.large_box_coef,
                                            args.fix_full_frame_box, args.store_uint8)
            rows.extend(clips)
            audit.append(info)
            print(info["detector_status"], flush=True)
            print("{}: {} clips; {:.3f} fps; {} tail frames dropped".format(
                info["recording"], info["clips"], info["fps"], info["dropped_tail_frames"]
            ), flush=True)
    else:
        rows, audit = index_cache(
            args.cache_dir, args.fps, args.input_representation, args.label_representation
        )
    write_csv(out / "recording.csv", AUDIT_FIELDS, audit)
    # A complete manifest appears only after every recording and audit is valid.
    temporary = out / "manifest.partial.csv"
    write_csv(temporary, MANIFEST_FIELDS, rows)
    temporary.rename(out / "manifest.csv")
    print("Saved {} clips from {} recordings: {}".format(len(rows), len(audit), out / "manifest.csv"))


if __name__ == "__main__":
    main()
