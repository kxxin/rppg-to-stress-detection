# Efficient Stress Detection using Remote Photoplethysmography (rPPG)

## Files

| File | Responsibility |
|---|---|
| `preprocess.py` | Stream raw AVI/BVP pairs into NumPy clips, or index an existing cache |
| `face_crop.py` | Reference-compatible Haar/YOLO selection, box expansion, and fallback |
| `data.py` | Validate CSV manifests, split subjects, normalize and load clips |
| `model.py` | RhythmMamba architecture; `forward_features()` exposes temporal features |
| `losses.py` | Hybrid negative-Pearson and spectral cross-entropy loss |
| `engine.py` | One training or validation epoch |
| `train.py` | Train and select the best validation checkpoint |
| `metrics.py` | Waveform correlation, spectral HR, and SNR |
| `test.py` | Evaluate the saved validation/test partition |
| `predict.py` | Predict from one preprocessed clip without BVP labels |
| `utils.py` | Checkpoint, CSV, seed, and output-directory helpers |
| `check.py` | Small synthetic integration check, including a CUDA optimizer step |

## Environment

For a fresh Linux/WSL environment:

```bash
git clone https://github.com/kxxin/rppg-to-stress-detection.git
cd rppg-to-stress-detection
conda create -n rppg-to-stress-detection python=3.8 -y
conda activate rppg-to-stress-detection
```

Install matching CUDA PyTorch/torchvision builds first. One reference 
combination is:

```bash
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y #if nvcc not found
python -m pip install causal-conv1d==1.4.0 mamba-ssm==2.2.2 --no-build-isolation
```

## 1. Preprocess raw UBFC-PHYS

Expected source layout:

```text
RawData/
  s1/
    vid_s1_T1.avi
    bvp_s1_T1.csv
    vid_s1_T2.avi
    bvp_s1_T2.csv
    ...
  s2/
    ...
```

```bash
python preprocess.py \
  --data_path /path/to/UBFC-PHYS/RawData \
  --cached_path runs/cache \
  --data_type Standardized --label_type Standardized --store_uint8
```

This matches the supplied command for the newer WSL
`preprocess_ubfc_phys_standalone.py`: 160 frames, 128x128 RGB, Standardized BVP,
and raw uint8 video storage. The old names `--data_path`, `--cached_path`, and
`--chunk_length` are accepted alongside `--raw-dir`, `--out`, and `--frames`.
Defaults here follow that supplied command, not the reference script's
DiffNormalized defaults. `--data_type Standardized` is bypassed when
`--store_uint8` is active; it does not standardize the saved RGB array.
Use `--store-float` for the reference's Standardized float32 video output instead.
That optional mode holds the complete resized recording in memory to reproduce
the reference's float32 normalization order. The default uint8 path remains streamed.

`--detector auto` (also `--crop auto`) is the default. It tries the default,
alt2, and alt Haar cascades in that order, using the reference's RGB detector
input and largest-width face selection. It searches 60 frames, expands the
selected box by 1.5, then holds the box fixed. `--detect_search_frames`,
`--large_box_coef`, `--cascade`, and `--yolo_device` match the reference options.
Auto tries YOLO on the first frame only if Haar finds no face during the scan.

Haar operation is self-contained. For the same optional YOLO5Face backend and
weights as your existing WSL installation, add:

```bash
--toolbox-dir /home/racha/rPPG-Toolbox
```

The adapter imports that installation without modifying it. YOLO weights and
Toolbox dependencies are not bundled. Auto reports when YOLO is unavailable and
continues with Haar, like the reference; `--detector yolo` requires working YOLO.
For numerical parity, use the same detector availability, cascade XML, weights,
OpenCV version, and crop options. The tested WSL Toolbox default cascade and
OpenCV default cascade have identical hashes. An explicit `--cascade` takes
precedence; otherwise the working-directory Toolbox cascade is used if present,
then OpenCV's default cascade.

For exact legacy cache reproduction, the no-face fallback deliberately retains
the WSL script's `[0,0,height,width]` box. This swaps width/height on nonsquare
videos and is recorded in `recording.csv`. Use `--fix-full-frame-box` to correct
only that fallback; this changes affected pixels. `--crop full-frame` explicitly
bypasses detection and always uses the correctly sized full frame. Inspect crops
and the recorded detector status before a full experiment.

Videos retain their reported sampling rate, usually 35 Hz for UBFC-PHYS.
Frames are stored by default as **raw RGB uint8 `[T,H,W,3]`**. The loader applies one scalar
mean/std over the whole cropped recording, matching upstream recording-level
standardization. The fusion stem computes its own temporal differences; do not
feed difference-normalized video into this pipeline.

BVP is linearly interpolated to the decoded frame count, centered, divided by
the centered recording's standard deviation, then chunked. It is saved as
**float64 `[T]`**, matching the supplied script; `data.py` converts it to float32
for the model. Non-numeric CSV headers/empty rows are skipped as in the reference.
This preserves the upstream assumption
that video and BVP recording endpoints align; it is **not timestamp-based
synchronization or correction of physiological delay**. Nonfinite or constant
targets, missing pairs, identity mismatches, and truncated decoding fail visibly.
No source files are deleted and no published quality exclusions are applied.
`--tasks T1 T2 T3` can select tasks; by default all available tasks are processed.

Output consists of paired `s1_T1_input0.npy` / `s1_T1_label0.npy`, `manifest.csv`,
and `recording.csv`. The latter records crop coordinates, normalization statistics,
decoded frames, and dropped tails. Clips are nonoverlapping; incomplete final
clips are dropped. Raw preprocessing includes the tail in recording statistics.
Frame counts and image sides must be at least 16 and divisible by 8.

Use a new/empty output directory. A completed manifest is written only after all
recordings succeed; a failed run may leave partial NumPy files for inspection.
These validation/output safeguards are retained: unlike the reference driver,
this tool does not silently skip existing outputs, continue after failed videos,
overwrite an existing cache, or implement `--delete_raw`. It writes CSV metadata
instead of the reference's append-only `DataFileLists/standalone_filelist.csv`.
DiffNormalized and raw-float modes are outside this waveform-training contract.

### Reuse an existing cache

This route indexes files without copying or modifying them:

```bash
python preprocess.py \
  --cache-dir /mnt/c/downloads2026/cp/PreprocessedData \
  --out runs/cache_index --fps 35 \
  --input-representation raw --label-representation standardized
```

Use `raw` only for uint8 RGB. For already standardized floating RGB, explicitly
choose `--input-representation standardized`. BVP labels must already be
standardized waveforms, not derivatives or binary stress labels. The program
cannot infer preprocessing semantics from floating-point values; these flags
are declarations of how the cache was produced.

Cache clips must be consecutive from index zero and nonoverlapping. For raw
caches, RGB statistics use available clips; discarded original tails cannot be
recovered. This can differ slightly from fresh raw preprocessing. The cache
manifest stores absolute paths; raw preprocessing uses paths relative to its
manifest. Replace the manifest path below when using this route.
Indexing searches nested directories, including `PreprocessedData01` through
`PreprocessedData08`. It preserves the existing crops and float64 BVP arrays.

## 2. Train

```bash
python train.py \
  --manifest runs/cache/manifest.csv \
  --out runs/train_01 \
  --epochs 30 --batch-size 2 --lr 0.0003 --seed 100
```

Default partitions are approximately 60/20/20 by subject, assigned before dataset
loading. With 56 subjects this gives 34 train, 11 validation, and 11 test subjects.
Every task and clip belonging to one person stays in one partition. Preprocessing
is deterministic per recording and does not fit any population-level statistics.
To preserve a particular protocol, supply `--splits /path/to/splits.csv`:

```csv
subject,split
s1,train
s2,valid
s3,test
```

The actual file must assign every subject exactly once and contain all three
partitions. This example is a schema illustration, not a complete dataset split.

Training uses FP32, AdamW with zero weight decay, OneCycleLR, gradient clipping
at 1.0, and upstream's `0.2 * negative Pearson + spectral CE` objective.
The smaller default batch size fits debugging on a laptop GPU; upstream uses 16.
`--depth` and `--embed-dim` allow explicitly smaller ablations, changing the model.
The original HR-conditioned augmentation is not enabled in this baseline.

`best.pt` is selected only by validation loss. `last.pt` keeps the final completed
epoch. Both contain model configuration, experiment arguments, exact subject
assignments, manifest hash, optimizer state, serializable scheduler values, and model weights.
`history.csv` and `splits.csv` are the human-readable logs. This version starts
fresh runs; optimizer state is retained for inspection, but there is no resume CLI.
Test subjects are not evaluated by the training script.

To fine-tune an explicitly supplied upstream checkpoint, add
`--init-weights /path/to/upstream.pth`. DataParallel `module.` prefixes are removed
and weights are loaded strictly. Match depth/width to the checkpoint. Do not use
a checkpoint trained on your held-out subjects. Loading pretrained weights is
fine-tuning, not KD. Without this option, all parameters start randomly initialized.

## 3. Test

```bash
python test.py \
  --manifest runs/cache/manifest.csv \
  --checkpoint runs/train_01/best.pt \
  --out runs/test_01 --window-seconds 10 --save-waveforms
```

Testing checks the original manifest hash and uses subjects saved in the checkpoint.
Do not edit the manifest between training and testing. `--split valid` is available
for debugging without accessing the test partition.

Predictions are standardized per clip, as in the upstream trainer. Consecutive
clips are concatenated within each recording; tasks and people are never joined.
Metrics use nonoverlapping 10-second windows by default. Evaluation tails are
dropped and recorded. Recordings shorter than one window raise an error instead
of silently disappearing. Windows of at least five seconds are configurable.

`windows.csv` contains subject/recording/task identifiers and per-window metrics.
`metrics.csv` contains overall and per-task summaries. `recordings.csv` reports
coverage. `--save-waveforms` additionally writes prediction/target NumPy arrays;
omit it to retain CSV results only.

Metrics use explicit linear detrending, a first-order 45-150 bpm bandpass, and a
zero-padded periodogram for HR. Waveform Pearson is calculated after that filtering,
without lag or polarity correction. SNR uses predicted power within +/-6 bpm of
the reference fundamental and first harmonic, restricted to the same cardiac band.
These evaluation choices differ from upstream and are documented in `metrics.py`;
do not compare numerical results across different protocols as if identical.

Undefined estimates remain NaN, with valid HR counts reported. Summaries weight
windows equally, not subjects. Use the identifiers in `windows.csv` for grouped
analysis and repeated subject splits. Good HR metrics do not establish accurate
beat intervals, HRV, or stress classification. Clip standardization can also create
boundary effects when waveforms are concatenated.

## 4. Predict one clip

```bash
python predict.py \
  --input runs/cache/s1_T1_input0.npy \
  --checkpoint runs/train_01/best.pt \
  --representation raw --out runs/prediction.npy
```

No BVP file is required. Raw input defaults to that clip's RGB mean/std; supply
`--input-mean` and `--input-std` from its manifest row to match recording-level
normalization exactly. Standardized floating input uses
`--representation standardized` without mean/std overrides. Preserve RGB order,
FPS, crop convention, spatial size, and clip length from training; NumPy alone
does not encode FPS. The output is a normalized pulse waveform, not calibrated
pulse amplitude, a stress score, or an HRV estimate.

## Debug and verification

```bash
python -B check.py
```

The check creates temporary synthetic AVI/BVP data, verifies preprocessing and
subject separation, runs a small Mamba training/test/prediction cycle, and removes
its generated files automatically. It checks implementation behavior; synthetic
data does not measure research accuracy. Each entry point also supports `--help`.

To additionally compare the preprocessor against the supplied WSL source:

```bash
python -B check.py --reference-preprocess /home/racha/rPPG-Toolbox/preprocess_ubfc_phys_standalone.py
```

This checks detector selection/expansion, the historical nonsquare fallback,
uint8 clips, standardized float32 clips, and float64 BVP against that source on
temporary synthetic video. The reference script is imported for its helpers;
its main routine, overwrite, and deletion options are never invoked.

The extracted default model was checked against the original implementation with
the same weights and `[1,160,3,128,128]` CUDA input: outputs matched exactly.
The hybrid loss matched within 4.77e-7 after correcting upstream's obsolete
floating-point Welch `nfft` argument for comparison. Gradient checks covered the
fusion stem, Mamba blocks, and waveform head. No real UBFC-PHYS training run or
accuracy claim is included.

Useful breakpoint locations are `convert_recording()` in preprocessing,
`PulseDataset.__getitem__()`, `RhythmMamba.forward_features()`,
`HybridLoss.forward()`, `run_epoch()`, and `evaluate_recording()`.