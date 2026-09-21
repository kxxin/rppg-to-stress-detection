"""Explicit waveform metrics for equally sampled, time-aligned evaluation windows.

Both waveforms are linearly detrended and zero-phase filtered with a first-order
Butterworth bandpass (0.75--2.5 Hz / 45--150 bpm). HR uses a periodogram with at
least 2048 FFT points; zero padding refines the frequency grid, not the physical
resolution of a short observation. SNR uses 10*log10 of power sums, with both
signal and noise restricted to that same band. These choices are intentionally
explicit and are not an exact reproduction of the upstream evaluation code.

Inputs must already represent waveforms: derivative predictions are not silently
integrated, resampled, shifted, or sign-flipped. Constant/degenerate signals have
undefined metrics (NaN), rather than an arbitrary heart rate.
"""

import numpy as np
from scipy.signal import butter, detrend, filtfilt, periodogram


LOW_HZ = 0.75
HIGH_HZ = 2.5
SNR_HALF_WIDTH_HZ = 6.0 / 60.0


def _prepare(signal, fs):
    """Validate, normalize numerical scale, detrend, and bandpass a waveform."""
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 2 * HIGH_HZ:
        raise ValueError("fs must be finite and greater than 5 Hz.")
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or values.size < 10:
        raise ValueError("Each waveform must be one-dimensional with at least 10 samples.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Waveforms must contain only finite samples.")
    scale = np.max(np.abs(values))
    if scale == 0:
        return np.zeros_like(values)
    values = detrend(values / scale, type="linear")
    if np.max(np.abs(values)) <= 100 * np.finfo(np.float64).eps:
        return np.zeros_like(values)
    b, a = butter(1, [LOW_HZ, HIGH_HZ], btype="bandpass", fs=fs)
    return filtfilt(b, a, values)


def _spectrum(values, fs):
    nfft = max(2048, 1 << (len(values) - 1).bit_length())
    return periodogram(values, fs=fs, nfft=nfft, detrend=False)


def _hr(values, fs):
    frequency, power = _spectrum(values, fs)
    band = (frequency >= LOW_HZ) & (frequency <= HIGH_HZ)
    if not np.any(power[band] > 0):
        return float("nan")
    return float(60.0 * frequency[band][np.argmax(power[band])])


def _pearson(first, second):
    if len(first) < 2:
        return float("nan")
    first = np.asarray(first, dtype=np.float64) - np.mean(first)
    second = np.asarray(second, dtype=np.float64) - np.mean(second)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        return float("nan")
    return float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))


def estimate_hr(signal, fs):
    """Return dominant frequency in bpm (45--150), or NaN for a flat waveform.

    Invalid shape, non-finite samples, insufficient length, or an invalid sample
    rate raises ValueError. Duration still governs whether an HR estimate is
    meaningful; the ten-sample minimum only satisfies the filter requirements.
    """
    return _hr(_prepare(signal, fs), float(fs))


def waveform_metrics(prediction, target, fs):
    """Compare aligned waveforms; return HR, absolute HR error, Pearson, and SNR.

    Pearson compares the filtered waveforms without lag correction. SNR counts
    predicted spectral power within +/-6 bpm of the target fundamental and its
    first harmonic (2*f), clipped to 0.75--2.5 Hz, as signal. Remaining power in
    that band is noise. The harmonic regions are unioned to avoid double counts.
    Equal-width periodogram bins make summed power ratios equivalent to area
    ratios. A nonzero signal with exactly zero noise has +inf SNR; an empty
    signal region with nonzero noise has -inf. A flat prediction has NaN SNR.
    """
    prediction = _prepare(prediction, fs)
    target = _prepare(target, fs)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target must have the same sample count.")
    fs = float(fs)
    hr_pred = _hr(prediction, fs)
    hr_true = _hr(target, fs)
    snr = float("nan")
    if np.isfinite(hr_pred) and np.isfinite(hr_true):
        frequency, power = _spectrum(prediction, fs)
        band = (frequency >= LOW_HZ) & (frequency <= HIGH_HZ)
        fundamental = hr_true / 60.0
        signal_bins = band & (
            (np.abs(frequency - fundamental) <= SNR_HALF_WIDTH_HZ)
            | (np.abs(frequency - 2 * fundamental) <= SNR_HALF_WIDTH_HZ)
        )
        signal_power = np.sum(power[signal_bins])
        noise_power = np.sum(power[band & ~signal_bins])
        if signal_power > 0 and noise_power > 0:
            snr = float(10.0 * np.log10(signal_power / noise_power))
        elif signal_power > 0:
            snr = float("inf")
        elif noise_power > 0:
            snr = float("-inf")
    return {
        "hr_pred_bpm": hr_pred,
        "hr_true_bpm": hr_true,
        "hr_abs_error_bpm": float(abs(hr_pred - hr_true)),
        "waveform_pearson": _pearson(prediction, target),
        "snr_db": snr,
    }


def summary_metrics(rows):
    """Aggregate equally weighted windows, with undefined values excluded.

    HR summaries use pairs with finite predicted and true HR and positive true
    HR. HR Pearson is across those windows, whereas mean_waveform_pearson is the
    mean of the individual waveform correlations. The means of correlation and
    SNR include finite values only; inspect per-window rows for NaN/inf values.
    No valid observations returns NaN rather than a misleading zero error.
    """
    rows = list(rows)
    pairs = np.asarray(
        [(row["hr_pred_bpm"], row["hr_true_bpm"]) for row in rows],
        dtype=np.float64,
    ).reshape(-1, 2)
    valid = np.all(np.isfinite(pairs), axis=1) & (pairs[:, 1] > 0)
    pairs = pairs[valid]
    error = pairs[:, 0] - pairs[:, 1]

    def finite_mean(key):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if values.size else float("nan")

    return {
        "count": len(rows),
        "valid_hr_count": int(len(pairs)),
        "hr_mae_bpm": float(np.mean(np.abs(error))) if len(pairs) else float("nan"),
        "hr_rmse_bpm": float(np.sqrt(np.mean(error ** 2))) if len(pairs) else float("nan"),
        "hr_mape_percent": float(100 * np.mean(np.abs(error) / pairs[:, 1]))
        if len(pairs) else float("nan"),
        "hr_pearson": _pearson(pairs[:, 0], pairs[:, 1]),
        "mean_waveform_pearson": finite_mean("waveform_pearson"),
        "mean_snr_db": finite_mean("snr_db"),
    }
