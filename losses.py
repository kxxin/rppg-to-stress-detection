"""Upstream RhythmMamba hybrid pulse loss with explicit device-safe numerics.

The objective remains 0.2 * negative Pearson + spectral cross entropy. As in
upstream, normalized spectral power is passed as CE logits (not log power).
Ground-truth HR follows the upstream detrend / bandpass / Welch pipeline.
Adapted from RhythmMamba's TorchLossComputer.py; see LICENSE.txt and README.md.
"""

from functools import lru_cache

import numpy as np
from scipy import sparse
from scipy.signal import butter, filtfilt, welch
from scipy.sparse.linalg import factorized
import torch
from torch import nn
from torch.nn import functional as F


def normalize_waveform(waveform, eps=1e-6):
    """Per-clip zero mean and sample standard deviation, matching the trainer."""
    centered = waveform - waveform.mean(dim=-1, keepdim=True)
    scale = waveform.std(dim=-1, keepdim=True, unbiased=True).clamp_min(eps)
    return centered / scale


def negative_pearson(prediction, target, eps=1e-8):
    """Return the batch mean of 1-r, with finite behavior for flat predictions."""
    prediction = prediction - prediction.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    numerator = (prediction * target).sum(dim=-1)
    denominator = (
        prediction.square().sum(dim=-1) * target.square().sum(dim=-1)
    ).clamp_min(eps).sqrt()
    return (1.0 - (numerator / denominator).clamp(-1.0, 1.0)).mean()


@lru_cache(maxsize=8)
def _detrend_solver(length):
    """Factor the same smoothness-prior system used by the upstream loss."""
    second_difference = sparse.diags(
        (np.ones(length - 2), -2 * np.ones(length - 2), np.ones(length - 2)),
        (0, 1, 2), shape=(length - 2, length), format="csc",
    )
    system = sparse.eye(length, format="csc") + 100.0 ** 2 * (
        second_difference.T @ second_difference
    )
    return factorized(system)


def target_heart_rate(target, fs):
    """Derive one HR label in bpm from a standardized, non-differenced BVP clip.

    Sparse solving replaces a dense matrix inverse without changing its
    mathematical operation. Welch's nfft is explicitly converted to an integer
    because upstream's float value is rejected by some SciPy versions.
    """
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if target.size < 16:
        raise ValueError("Pulse loss requires at least 16 waveform samples")
    if not np.isfinite(target).all() or np.ptp(target) <= 1e-8:
        raise ValueError("BVP target must be finite and nonconstant")
    if not np.isfinite(fs) or fs <= 5:
        raise ValueError("Sampling rate must exceed 5 Hz for the 45-150 bpm band")
    detrended = target - _detrend_solver(target.size)(target)
    b, a = butter(1, (0.75, 2.5), btype="bandpass", fs=fs)
    filtered = filtfilt(b, a, detrended)
    nperseg = min(target.size - 1, 256)
    frequencies, powers = welch(
        filtered, fs=fs, nfft=max(nperseg, int(1e5 / fs)), nperseg=nperseg
    )
    usable = (frequencies > 0.75) & (frequencies < 2.5)
    return float(frequencies[usable][np.argmax(powers[usable])] * 60.0)


def normalized_power(waveform, fs, eps=1e-12):
    """Evaluate the Hann-windowed spectrum at the upstream 45..149 bpm bins."""
    frames = waveform.shape[-1]
    time = torch.arange(frames, dtype=waveform.dtype, device=waveform.device) / fs
    frequencies = torch.arange(45, 150, dtype=waveform.dtype, device=waveform.device) / 60
    angles = 2 * torch.pi * frequencies[:, None] * time[None, :]
    hann = torch.hann_window(
        frames, periodic=False, dtype=waveform.dtype, device=waveform.device
    )
    windowed = waveform * hann
    power = (windowed @ angles.sin().T).square() + (windowed @ angles.cos().T).square()
    return power / power.sum(dim=-1, keepdim=True).clamp_min(eps)


class HybridLoss(nn.Module):
    """Compute the upstream waveform objective for prediction/target [B,T]."""

    def forward(self, prediction, target, fs):
        if prediction.ndim != 2 or prediction.shape != target.shape:
            raise ValueError("Prediction and BVP target must have identical [B,T] shape")
        if prediction.shape[1] < 16:
            raise ValueError("Pulse loss requires at least 16 frames per clip")
        prediction = normalize_waveform(prediction)
        time_loss = negative_pearson(prediction, target)
        hr_values = [
            target_heart_rate(sample, float(fs))
            for sample in target.detach().cpu().numpy()
        ]
        # Upstream casts hr-45 to long, truncating toward zero.
        hr_classes = torch.as_tensor(
            [int(hr - 45) for hr in hr_values], device=prediction.device, dtype=torch.long
        ).clamp_(0, 104)
        spectral_loss = F.cross_entropy(normalized_power(prediction, fs), hr_classes)
        return 0.2 * time_loss + spectral_loss
