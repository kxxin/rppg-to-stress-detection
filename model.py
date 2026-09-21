"""RhythmMamba waveform model, separated from the upstream Toolbox trainer.

Adapted from https://github.com/zizheng-guo/RhythmMamba (MIT, Zizheng Guo,
2024). See LICENSE.txt. Module names and parameter shapes intentionally match
the upstream model so its state dictionaries can be loaded strictly.

Input: standardized RGB float32 [batch, frames, 3, height, width].
Output: unnormalized pulse waveform [batch, frames]. Use recording-level RGB
normalization from preprocessing; this module does not normalize input pixels.
The original spatial attention mask is part of RhythmMamba itself.
"""

import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from timm.models.layers import DropPath, lecun_normal_, trunc_normal_
from mamba_ssm.modules.mamba_simple import Mamba


class Fusion_Stem(nn.Module):
    """Fuse appearance and four adjacent frame differences at 1/8 resolution."""

    def __init__(self, apha=0.5, belta=0.5, dim=24):
        super().__init__()
        # Keep upstream spellings for inspection and checkpoint compatibility.
        self.apha = apha
        self.belta = belta
        self.stem11 = nn.Sequential(
            nn.Conv2d(3, dim // 2, 7, stride=2, padding=3),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
        self.stem12 = nn.Sequential(
            nn.Conv2d(12, dim // 2, 7, stride=2, padding=3),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
        self.stem21 = nn.Sequential(
            nn.Conv2d(dim // 2, dim, 7, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
        self.stem22 = nn.Sequential(
            nn.Conv2d(dim // 2, dim, 7, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )

    def forward(self, x):
        batch, frames, channels, height, width = x.shape
        previous2 = torch.cat((x[:, :1], x[:, :1], x[:, :-2]), dim=1)
        previous1 = torch.cat((x[:, :1], x[:, :-1]), dim=1)
        next1 = torch.cat((x[:, 1:], x[:, -1:]), dim=1)
        next2 = torch.cat((x[:, 2:], x[:, -1:], x[:, -1:]), dim=1)
        differences = torch.cat(
            (previous1 - previous2, x - previous1, next1 - x, next2 - next1),
            dim=2,
        )
        difference_features = self.stem12(
            differences.reshape(batch * frames, 12, height, width)
        )
        appearance = self.stem11(x.reshape(batch * frames, channels, height, width))
        first_path = self.stem21(
            self.apha * appearance + self.belta * difference_features
        )
        second_path = self.stem22(difference_features)
        return self.apha * first_path + self.belta * second_path


class Attention_mask(nn.Module):
    """Original positive spatial mask, with a guarded normalization divisor."""

    def forward(self, x):
        spatial_sum = x.sum(dim=3, keepdim=True).sum(dim=4, keepdim=True)
        # Only changes saturated/degenerate masks that previously produced NaNs.
        spatial_sum = spatial_sum.clamp_min(torch.finfo(x.dtype).tiny)
        return x / spatial_sum * x.shape[3] * x.shape[4] * 0.5


class Frequencydomain_FFN(nn.Module):
    """The upstream frequency-domain feed-forward network, including its einsum."""

    def __init__(self, dim, mlp_ratio):
        super().__init__()
        self.scale = 0.02
        self.dim = dim * mlp_ratio
        self.r = nn.Parameter(self.scale * torch.randn(self.dim, self.dim))
        self.i = nn.Parameter(self.scale * torch.randn(self.dim, self.dim))
        self.rb = nn.Parameter(self.scale * torch.randn(self.dim))
        self.ib = nn.Parameter(self.scale * torch.randn(self.dim))
        self.fc1 = nn.Sequential(
            nn.Conv1d(dim, self.dim, 1, bias=False),
            nn.BatchNorm1d(self.dim),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(self.dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x):
        x = self.fc1(x.transpose(1, 2)).transpose(1, 2)
        spectrum = torch.fft.fft(x, dim=1, norm="ortho")
        # "cc" selects the diagonal, rather than a dense channel multiplication.
        # Preserve this upstream operation and parameter layout deliberately.
        real = F.relu(
            torch.einsum("bnc,cc->bnc", spectrum.real, self.r)
            - torch.einsum("bnc,cc->bnc", spectrum.imag, self.i)
            + self.rb
        )
        imag = F.relu(
            torch.einsum("bnc,cc->bnc", spectrum.imag, self.r)
            + torch.einsum("bnc,cc->bnc", spectrum.real, self.i)
            + self.ib
        )
        spectrum = torch.view_as_complex(torch.stack((real, imag), dim=-1).float())
        # Upstream casts the complex IFFT to float32, discarding its imaginary part.
        x = torch.fft.ifft(spectrum, dim=1, norm="ortho").real.float()
        return self.fc2(x.transpose(1, 2)).transpose(1, 2)


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=48, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand
        )

    def forward(self, x):
        return self.mamba(self.norm(x))


class Block_mamba(nn.Module):
    """Multi-temporal Mamba aggregation followed by a frequency-domain FFN."""

    def __init__(self, dim, mlp_ratio, drop_path=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = MambaLayer(dim)
        self.mlp = Frequencydomain_FFN(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        elif isinstance(module, nn.Conv2d):
            fan_out = math.prod(module.kernel_size) * module.out_channels // module.groups
            nn.init.normal_(module.weight, std=math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        batch, frames, _ = x.shape
        segments = 4
        segment_length = frames // segments
        repeated = x.repeat(segments, 1, 1)
        shifted = repeated.clone()
        for offset in range(1, segments):
            shifted[offset * batch : (offset + 1) * batch, : frames - offset * segment_length] = (
                repeated[offset * batch : (offset + 1) * batch, offset * segment_length :]
            )
        outputs = self.attn(shifted)
        for segment in range(1, segments):
            destination = slice(segment * segment_length, (segment + 1) * segment_length)
            for earlier in range(segment):
                source = slice(
                    (segment - earlier - 1) * segment_length,
                    (segment - earlier) * segment_length,
                )
                outputs[:batch, destination] = (
                    outputs[:batch, destination]
                    + outputs[(earlier + 1) * batch : (earlier + 2) * batch, source]
                )
            outputs[:batch, destination] = outputs[:batch, destination] / (segment + 1)
        x = x + self.drop_path(self.norm1(outputs[:batch]))
        return x + self.drop_path(self.mlp(self.norm2(x)))


def _init_weights(module, n_layer):
    """Preserve the upstream Mamba residual-projection initialization."""
    if isinstance(module, nn.Linear) and module.bias is not None:
        if not getattr(module.bias, "_no_reinit", False):
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)
    for name, parameter in module.named_parameters():
        if name in ("out_proj.weight", "fc2.weight"):
            nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
            with torch.no_grad():
                parameter /= math.sqrt(n_layer)


def segm_init_weights(module):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


class RhythmMamba(nn.Module):
    """Default upstream architecture: 24 blocks, 96 channels, Mamba state size 48.

    Frames must be even and >= 8; multiples of 8 give equal temporal partitions.
    Spatial dimensions must be multiples of 8. The reference configuration uses
    160 frames at 128x128. Keep training in float32 for the FFT and Mamba kernels.
    """

    def __init__(self, depth=24, embed_dim=96, mlp_ratio=2, drop_path_rate=0.1):
        super().__init__()
        if not isinstance(depth, int) or depth < 1:
            raise ValueError("depth must be a positive integer")
        if not isinstance(embed_dim, int) or embed_dim < 8 or embed_dim % 8:
            raise ValueError("embed_dim must be a positive multiple of 8")
        if not isinstance(mlp_ratio, int) or mlp_ratio < 1:
            raise ValueError("mlp_ratio must be a positive integer")
        if not 0 <= drop_path_rate < 1:
            raise ValueError("drop_path_rate must be in [0, 1)")
        self.embed_dim = embed_dim
        self.Fusion_Stem = Fusion_Stem(dim=embed_dim // 4)
        self.attn_mask = Attention_mask()
        self.stem3 = nn.Sequential(
            nn.Conv3d(
                embed_dim // 4, embed_dim, (2, 5, 5),
                stride=(2, 1, 1), padding=(0, 2, 2),
            ),
            nn.BatchNorm3d(embed_dim),
        )
        # Upstream prepends zero and indexes this list: the first two blocks have
        # zero drop probability. Retain this detail for training reproducibility.
        drop_probabilities = [0.0] + torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            Block_mamba(embed_dim, mlp_ratio, drop_probabilities[index])
            for index in range(depth)
        ])
        self.upsample = nn.Upsample(scale_factor=2)
        self.ConvBlockLast = nn.Conv1d(embed_dim, 1, 1)
        self.apply(segm_init_weights)
        self.apply(partial(_init_weights, n_layer=depth))

    def forward_features(self, x):
        """Return [batch, frames/2, embed_dim] for debugging or future heads."""
        if x.ndim != 5:
            raise ValueError("Expected RGB tensor [batch, frames, 3, height, width]")
        batch, frames, channels, height, width = x.shape
        if channels != 3 or frames < 8 or frames % 2:
            raise ValueError("Expected 3 RGB channels and an even frame count >= 8")
        if height < 8 or width < 8 or height % 8 or width % 8:
            raise ValueError("Height and width must be positive multiples of 8")
        if x.dtype != torch.float32:
            raise ValueError("RhythmMamba expects float32 standardized RGB")
        x = self.Fusion_Stem(x)
        x = x.reshape(batch, frames, self.embed_dim // 4, height // 8, width // 8)
        x = self.stem3(x.permute(0, 2, 1, 3, 4))
        x = x * self.attn_mask(torch.sigmoid(x))
        x = x.mean(dim=4).mean(dim=3).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x):
        sequence = self.forward_features(x)
        waveform = self.ConvBlockLast(self.upsample(sequence.transpose(1, 2)))
        return waveform.squeeze(1)
