#!/usr/bin/env python3
"""Minimal neural core for the public ISMIR 2020 TFC-TDF checkpoint.

The model structure in this file is adapted from
https://github.com/ws-choi/ISMIR2020_U_Nets_SVS at commit
aafcb69c43675713b86cd4f96eb659cf7eb55d16.

MIT License

Copyright (c) 2020 ws-choi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


UPSTREAM_REPOSITORY = "https://github.com/ws-choi/ISMIR2020_U_Nets_SVS"
UPSTREAM_COMMIT = "aafcb69c43675713b86cd4f96eb659cf7eb55d16"
UPSTREAM_CHECKPOINT_PATH = (
    "etc/checkpoints/tfc_tdf_net/debug/vocals_epoch=891.ckpt"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "101921dac943e1683452f293dc0d32df53b694886541952be26d9592d819c20d"
)
EXPECTED_CHECKPOINT_BYTES = 12_002_436


@dataclass(frozen=True)
class DefaultTfcTdfConfig:
    sample_rate: int = 44_100
    n_fft: int = 2_048
    hop_length: int = 1_024
    num_frames: int = 128
    input_channels: int = 4
    n_blocks: int = 7
    internal_channels: int = 24
    n_internal_layers: int = 5
    kernel_size_t: int = 3
    kernel_size_f: int = 3
    bottleneck_factor: int = 16
    minimum_bottleneck_units: int = 16

    @property
    def frequency_bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def model_input_samples(self) -> int:
        return self.hop_length * (self.num_frames - 1)

    @property
    def trim_samples(self) -> int:
        return 5 * self.hop_length

    @property
    def useful_samples(self) -> int:
        return self.model_input_samples - 2 * self.trim_samples


DEFAULT_CONFIG = DefaultTfcTdfConfig()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TFC(nn.Module):
    """Densely connected time-frequency convolution block."""

    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        growth_rate: int,
        kernel_t: int,
        kernel_f: int,
    ) -> None:
        super().__init__()
        channels = in_channels
        self.H = nn.ModuleList()
        for _ in range(num_layers):
            self.H.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        growth_rate,
                        kernel_size=(kernel_t, kernel_f),
                        stride=1,
                        padding=(kernel_t // 2, kernel_f // 2),
                    ),
                    nn.BatchNorm2d(growth_rate),
                    nn.ReLU(),
                )
            )
            channels += growth_rate

    def forward(self, value: Tensor) -> Tensor:
        output = self.H[0](value)
        for layer in self.H[1:]:
            value = torch.cat((output, value), dim=1)
            output = layer(value)
        return output


class TIF(nn.Module):
    """Frequency-axis bottleneck used as the TDF branch."""

    def __init__(
        self,
        channels: int,
        frequency_bins: int,
        bottleneck_factor: int,
        minimum_bottleneck_units: int,
    ) -> None:
        super().__init__()
        bottleneck_units = max(
            frequency_bins // bottleneck_factor,
            minimum_bottleneck_units,
        )
        self.tif = nn.Sequential(
            nn.Linear(frequency_bins, bottleneck_units, bias=False),
            nn.BatchNorm2d(channels, affine=False),
            nn.ReLU(),
            nn.Linear(bottleneck_units, frequency_bins, bias=False),
            nn.BatchNorm2d(channels, affine=False),
            nn.ReLU(),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.tif(value)


class TfcTdfBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        config: DefaultTfcTdfConfig,
        frequency_bins: int,
    ) -> None:
        super().__init__()
        self.tfc = TFC(
            in_channels,
            config.n_internal_layers,
            config.internal_channels,
            config.kernel_size_t,
            config.kernel_size_f,
        )
        self.tif = TIF(
            config.internal_channels,
            frequency_bins,
            config.bottleneck_factor,
            config.minimum_bottleneck_units,
        )

    def forward(self, value: Tensor) -> Tensor:
        value = self.tfc(value)
        return value + self.tif(value)


class TfcTdfNeuralCore(nn.Module):
    """Checkpoint-compatible core with upstream [B, C, T, F] layout."""

    def __init__(self, config: DefaultTfcTdfConfig = DEFAULT_CONFIG) -> None:
        super().__init__()
        if config.n_blocks != 7:
            raise ValueError("The frozen default checkpoint requires seven blocks")

        self.first_conv = nn.Sequential(
            nn.Conv2d(
                config.input_channels,
                config.internal_channels,
                kernel_size=(1, 2),
                stride=1,
            ),
            nn.BatchNorm2d(config.internal_channels),
            nn.ReLU(),
        )
        self.encoders = nn.ModuleList()
        self.downsamplings = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upsamplings = nn.ModuleList()

        frequency_bins = config.n_fft // 2
        for _ in range(3):
            self.encoders.append(
                TfcTdfBlock(config.internal_channels, config, frequency_bins)
            )
            self.downsamplings.append(
                nn.Sequential(
                    nn.Conv2d(
                        config.internal_channels,
                        config.internal_channels,
                        kernel_size=(2, 2),
                        stride=(2, 2),
                    ),
                    nn.BatchNorm2d(config.internal_channels),
                )
            )
            frequency_bins //= 2

        self.mid_block = TfcTdfBlock(
            config.internal_channels,
            config,
            frequency_bins,
        )

        for _ in range(3):
            self.upsamplings.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        config.internal_channels,
                        config.internal_channels,
                        kernel_size=(2, 2),
                        stride=(2, 2),
                    ),
                    nn.BatchNorm2d(config.internal_channels),
                )
            )
            frequency_bins *= 2
            self.decoders.append(
                TfcTdfBlock(
                    2 * config.internal_channels,
                    config,
                    frequency_bins,
                )
            )

        self.last_conv = nn.Sequential(
            nn.Conv2d(
                config.internal_channels,
                config.input_channels,
                kernel_size=(1, 2),
                stride=1,
                padding=(0, 1),
            ),
            nn.Identity(),
        )

    def forward(self, value: Tensor) -> Tensor:
        value = self.first_conv(value)
        encoder_outputs: list[Tensor] = []
        for encoder, downsampling in zip(self.encoders, self.downsamplings):
            value = encoder(value)
            encoder_outputs.append(value)
            value = downsampling(value)

        value = self.mid_block(value)
        for index, (upsampling, decoder) in enumerate(
            zip(self.upsamplings, self.decoders)
        ):
            value = upsampling(value)
            value = torch.cat((value, encoder_outputs[-index - 1]), dim=1)
            value = decoder(value)
        return self.last_conv(value)


class TfcTdfNchwWrapper(nn.Module):
    """Expose [B, feature, frequency, frame] for ONNX and Booming SS."""

    def __init__(self, core: TfcTdfNeuralCore) -> None:
        super().__init__()
        self.core = core

    def forward(self, value: Tensor) -> Tensor:
        value = value.transpose(2, 3)
        value = self.core(value)
        return value.transpose(2, 3)


class _LightningAttributeDict(dict):
    """Minimal pickle compatibility for the old Lightning checkpoint."""

    def __getattr__(self, key: str) -> Any:
        return self[key]


_LightningAttributeDict.__module__ = "pytorch_lightning.utilities.parsing"
_LightningAttributeDict.__name__ = "AttributeDict"
_LightningAttributeDict.__qualname__ = "AttributeDict"


def _install_lightning_pickle_shim() -> None:
    # The local compatibility modules intentionally have no import spec.  A
    # second checkpoint load in the same process must therefore return before
    # importlib.util.find_spec() inspects the synthetic module.
    if "pytorch_lightning" in sys.modules:
        return
    if importlib.util.find_spec("pytorch_lightning") is not None:
        return
    lightning = types.ModuleType("pytorch_lightning")
    utilities = types.ModuleType("pytorch_lightning.utilities")
    parsing = types.ModuleType("pytorch_lightning.utilities.parsing")
    parsing.AttributeDict = _LightningAttributeDict
    utilities.parsing = parsing
    lightning.utilities = utilities
    sys.modules["pytorch_lightning"] = lightning
    sys.modules["pytorch_lightning.utilities"] = utilities
    sys.modules["pytorch_lightning.utilities.parsing"] = parsing


def _validate_hyperparameters(hyperparameters: dict[str, Any]) -> None:
    expected = {
        "target_name": "vocals",
        "n_fft": 2048,
        "hop_length": 1024,
        "num_frame": 128,
        "spec_type": "complex",
        "spec_est_mode": "mapping",
        "n_internal_layers": 5,
        "kernel_size_t": 3,
        "kernel_size_f": 3,
        "bn_factor": 16,
        "min_bn_units": 16,
        "tfc_tdf_bias": False,
        "tfc_tdf_activation": "relu",
        "n_blocks": 7,
        "input_channels": 4,
        "internal_channels": 24,
        "first_conv_activation": "relu",
        "last_activation": "identity",
        "t_down_layers": None,
        "f_down_layers": None,
    }
    mismatches = {
        key: {"expected": expected_value, "actual": hyperparameters.get(key)}
        for key, expected_value in expected.items()
        if hyperparameters.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"Unexpected checkpoint hyperparameters: {mismatches}")


def load_default_checkpoint(
    checkpoint_path: Path,
) -> tuple[TfcTdfNeuralCore, dict[str, Any]]:
    checkpoint_path = checkpoint_path.resolve()
    if checkpoint_path.stat().st_size != EXPECTED_CHECKPOINT_BYTES:
        raise ValueError(
            f"Unexpected checkpoint size: {checkpoint_path.stat().st_size}"
        )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"Unexpected checkpoint SHA-256: {checkpoint_sha256}")

    _install_lightning_pickle_shim()
    # The exact trusted public checkpoint is digest-verified before unpickling.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    hyperparameters = dict(checkpoint["hyper_parameters"])
    _validate_hyperparameters(hyperparameters)

    source_state = checkpoint["state_dict"]
    ignored_keys = [key for key in source_state if key.startswith("stft.")]
    if ignored_keys != ["stft.stft.window"]:
        raise ValueError(f"Unexpected non-core checkpoint keys: {ignored_keys}")
    neural_state = OrderedDict(
        (key.removeprefix("spec2spec."), value)
        for key, value in source_state.items()
        if key.startswith("spec2spec.")
    )
    unclaimed_keys = [
        key
        for key in source_state
        if not key.startswith("spec2spec.") and key not in ignored_keys
    ]
    if unclaimed_keys:
        raise ValueError(f"Unclaimed checkpoint keys: {unclaimed_keys}")

    model = TfcTdfNeuralCore()
    model.load_state_dict(neural_state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    state_values = list(neural_state.values())
    state_elements = sum(value.numel() for value in state_values)
    float32_elements = sum(
        value.numel() for value in state_values if value.dtype == torch.float32
    )
    int64_elements = sum(
        value.numel() for value in state_values if value.dtype == torch.int64
    )
    metadata = {
        "checkpoint": {
            "path": UPSTREAM_CHECKPOINT_PATH,
            "bytes": checkpoint_path.stat().st_size,
            "sha256": checkpoint_sha256,
            "upstreamRepository": UPSTREAM_REPOSITORY,
            "upstreamCommit": UPSTREAM_COMMIT,
            "pytorchLightningVersion": checkpoint.get("pytorch-lightning_version"),
            "epoch": checkpoint.get("epoch"),
            "globalStep": checkpoint.get("global_step"),
        },
        "config": asdict(DEFAULT_CONFIG),
        "state": {
            "sourceTensorCount": len(source_state),
            "neuralTensorCount": len(neural_state),
            "ignoredTensorNames": ignored_keys,
            "stateElements": state_elements,
            "float32Elements": float32_elements,
            "int64Elements": int64_elements,
            "float32ConstantBytes": float32_elements * 4,
            "parameterElements": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        },
        "hyperparameters": {
            key: hyperparameters[key]
            for key in (
                "target_name",
                "n_fft",
                "hop_length",
                "num_frame",
                "spec_type",
                "spec_est_mode",
                "n_internal_layers",
                "kernel_size_t",
                "kernel_size_f",
                "bn_factor",
                "min_bn_units",
                "tfc_tdf_bias",
                "n_blocks",
                "input_channels",
                "internal_channels",
            )
        },
    }
    return model, metadata
