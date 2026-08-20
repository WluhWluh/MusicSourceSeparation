"""Host-side contracts and diagnostics for static short TFC-TDF windows."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


@dataclass(frozen=True)
class ShortWindowContract:
    """One overlap-save contract for a static TFC-TDF time dimension."""

    num_frames: int = 24
    sample_rate: int = 44_100
    n_fft: int = 2_048
    hop_length: int = 1_024
    left_context_hops: int = 2
    right_context_hops: int = 2

    def __post_init__(self) -> None:
        if self.num_frames < 8 or self.num_frames % 8 != 0:
            raise ValueError("num_frames must be at least 8 and divisible by 8")
        total_hops = self.num_frames - 1
        if self.left_context_hops < 0 or self.right_context_hops < 0:
            raise ValueError("context hops cannot be negative")
        if self.left_context_hops + self.right_context_hops >= total_hops:
            raise ValueError("context must leave at least one output hop")

    @property
    def frequency_bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def input_samples(self) -> int:
        return self.hop_length * (self.num_frames - 1)

    @property
    def left_context_samples(self) -> int:
        return self.left_context_hops * self.hop_length

    @property
    def right_context_samples(self) -> int:
        return self.right_context_hops * self.hop_length

    @property
    def stride_hops(self) -> int:
        return (
            self.num_frames
            - 1
            - self.left_context_hops
            - self.right_context_hops
        )

    @property
    def stride_samples(self) -> int:
        return self.stride_hops * self.hop_length

    @property
    def useful_samples(self) -> int:
        return self.stride_samples

    @property
    def input_to_output_ratio(self) -> float:
        return self.input_samples / self.stride_samples

    @property
    def output_efficiency(self) -> float:
        return self.stride_samples / self.input_samples

    def as_dict(
        self,
        assembly: str = "continuous-context-overlap-save",
    ) -> dict[str, int | float | str]:
        return {
            "sampleRate": self.sample_rate,
            "nFft": self.n_fft,
            "hopLength": self.hop_length,
            "numFrames": self.num_frames,
            "inputSamples": self.input_samples,
            "leftContextHops": self.left_context_hops,
            "rightContextHops": self.right_context_hops,
            "leftContextSamples": self.left_context_samples,
            "rightContextSamples": self.right_context_samples,
            "strideHops": self.stride_hops,
            "strideSamples": self.stride_samples,
            "usefulSamples": self.useful_samples,
            "inputToOutputRatio": self.input_to_output_ratio,
            "outputEfficiency": self.output_efficiency,
            "assembly": assembly,
        }


@dataclass(frozen=True)
class RenderResult:
    audio: np.ndarray
    window_count: int
    boundaries: tuple[int, ...]
    elapsed_seconds: float


def periodic_hann(size: int) -> np.ndarray:
    return np.hanning(size + 1)[:-1].astype(np.float32)


def stft_centered(wave: np.ndarray, contract: ShortWindowContract) -> np.ndarray:
    expected = (contract.input_samples, 2)
    if wave.shape != expected:
        raise ValueError(f"Expected waveform {expected}, got {wave.shape}")
    channels = wave.T
    pad = contract.n_fft // 2
    padded = np.pad(channels, ((0, 0), (pad, pad)), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(
        padded,
        contract.n_fft,
        axis=1,
    )[:, :: contract.hop_length, :]
    if frames.shape[1] != contract.num_frames:
        raise ValueError(f"Unexpected STFT frame count: {frames.shape}")
    windowed = frames * periodic_hann(contract.n_fft)[None, None, :]
    spectrum = np.fft.rfft(windowed, n=contract.n_fft, axis=-1)
    spectrum = spectrum.astype(np.complex64).transpose(0, 2, 1)
    packed = np.stack(
        [
            spectrum[0].real,
            spectrum[1].real,
            spectrum[0].imag,
            spectrum[1].imag,
        ],
        axis=0,
    )
    return packed[None].astype(np.float32)


def istft_centered(packed: np.ndarray, contract: ShortWindowContract) -> np.ndarray:
    expected = (1, 4, contract.frequency_bins, contract.num_frames)
    if packed.shape != expected:
        raise ValueError(f"Expected spectrum {expected}, got {packed.shape}")
    left = packed[0, 0] + 1j * packed[0, 2]
    right = packed[0, 1] + 1j * packed[0, 3]
    spectrum = np.stack([left, right], axis=0).transpose(0, 2, 1)
    frames = np.fft.irfft(spectrum, n=contract.n_fft, axis=-1)
    frames = frames.real.astype(np.float32)

    window = periodic_hann(contract.n_fft)
    padded_samples = contract.n_fft + contract.hop_length * (contract.num_frames - 1)
    output = np.zeros((2, padded_samples), dtype=np.float32)
    window_sum = np.zeros(padded_samples, dtype=np.float32)
    for frame_index in range(contract.num_frames):
        start = frame_index * contract.hop_length
        output[:, start : start + contract.n_fft] += (
            frames[:, frame_index, :] * window
        )
        window_sum[start : start + contract.n_fft] += window * window
    nonzero = window_sum > 1e-8
    output[:, nonzero] /= window_sum[nonzero]
    pad = contract.n_fft // 2
    output = output[:, pad : pad + contract.input_samples]
    return output.T


def load_audio(path: Path, sample_rate: int = 44_100) -> tuple[np.ndarray, int]:
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, repeats=2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    if source_rate != sample_rate:
        divisor = math.gcd(source_rate, sample_rate)
        audio = resample_poly(
            audio,
            sample_rate // divisor,
            source_rate // divisor,
            axis=0,
        ).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32), sample_rate


def assemble_input(
    audio: np.ndarray,
    output_start: int,
    output_length: int,
    contract: ShortWindowContract,
    mode: str = "continuous",
) -> np.ndarray:
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError(f"Expected stereo audio, got {audio.shape}")
    if output_length < 0 or output_length > contract.stride_samples:
        raise ValueError(f"Invalid output length: {output_length}")
    window = np.zeros((contract.input_samples, 2), dtype=np.float32)
    if mode == "isolated":
        source_start = max(0, output_start)
        source_end = min(audio.shape[0], output_start + output_length)
        if source_end > source_start:
            destination_start = contract.left_context_samples + source_start - output_start
            window[destination_start : destination_start + source_end - source_start] = (
                audio[source_start:source_end]
            )
        return window
    if mode != "continuous":
        raise ValueError(f"Unknown assembly mode: {mode}")

    window_start = output_start - contract.left_context_samples
    window_end = window_start + contract.input_samples
    source_start = max(0, window_start)
    source_end = min(audio.shape[0], window_end)
    if source_end > source_start:
        destination_start = source_start - window_start
        window[destination_start : destination_start + source_end - source_start] = (
            audio[source_start:source_end]
        )
    return window


def render_with_backend(
    audio: np.ndarray,
    contract: ShortWindowContract,
    backend: Callable[[np.ndarray], np.ndarray],
    assembly_mode: str = "continuous",
) -> RenderResult:
    output = np.empty_like(audio)
    window_count = math.ceil(audio.shape[0] / contract.stride_samples)
    boundaries = tuple(
        index * contract.stride_samples
        for index in range(1, window_count)
        if index * contract.stride_samples < audio.shape[0]
    )
    started = time.perf_counter()
    for window_index in range(window_count):
        output_start = window_index * contract.stride_samples
        output_length = min(contract.stride_samples, audio.shape[0] - output_start)
        window = assemble_input(
            audio,
            output_start,
            output_length,
            contract,
            mode=assembly_mode,
        )
        prediction = backend(stft_centered(window, contract))
        reconstructed = istft_centered(prediction, contract)
        output[output_start : output_start + output_length] = reconstructed[
            contract.left_context_samples : contract.left_context_samples + output_length
        ]
    return RenderResult(
        audio=output,
        window_count=window_count,
        boundaries=boundaries,
        elapsed_seconds=time.perf_counter() - started,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value.astype("<f4", copy=False)).tobytes()
    ).hexdigest()


def _iter_chunks(value: np.ndarray, chunk_size: int = 262_144) -> Iterable[np.ndarray]:
    for start in range(0, value.shape[0], chunk_size):
        yield value[start : start + chunk_size]


def waveform_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Metric shape mismatch: {reference.shape} != {candidate.shape}")
    error_power = 0.0
    reference_power = 0.0
    candidate_power = 0.0
    dot_product = 0.0
    absolute_error = 0.0
    maximum_error = 0.0
    count = 0
    for reference_chunk, candidate_chunk in zip(
        _iter_chunks(reference), _iter_chunks(candidate)
    ):
        if not np.isfinite(candidate_chunk).all():
            raise ValueError("Candidate contains non-finite values")
        error = candidate_chunk.astype(np.float64) - reference_chunk.astype(np.float64)
        ref64 = reference_chunk.astype(np.float64)
        cand64 = candidate_chunk.astype(np.float64)
        count += error.size
        error_power += float(np.sum(error * error))
        reference_power += float(np.sum(ref64 * ref64))
        candidate_power += float(np.sum(cand64 * cand64))
        dot_product += float(np.sum(ref64 * cand64))
        absolute_error += float(np.sum(np.abs(error)))
        maximum_error = max(maximum_error, float(np.max(np.abs(error))))
    return {
        "maxAbsError": maximum_error,
        "meanAbsError": absolute_error / count,
        "rmse": math.sqrt(error_power / count),
        "signalRms": math.sqrt(reference_power / count),
        "snrDb": float("inf")
        if error_power == 0.0
        else 10.0 * math.log10(reference_power / error_power),
        "cosineSimilarity": dot_product
        / math.sqrt(reference_power * candidate_power),
    }


def seam_metrics(
    audio: np.ndarray,
    boundaries: Iterable[int],
    radius: int = 2_048,
) -> dict[str, object]:
    differences = np.abs(np.diff(audio, axis=0)).astype(np.float64)
    observations: list[dict[str, object]] = []
    ratios: list[float] = []
    for boundary in boundaries:
        if boundary <= 0 or boundary >= audio.shape[0]:
            continue
        index = boundary - 1
        local_start = max(0, index - radius)
        local_end = min(differences.shape[0], index + radius + 1)
        local = np.concatenate(
            [differences[local_start:index], differences[index + 1 : local_end]],
            axis=0,
        )
        if local.shape[0] == 0:
            continue
        boundary_delta = differences[index]
        channel_ratios = [
            float(boundary_delta[channel] / max(np.percentile(local[:, channel], 95), 1e-9))
            for channel in range(audio.shape[1])
        ]
        ratio = max(channel_ratios)
        ratios.append(ratio)
        observations.append(
            {
                "sample": boundary,
                "boundaryDelta": boundary_delta.tolist(),
                "localDerivativeP95": np.percentile(local, 95, axis=0).tolist(),
                "ratio": ratio,
            }
        )
    if not ratios:
        return {"count": 0, "observations": []}
    return {
        "count": len(ratios),
        "p50Ratio": float(np.percentile(ratios, 50)),
        "p95Ratio": float(np.percentile(ratios, 95)),
        "maxRatio": float(np.max(ratios)),
        "countRatioAtLeast2": int(sum(value >= 2.0 for value in ratios)),
        "countRatioAtLeast4": int(sum(value >= 4.0 for value in ratios)),
        "observations": observations,
    }


def vocal_residual_proxy(
    mixture: np.ndarray,
    reference_vocals: np.ndarray,
    candidate_vocals: np.ndarray,
) -> dict[str, float]:
    """Compare instrumental vocal projection without claiming ground-truth SDR."""
    reference_instrumental = mixture - reference_vocals
    candidate_instrumental = mixture - candidate_vocals
    ref_vocal_power = 0.0
    candidate_vocal_power = 0.0
    candidate_vocal_dot = 0.0
    candidate_instr_power = 0.0
    candidate_instr_vocal_dot = 0.0
    reference_instr_vocal_dot = 0.0
    unexplained_power = 0.0
    for ref_vocal, cand_vocal, ref_instr, cand_instr in zip(
        _iter_chunks(reference_vocals),
        _iter_chunks(candidate_vocals),
        _iter_chunks(reference_instrumental),
        _iter_chunks(candidate_instrumental),
    ):
        ref = ref_vocal.astype(np.float64)
        cand = cand_vocal.astype(np.float64)
        ref_i = ref_instr.astype(np.float64)
        cand_i = cand_instr.astype(np.float64)
        ref_vocal_power += float(np.sum(ref * ref))
        candidate_vocal_power += float(np.sum(cand * cand))
        candidate_vocal_dot += float(np.sum(cand * ref))
        candidate_instr_power += float(np.sum(cand_i * cand_i))
        candidate_instr_vocal_dot += float(np.sum(cand_i * ref))
        reference_instr_vocal_dot += float(np.sum(ref_i * ref))
        unexplained = ref - cand
        unexplained_power += float(np.sum(unexplained * unexplained))
    projection_gain = candidate_vocal_dot / max(ref_vocal_power, 1e-12)
    projection_power = (candidate_instr_vocal_dot**2) / max(ref_vocal_power, 1e-12)
    reference_projection_power = (
        reference_instr_vocal_dot**2 / max(ref_vocal_power, 1e-12)
    )
    return {
        "candidateToReferenceVocalProjectionGain": projection_gain,
        "candidateUnexplainedReferenceVocalDb": 10.0
        * math.log10(max(unexplained_power, 1e-12) / max(ref_vocal_power, 1e-12)),
        "candidateInstrumentalReferenceVocalProjectionDb": 10.0
        * math.log10(max(projection_power, 1e-12) / max(candidate_instr_power, 1e-12)),
        "referenceInstrumentalReferenceVocalProjectionDb": 10.0
        * math.log10(
            max(reference_projection_power, 1e-12)
            / max(float(np.sum(reference_instrumental.astype(np.float64) ** 2)), 1e-12)
        ),
        "candidateToReferenceVocalEnergyRatioDb": 10.0
        * math.log10(max(candidate_vocal_power, 1e-12) / max(ref_vocal_power, 1e-12)),
    }


def write_flac(path: Path, audio: np.ndarray, sample_rate: int) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(audio, -1.0, 1.0), sample_rate, format="FLAC", subtype="PCM_16")
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rawFloat32Sha256": sha256_array(audio),
    }
