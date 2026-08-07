#!/usr/bin/env python3
"""Run a controlled HTDemucs CPU matrix on one explicitly selected device."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Sequence
import wave

import run_htdemucs_4s_s25_batch as common


SAMPLE_RATE = 44_100
STRIDE_SAMPLES = 257_985
DEFAULT_DURATION_SECONDS = 30
DEFAULT_RUNTIME_SHA256 = (
    "a162d1ddbdad87c002b7ec7eb31a703f2761335e693f292f94091b3569d8aa37"
)
TRACKS = common.TRACKS


@dataclass(frozen=True)
class ModelIdentity:
    variant: str
    model_id: str
    file_name: str
    byte_size: int
    sha256: str
    artifact_path: str
    stems: tuple[str, ...]
    diagnostic_only: bool
    research_only: bool
    host_admission_status: str


FOUR_STEMS = ("drums", "bass", "other", "vocals")
SIX_STEMS = FOUR_STEMS + ("guitar", "piano")
MODELS = {
    "official": ModelIdentity(
        variant="official",
        model_id="htdemucs_6s_core_canonical_7p8s_fp32_v1_0_0",
        file_name="htdemucs_6s.core.canonical_7p8s.fp32.tflite",
        byte_size=117_624_880,
        sha256="8b19e919dd17c6a93d862ca9b1158ed72f09feb4c52745819346369506ba4ed7",
        artifact_path=(
            "models/demucs/generated/"
            "htdemucs_6s_core_canonical_7p8s_fp32_v1_0_0/"
            "htdemucs_6s.core.canonical_7p8s.fp32.tflite"
        ),
        stems=SIX_STEMS,
        diagnostic_only=False,
        research_only=False,
        host_admission_status="admitted",
    ),
    "official-4s": ModelIdentity(
        variant="official-4s",
        model_id="htdemucs_4s_core_canonical_7p8s_fp32_v1_0_0",
        file_name="htdemucs_4s.core.canonical_7p8s.fp32.tflite",
        byte_size=178_042_000,
        sha256="9855718072ee819bacacdb6b670bd6257feca172bf27ac1d72dff994cdbeed81",
        artifact_path=(
            "models/demucs/generated/"
            "htdemucs_4s_core_canonical_7p8s_fp32_v1_0_0/"
            "htdemucs_4s.core.canonical_7p8s.fp32.tflite"
        ),
        stems=FOUR_STEMS,
        diagnostic_only=False,
        research_only=False,
        host_admission_status="admitted",
    ),
    "guitar-ft": ModelIdentity(
        variant="guitar-ft",
        model_id="htdemucs_6s_guitar_ft_core_canonical_7p8s_fp32_v1_0_0",
        file_name="htdemucs_6s_guitar_ft.core.canonical_7p8s.fp32.tflite",
        byte_size=117_729_544,
        sha256="ab632a5a024033d557eabb716f8829230532e8e5b4cd7ba146812a301f89b9a5",
        artifact_path=(
            "models/demucs/generated/"
            "htdemucs_6s_guitar_ft_core_canonical_7p8s_fp32_v1_0_0/"
            "htdemucs_6s_guitar_ft.core.canonical_7p8s.fp32.tflite"
        ),
        stems=SIX_STEMS,
        diagnostic_only=True,
        research_only=True,
        host_admission_status="not-admitted",
    ),
}

MODEL_REPORT_EXTENSION_FIELDS = (
    "variant",
    "diagnosticOnly",
    "researchOnly",
    "hostAdmissionStatus",
)


def percentile_nearest_rank(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile for an empty sequence")
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def timing_summary(values: Sequence[float]) -> dict[str, Any] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "minimumMs": min(values),
        "medianMs": statistics.median(values),
        "meanMs": statistics.fmean(values),
        "p95NearestRankMs": percentile_nearest_rank(values, 0.95),
        "maximumMs": max(values),
    }


def validate_model_report(model_report: dict[str, Any], model: ModelIdentity) -> None:
    expected_base = {
        "modelId": model.model_id,
        "fileName": model.file_name,
        "byteSize": model.byte_size,
        "sha256": model.sha256,
    }
    if any(model_report.get(key) != value for key, value in expected_base.items()):
        raise ValueError(f"{model.variant} report model identity mismatch")

    # The long-standing official report schema omits the later candidate-status
    # extension. Keep that omission explicit instead of treating missing values
    # as false/admitted values.
    if model.variant == "official":
        unexpected = [key for key in MODEL_REPORT_EXTENSION_FIELDS if key in model_report]
        if unexpected:
            raise ValueError(
                f"official report unexpectedly contains candidate fields: {unexpected}"
            )
        return

    expected_extension = {
        "variant": model.variant,
        "diagnosticOnly": model.diagnostic_only,
        "researchOnly": model.research_only,
        "hostAdmissionStatus": model.host_admission_status,
    }
    if any(model_report.get(key) != value for key, value in expected_extension.items()):
        raise ValueError(f"{model.variant} report candidate status mismatch")


def archive_incomplete_path(path: Path) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    for index in range(1_000):
        suffix = f".invalid-{stamp}" if index == 0 else f".invalid-{stamp}-{index}"
        archived = path.with_name(path.name + suffix)
        if not archived.exists():
            path.replace(archived)
            return archived
    raise RuntimeError(f"Could not reserve an archive path for {path}")


def read_git_revision() -> str:
    result = common.run_command(
        ["git", "rev-parse", "HEAD"],
        timeout=30,
    )
    return result.stdout.strip()


def apk_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing APK: {path}")
    return {
        "path": str(path),
        "byteSize": path.stat().st_size,
        "sha256": common.sha256_file(path),
    }


def installed_package_identity(
    args: argparse.Namespace,
    package_name: str,
    local_identity: dict[str, Any],
) -> dict[str, Any]:
    output = common.adb(args, "shell", "pm", "path", package_name).stdout
    paths = [line.removeprefix("package:").strip() for line in output.splitlines() if line]
    if len(paths) != 1 or not paths[0].endswith("/base.apk"):
        raise ValueError(f"Unexpected installed APK paths for {package_name}: {paths}")
    remote_path = paths[0]
    remote_sha256 = common.adb(
        args,
        "shell",
        "sha256sum",
        remote_path,
        timeout=120,
    ).stdout.split()[0]
    remote_size = int(
        common.adb(
            args,
            "shell",
            "stat",
            "-c",
            "%s",
            remote_path,
        ).stdout.strip()
    )
    if (
        remote_sha256 != local_identity["sha256"]
        or remote_size != local_identity["byteSize"]
    ):
        raise ValueError(f"Installed {package_name} APK differs from the local APK")
    return {
        "packageName": package_name,
        "remotePath": remote_path,
        "byteSize": remote_size,
        "sha256": remote_sha256,
    }


def mem_available_kib(args: argparse.Namespace) -> int:
    output = common.adb(args, "shell", "cat", "/proc/meminfo").stdout
    for line in output.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise ValueError("S10 /proc/meminfo did not report MemAvailable")


def device_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    snapshot = common.device_snapshot(args)
    snapshot["memAvailableKiB"] = mem_available_kib(args)
    snapshot["dataFilesystem"] = common.adb(
        args,
        "shell",
        "df",
        "-k",
        "/data",
    ).stdout
    return snapshot


def device_identity(args: argparse.Namespace) -> dict[str, Any]:
    identity = common.device_identity(args)
    identity["initialSnapshot"] = device_snapshot(args)
    hardware_serial = common.adb(
        args,
        "shell",
        "getprop",
        "ro.serialno",
    ).stdout.strip()
    if not hardware_serial:
        raise ValueError("Selected device did not report a hardware serial")
    identity.pop("serial", None)
    identity["hardwareSerialSha256"] = sha256_text(hardware_serial)
    return identity


def resolve_artifact(repo: Path, model: ModelIdentity) -> Path:
    path = (repo / model.artifact_path).resolve()
    if path.stat().st_size != model.byte_size:
        raise ValueError(f"{model.variant} local artifact byte-size mismatch")
    actual_sha = common.sha256_file(path)
    if actual_sha != model.sha256:
        raise ValueError(f"{model.variant} local artifact SHA-256 mismatch: {actual_sha}")
    return path


def stage_model(
    args: argparse.Namespace,
    artifact: Path,
    model: ModelIdentity,
) -> dict[str, Any]:
    remote = f"{common.REMOTE_ROOT}/models/{model.file_name}"
    common.adb(args, "shell", "mkdir", "-p", f"{common.REMOTE_ROOT}/models")
    current_result = common.run_command(
        [args.adb, "-s", args.serial, "shell", "sha256sum", remote],
        timeout=120,
        check=False,
    )
    current = current_result.stdout.strip() if current_result.returncode == 0 else ""
    push_stdout = None
    if not current.startswith(model.sha256):
        push_stdout = common.adb(
            args,
            "push",
            str(artifact),
            remote,
            timeout=300,
        ).stdout
    remote_hash = common.adb(
        args,
        "shell",
        "sha256sum",
        remote,
        timeout=120,
    ).stdout.strip()
    remote_size = common.adb(
        args,
        "shell",
        "stat",
        "-c",
        "%s",
        remote,
    ).stdout.strip()
    if not remote_hash.startswith(model.sha256) or remote_size != str(model.byte_size):
        raise ValueError(f"{model.variant} remote artifact identity mismatch")
    return {
        "variant": model.variant,
        "modelId": model.model_id,
        "localPath": str(artifact),
        "remotePath": remote,
        "byteSize": model.byte_size,
        "sha256": model.sha256,
        "pushStdout": push_stdout,
    }


def remote_run_path(
    model: ModelIdentity,
    duration_seconds: int,
    run_id: str,
    istft_mode: str,
    istft_workers: int,
    postprocess_mode: str,
) -> str:
    family = f"{common.REMOTE_ROOT}/htdemucs-canonical-e2e/cpu"
    if model.variant != "official":
        family += f"/{model.variant}"
    if istft_mode != "serial":
        family += f"/istft-{istft_mode}-w{istft_workers}"
    family += f"/postprocess-{postprocess_mode}"
    return f"{family}/{duration_seconds}s/{run_id}"


def validate_run(
    args: argparse.Namespace,
    target: Path,
    model: ModelIdentity,
    source_sha256: str,
    selected_pcm_sha256: str,
    duration_seconds: int,
) -> dict[str, Any]:
    selected_frames = SAMPLE_RATE * duration_seconds
    expected_windows = math.ceil(selected_frames / STRIDE_SAMPLES)
    report_path = target / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    lane_count = len(model.stems) * 2
    expected_istft_fields = {
        "mode": args.istft_mode,
        "requestedWorkers": args.istft_workers,
        "effectiveWorkers": args.istft_workers,
        "laneCount": lane_count,
        "fftImplementation": "JTransforms-3.1-FloatFFT_1D-complexInverse",
        "jTransformsInternalThreadPolicy": "library-default-global",
        "complexTransformArrayLength": 8192,
        "executorOwned": args.istft_mode == "parallel-lanes",
        "floatParityCheckRequested": args.validate_istft_float_parity,
        "postprocessMode": args.postprocess_mode,
        "reuseWaveformWorkspace": args.postprocess_mode == "fused-reuse",
        "reuseDspIoWorkspaces": args.postprocess_mode == "fused-reuse",
        "reusePcmByteBuffers": args.postprocess_mode == "fused-reuse",
        "fusedBranchOlaPcmWrite": args.postprocess_mode == "fused-reuse",
        "tensorBufferReadIntoAvailable": False,
    }
    execution = report.get("execution", {})
    if (
        report.get("schemaVersion") != 3
        or any(execution.get(key) != value for key, value in expected_istft_fields.items())
        or execution.get("jTransformsGlobalThreads", 0) < 1
        or execution.get("jTransformsProcessorCount", 0) < 1
        or execution.get("jTransforms1dFft2ThreadsThreshold") != 8192
        or execution.get("jTransforms1dFft4ThreadsThreshold") != 65536
    ):
        raise ValueError(f"{model.variant} iSTFT execution identity mismatch")
    if report.get("status") != "complete":
        raise ValueError(f"{model.variant} report is incomplete")

    runtime = report["runtime"]
    expected_device = args.expected_device_model
    if (
        report.get("backend") != "litert-compiled-model-cpu"
        or runtime.get("runtimeId") != "com.google.ai.edge.litert:litert:2.1.5"
        or runtime.get("litertVersion") != "2.1.5"
        or runtime.get("runtimeArtifactSha256") != args.expected_runtime_sha256
        or runtime.get("sourceRevision") != args.expected_source_revision
        or runtime.get("sourceDirty") != args.expected_source_dirty
        or runtime.get("model") != expected_device
        or (
            args.expected_soc_model is not None
            and runtime.get("socModel") != args.expected_soc_model
        )
        or (args.expected_sdk is not None and runtime.get("sdk") != args.expected_sdk)
        or (
            args.expected_manufacturer is not None
            and runtime.get("manufacturer") != args.expected_manufacturer
        )
        or (
            args.expected_device_codename is not None
            and runtime.get("device") != args.expected_device_codename
        )
    ):
        raise ValueError(f"{model.variant} runtime/device identity mismatch: {runtime}")

    model_report = report["model"]
    validate_model_report(model_report, model)

    source = report["source"]
    if (
        source.get("fileSha256") != source_sha256
        or source.get("selectedFrames") != selected_frames
        or source.get("selectedPcmSha256") != selected_pcm_sha256
    ):
        raise ValueError(f"{model.variant} selected PCM identity mismatch")

    attempts = report["attempts"]
    if len(attempts) != 1:
        raise ValueError(f"{model.variant} expected exactly one attempt")
    attempt = attempts[0]
    if (
        attempt.get("status") != "complete"
        or attempt.get("backend") != "CPU"
        or attempt.get("threads") != args.threads
        or attempt.get("litertCpuThreads") != args.threads
        or attempt.get("dspIstft") != execution
        or attempt.get("expectedWindows") != expected_windows
        or attempt.get("completedWindows") != expected_windows
        or attempt.get("outputFrames") != selected_frames
    ):
        raise ValueError(f"{model.variant} attempt/window contract mismatch")

    core_benchmark = attempt.get("coreBenchmark")
    if args.core_warmup_runs or args.core_measured_runs:
        if (
            not isinstance(core_benchmark, dict)
            or core_benchmark.get("warmupRuns") != args.core_warmup_runs
            or core_benchmark.get("measuredRuns") != args.core_measured_runs
            or len(core_benchmark.get("samples", [])) != args.core_measured_runs
            or core_benchmark.get("summary", {}).get("count") != args.core_measured_runs
            or core_benchmark.get("perOpProfiling", {}).get("status")
            != "unsupported-by-litert-2.1.5-java-api"
        ):
            raise ValueError(f"{model.variant} core benchmark contract mismatch")
    elif core_benchmark is not None:
        raise ValueError(f"{model.variant} contains an unrequested core benchmark")

    windows = attempt["windows"]
    if len(windows) != expected_windows:
        raise ValueError(f"{model.variant} window evidence inventory mismatch")
    if [window.get("index") for window in windows] != list(range(expected_windows)):
        raise ValueError(f"{model.variant} window evidence indices are not contiguous")

    float_parity = None
    parity_stage = windows[0]["stages"].get("iSTFTParityComparison")
    if args.validate_istft_float_parity:
        if parity_stage is None or "iSTFTParityReference" not in windows[0]["stages"]:
            raise ValueError(f"{model.variant} raw-float iSTFT parity evidence is missing")
        float_parity = parity_stage["result"]
        expected_elements = len(model.stems) * 2 * 343_980
        if (
            float_parity.get("elementCount") != expected_elements
            or float_parity.get("byteCount") != expected_elements * 4
            or float_parity.get("byteOrder") != "little-endian-raw-float-bits"
            or float_parity.get("mismatchCount") != 0
            or float_parity.get("firstMismatchIndex") is not None
            or float_parity.get("candidateNonFiniteCount") != 0
            or float_parity.get("referenceNonFiniteCount") != 0
            or float_parity.get("candidateRawSha256")
            != float_parity.get("referenceRawSha256")
        ):
            raise ValueError(f"{model.variant} raw-float iSTFT parity gate failed")
    elif parity_stage is not None:
        raise ValueError(f"{model.variant} contains an unrequested iSTFT parity stage")

    if set(attempt["outputs"]) != set(model.stems):
        raise ValueError(f"{model.variant} output stem inventory mismatch")
    output_root = target / "full" / "outputs"
    output_hashes = {}
    for stem in model.stems:
        path = output_root / f"{stem}.wav"
        with wave.open(str(path), "rb") as audio:
            contract = (
                audio.getframerate(),
                audio.getnchannels(),
                audio.getsampwidth(),
                audio.getnframes(),
        )
        if contract != (SAMPLE_RATE, 2, 2, selected_frames):
            raise ValueError(f"Unexpected {model.variant} WAV contract: {path}: {contract}")
        actual_size = path.stat().st_size
        actual_sha256 = common.sha256_file(path)
        output_evidence = attempt["outputs"][stem]
        if (
            output_evidence.get("byteSize") != actual_size
            or output_evidence.get("sha256") != actual_sha256
        ):
            raise ValueError(f"{model.variant}/{stem} WAV differs from device evidence")
        output_hashes[stem] = {
            "byteSize": actual_size,
            "sha256": actual_sha256,
        }
    non_finite = sum(value["nonFiniteCount"] for value in attempt["outputs"].values())
    if non_finite != 0:
        raise ValueError(f"{model.variant} produced {non_finite} non-finite samples")

    inference_ms = [window["stages"]["inference"]["wallMs"] for window in windows]
    warm_inference_ms = inference_ms[1:]
    istft_ms = [window["stages"]["iSTFT"]["wallMs"] for window in windows]
    warm_istft_ms = istft_ms[1:]
    thermal_statuses = sorted({window["thermalStatus"] for window in windows})
    performance = {
        "prepareWallMs": attempt["prepare"]["total"]["wallMs"],
        "attemptWallMs": attempt["total"]["wallMs"],
        "e2eWallMs": attempt["e2eTotal"]["wallMs"],
        "realtimeFactor": attempt["realtimeFactor"],
        "coreBenchmark": core_benchmark,
        "windowCount": len(windows),
        "firstWindowInferenceMs": inference_ms[0],
        "allWindowInference": timing_summary(inference_ms),
        "warmOnlyInference": timing_summary(warm_inference_ms),
        "firstWindowIstftMs": istft_ms[0],
        "allWindowIstft": timing_summary(istft_ms),
        "warmOnlyIstft": timing_summary(warm_istft_ms),
        "stageSummary": attempt["stageSummary"],
        "peakWindowPssKiB": max(window["process"]["pssKb"] for window in windows),
        "peakWindowNativeAllocatedBytes": max(
            window["process"]["nativeHeapAllocatedBytes"] for window in windows
        ),
        "peakWindowJavaUsedBytes": max(
            window["process"]["javaHeapUsedBytes"] for window in windows
        ),
        "thermalStatuses": thermal_statuses,
        "nonFiniteSampleCount": non_finite,
        "clippedSampleCount": sum(
            value["clippedSampleCount"] for value in attempt["outputs"].values()
        ),
    }
    return {
        "reportSha256": common.sha256_file(report_path),
        "selectedPcmSha256": source["selectedPcmSha256"],
        "canonicalPcmSha256": source["canonicalPcm"]["pcmSha256"],
        "durationSeconds": source["durationSeconds"],
        "performance": performance,
        "outputSha256": output_hashes,
        "floatParity": float_parity,
        "istftExecution": execution,
    }


def run_variant(
    args: argparse.Namespace,
    output_root: Path,
    model: ModelIdentity,
    category: str,
    source: Path,
    remote_audio_name: str,
    remote_audio: str,
    selected_pcm_sha256: str,
) -> dict[str, Any]:
    duration_seconds = args.duration_seconds
    slug = common.slugify(source.name)
    source_sha256 = common.sha256_file(source)
    target = (
        output_root
        / "tracks"
        / slug
        / model.variant
        / (
            f"{args.device_label}-cpu-{duration_seconds}s-t{args.threads}"
            f"-istft-{args.istft_mode}-w{args.istft_workers}"
            f"-core-w{args.core_warmup_runs}-m{args.core_measured_runs}"
            f"-postprocess-{args.postprocess_mode}"
            f"{'-float-parity' if args.validate_istft_float_parity else ''}"
        )
    )
    archived_invalid_target = None
    if target.is_dir():
        try:
            validation = validate_run(
                args,
                target,
                model,
                source_sha256,
                selected_pcm_sha256,
                duration_seconds,
            )
        except (KeyError, OSError, TypeError, ValueError):
            archived_invalid_target = archive_incomplete_path(target)
        else:
            return {
                "category": category,
                "track": slug,
                "variant": model.variant,
                "status": "skipped-valid-existing",
                "validation": validation,
            }

    staging_target = target.with_name(target.name + ".partial")
    archived_partial = None
    if staging_target.exists():
        archived_partial = archive_incomplete_path(staging_target)

    run_id = (
        f"{args.device_label}-{model.variant}-{slug}-{duration_seconds}s"
        f"-istft-{args.istft_mode}-w{args.istft_workers}"
        f"{'-float-parity' if args.validate_istft_float_parity else ''}"
    )
    remote_run = remote_run_path(
        model,
        duration_seconds,
        run_id,
        args.istft_mode,
        args.istft_workers,
        args.postprocess_mode,
    )
    common.adb(args, "shell", "rm", "-rf", remote_run)
    available_before = mem_available_kib(args)
    if available_before < args.minimum_mem_available_kib:
        raise RuntimeError(
            f"MemAvailable {available_before} KiB is below the safety threshold "
            f"{args.minimum_mem_available_kib} KiB"
        )
    before = device_snapshot(args)
    command = [
        args.adb,
        "-s",
        args.serial,
        "shell",
        "am",
        "instrument",
        "-w",
        "-r",
        "-e",
        "class",
        common.TEST_CLASS,
        "-e",
        "canonicalE2eAudioFile",
        remote_audio_name,
        "-e",
        "canonicalE2eAudioSha256",
        source_sha256,
        "-e",
        "canonicalE2eDurationSeconds",
        str(duration_seconds),
        "-e",
        "canonicalE2eThreads",
        str(args.threads),
        "-e",
        "canonicalE2eIstftMode",
        args.istft_mode,
        "-e",
        "canonicalE2eIstftWorkers",
        str(args.istft_workers),
        "-e",
        "canonicalE2eValidateIstftFloatParity",
        str(args.validate_istft_float_parity).lower(),
        "-e",
        "canonicalE2eCoreWarmupRuns",
        str(args.core_warmup_runs),
        "-e",
        "canonicalE2eCoreMeasuredRuns",
        str(args.core_measured_runs),
        "-e",
        "canonicalE2ePostprocessMode",
        args.postprocess_mode,
        "-e",
        "canonicalE2eRunId",
        run_id,
        "-e",
        "canonicalE2eModelVariant",
        model.variant,
        common.RUNNER,
    ]
    started = time.perf_counter()
    instrumentation = common.run_command(
        command,
        timeout=args.track_timeout,
        check=False,
    )
    instrumentation_wall_seconds = time.perf_counter() - started
    log_root = output_root / "tracks" / slug / model.variant
    log_root.mkdir(parents=True, exist_ok=True)
    log_stem = (
        f"instrumentation-{args.device_label}-{duration_seconds}s"
        f"-istft-{args.istft_mode}-w{args.istft_workers}"
        f"-core-w{args.core_warmup_runs}-m{args.core_measured_runs}"
        f"-postprocess-{args.postprocess_mode}"
        f"{'-float-parity' if args.validate_istft_float_parity else ''}"
    )
    (log_root / f"{log_stem}.stdout.txt").write_text(
        instrumentation.stdout,
        encoding="utf-8",
    )
    (log_root / f"{log_stem}.stderr.txt").write_text(
        instrumentation.stderr,
        encoding="utf-8",
    )
    if instrumentation.returncode != 0 or "OK (1 test)" not in instrumentation.stdout:
        raise RuntimeError(
            f"Instrumentation failed for {model.variant}/{source.name}; "
            f"remote evidence retained at {remote_run}"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    common.adb(args, "pull", remote_run, str(staging_target), timeout=300)
    after = device_snapshot(args)
    validation = validate_run(
        args,
        staging_target,
        model,
        source_sha256,
        selected_pcm_sha256,
        duration_seconds,
    )
    staging_target.replace(target)
    common.write_json(
        log_root / f"device-envelope-{args.device_label}-{duration_seconds}s.json",
        {
            "category": category,
            "track": slug,
            "variant": model.variant,
            "sourceFileName": source.name,
            "sourceByteSize": source.stat().st_size,
            "sourceSha256": source_sha256,
            "expectedSelectedPcmSha256": selected_pcm_sha256,
            "remoteAudio": remote_audio,
            "remoteRun": remote_run,
            "instrumentationWallSeconds": instrumentation_wall_seconds,
            "before": before,
            "after": after,
            "archivedInvalidTarget": (
                str(archived_invalid_target) if archived_invalid_target is not None else None
            ),
            "archivedPartialTarget": (
                str(archived_partial) if archived_partial is not None else None
            ),
            "validation": validation,
        },
    )
    common.adb(args, "shell", "rm", "-rf", remote_run)
    return {
        "category": category,
        "track": slug,
        "variant": model.variant,
        "status": "complete",
        "instrumentationWallSeconds": instrumentation_wall_seconds,
        "archivedInvalidTarget": (
            str(archived_invalid_target) if archived_invalid_target is not None else None
        ),
        "archivedPartialTarget": (
            str(archived_partial) if archived_partial is not None else None
        ),
        "validation": validation,
    }


def rotated(values: Sequence[ModelIdentity], offset: int) -> tuple[ModelIdentity, ...]:
    if not values:
        return ()
    index = offset % len(values)
    return tuple(values[index:]) + tuple(values[:index])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--serial", required=True)
    parser.add_argument("--device-label", required=True)
    parser.add_argument("--expected-device-model", required=True)
    parser.add_argument("--expected-hardware-serial-sha256")
    parser.add_argument("--expected-soc-model")
    parser.add_argument("--expected-sdk", type=int)
    parser.add_argument("--expected-os-release")
    parser.add_argument("--expected-manufacturer")
    parser.add_argument("--expected-device-codename")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--istft-mode",
        choices=("serial", "parallel-lanes"),
        default="serial",
    )
    parser.add_argument("--istft-workers", type=int, default=1)
    parser.add_argument("--validate-istft-float-parity", action="store_true")
    parser.add_argument("--core-warmup-runs", type=int, default=0)
    parser.add_argument("--core-measured-runs", type=int, default=0)
    parser.add_argument(
        "--postprocess-mode",
        choices=("legacy", "fused-reuse"),
        default="legacy",
    )
    parser.add_argument("--duration-seconds", type=int, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--track-timeout", type=int, default=1200)
    parser.add_argument("--cooldown-seconds", type=int, default=10)
    parser.add_argument("--minimum-mem-available-kib", type=int, default=2_000_000)
    parser.add_argument(
        "--variant",
        action="append",
        choices=tuple(MODELS),
        dest="variants",
        help="Select a model; repeat to set the base rotation order.",
    )
    parser.add_argument(
        "--track",
        action="append",
        dest="track_slugs",
        choices=tuple(common.slugify(file_name) for _, file_name in TRACKS),
        help="Select a track; repeat to preserve an explicit track order.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Directory containing the explicitly selected source tracks.",
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=Path("outputs/htdemucs6-mp3-no-gapless-s25-onnx-original-20260804"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/htdemucs-cpu-device-matrix"),
    )
    parser.add_argument(
        "--app-apk",
        type=Path,
        default=Path("app/build/outputs/apk/standard/debug/app-standard-debug.apk"),
    )
    parser.add_argument(
        "--test-apk",
        type=Path,
        default=Path(
            "app/build/outputs/apk/androidTest/standard/debug/"
            "app-standard-debug-androidTest.apk"
        ),
    )
    parser.add_argument("--expected-source-revision")
    parser.add_argument("--expected-source-dirty", choices=("true", "false"), default="false")
    parser.add_argument("--expected-runtime-sha256", default=DEFAULT_RUNTIME_SHA256)
    parser.add_argument("--keep-remote-models", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.duration_seconds <= 0:
        raise ValueError("--duration-seconds must be positive")
    if args.threads not in range(1, 17):
        raise ValueError("--threads must be in [1, 16]")
    if args.istft_workers not in range(1, 17):
        raise ValueError("--istft-workers must be in [1, 16]")
    if args.istft_mode == "serial" and args.istft_workers != 1:
        raise ValueError("--istft-mode serial requires --istft-workers 1")
    if args.istft_mode == "parallel-lanes" and args.istft_workers < 2:
        raise ValueError("--istft-mode parallel-lanes requires at least 2 workers")
    if args.validate_istft_float_parity and args.istft_mode != "parallel-lanes":
        raise ValueError("--validate-istft-float-parity requires parallel-lanes mode")
    if not 0 <= args.core_warmup_runs <= 100:
        raise ValueError("--core-warmup-runs must be in [0, 100]")
    if not 0 <= args.core_measured_runs <= 100:
        raise ValueError("--core-measured-runs must be in [0, 100]")
    if (args.core_warmup_runs == 0) != (args.core_measured_runs == 0):
        raise ValueError("core warmup and measured counts must both be zero or both positive")
    if args.cooldown_seconds < 0:
        raise ValueError("--cooldown-seconds must be non-negative")
    if args.expected_hardware_serial_sha256 is not None:
        expected_hardware_serial_sha256 = args.expected_hardware_serial_sha256.lower()
        if (
            len(expected_hardware_serial_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_hardware_serial_sha256
            )
        ):
            raise ValueError("--expected-hardware-serial-sha256 must be 64 hex characters")
        args.expected_hardware_serial_sha256 = expected_hardware_serial_sha256

    repo = Path.cwd().resolve()
    args.source_root = args.source_root.resolve()
    args.canonical_root = (repo / args.canonical_root).resolve()
    args.output_root = (repo / args.output_root).resolve()
    args.app_apk = (repo / args.app_apk).resolve()
    args.test_apk = (repo / args.test_apk).resolve()
    args.expected_source_revision = args.expected_source_revision or read_git_revision()

    variants = tuple(MODELS[name] for name in (args.variants or tuple(MODELS)))
    selected_slugs = tuple(args.track_slugs or ())
    if selected_slugs:
        tracks_by_slug = {common.slugify(file_name): (category, file_name) for category, file_name in TRACKS}
        selected_tracks = tuple(tracks_by_slug[slug] for slug in selected_slugs)
    else:
        selected_tracks = TRACKS

    args.output_root.mkdir(parents=True, exist_ok=True)
    identity = device_identity(args)
    actual_model = identity["properties"]["ro.product.model"]
    if actual_model != args.expected_device_model:
        raise ValueError(f"Selected device is {actual_model}, expected {args.expected_device_model}")
    expected_properties = {
        "ro.soc.model": args.expected_soc_model,
        "ro.build.version.sdk": (
            str(args.expected_sdk) if args.expected_sdk is not None else None
        ),
        "ro.build.version.release": args.expected_os_release,
        "ro.product.manufacturer": args.expected_manufacturer,
        "ro.product.device": args.expected_device_codename,
    }
    for property_name, expected_value in expected_properties.items():
        if (
            expected_value is not None
            and identity["properties"].get(property_name) != expected_value
        ):
            raise ValueError(
                f"Selected device {property_name} is "
                f"{identity['properties'].get(property_name)!r}, expected {expected_value!r}"
            )
    if (
        args.expected_hardware_serial_sha256 is not None
        and identity["hardwareSerialSha256"] != args.expected_hardware_serial_sha256
    ):
        raise ValueError(
            "Selected hardware serial SHA-256 is "
            f"{identity['hardwareSerialSha256']!r}, "
            f"expected {args.expected_hardware_serial_sha256!r}"
        )
    local_apks = {
        "app": apk_identity(args.app_apk),
        "test": apk_identity(args.test_apk),
    }
    identity["apks"] = {
        "local": local_apks,
        "installed": {
            "app": installed_package_identity(
                args,
                common.PACKAGE,
                local_apks["app"],
            ),
            "test": installed_package_identity(
                args,
                common.TEST_PACKAGE,
                local_apks["test"],
            ),
        },
    }
    identity["expectedRuntime"] = {
        "litertVersion": "2.1.5",
        "runtimeArtifactSha256": args.expected_runtime_sha256,
        "sourceRevision": args.expected_source_revision,
    }

    artifacts = {}
    try:
        for model in variants:
            artifacts[model.variant] = stage_model(
                args,
                resolve_artifact(repo, model),
                model,
            )
        identity["artifacts"] = artifacts
        common.write_json(args.output_root / "device.json", identity)

        progress: dict[str, Any] = {
            "schemaVersion": 1,
            "status": "running",
            "scope": "device-cpu-performance-matrix-not-product-qualification",
            "hardwareSerialSha256": identity["hardwareSerialSha256"],
            "deviceLabel": args.device_label,
            "durationSeconds": args.duration_seconds,
            "threads": args.threads,
            "litertCpuThreads": args.threads,
            "istftMode": args.istft_mode,
            "istftWorkers": args.istft_workers,
            "validateIstftFloatParity": args.validate_istft_float_parity,
            "coreWarmupRuns": args.core_warmup_runs,
            "coreMeasuredRuns": args.core_measured_runs,
            "postprocessMode": args.postprocess_mode,
            "baseVariantOrder": [model.variant for model in variants],
            "trackOrder": [common.slugify(file_name) for _, file_name in selected_tracks],
            "runs": [],
        }
        progress_name = (
            f"matrix-{args.duration_seconds}s-t{args.threads}-istft-{args.istft_mode}"
            f"-w{args.istft_workers}-core-w{args.core_warmup_runs}"
            f"-m{args.core_measured_runs}-postprocess-{args.postprocess_mode}"
            f"{'-float-parity' if args.validate_istft_float_parity else ''}.json"
        )
        progress_path = args.output_root / progress_name
        common.write_json(progress_path, progress)

        run_index = 0
        for track_index, (category, file_name) in enumerate(selected_tracks):
            source = args.source_root / file_name
            if not source.is_file():
                raise FileNotFoundError(source)
            slug = common.slugify(source.name)
            canonical = (
                args.canonical_root
                / "tracks"
                / slug
                / "s25-cpu"
                / "canonical-input-44100-stereo-pcm16.wav"
            )
            selected_frames = SAMPLE_RATE * args.duration_seconds
            selected_pcm_sha256 = common.pcm_prefix_sha256(canonical, selected_frames)
            source_sha256 = common.sha256_file(source)
            remote_audio_name = (
                f"{args.device_label}-{args.duration_seconds}s-{slug}{source.suffix.lower()}"
            )
            remote_audio = f"{common.REMOTE_ROOT}/audio/{remote_audio_name}"
            common.adb(args, "shell", "mkdir", "-p", f"{common.REMOTE_ROOT}/audio")
            common.adb(args, "shell", "rm", "-rf", remote_audio)
            push = common.adb(args, "push", str(source), remote_audio, timeout=180)
            remote_hash = common.adb(
                args,
                "shell",
                "sha256sum",
                remote_audio,
                timeout=120,
            ).stdout.strip()
            if not remote_hash.startswith(source_sha256):
                raise ValueError(f"Remote source SHA mismatch for {source.name}")
            try:
                for model in rotated(variants, track_index):
                    if run_index and args.cooldown_seconds:
                        time.sleep(args.cooldown_seconds)
                    result = run_variant(
                        args,
                        args.output_root,
                        model,
                        category,
                        source,
                        remote_audio_name,
                        remote_audio,
                        selected_pcm_sha256,
                    )
                    result["sourcePushStdout"] = push.stdout
                    progress["runs"].append(result)
                    common.write_json(progress_path, progress)
                    run_index += 1
            finally:
                common.adb(args, "shell", "rm", "-rf", remote_audio)

        progress["status"] = "complete"
        common.write_json(progress_path, progress)
    finally:
        if not args.keep_remote_models:
            for model in variants:
                common.run_command(
                    [
                        args.adb,
                        "-s",
                        args.serial,
                        "shell",
                        "rm",
                        "-f",
                        f"{common.REMOTE_ROOT}/models/{model.file_name}",
                    ],
                    timeout=120,
                    check=False,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
