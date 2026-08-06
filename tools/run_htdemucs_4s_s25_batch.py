#!/usr/bin/env python3
"""Run three bounded official four-stem LiteRT CPU evaluations on the S25."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Sequence
import wave


PACKAGE = "com.example.musicsourceseparation"
TEST_PACKAGE = f"{PACKAGE}.test"
RUNNER = f"{TEST_PACKAGE}/androidx.test.runner.AndroidJUnitRunner"
TEST_CLASS = (
    "com.example.musicsourceseparation.benchmark."
    "HtdemucsCanonicalE2eInstrumentedTest"
)
REMOTE_ROOT = f"/sdcard/Android/data/{PACKAGE}/files/benchmark"
MODEL_ID = "htdemucs_4s_core_canonical_7p8s_fp32_v1_0_0"
MODEL_FILE = "htdemucs_4s.core.canonical_7p8s.fp32.tflite"
MODEL_BYTES = 178_042_000
MODEL_SHA256 = "9855718072ee819bacacdb6b670bd6257feca172bf27ac1d72dff994cdbeed81"
SAMPLE_RATE = 44_100
DEFAULT_DURATION_SECONDS = 30
STRIDE_SAMPLES = 257_985
STEMS = ("drums", "bass", "other", "vocals")
TRACKS = (
    ("guitar-heavy", "Athletics - II.mp3"),
    ("piano-heavy", "John Lennon - Imagine.MP3"),
    ("vocals-drums-sensitive", "Josiah James - Chasing The Wind.MP3"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def slugify(name: str) -> str:
    return re.sub(
        r"(^-+|-+$)",
        "",
        re.sub(r"[^a-z0-9]+", "-", Path(name).stem.lower()),
    )


def run_command(
    command: Sequence[str],
    *,
    timeout: int,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {command}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def adb(
    args: argparse.Namespace,
    *command: str,
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    return run_command([args.adb, "-s", args.serial, *command], timeout=timeout)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def pcm_prefix_sha256(path: Path, frames: int) -> str:
    with wave.open(str(path), "rb") as source:
        if (
            source.getframerate() != SAMPLE_RATE
            or source.getnchannels() != 2
            or source.getsampwidth() != 2
            or source.getnframes() < frames
        ):
            raise ValueError(f"Unexpected canonical WAV contract: {path}")
        digest = hashlib.sha256()
        remaining = frames
        while remaining:
            block_frames = min(16_384, remaining)
            payload = source.readframes(block_frames)
            if len(payload) != block_frames * 4:
                raise ValueError(f"Short canonical WAV prefix: {path}")
            digest.update(payload)
            remaining -= block_frames
    return digest.hexdigest()


def device_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    battery = adb(args, "shell", "dumpsys", "battery").stdout
    thermal = adb(args, "shell", "dumpsys", "thermalservice").stdout
    return {
        "epochMs": int(time.time() * 1000),
        "battery": battery,
        "thermalService": thermal,
        "thermalServiceSha256": hashlib.sha256(thermal.encode()).hexdigest(),
    }


def device_identity(args: argparse.Namespace) -> dict[str, Any]:
    properties = {}
    for key in (
        "ro.product.manufacturer",
        "ro.product.model",
        "ro.product.device",
        "ro.board.platform",
        "ro.soc.model",
        "ro.build.version.release",
        "ro.build.version.sdk",
        "ro.build.fingerprint",
    ):
        properties[key] = adb(args, "shell", "getprop", key).stdout.strip()
    return {
        "serial": args.serial,
        "properties": properties,
        "initialSnapshot": device_snapshot(args),
    }


def stage_model(args: argparse.Namespace, artifact: Path) -> dict[str, Any]:
    if artifact.stat().st_size != MODEL_BYTES or sha256_file(artifact) != MODEL_SHA256:
        raise ValueError("Local official four-stem LiteRT artifact identity mismatch")
    remote = f"{REMOTE_ROOT}/models/{MODEL_FILE}"
    adb(args, "shell", "mkdir", "-p", f"{REMOTE_ROOT}/models")
    current_result = run_command(
        [args.adb, "-s", args.serial, "shell", "sha256sum", remote],
        timeout=120,
        check=False,
    )
    current = current_result.stdout.strip() if current_result.returncode == 0 else ""
    if not current.startswith(MODEL_SHA256):
        adb(args, "push", str(artifact), remote, timeout=300)
    remote_hash = adb(args, "shell", "sha256sum", remote, timeout=120).stdout.strip()
    if not remote_hash.startswith(MODEL_SHA256):
        raise ValueError(f"Remote official four-stem model SHA mismatch: {remote_hash}")
    size = adb(args, "shell", "stat", "-c", "%s", remote).stdout.strip()
    if size != str(MODEL_BYTES):
        raise ValueError(f"Remote official four-stem model size mismatch: {size}")
    return {
        "modelId": MODEL_ID,
        "localPath": str(artifact),
        "remotePath": remote,
        "byteSize": MODEL_BYTES,
        "sha256": MODEL_SHA256,
    }


def validate_run(
    target: Path,
    source_sha: str,
    expected_selected_pcm_sha: str,
    duration_seconds: int,
) -> dict[str, Any]:
    selected_frames = SAMPLE_RATE * duration_seconds
    expected_windows = (selected_frames + STRIDE_SAMPLES - 1) // STRIDE_SAMPLES
    report_path = target / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    model = report["model"]
    if report.get("status") != "complete":
        raise ValueError("S25 report is incomplete")
    if (
        model.get("variant") != "official-4s"
        or model.get("modelId") != MODEL_ID
        or model.get("sha256") != MODEL_SHA256
        or model.get("diagnosticOnly") is not False
        or model.get("researchOnly") is not False
        or model.get("hostAdmissionStatus") != "admitted"
    ):
        raise ValueError("S25 report does not bind the admitted official four-stem identity")
    source = report["source"]
    if source["fileSha256"] != source_sha:
        raise ValueError("S25 source MP3 SHA mismatch")
    if source["selectedFrames"] != selected_frames:
        raise ValueError(f"S25 run did not select exactly {duration_seconds} seconds")
    if source["selectedPcmSha256"] != expected_selected_pcm_sha:
        raise ValueError("S25 selected PCM differs from the existing canonical decode")
    attempt = report["attempts"][0]
    if (
        attempt.get("status") != "complete"
        or attempt.get("expectedWindows") != expected_windows
    ):
        raise ValueError(
            f"S25 attempt did not complete the expected {expected_windows} windows"
        )
    output_root = target / "full" / "outputs"
    output_hashes = {}
    for stem in STEMS:
        path = output_root / f"{stem}.wav"
        with wave.open(str(path), "rb") as audio:
            contract = (
                audio.getframerate(),
                audio.getnchannels(),
                audio.getsampwidth(),
                audio.getnframes(),
            )
        if contract != (SAMPLE_RATE, 2, 2, selected_frames):
            raise ValueError(f"Unexpected S25 stem WAV contract for {path}: {contract}")
        output_hashes[stem] = sha256_file(path)
    return {
        "reportSha256": sha256_file(report_path),
        "selectedPcmSha256": source["selectedPcmSha256"],
        "canonicalPcmSha256": source["canonicalPcm"]["pcmSha256"],
        "durationSeconds": source["durationSeconds"],
        "windowCount": attempt["expectedWindows"],
        "realtimeFactor": attempt["realtimeFactor"],
        "attempt": attempt,
        "outputSha256": output_hashes,
    }


def process_track(
    args: argparse.Namespace,
    category: str,
    source: Path,
    output_root: Path,
) -> dict[str, Any]:
    duration_seconds = args.duration_seconds
    selected_frames = SAMPLE_RATE * duration_seconds
    slug = slugify(source.name)
    source_sha = sha256_file(source)
    prior_canonical = (
        args.canonical_root / "tracks" / slug / "s25-cpu" /
        "canonical-input-44100-stereo-pcm16.wav"
    )
    expected_selected_pcm_sha = pcm_prefix_sha256(prior_canonical, selected_frames)
    target = output_root / "tracks" / slug / f"s25-cpu-{duration_seconds}s"
    if target.is_dir():
        validation = validate_run(
            target,
            source_sha,
            expected_selected_pcm_sha,
            duration_seconds,
        )
        return {
            "category": category,
            "slug": slug,
            "sourceFileName": source.name,
            "sourceSha256": source_sha,
            "status": "skipped-valid-existing",
            "validation": validation,
        }

    remote_audio_name = (
        f"official-4s-{duration_seconds}s-{slug}{source.suffix.lower()}"
    )
    remote_audio = f"{REMOTE_ROOT}/audio/{remote_audio_name}"
    run_id = f"official-4s-{slug}-{duration_seconds}s"
    remote_run = (
        f"{REMOTE_ROOT}/htdemucs-canonical-e2e/cpu/official-4s/"
        f"{duration_seconds}s/{run_id}"
    )
    adb(args, "shell", "mkdir", "-p", f"{REMOTE_ROOT}/audio")
    adb(args, "shell", "rm", "-rf", remote_audio, remote_run)
    push = adb(args, "push", str(source), remote_audio, timeout=180)
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
        TEST_CLASS,
        "-e",
        "canonicalE2eAudioFile",
        remote_audio_name,
        "-e",
        "canonicalE2eAudioSha256",
        source_sha,
        "-e",
        "canonicalE2eDurationSeconds",
        str(duration_seconds),
        "-e",
        "canonicalE2eThreads",
        str(args.threads),
        "-e",
        "canonicalE2eRunId",
        run_id,
        "-e",
        "canonicalE2eModelVariant",
        "official-4s",
        RUNNER,
    ]
    started = time.perf_counter()
    instrumentation = run_command(command, timeout=args.track_timeout, check=False)
    elapsed = time.perf_counter() - started
    log_root = output_root / "tracks" / slug
    log_root.mkdir(parents=True, exist_ok=True)
    evidence_suffix = (
        "" if duration_seconds == DEFAULT_DURATION_SECONDS
        else f"-{duration_seconds}s"
    )
    (log_root / f"instrumentation{evidence_suffix}.stdout.txt").write_text(
        instrumentation.stdout,
        encoding="utf-8",
    )
    (log_root / f"instrumentation{evidence_suffix}.stderr.txt").write_text(
        instrumentation.stderr,
        encoding="utf-8",
    )
    if instrumentation.returncode != 0 or "OK (1 test)" not in instrumentation.stdout:
        raise RuntimeError(
            f"Instrumentation failed for {source.name}; evidence retained at {remote_run}"
        )
    adb(args, "pull", remote_run, str(target), timeout=300)
    after = device_snapshot(args)
    validation = validate_run(
        target,
        source_sha,
        expected_selected_pcm_sha,
        duration_seconds,
    )
    write_json(log_root / f"device-envelope{evidence_suffix}.json", {
        "category": category,
        "sourceFileName": source.name,
        "sourceByteSize": source.stat().st_size,
        "sourceSha256": source_sha,
        "expectedSelectedPcmSha256": expected_selected_pcm_sha,
        "remoteAudio": remote_audio,
        "remoteRun": remote_run,
        "pushStdout": push.stdout,
        "instrumentationWallSeconds": elapsed,
        "before": before,
        "after": after,
        "validation": validation,
    })
    adb(args, "shell", "rm", "-rf", remote_audio, remote_run)
    return {
        "category": category,
        "slug": slug,
        "sourceFileName": source.name,
        "sourceSha256": source_sha,
        "status": "complete",
        "instrumentationWallSeconds": elapsed,
        "validation": validation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--serial", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--track-timeout", type=int, default=600)
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=DEFAULT_DURATION_SECONDS,
    )
    parser.add_argument(
        "--track",
        action="append",
        dest="track_slugs",
        choices=tuple(slugify(file_name) for _, file_name in TRACKS),
        help="Run only the selected track slug; repeat to select multiple tracks.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path(
            "models/demucs/generated/"
            "htdemucs_4s_core_canonical_7p8s_fp32_v1_0_0/"
            f"{MODEL_FILE}"
        ),
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=Path("outputs/htdemucs6-mp3-no-gapless-s25-onnx-original-20260804"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/htdemucs4-official-s25-20260805"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.duration_seconds <= 0:
        raise ValueError("--duration-seconds must be positive")
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    if args.track_timeout <= 0:
        raise ValueError("--track-timeout must be positive")
    repo = Path.cwd().resolve()
    args.source_root = args.source_root.resolve()
    args.artifact = (repo / args.artifact).resolve() if not args.artifact.is_absolute() else args.artifact
    args.canonical_root = (
        (repo / args.canonical_root).resolve()
        if not args.canonical_root.is_absolute()
        else args.canonical_root.resolve()
    )
    output_root = (
        (repo / args.output_root).resolve()
        if not args.output_root.is_absolute()
        else args.output_root.resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)
    selected_slugs = set(args.track_slugs or ())
    selected_tracks = tuple(
        track for track in TRACKS
        if not selected_slugs or slugify(track[1]) in selected_slugs
    )
    if not selected_tracks:
        raise ValueError("No tracks selected")
    run_suffix = ""
    if args.duration_seconds != DEFAULT_DURATION_SECONDS or selected_slugs:
        suffix_slugs = "-".join(slugify(track[1]) for track in selected_tracks)
        run_suffix = f"-{args.duration_seconds}s-{suffix_slugs}"
    device = device_identity(args)
    device["artifact"] = stage_model(args, args.artifact)
    write_json(output_root / f"device{run_suffix}.json", device)
    progress: dict[str, Any] = {
        "schemaVersion": 1,
        "status": "running",
        "scope": "host-admitted-official-four-stem-device-evaluation",
        "durationSeconds": args.duration_seconds,
        "threads": args.threads,
        "tracks": [],
    }
    progress_path = output_root / f"batch-progress{run_suffix}.json"
    write_json(progress_path, progress)
    try:
        for category, file_name in selected_tracks:
            result = process_track(
                args,
                category,
                args.source_root / file_name,
                output_root,
            )
            progress["tracks"].append(result)
            write_json(progress_path, progress)
            print(json.dumps(result, sort_keys=True), flush=True)
    except Exception as error:
        progress["status"] = "failed"
        progress["error"] = str(error)
        progress["finalSnapshot"] = device_snapshot(args)
        write_json(progress_path, progress)
        raise
    progress["status"] = "complete"
    progress["finalSnapshot"] = device_snapshot(args)
    progress["aggregateAudioSeconds"] = (
        args.duration_seconds * len(selected_tracks)
    )
    progress["meanRealtimeFactor"] = sum(
        item["validation"]["realtimeFactor"] for item in progress["tracks"]
    ) / len(progress["tracks"])
    write_json(progress_path, progress)
    print(json.dumps({
        "status": "complete",
        "progress": str(progress_path),
        "trackCount": len(progress["tracks"]),
        "meanRealtimeFactor": progress["meanRealtimeFactor"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
