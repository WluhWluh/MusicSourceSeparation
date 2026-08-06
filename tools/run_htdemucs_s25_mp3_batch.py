#!/usr/bin/env python3
"""Run full-song canonical HTDemucs CPU separation for a directory of MP3 files on S25."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Sequence


PACKAGE = "com.example.musicsourceseparation"
TEST_PACKAGE = f"{PACKAGE}.test"
RUNNER = f"{TEST_PACKAGE}/androidx.test.runner.AndroidJUnitRunner"
TEST_CLASS = (
    "com.example.musicsourceseparation.benchmark."
    "HtdemucsCanonicalE2eInstrumentedTest"
)
REMOTE_ROOT = f"/sdcard/Android/data/{PACKAGE}/files/benchmark"
STEMS = ("drums", "bass", "other", "vocals", "guitar", "piano")
ORDER = (
    "Athletics - II.mp3",
    "Kygo Ed Sheeran - I See Fire (Kygo Remix).mp3",
    "Sleeping at Last - North.mp3",
    "Sleeping at Last - Already Gone.MP3",
    "Josiah James - Chasing The Wind.MP3",
    "Joel Hanson - Traveling Light.mp3",
    "Nylon - Eventide.mp3",
    "John Lennon - Imagine.MP3",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def slugify(name: str) -> str:
    stem = Path(name).stem.lower()
    return re.sub(r"(^-+|-+$)", "", re.sub(r"[^a-z0-9]+", "-", stem))


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


def adb(args: argparse.Namespace, *command: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return run_command(
        [args.adb, "-s", args.serial, *command],
        timeout=timeout,
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


def parse_key_values(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for line in text.splitlines():
        key, separator, value = line.strip().partition(":")
        if not separator:
            continue
        normalized = re.sub(r"[^a-zA-Z0-9]+", "_", key).strip("_")
        value = value.strip()
        try:
            result[normalized] = int(value)
        except ValueError:
            result[normalized] = value
    return result


def device_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    battery = adb(args, "shell", "dumpsys", "battery").stdout
    thermal = adb(args, "shell", "dumpsys", "thermalservice").stdout
    return {
        "epochMs": int(time.time() * 1000),
        "battery": parse_key_values(battery),
        "thermalServiceSha256": hashlib.sha256(thermal.encode("utf-8")).hexdigest(),
        "thermalService": thermal,
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
    model_path = f"{REMOTE_ROOT}/models/htdemucs_6s.core.canonical_7p8s.fp32.tflite"
    model_hash = adb(args, "shell", "sha256sum", model_path, timeout=120).stdout.strip()
    return {
        "serial": args.serial,
        "properties": properties,
        "modelSha256Output": model_hash,
        "initialSnapshot": device_snapshot(args),
    }


def existing_run_is_valid(target: Path, source_sha: str) -> bool:
    report_path = target / "report.json"
    if not report_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if report.get("status") != "complete":
        return False
    if report.get("source", {}).get("fileSha256") != source_sha:
        return False
    frames = report.get("source", {}).get("selectedFrames")
    if not isinstance(frames, int) or frames <= 0:
        return False
    expected_bytes = 44 + frames * 4
    if not (target / "canonical-input-44100-stereo-pcm16.wav").is_file():
        return False
    return all(
        (target / "full" / "outputs" / f"{stem}.wav").stat().st_size == expected_bytes
        for stem in STEMS
    )


def validate_pulled_run(target: Path, source_sha: str) -> dict[str, Any]:
    report_path = target / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise ValueError(f"Pulled report is not complete: {report_path}")
    source = report["source"]
    if source["fileSha256"] != source_sha:
        raise ValueError("Pulled report source SHA does not match the host MP3")
    if source["selectedFrames"] != source["canonicalPcm"]["frames"]:
        raise ValueError("S25 run did not select the complete canonical decode")
    if source["selectedPcmSha256"] != source["canonicalPcm"]["pcmSha256"]:
        raise ValueError("Selected PCM SHA differs from the complete canonical decode")
    export = report["canonicalInputExport"]
    if not export.get("retained") or export.get("pcmSha256") != source["selectedPcmSha256"]:
        raise ValueError("Canonical input export is missing or has the wrong PCM SHA")
    frames = source["selectedFrames"]
    expected_bytes = 44 + frames * 4
    outputs = target / "full" / "outputs"
    output_hashes = {}
    for stem in STEMS:
        path = outputs / f"{stem}.wav"
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"Unexpected output byte size for {path}")
        output_hashes[stem] = sha256_file(path)
    return {
        "reportSha256": sha256_file(report_path),
        "canonicalInputFileSha256": sha256_file(
            target / "canonical-input-44100-stereo-pcm16.wav"
        ),
        "canonicalPcmSha256": source["selectedPcmSha256"],
        "frames": frames,
        "durationSeconds": source["durationSeconds"],
        "windowCount": report["attempts"][0]["expectedWindows"],
        "realtimeFactor": report["attempts"][0]["realtimeFactor"],
        "outputSha256": output_hashes,
    }


def process_track(
    args: argparse.Namespace,
    source: Path,
    output_root: Path,
) -> dict[str, Any]:
    slug = slugify(source.name)
    source_sha = sha256_file(source)
    target = output_root / "tracks" / slug / "s25-cpu"
    if existing_run_is_valid(target, source_sha):
        return {
            "slug": slug,
            "sourceFileName": source.name,
            "sourceSha256": source_sha,
            "status": "skipped-valid-existing",
            "validation": validate_pulled_run(target, source_sha),
        }
    if target.exists():
        raise FileExistsError(f"Invalid existing run must be inspected manually: {target}")

    remote_audio = f"{REMOTE_ROOT}/audio/batch-{slug}.mp3"
    run_id = f"batch-{slug}-full"
    remote_run = (
        f"{REMOTE_ROOT}/htdemucs-canonical-e2e/cpu/full-song/{run_id}"
    )
    adb(args, "shell", "rm", "-rf", remote_audio, remote_run)
    push = adb(args, "push", str(source), remote_audio, timeout=180)
    before = device_snapshot(args)
    instrument_command = [
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
        f"batch-{slug}.mp3",
        "-e",
        "canonicalE2eAudioSha256",
        source_sha,
        "-e",
        "canonicalE2eDurationSeconds",
        "0",
        "-e",
        "canonicalE2eThreads",
        str(args.threads),
        "-e",
        "canonicalE2eRunId",
        run_id,
        "-e",
        "canonicalE2eExportCanonicalInput",
        "true",
        RUNNER,
    ]
    instrument_started = time.perf_counter()
    instrument = run_command(instrument_command, timeout=args.track_timeout, check=False)
    instrument_seconds = time.perf_counter() - instrument_started
    log_root = output_root / "tracks" / slug
    log_root.mkdir(parents=True, exist_ok=True)
    (log_root / "instrumentation.stdout.txt").write_text(
        instrument.stdout, encoding="utf-8"
    )
    (log_root / "instrumentation.stderr.txt").write_text(
        instrument.stderr, encoding="utf-8"
    )
    if instrument.returncode != 0 or "OK (1 test)" not in instrument.stdout:
        raise RuntimeError(
            f"Instrumentation failed for {source.name}; remote evidence retained at {remote_run}"
        )

    pull = adb(args, "pull", remote_run, str(target), timeout=300)
    after = device_snapshot(args)
    validation = validate_pulled_run(target, source_sha)
    envelope = {
        "sourceFileName": source.name,
        "sourceByteSize": source.stat().st_size,
        "sourceSha256": source_sha,
        "remoteAudio": remote_audio,
        "remoteRun": remote_run,
        "instrumentationCommand": instrument_command,
        "instrumentationReturnCode": instrument.returncode,
        "instrumentationWallSeconds": instrument_seconds,
        "pushStdout": push.stdout,
        "pullStdout": pull.stdout,
        "before": before,
        "after": after,
        "validation": validation,
    }
    write_json(log_root / "s25-device-envelope.json", envelope)
    adb(args, "shell", "rm", "-rf", remote_audio, remote_run)
    return {
        "slug": slug,
        "sourceFileName": source.name,
        "sourceSha256": source_sha,
        "status": "complete",
        "validation": validation,
        "deviceEnvelope": str((log_root / "s25-device-envelope.json").resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--track-timeout", type=int, default=900)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    if args.track_timeout <= 0:
        raise ValueError("--track-timeout must be positive")
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    sources = {
        path.name: path
        for path in input_root.iterdir()
        if path.is_file() and path.suffix.lower() == ".mp3"
    }
    missing = [name for name in ORDER if name not in sources]
    extra = sorted(name for name in sources if name not in ORDER)
    if missing or extra:
        raise ValueError(f"Unexpected input inventory: missing={missing}, extra={extra}")
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "s25-device-identity.json", device_identity(args))

    progress_path = output_root / "s25-batch-progress.json"
    progress: dict[str, Any] = {
        "schemaVersion": 1,
        "status": "running",
        "serial": args.serial,
        "threads": args.threads,
        "trackOrder": list(ORDER),
        "tracks": [],
    }
    write_json(progress_path, progress)
    try:
        for name in ORDER:
            item = process_track(args, sources[name], output_root)
            progress["tracks"].append(item)
            write_json(progress_path, progress)
    except BaseException as exc:
        progress["status"] = "failed"
        progress["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        write_json(progress_path, progress)
        raise
    progress["status"] = "complete"
    write_json(progress_path, progress)
    print(json.dumps({
        "status": "complete",
        "progress": str(progress_path),
        "trackCount": len(progress["tracks"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
