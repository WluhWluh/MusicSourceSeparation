#!/usr/bin/env python3
"""Run desktop ONNX and original-safetensors references as S25 tracks become available."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence


STEMS = ("drums", "bass", "other", "vocals", "guitar", "piano")
ORDER = (
    "john-lennon-imagine",
    "athletics-ii",
    "kygo-ed-sheeran-i-see-fire-kygo-remix",
    "sleeping-at-last-north",
    "sleeping-at-last-already-gone",
    "josiah-james-chasing-the-wind",
    "joel-hanson-traveling-light",
    "nylon-eventide",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def s25_ready(track_root: Path) -> bool:
    report_path = track_root / "s25-cpu" / "report.json"
    if not report_path.is_file():
        return False
    try:
        report = load_json(report_path)
        frames = int(report["source"]["selectedFrames"])
    except (KeyError, ValueError, TypeError, json.JSONDecodeError, OSError):
        return False
    if report.get("status") != "complete" or frames <= 0:
        return False
    expected_bytes = 44 + frames * 4
    canonical = track_root / "s25-cpu" / "canonical-input-44100-stereo-pcm16.wav"
    if not canonical.is_file() or canonical.stat().st_size != expected_bytes:
        return False
    output_root = track_root / "s25-cpu" / "full" / "outputs"
    return all(
        (output_root / f"{stem}.wav").is_file()
        and (output_root / f"{stem}.wav").stat().st_size == expected_bytes
        for stem in STEMS
    )


def reference_ready(track_root: Path, backend: str, pcm_sha: str, frames: int) -> bool:
    root = track_root / backend
    report_path = root / "report.json"
    if not report_path.is_file():
        return False
    try:
        report = load_json(report_path)
    except (json.JSONDecodeError, OSError):
        return False
    if report.get("status") != "complete":
        return False
    source = report.get("source", {})
    if source.get("selectedPcmSha256") != pcm_sha or source.get("selectedFrames") != frames:
        return False
    expected_bytes = 44 + frames * 4
    return all(
        (root / f"{stem}.wav").is_file()
        and (root / f"{stem}.wav").stat().st_size == expected_bytes
        for stem in STEMS
    )


def run_logged(
    command: Sequence[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(
        list(command),
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    elapsed = time.perf_counter() - started
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {command}; see {stderr_path}"
        )
    return {
        "command": list(command),
        "returnCode": result.returncode,
        "wallSeconds": elapsed,
        "stdout": str(stdout_path.resolve()),
        "stderr": str(stderr_path.resolve()),
    }


def wait_for_s25(track_root: Path, timeout: int, poll_seconds: int) -> None:
    deadline = time.monotonic() + timeout
    while not s25_ready(track_root):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for S25 output: {track_root}")
        time.sleep(poll_seconds)


def process_track(args: argparse.Namespace, repo: Path, track_root: Path) -> dict[str, Any]:
    wait_for_s25(track_root, args.wait_timeout, args.poll_seconds)
    s25_report_path = track_root / "s25-cpu" / "report.json"
    s25_report = load_json(s25_report_path)
    frames = int(s25_report["source"]["selectedFrames"])
    pcm_sha = s25_report["source"]["selectedPcmSha256"]
    canonical_input = (
        track_root / "s25-cpu" / "canonical-input-44100-stereo-pcm16.wav"
    )
    actions = []

    if not reference_ready(track_root, "desktop-onnx", pcm_sha, frames):
        actions.append(
            run_logged(
                [
                    str(args.onnx_python.resolve()),
                    "tools/run_htdemucs_canonical_onnx.py",
                    "--input",
                    str(canonical_input),
                    "--output-dir",
                    str(track_root / "desktop-onnx"),
                    "--threads",
                    str(args.threads),
                    "--force",
                ],
                cwd=repo,
                stdout_path=track_root / "desktop-onnx.stdout.txt",
                stderr_path=track_root / "desktop-onnx.stderr.txt",
                timeout=args.backend_timeout,
            )
        )
    if not reference_ready(track_root, "desktop-onnx", pcm_sha, frames):
        raise ValueError(f"Desktop ONNX validation failed after completion: {track_root}")

    if not reference_ready(track_root, "original-safetensors-torch", pcm_sha, frames):
        relative_input = canonical_input.relative_to(repo).as_posix()
        relative_output = (track_root / "original-safetensors-torch").relative_to(repo).as_posix()
        shell_command = (
            "source /root/.venvs/mss-demucs-litert/bin/activate && "
            "cd /mnt/c/Users/User/Documents/MusicSourceSeparation && "
            f"python tools/run_htdemucs_canonical_torch.py --input {relative_input} "
            f"--output-dir {relative_output} --threads {args.threads} --force"
        )
        actions.append(
            run_logged(
                [args.wsl, "-e", "bash", "-lc", shell_command],
                cwd=repo,
                stdout_path=track_root / "original-safetensors-torch.stdout.txt",
                stderr_path=track_root / "original-safetensors-torch.stderr.txt",
                timeout=args.backend_timeout,
            )
        )
    if not reference_ready(track_root, "original-safetensors-torch", pcm_sha, frames):
        raise ValueError(f"Original safetensors validation failed after completion: {track_root}")

    comparison_path = track_root / "comparison.json"
    comparison_command = [
        sys.executable,
        "tools/compare_htdemucs_stem_sets.py",
        "--set",
        f"s25-cpu={track_root / 's25-cpu' / 'full' / 'outputs'}",
        "--set",
        f"desktop-onnx={track_root / 'desktop-onnx'}",
        "--set",
        f"original-safetensors-torch={track_root / 'original-safetensors-torch'}",
        "--set-report",
        f"s25-cpu={s25_report_path}",
        "--set-report",
        f"desktop-onnx={track_root / 'desktop-onnx' / 'report.json'}",
        "--set-report",
        (
            "original-safetensors-torch="
            f"{track_root / 'original-safetensors-torch' / 'report.json'}"
        ),
        "--primary-reference",
        "original-safetensors-torch",
        "--output",
        str(comparison_path),
    ]
    actions.append(
        run_logged(
            comparison_command,
            cwd=repo,
            stdout_path=track_root / "comparison.stdout.txt",
            stderr_path=track_root / "comparison.stderr.txt",
            timeout=600,
        )
    )
    comparison = load_json(comparison_path)
    if comparison.get("sharedSelectedPcmSha256") != pcm_sha:
        raise ValueError("Comparison does not preserve the S25 canonical PCM identity")
    return {
        "slug": track_root.name,
        "status": "complete",
        "frames": frames,
        "durationSeconds": s25_report["source"]["durationSeconds"],
        "selectedPcmSha256": pcm_sha,
        "actions": actions,
        "comparison": str(comparison_path.resolve()),
        "aggregatePairs": {
            name: value["aggregate"] for name, value in comparison["pairs"].items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--onnx-python", type=Path, default=Path(".venv/Scripts/python.exe"))
    parser.add_argument("--wsl", default="wsl.exe")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--backend-timeout", type=int, default=1800)
    parser.add_argument("--wait-timeout", type=int, default=3600)
    parser.add_argument("--poll-seconds", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path.cwd().resolve()
    output_root = args.output_root.resolve()
    if not args.onnx_python.is_file():
        raise FileNotFoundError(args.onnx_python)
    progress_path = output_root / "host-reference-progress.json"
    progress: dict[str, Any] = {
        "schemaVersion": 1,
        "status": "running",
        "threads": args.threads,
        "trackOrder": list(ORDER),
        "tracks": [],
    }
    write_json(progress_path, progress)
    try:
        for slug in ORDER:
            item = process_track(args, repo, output_root / "tracks" / slug)
            progress["tracks"].append(item)
            write_json(progress_path, progress)
    except BaseException as exc:
        progress["status"] = "failed"
        progress["failure"] = {"type": type(exc).__name__, "message": str(exc)}
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
