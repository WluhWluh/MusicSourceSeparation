#!/usr/bin/env python3
"""Run the audited guitar-ft Torch model over the existing eight-track corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


STEMS = ("drums", "bass", "other", "vocals", "guitar", "piano")
TRACKS = (
    "john-lennon-imagine",
    "athletics-ii",
    "kygo-ed-sheeran-i-see-fire-kygo-remix",
    "sleeping-at-last-north",
    "sleeping-at-last-already-gone",
    "josiah-james-chasing-the-wind",
    "joel-hanson-traveling-light",
    "nylon-eventide",
)
WEIGHTS = (
    "models/demucs/candidates/htdemucs-6s-guitar-ft/"
    "163ec83135ee06e6f10cb8cd94d2ecef8f3f34ad/"
    "htdemucs_6s_guitar_ft_fp32.safetensors"
)
MANIFEST = (
    "models/demucs/candidates/htdemucs-6s-guitar-ft/"
    "163ec83135ee06e6f10cb8cd94d2ecef8f3f34ad/candidate-manifest.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


def valid_output(output: Path, base_report: dict[str, Any]) -> bool:
    report_path = output / "report.json"
    if not report_path.is_file():
        return False
    try:
        report = load_json(report_path)
    except (OSError, json.JSONDecodeError):
        return False
    frames = base_report["source"]["selectedFrames"]
    expected_bytes = 44 + frames * 4
    return bool(
        report.get("status") == "complete"
        and report.get("variant", {}).get("candidateId")
        == "htdemucs_6s_guitar_ft_host_fp32_v1_0_0"
        and report.get("source", {}).get("selectedPcmSha256")
        == base_report["source"]["selectedPcmSha256"]
        and report.get("source", {}).get("selectedFrames") == frames
        and all(
            (output / f"{stem}.wav").is_file()
            and (output / f"{stem}.wav").stat().st_size == expected_bytes
            for stem in STEMS
        )
    )


def wsl_path(repo: Path) -> str:
    result = subprocess.run(
        ["wsl", "-d", "Ubuntu-LiteRT", "--", "pwd"],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo,
    )
    return result.stdout.strip()


def run_track(args: argparse.Namespace, repo: Path, slug: str) -> dict[str, Any]:
    source_track = args.source_root / "tracks" / slug
    base_report_path = source_track / "original-safetensors-torch" / "report.json"
    canonical = source_track / "s25-cpu" / "canonical-input-44100-stereo-pcm16.wav"
    if not base_report_path.is_file() or not canonical.is_file():
        raise FileNotFoundError(f"Missing canonical input or official report for {slug}")
    base_report = load_json(base_report_path)
    if base_report.get("status") != "complete":
        raise ValueError(f"Official report is incomplete for {slug}")

    track_root = args.output_root / "tracks" / slug
    output = track_root / "guitar-ft-torch"
    if valid_output(output, base_report):
        report = load_json(output / "report.json")
        return {
            "slug": slug,
            "status": "skipped-valid-existing",
            "selectedPcmSha256": report["source"]["selectedPcmSha256"],
            "durationSeconds": report["source"]["durationSeconds"],
            "windowCount": report["contract"]["windowCount"],
            "separationWallMs": report["separationWallMs"],
            "separationRealtimeFactor": report["separationRealtimeFactor"],
            "totalWallMs": report["totalWallMs"],
            "peakVmHwmKiB": max(
                item.get("process", {}).get("VmHWMKiB", 0)
                for item in report["windows"]
            ),
        }
    if output.exists() or output.with_name(output.name + ".partial").exists():
        raise FileExistsError(f"Invalid existing output requires inspection: {output}")

    track_root.mkdir(parents=True, exist_ok=True)
    repo_linux = wsl_path(repo)
    canonical_rel = canonical.resolve().relative_to(repo).as_posix()
    output_rel = output.resolve().relative_to(repo).as_posix()
    command = (
        "source /root/.venvs/mss-demucs-litert/bin/activate && "
        f"cd {repo_linux} && "
        "python tools/run_htdemucs_guitar_ft_torch.py "
        f"--input {canonical_rel} --output-dir {output_rel} "
        f"--weights {WEIGHTS} --manifest {MANIFEST} "
        f"--threads {args.threads} --skip-fixture-validation"
    )
    started = time.perf_counter()
    completed = subprocess.run(
        ["wsl", "-d", "Ubuntu-LiteRT", "--", "bash", "-lc", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=args.track_timeout,
    )
    (track_root / "guitar-ft-torch.stdout.txt").write_text(
        completed.stdout, encoding="utf-8"
    )
    (track_root / "guitar-ft-torch.stderr.txt").write_text(
        completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"guitar-ft runner failed for {slug}: {completed.returncode}\n"
            f"{completed.stderr[-2000:]}"
        )
    if not valid_output(output, base_report):
        raise ValueError(f"Completed guitar-ft output failed validation for {slug}")
    report = load_json(output / "report.json")
    return {
        "slug": slug,
        "status": "complete",
        "hostProcessWallSeconds": time.perf_counter() - started,
        "selectedPcmSha256": report["source"]["selectedPcmSha256"],
        "durationSeconds": report["source"]["durationSeconds"],
        "windowCount": report["contract"]["windowCount"],
        "separationWallMs": report["separationWallMs"],
        "separationRealtimeFactor": report["separationRealtimeFactor"],
        "totalWallMs": report["totalWallMs"],
        "peakVmHwmKiB": max(
            item.get("process", {}).get("VmHWMKiB", 0) for item in report["windows"]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("outputs/htdemucs6-mp3-no-gapless-s25-onnx-original-20260804"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/htdemucs6-guitar-ft-host-20260804"),
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--track-timeout", type=int, default=1800)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    args.source_root = (repo / args.source_root).resolve()
    args.output_root = (repo / args.output_root).resolve()
    progress_path = args.output_root / "batch-progress.json"
    progress: dict[str, Any] = {
        "schemaVersion": 1,
        "status": "running",
        "trackOrder": list(TRACKS),
        "threads": args.threads,
        "tracks": [],
    }
    write_json(progress_path, progress)
    try:
        for slug in TRACKS:
            result = run_track(args, repo, slug)
            progress["tracks"].append(result)
            write_json(progress_path, progress)
            print(json.dumps(result, sort_keys=True), flush=True)
    except Exception as exc:
        progress["status"] = "failed"
        progress["error"] = str(exc)
        write_json(progress_path, progress)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    progress["status"] = "complete"
    progress["aggregateAudioSeconds"] = sum(
        item["durationSeconds"] for item in progress["tracks"]
    )
    progress["aggregateSeparationWallMs"] = sum(
        item.get("separationWallMs", 0.0) for item in progress["tracks"]
    )
    write_json(progress_path, progress)
    print(json.dumps({
        "status": "complete",
        "progress": str(progress_path),
        "trackCount": len(progress["tracks"]),
        "aggregateAudioSeconds": progress["aggregateAudioSeconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
