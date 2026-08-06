#!/usr/bin/env python3
"""Generate same-PCM official and guitar-ft Torch references for S25 diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


TRACKS = (
    "athletics-ii",
    "john-lennon-imagine",
    "josiah-james-chasing-the-wind",
)
FRAME_COUNT = 1_323_000
GUITAR_WEIGHTS = (
    "models/demucs/candidates/htdemucs-6s-guitar-ft/"
    "163ec83135ee06e6f10cb8cd94d2ecef8f3f34ad/"
    "htdemucs_6s_guitar_ft_fp32.safetensors"
)
GUITAR_MANIFEST = (
    "models/demucs/candidates/htdemucs-6s-guitar-ft/"
    "163ec83135ee06e6f10cb8cd94d2ecef8f3f34ad/candidate-manifest.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def valid_reference(output: Path, selected_pcm_sha: str, variant: str) -> bool:
    report_path = output / "report.json"
    if not report_path.is_file():
        return False
    try:
        report = load_json(report_path)
    except (OSError, json.JSONDecodeError):
        return False
    if (
        report.get("status") != "complete"
        or report.get("source", {}).get("selectedFrames") != FRAME_COUNT
        or report.get("source", {}).get("selectedPcmSha256") != selected_pcm_sha
    ):
        return False
    if variant == "guitar-ft" and (
        report.get("variant", {}).get("candidateId")
        != "htdemucs_6s_guitar_ft_host_fp32_v1_0_0"
    ):
        return False
    return all(
        (output / f"{stem}.wav").is_file()
        and (output / f"{stem}.wav").stat().st_size == 44 + FRAME_COUNT * 4
        for stem in ("drums", "bass", "other", "vocals", "guitar", "piano")
    )


def run_reference(
    args: argparse.Namespace,
    repo: Path,
    slug: str,
    variant: str,
    selected_pcm_sha: str,
) -> dict[str, Any]:
    track_root = args.output_root / "tracks" / slug
    output = track_root / f"{variant}-torch-30s"
    if valid_reference(output, selected_pcm_sha, variant):
        report = load_json(output / "report.json")
        return {
            "slug": slug,
            "variant": variant,
            "status": "skipped-valid-existing",
            "selectedPcmSha256": selected_pcm_sha,
            "realtimeFactor": report["separationRealtimeFactor"],
        }
    if output.exists() or output.with_name(output.name + ".partial").exists():
        raise FileExistsError(f"Invalid reference output requires inspection: {output}")
    canonical = (
        args.canonical_root / "tracks" / slug / "s25-cpu" /
        "canonical-input-44100-stereo-pcm16.wav"
    )
    if not canonical.is_file():
        raise FileNotFoundError(canonical)
    command = [
        "python",
        (
            "tools/run_htdemucs_guitar_ft_torch.py"
            if variant == "guitar-ft"
            else "tools/run_htdemucs_canonical_torch.py"
        ),
        "--input",
        canonical.resolve().relative_to(repo).as_posix(),
        "--output-dir",
        output.resolve().relative_to(repo).as_posix(),
        "--duration-seconds",
        "30",
        "--threads",
        str(args.threads),
        "--skip-fixture-validation",
    ]
    if variant == "guitar-ft":
        command.extend(["--weights", GUITAR_WEIGHTS, "--manifest", GUITAR_MANIFEST])
    shell = (
        "source /root/.venvs/mss-demucs-litert/bin/activate && "
        f"cd {args.repo_linux} && "
        + " ".join(command)
    )
    started = time.perf_counter()
    completed = subprocess.run(
        ["wsl", "-d", "Ubuntu-LiteRT", "--", "bash", "-lc", shell],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=args.timeout,
    )
    track_root.mkdir(parents=True, exist_ok=True)
    (track_root / f"{variant}-torch-30s.stdout.txt").write_text(
        completed.stdout,
        encoding="utf-8",
    )
    (track_root / f"{variant}-torch-30s.stderr.txt").write_text(
        completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{variant} Torch failed for {slug}: {completed.stderr[-2000:]}"
        )
    if not valid_reference(output, selected_pcm_sha, variant):
        raise ValueError(f"Invalid completed {variant} reference for {slug}")
    report = load_json(output / "report.json")
    return {
        "slug": slug,
        "variant": variant,
        "status": "complete",
        "hostWallSeconds": time.perf_counter() - started,
        "selectedPcmSha256": selected_pcm_sha,
        "realtimeFactor": report["separationRealtimeFactor"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=Path("outputs/htdemucs6-mp3-no-gapless-s25-onnx-original-20260804"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/htdemucs6-guitar-ft-s25-20260805"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    args.canonical_root = (repo / args.canonical_root).resolve()
    args.output_root = (repo / args.output_root).resolve()
    args.repo_linux = subprocess.run(
        ["wsl", "-d", "Ubuntu-LiteRT", "--", "pwd"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    progress: dict[str, Any] = {"status": "running", "references": []}
    progress_path = args.output_root / "host-reference-progress.json"
    write_json(progress_path, progress)
    try:
        for slug in TRACKS:
            s25_report = load_json(
                args.output_root / "tracks" / slug / "s25-cpu-30s" / "report.json"
            )
            selected_pcm_sha = s25_report["source"]["selectedPcmSha256"]
            for variant in ("official", "guitar-ft"):
                result = run_reference(
                    args,
                    repo,
                    slug,
                    variant,
                    selected_pcm_sha,
                )
                progress["references"].append(result)
                write_json(progress_path, progress)
                print(json.dumps(result, sort_keys=True), flush=True)
    except Exception as error:
        progress["status"] = "failed"
        progress["error"] = str(error)
        write_json(progress_path, progress)
        print(f"error: {error}", file=sys.stderr)
        return 1
    progress["status"] = "complete"
    write_json(progress_path, progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
