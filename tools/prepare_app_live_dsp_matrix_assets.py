#!/usr/bin/env python3
"""Prepare private relay configuration for the App Live MDX DSP matrix APK."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = 1
CONTRACT_VERSION = 1
SHAPES = (
    {"id": "uvr_mdxnet_3_9662", "nFft": 6144, "hopLength": 1024, "dimF": 2048, "dimTPower": 8},
    {"id": "kim_inst", "nFft": 7680, "hopLength": 1024, "dimF": 3072, "dimTPower": 8},
    {
        "id": "uvr_mdxnet_inst_hq_4",
        "nFft": 5120,
        "hopLength": 1024,
        "dimF": 2560,
        "dimTPower": 8,
    },
)


def parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator:
            raise ValueError(f"Invalid relay environment entry: {raw_line}")
        values[key] = value
    return values


def git_output(repository: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repository, text=True, encoding="utf-8"
    ).strip()


def parse_args(repository: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--relay-env",
        type=Path,
        default=repository.parent / "BSSUploadRelay/relay-client.env",
    )
    parser.add_argument("--campaign", default="app-live-mdx-dsp-matrix-v1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository / "app/src/dspMatrix/assets/app-live-dsp",
    )
    parser.add_argument("--worker-count", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--measured-runs", type=int, default=10)
    parser.add_argument("--no-auto-start", action="store_true")
    return parser.parse_args()


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    args = parse_args(repository)
    if not 1 <= args.worker_count <= 8:
        raise ValueError("worker count must be between 1 and 8")
    if not 1 <= args.warmups <= 5:
        raise ValueError("warmups must be between 1 and 5")
    if not 2 <= args.measured_runs <= 50 or args.measured_runs % 2:
        raise ValueError("measured runs must be an even number from 2 through 50")
    safe = lambda value: len(value) <= 96 and all(
        character.isalnum() or character in "._-" for character in value
    )
    if not safe(args.campaign):
        raise ValueError("campaign must use at most 96 safe ASCII characters")
    relay_path = args.relay_env.expanduser().resolve()
    if not relay_path.is_file():
        raise FileNotFoundError(f"relay environment not found: {relay_path}")
    relay = parse_env(relay_path)
    base_url = relay.get("BSS_RELAY_BASE_URL", "").rstrip("/")
    upload_token = relay.get("BSS_RELAY_UPLOAD_TOKEN", "")
    if not base_url.startswith("https://"):
        raise ValueError("BSS_RELAY_BASE_URL must use HTTPS")
    if len(upload_token) < 32:
        raise ValueError("BSS_RELAY_UPLOAD_TOKEN is missing or too short")

    commit = git_output(repository, "rev-parse", "HEAD")
    dirty = bool(git_output(repository, "status", "--short"))
    matrix = {
        "profiles": ["kotlin-jtransforms", "native-full", "native-packed"],
        "workerCount": args.worker_count,
        "warmups": args.warmups,
        "measuredRuns": args.measured_runs,
        "order": "balanced-cross-over",
        "minimumSnrDb": 80.0,
        "maximumAbsoluteError": 0.001,
        "fixture": "deterministic-stereo-multisine-v1",
        "shapes": SHAPES,
    }
    bundle_material = json.dumps(
        {"contractVersion": CONTRACT_VERSION, "sourceCommit": commit, "matrix": matrix},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    bundle_id = hashlib.sha256(bundle_material).hexdigest()
    manifest = {
        "schemaVersion": SCHEMA_VERSION,
        "contractVersion": CONTRACT_VERSION,
        "bundleId": bundle_id,
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": {"commit": commit, "dirty": dirty},
        "matrix": matrix,
        "uploadPolicy": {
            "deferredUntilMeasurementsComplete": True,
            "files": [
                "artifact-manifest.json",
                "identity.json",
                "dsp-matrix-report.json",
                "dsp-matrix-logcat.txt",
                "app.log",
                "complete.json",
            ],
        },
    }
    config = {
        "schemaVersion": SCHEMA_VERSION,
        "baseUrl": base_url,
        "campaign": args.campaign,
        "uploadToken": upload_token,
        "autoStart": not args.no_auto_start,
    }

    output_dir = args.output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".app-live-dsp-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "app-live-dsp"
        staging.mkdir()
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        (staging / "config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.move(str(staging), output_dir)

    print(f"Prepared App Live DSP bundle {bundle_id[:12]}")
    print(f"Source: {commit} (dirty={str(dirty).lower()})")
    print(f"Campaign: {args.campaign}")
    print(f"Assets: {output_dir}")
    print("Relay upload credential embedded: yes (write-only; value not printed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
