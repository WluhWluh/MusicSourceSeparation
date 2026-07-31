#!/usr/bin/env python3
"""Prepare ignored, self-contained assets for the App Live QNN validation APK."""

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
DIAGNOSTIC_CONTRACT_VERSION = 2
MODEL_SHA256 = "f74eee1ac06845a7cf277416138b19a6203f34316a3a74b2bde19acbfb2f8378"
WINDOW_INPUT_SHA256 = "cfe4f6cd85c2f91e94a259886d96eed3c7dcce604642dcfcb9ea1873fb254201"
WINDOW_REFERENCE_SHA256 = "4d6e23a417428439cf010b54e690e645445c82a4d89255d83824b80dded933f0"
QUICK_AUDIO_SHA256 = "0fa980cace732c47647043d1a5fce3589e54771ca3574b003f152acfa6ceec96"
FULL_AUDIO_SHA256 = "e845e52aeeeb69be702d3a28d756eaf7a3137dbe5338ea50fb0ac3d8c4f9bd89"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checked_file(path: Path, label: str, expected_sha256: str | None = None) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise FileNotFoundError(f"{label} not found: {resolved}")
    actual_sha256 = sha256_file(resolved)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    return resolved


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
    documents = repository.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=documents
        / "BSSModels/bss-tflite/artifacts/all-candidates-fp32/"
        "UVR_MDXNET_3_9662_static_float32.tflite",
    )
    parser.add_argument(
        "--window-input",
        type=Path,
        default=repository
        / "data/samples/multi-model-inputs/"
        "uvr_mdxnet_3_9662_coast_town_window0_nchw_f32.bin",
    )
    parser.add_argument(
        "--window-reference",
        type=Path,
        default=repository
        / ".tmp/android-benchmark-references/"
        "ort-uvr_mdxnet_3_9662-9482-"
        "uvr_mdxnet_3_9662_coast_town_window0_nchw_f32-fp32.bin",
    )
    parser.add_argument(
        "--quick-audio",
        type=Path,
        default=repository / "data/samples/coast_town_vocal_entry_37s_12s.wav",
    )
    parser.add_argument(
        "--full-audio",
        type=Path,
        default=repository / "data/samples/_-_Coast_Town__decoded.wav",
    )
    parser.add_argument(
        "--htp-version",
        type=int,
        choices=(69, 73, 75, 79, 81),
        default=79,
        help="Qualcomm HTP runtime generation packaged by this bundle (default: 79).",
    )
    parser.add_argument(
        "--runtime-manifest",
        type=Path,
        help="Runtime manifest (default: .tmp/litert-qnn-v<version>-runtime/runtime-manifest.json).",
    )
    parser.add_argument(
        "--relay-env",
        type=Path,
        default=documents / "BSSUploadRelay/relay-client.env",
    )
    parser.add_argument(
        "--campaign",
        help="Relay campaign (default: app-live-qnn-v<version>-v1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Asset output (default: app/src/qnnV<version>/assets/app-live).",
    )
    return parser.parse_args()


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    args = parse_args(repository)
    campaign = args.campaign or f"app-live-qnn-v{args.htp_version}-v1"
    if len(campaign) > 96 or not all(
        character.isalnum() or character in "._-" for character in campaign
    ):
        raise ValueError("Campaign must use at most 96 ASCII letters, digits, dots, underscores, or hyphens")
    runtime_manifest = args.runtime_manifest or (
        repository / f".tmp/litert-qnn-v{args.htp_version}-runtime/runtime-manifest.json"
    )
    runtime_manifest = checked_file(runtime_manifest, "QNN runtime manifest")
    runtime_metadata = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    packaged_htp_version = runtime_metadata.get("runtime", {}).get("htpVersion")
    if packaged_htp_version != args.htp_version:
        raise ValueError(
            "QNN runtime manifest HTP mismatch: "
            f"expected v{args.htp_version}, got {packaged_htp_version!r}"
        )

    sources = {
        "model": checked_file(args.model, "9662 TFLite model", MODEL_SHA256),
        "windowInput": checked_file(args.window_input, "Coast Town tensor", WINDOW_INPUT_SHA256),
        "windowReference": checked_file(
            args.window_reference, "Coast Town ORT reference", WINDOW_REFERENCE_SHA256
        ),
        "quickAudio": checked_file(args.quick_audio, "12-second Coast Town WAV", QUICK_AUDIO_SHA256),
        "fullAudio": checked_file(args.full_audio, "full Coast Town WAV", FULL_AUDIO_SHA256),
        "runtimeManifest": runtime_manifest,
    }
    relay_env = checked_file(args.relay_env, "relay environment")
    relay = parse_env(relay_env)
    base_url = relay.get("BSS_RELAY_BASE_URL", "").rstrip("/")
    upload_token = relay.get("BSS_RELAY_UPLOAD_TOKEN", "")
    if not base_url.startswith("https://"):
        raise ValueError("BSS_RELAY_BASE_URL must use HTTPS")
    if len(upload_token) < 32:
        raise ValueError("BSS_RELAY_UPLOAD_TOKEN is missing or too short")

    commit = git_output(repository, "rev-parse", "HEAD")
    dirty = bool(git_output(repository, "status", "--short"))
    output_dir = (
        args.output_dir
        or repository / f"app/src/qnnV{args.htp_version}/assets/app-live"
    ).resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    file_specs = (
        ("model", "model.tflite", "models", "UVR_MDXNET_3_9662_static_float32.tflite"),
        (
            "windowInput",
            "window0-input.bin",
            "inputs",
            "uvr_mdxnet_3_9662_coast_town_window0_nchw_f32.bin",
        ),
        (
            "windowReference",
            "window0-reference.bin",
            "reference",
            "ort-uvr_mdxnet_3_9662-9482-"
            "uvr_mdxnet_3_9662_coast_town_window0_nchw_f32-fp32.bin",
        ),
        ("quickAudio", "quick.wav", "audio-input", "coast_town_vocal_entry_37s_12s.wav"),
        ("fullAudio", "full.wav", "audio-input", "coast_town_full.wav"),
        ("runtimeManifest", "runtime-manifest.json", "evidence", "runtime-manifest.json"),
    )

    with tempfile.TemporaryDirectory(prefix=".app-live-assets-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "app-live"
        staging.mkdir()
        records: list[dict[str, object]] = []
        for key, asset_name, target_dir, target_name in file_specs:
            source = sources[key]
            destination = staging / asset_name
            shutil.copyfile(source, destination)
            records.append(
                {
                    "id": key,
                    "asset": f"app-live/{asset_name}",
                    "targetDirectory": target_dir,
                    "targetName": target_name,
                    "bytes": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                }
            )

        bundle_material = json.dumps(
            {
                "diagnosticContractVersion": DIAGNOSTIC_CONTRACT_VERSION,
                "files": records,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        bundle_id = hashlib.sha256(bundle_material).hexdigest()
        manifest = {
            "schemaVersion": SCHEMA_VERSION,
            "diagnosticContractVersion": DIAGNOSTIC_CONTRACT_VERSION,
            "bundleId": bundle_id,
            "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source": {"commit": commit, "dirty": dirty},
            "model": {
                "id": "uvr_mdxnet_3_9662",
                "contractId": "uvr_mdxnet_3_9662@2",
                "outputStem": "vocals",
                "outputScale": 1.035,
            },
            "runtime": {
                "accelerator": f"Qualcomm HTP v{args.htp_version}",
                "htpVersion": args.htp_version,
            },
            "profiles": {
                "quick": ["qnn-window", "cpu-window", "bounded-gpu-window", "qnn-audio-12s"],
                "full": ["qnn-window", "qnn-audio-full"],
            },
            "files": records,
        }
        config = {
            "schemaVersion": SCHEMA_VERSION,
            "baseUrl": base_url,
            "campaign": campaign,
            "uploadToken": upload_token,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        (staging / "config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.move(str(staging), output_dir)

    total_bytes = sum(int(record["bytes"]) for record in records)
    print(f"Prepared App Live bundle {bundle_id[:12]} with {total_bytes} fixture bytes")
    print(f"Assets: {output_dir}")
    print("Relay upload credential embedded: yes (write-only; value not printed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
