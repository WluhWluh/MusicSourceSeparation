#!/usr/bin/env python3
"""Create verified FLAC references and per-track randomized A/B/C blind folders."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
from typing import Any, BinaryIO, Callable
import wave


STEMS = ("drums", "bass", "other", "vocals", "guitar", "piano")
BACKENDS = ("s25-cpu", "desktop-onnx", "original-safetensors-torch")
TRACKS = (
    "athletics-ii",
    "joel-hanson-traveling-light",
    "john-lennon-imagine",
    "josiah-james-chasing-the-wind",
    "kygo-ed-sheeran-i-see-fire-kygo-remix",
    "nylon-eventide",
    "sleeping-at-last-already-gone",
    "sleeping-at-last-north",
)
GROUPS = ("A", "B", "C")
FIXED_MTIME = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc).timestamp()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


def update_digests(stream: BinaryIO, block_reader: Callable[[], bytes]) -> dict[str, Any]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    byte_count = 0
    for block in iter(block_reader, b""):
        md5.update(block)
        sha256.update(block)
        byte_count += len(block)
    return {
        "pcmByteCount": byte_count,
        "pcmMd5": md5.hexdigest(),
        "pcmSha256": sha256.hexdigest(),
    }


def wav_pcm_identity(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as reader:
        if (
            reader.getframerate() != 44_100
            or reader.getnchannels() != 2
            or reader.getsampwidth() != 2
            or reader.getcomptype() != "NONE"
        ):
            raise ValueError(f"Unexpected WAV contract: {path}")
        frames = reader.getnframes()
        identity = update_digests(reader, lambda: reader.readframes(65_536))
    if identity["pcmByteCount"] != frames * 4:
        raise ValueError(f"WAV PCM byte count mismatch: {path}")
    return {
        **identity,
        "frameCount": frames,
        "sampleRate": 44_100,
        "channelCount": 2,
        "sampleFormat": "signed-pcm16-le",
    }


def flac_pcm_identity(ffmpeg: str, path: Path, expected_frames: int) -> dict[str, Any]:
    process = subprocess.Popen(
        [
            ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-c:a",
            "pcm_s16le",
            "-f",
            "s16le",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None:
        raise RuntimeError("ffmpeg stdout pipe is unavailable")
    identity = update_digests(process.stdout, lambda: process.stdout.read(1 << 20))
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"FLAC decode failed for {path}: {stderr}")
    if identity["pcmByteCount"] != expected_frames * 4:
        raise ValueError(f"FLAC decoded byte count mismatch: {path}")
    return {
        **identity,
        "frameCount": expected_frames,
        "sampleRate": 44_100,
        "channelCount": 2,
        "sampleFormat": "signed-pcm16-le",
    }


def compress_flac(ffmpeg: str, source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.stem + ".partial.flac")
    if partial.exists():
        partial.unlink()
    result = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-map_metadata",
            "-1",
            "-c:a",
            "flac",
            "-compression_level",
            "8",
            "-sample_fmt",
            "s16",
            "-fflags",
            "+bitexact",
            "-flags:a",
            "+bitexact",
            str(partial),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"FLAC encode failed for {source}: {result.stderr}")
    partial.replace(target)


def source_wav(track_root: Path, backend: str, stem: str) -> Path:
    if backend == "s25-cpu":
        return track_root / backend / "full" / "outputs" / f"{stem}.wav"
    return track_root / backend / f"{stem}.wav"


def load_or_create_mapping(path: Path, track_evidence: dict[str, Any]) -> dict[str, Any]:
    if path.is_file():
        mapping = json.loads(path.read_text(encoding="utf-8"))
        for slug in TRACKS:
            values = set(mapping["tracks"][slug]["groups"].values())
            if values != set(BACKENDS):
                raise ValueError(f"Invalid existing blind mapping for {slug}")
            # Preserve the original blind assignment, but refresh evidence fields
            # when the source display name is corrected after the device run.
            item = mapping["tracks"][slug]
            evidence = track_evidence[slug]
            if item.get("sourceMp3Sha256") != evidence["sourceMp3Sha256"]:
                raise ValueError(f"Source SHA changed for existing mapping: {slug}")
            if item.get("canonicalPcmSha256") != evidence["canonicalPcmSha256"]:
                raise ValueError(f"Canonical PCM SHA changed for existing mapping: {slug}")
            item["sourceFileName"] = evidence["sourceFileName"]
        write_json(path, mapping)
        return mapping
    random_source = secrets.SystemRandom()
    tracks = {}
    for slug in TRACKS:
        assignment = random_source.sample(list(BACKENDS), k=len(BACKENDS))
        tracks[slug] = {
            "sourceFileName": track_evidence[slug]["sourceFileName"],
            "sourceMp3Sha256": track_evidence[slug]["sourceMp3Sha256"],
            "canonicalPcmSha256": track_evidence[slug]["canonicalPcmSha256"],
            "groups": dict(zip(GROUPS, assignment, strict=True)),
        }
    mapping = {
        "schemaVersion": 1,
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
        "randomAssignment": "Python secrets.SystemRandom.sample; independent permutation per track",
        "groups": list(GROUPS),
        "backends": list(BACKENDS),
        "tracks": tracks,
    }
    write_json(path, mapping)
    return mapping


def write_mapping_markdown(path: Path, mapping: dict[str, Any], json_path: Path) -> None:
    lines = [
        "# HTDemucs six-stem A/B/C blind mapping",
        "",
        f"Generated UTC: `{mapping['generatedUtc']}`",
        "",
        "This document is the only source-to-group key. Keep it outside the blind audio folder.",
        "",
        f"Machine-readable mapping: `{json_path}`",
        "",
        "| Track | A | B | C | Source MP3 SHA-256 | Canonical PCM SHA-256 |",
        "|---|---|---|---|---|---|",
    ]
    for slug in TRACKS:
        item = mapping["tracks"][slug]
        groups = item["groups"]
        lines.append(
            f"| {item['sourceFileName']} | {groups['A']} | {groups['B']} | {groups['C']} | "
            f"`{item['sourceMp3Sha256']}` | `{item['canonicalPcmSha256']}` |"
        )
    lines.extend(
        [
            "",
            "Backends:",
            "",
            "- `s25-cpu`: LiteRT 2.1.5 CPU, canonical project-owned neural core.",
            "- `desktop-onnx`: pinned FP16-weight waveform ONNX under ORT CPU.",
            "- `original-safetensors-torch`: official safetensors loaded in Torch FP32.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text("\n".join(lines), encoding="utf-8")
    partial.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mapping-json", type=Path, required=True)
    parser.add_argument("--mapping-markdown", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser.parse_args()


def load_source_name_overrides(root: Path) -> dict[str, str]:
    """Map source MP3 SHA-256 values to the original host filenames.

    The device runner intentionally uses collision-free transport names such as
    ``batch-athletics-ii.mp3``.  The batch progress sidecar retains the actual
    input names and SHA values, so reports can display the user-facing names
    without changing the device evidence.
    """
    progress_path = root / "s25-batch-progress.json"
    if not progress_path.is_file():
        return {}
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read source-name sidecar: {progress_path}") from exc
    overrides: dict[str, str] = {}
    for item in progress.get("tracks", []):
        source_sha = item.get("sourceSha256")
        source_name = item.get("sourceFileName")
        if not isinstance(source_sha, str) or not isinstance(source_name, str):
            continue
        previous = overrides.setdefault(source_sha, source_name)
        if previous != source_name:
            raise ValueError(f"Conflicting source names for SHA {source_sha}")
    return overrides


def main() -> int:
    args = parse_args()
    root = args.output_root.resolve()
    source_name_overrides = load_source_name_overrides(root)
    references = root / "reference-audio"
    manifest: dict[str, Any] = {
        "schemaVersion": 1,
        "status": "running",
        "encoding": "FLAC, signed PCM16, 44.1 kHz stereo, lossless PCM identity verified",
        "tracks": {},
    }
    track_evidence = {}
    for slug in TRACKS:
        track_root = root / "tracks" / slug
        s25_report = json.loads(
            (track_root / "s25-cpu" / "report.json").read_text(encoding="utf-8")
        )
        if s25_report.get("status") != "complete":
            raise ValueError(f"S25 report is not complete for {slug}")
        if not (track_root / "comparison.json").is_file():
            raise FileNotFoundError(track_root / "comparison.json")
        source = s25_report["source"]
        track_evidence[slug] = {
            "sourceFileName": source_name_overrides.get(
                source["fileSha256"], source["fileName"]
            ),
            "sourceMp3Sha256": source["fileSha256"],
            "canonicalPcmSha256": source["selectedPcmSha256"],
        }
        manifest["tracks"][slug] = {
            **track_evidence[slug],
            "frameCount": source["selectedFrames"],
            "durationSeconds": source["durationSeconds"],
            "backends": {},
        }
        for backend in BACKENDS:
            backend_files = {}
            for stem in STEMS:
                wav_path = source_wav(track_root, backend, stem)
                flac_path = references / slug / backend / f"{stem}.flac"
                wav_identity = wav_pcm_identity(wav_path)
                if not flac_path.is_file():
                    compress_flac(args.ffmpeg, wav_path, flac_path)
                flac_identity = flac_pcm_identity(
                    args.ffmpeg, flac_path, wav_identity["frameCount"]
                )
                if flac_identity["pcmSha256"] != wav_identity["pcmSha256"]:
                    raise ValueError(f"FLAC PCM identity mismatch: {flac_path}")
                os.utime(flac_path, (FIXED_MTIME, FIXED_MTIME))
                backend_files[stem] = {
                    "sourceWav": str(wav_path.resolve()),
                    "sourceWavSha256": sha256_file(wav_path),
                    "flac": str(flac_path.resolve()),
                    "flacByteSize": flac_path.stat().st_size,
                    "flacSha256": sha256_file(flac_path),
                    "pcmIdentity": flac_identity,
                }
            manifest["tracks"][slug]["backends"][backend] = backend_files
        write_json(root / "reference-audio-manifest.json", manifest)

    mapping_path = args.mapping_json.resolve()
    mapping = load_or_create_mapping(mapping_path, track_evidence)
    blind_root = root / "blind"
    for slug in TRACKS:
        for group, backend in mapping["tracks"][slug]["groups"].items():
            group_root = blind_root / slug / group
            group_root.mkdir(parents=True, exist_ok=True)
            for stem in STEMS:
                source = references / slug / backend / f"{stem}.flac"
                target = group_root / f"{stem}.flac"
                if target.exists():
                    if not os.path.samefile(source, target):
                        raise FileExistsError(f"Blind target is not the expected hard link: {target}")
                else:
                    os.link(source, target)
                os.utime(target, (FIXED_MTIME, FIXED_MTIME))

    manifest["status"] = "complete"
    manifest["blindRoot"] = str(blind_root.resolve())
    manifest["mappingDocumentOutsideBlindRoot"] = str(args.mapping_markdown.resolve())
    write_json(root / "reference-audio-manifest.json", manifest)
    write_mapping_markdown(args.mapping_markdown.resolve(), mapping, mapping_path)
    print(json.dumps({
        "status": "complete",
        "blindRoot": str(blind_root.resolve()),
        "mappingJson": str(mapping_path),
        "mappingMarkdown": str(args.mapping_markdown.resolve()),
        "trackCount": len(TRACKS),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
