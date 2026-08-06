#!/usr/bin/env python3
"""Create a lossless blind A/B for guitar-ft Torch versus S25 LiteRT CPU."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
from typing import Any

from finalize_htdemucs_blind_abc import (
    FIXED_MTIME,
    compress_flac,
    flac_pcm_identity,
    sha256_file,
    wav_pcm_identity,
    write_json,
)


TRACKS = (
    "athletics-ii",
    "john-lennon-imagine",
    "josiah-james-chasing-the-wind",
)
STEMS = ("drums", "bass", "other", "vocals", "guitar", "piano")
BACKENDS = ("guitar-ft-torch", "s25-litert-cpu")
GROUPS = ("A", "B")


def source_wav(root: Path, slug: str, backend: str, stem: str) -> Path:
    if backend == "guitar-ft-torch":
        return root / "tracks" / slug / "guitar-ft-torch-30s" / f"{stem}.wav"
    return root / "tracks" / slug / "s25-cpu-30s" / "full" / "outputs" / f"{stem}.wav"


def report_path(root: Path, slug: str, backend: str) -> Path:
    if backend == "guitar-ft-torch":
        return root / "tracks" / slug / "guitar-ft-torch-30s" / "report.json"
    return root / "tracks" / slug / "s25-cpu-30s" / "report.json"


def load_or_create_mapping(
    path: Path,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    if path.is_file():
        mapping = json.loads(path.read_text(encoding="utf-8"))
        for slug in TRACKS:
            if set(mapping["tracks"][slug]["groups"].values()) != set(BACKENDS):
                raise ValueError(f"Invalid existing A/B mapping for {slug}")
            if (
                mapping["tracks"][slug]["selectedPcmSha256"]
                != evidence[slug]["selectedPcmSha256"]
            ):
                raise ValueError(f"Selected PCM identity changed for {slug}")
        return mapping
    random_source = secrets.SystemRandom()
    tracks = {}
    for slug in TRACKS:
        assignment = random_source.sample(list(BACKENDS), k=2)
        tracks[slug] = {
            **evidence[slug],
            "groups": dict(zip(GROUPS, assignment, strict=True)),
        }
    mapping = {
        "schemaVersion": 1,
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
        "scope": "actual-s25-litert-cpu-vs-same-weight-guitar-ft-torch",
        "randomAssignment": "Python secrets.SystemRandom.sample; independent per track",
        "groups": list(GROUPS),
        "backends": list(BACKENDS),
        "tracks": tracks,
    }
    write_json(path, mapping)
    return mapping


def write_mapping_markdown(
    path: Path,
    mapping: dict[str, Any],
    json_path: Path,
) -> None:
    lines = [
        "# Guitar-ft LiteRT S25 blind A/B key",
        "",
        f"Generated UTC: `{mapping['generatedUtc']}`",
        "",
        "Keep this key outside the blind audio directory.",
        "",
        f"Machine-readable mapping: `{json_path}`",
        "",
        "| Track | A | B | Selected 30-second PCM SHA-256 |",
        "|---|---|---|---|",
    ]
    for slug in TRACKS:
        item = mapping["tracks"][slug]
        lines.append(
            f"| {slug} | {item['groups']['A']} | {item['groups']['B']} | "
            f"`{item['selectedPcmSha256']}` |"
        )
    lines.extend([
        "",
        "Backends:",
        "",
        "- `guitar-ft-torch`: audited guitar-ft safetensors, Torch FP32 host reference.",
        "- `s25-litert-cpu`: the same guitar-ft weights converted to the diagnostic 7.8-second LiteRT artifact and run on S25 CPU.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text("\n".join(lines), encoding="utf-8")
    partial.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mapping-json", type=Path, required=True)
    parser.add_argument("--mapping-markdown", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    references = root / "reference-audio"
    evidence: dict[str, Any] = {}
    manifest: dict[str, Any] = {
        "schemaVersion": 1,
        "status": "running",
        "scope": "diagnostic-research-only-not-product-admitted",
        "encoding": "FLAC lossless from signed PCM16 44.1 kHz stereo WAV",
        "tracks": {},
    }
    for slug in TRACKS:
        reports = {
            backend: json.loads(report_path(root, slug, backend).read_text(encoding="utf-8"))
            for backend in BACKENDS
        }
        identities = {
            report["source"]["selectedPcmSha256"] for report in reports.values()
        }
        frames = {report["source"]["selectedFrames"] for report in reports.values()}
        if len(identities) != 1 or frames != {1_323_000}:
            raise ValueError(f"The A/B sources do not share one 30-second PCM for {slug}")
        s25_model = reports["s25-litert-cpu"]["model"]
        if (
            s25_model.get("variant") != "guitar-ft"
            or s25_model.get("diagnosticOnly") is not True
            or s25_model.get("researchOnly") is not True
        ):
            raise ValueError(f"S25 report is not the diagnostic guitar-ft variant: {slug}")
        evidence[slug] = {
            "selectedPcmSha256": next(iter(identities)),
            "frameCount": 1_323_000,
            "durationSeconds": 30.0,
        }
        manifest["tracks"][slug] = {**evidence[slug], "backends": {}}
        for backend in BACKENDS:
            files = {}
            for stem in STEMS:
                wav_path = source_wav(root, slug, backend, stem)
                flac_path = references / slug / backend / f"{stem}.flac"
                wav_identity = wav_pcm_identity(wav_path)
                if wav_identity["frameCount"] != 1_323_000:
                    raise ValueError(f"Unexpected frame count for {wav_path}")
                if not flac_path.is_file():
                    compress_flac(args.ffmpeg, wav_path, flac_path)
                flac_identity = flac_pcm_identity(args.ffmpeg, flac_path, 1_323_000)
                if flac_identity["pcmSha256"] != wav_identity["pcmSha256"]:
                    raise ValueError(f"FLAC PCM identity mismatch: {flac_path}")
                os.utime(flac_path, (FIXED_MTIME, FIXED_MTIME))
                files[stem] = {
                    "sourceWav": str(wav_path.resolve()),
                    "sourceWavSha256": sha256_file(wav_path),
                    "flac": str(flac_path.resolve()),
                    "flacByteSize": flac_path.stat().st_size,
                    "flacSha256": sha256_file(flac_path),
                    "pcmIdentity": flac_identity,
                }
            manifest["tracks"][slug]["backends"][backend] = files
        write_json(root / "blind-reference-manifest.json", manifest)

    mapping_path = args.mapping_json.resolve()
    mapping = load_or_create_mapping(mapping_path, evidence)
    blind_root = root / "blind-litert-vs-torch"
    for slug in TRACKS:
        for group, backend in mapping["tracks"][slug]["groups"].items():
            group_root = blind_root / slug / group
            group_root.mkdir(parents=True, exist_ok=True)
            for stem in STEMS:
                source = references / slug / backend / f"{stem}.flac"
                target = group_root / f"{stem}.flac"
                if target.exists():
                    if not os.path.samefile(source, target):
                        raise FileExistsError(f"Unexpected existing blind file: {target}")
                else:
                    os.link(source, target)
                os.utime(target, (FIXED_MTIME, FIXED_MTIME))

    manifest["status"] = "complete"
    manifest["blindRoot"] = str(blind_root.resolve())
    manifest["mappingOutsideBlindRoot"] = str(args.mapping_markdown.resolve())
    write_json(root / "blind-reference-manifest.json", manifest)
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
