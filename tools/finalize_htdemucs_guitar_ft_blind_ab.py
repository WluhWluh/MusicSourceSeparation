#!/usr/bin/env python3
"""Create lossless per-track A/B blind folders for official versus guitar-ft."""

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
    "john-lennon-imagine",
    "athletics-ii",
    "kygo-ed-sheeran-i-see-fire-kygo-remix",
    "sleeping-at-last-north",
    "sleeping-at-last-already-gone",
    "josiah-james-chasing-the-wind",
    "joel-hanson-traveling-light",
    "nylon-eventide",
)
STEMS = ("drums", "bass", "other", "vocals", "guitar", "piano")
BACKENDS = ("official-safetensors-torch", "guitar-ft-torch")
GROUPS = ("A", "B")


def source_wav(source_root: Path, ft_root: Path, slug: str, backend: str, stem: str) -> Path:
    if backend == "official-safetensors-torch":
        return source_root / "tracks" / slug / "original-safetensors-torch" / f"{stem}.wav"
    return ft_root / "tracks" / slug / "guitar-ft-torch" / f"{stem}.wav"


def load_or_create_mapping(path: Path, evidence: dict[str, Any]) -> dict[str, Any]:
    if path.is_file():
        mapping = json.loads(path.read_text(encoding="utf-8"))
        for slug in TRACKS:
            if set(mapping["tracks"][slug]["groups"].values()) != set(BACKENDS):
                raise ValueError(f"Invalid existing A/B mapping for {slug}")
            if mapping["tracks"][slug]["canonicalPcmSha256"] != evidence[slug]["canonicalPcmSha256"]:
                raise ValueError(f"Canonical PCM identity changed for {slug}")
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
        "randomAssignment": "Python secrets.SystemRandom.sample; independent per track",
        "groups": list(GROUPS),
        "backends": list(BACKENDS),
        "tracks": tracks,
    }
    write_json(path, mapping)
    return mapping


def write_mapping_markdown(path: Path, mapping: dict[str, Any], json_path: Path) -> None:
    lines = [
        "# HTDemucs-6s guitar-ft blind A/B key",
        "",
        f"Generated UTC: `{mapping['generatedUtc']}`",
        "",
        "Keep this key outside the blind audio directory.",
        "",
        f"Machine-readable mapping: `{json_path}`",
        "",
        "| Track | A | B | Canonical PCM SHA-256 |",
        "|---|---|---|---|",
    ]
    for slug in TRACKS:
        item = mapping["tracks"][slug]
        lines.append(
            f"| {slug} | {item['groups']['A']} | {item['groups']['B']} | "
            f"`{item['canonicalPcmSha256']}` |"
        )
    lines.extend([
        "",
        "Backends:",
        "",
        "- `official-safetensors-torch`: official 5c90dfd2 HTDemucs-6s control.",
        "- `guitar-ft-torch`: audited third-party full-model guitar fine-tune.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text("\n".join(lines), encoding="utf-8")
    partial.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--guitar-ft-root", type=Path, required=True)
    parser.add_argument("--mapping-json", type=Path, required=True)
    parser.add_argument("--mapping-markdown", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    ft_root = args.guitar_ft_root.resolve()
    references = ft_root / "reference-audio"
    evidence: dict[str, Any] = {}
    manifest: dict[str, Any] = {
        "schemaVersion": 1,
        "status": "running",
        "encoding": "FLAC lossless from signed PCM16 44.1 kHz stereo WAV",
        "tracks": {},
    }
    for slug in TRACKS:
        official_report = json.loads(
            (source_root / "tracks" / slug / "original-safetensors-torch" / "report.json").read_text(
                encoding="utf-8"
            )
        )
        ft_report = json.loads(
            (ft_root / "tracks" / slug / "guitar-ft-torch" / "report.json").read_text(
                encoding="utf-8"
            )
        )
        if official_report.get("status") != "complete" or ft_report.get("status") != "complete":
            raise ValueError(f"Incomplete source report for {slug}")
        pcm_identities = {
            official_report["source"]["selectedPcmSha256"],
            ft_report["source"]["selectedPcmSha256"],
        }
        if len(pcm_identities) != 1:
            raise ValueError(f"Canonical PCM mismatch for {slug}")
        frames = official_report["source"]["selectedFrames"]
        evidence[slug] = {
            "canonicalPcmSha256": next(iter(pcm_identities)),
            "frameCount": frames,
            "durationSeconds": official_report["source"]["durationSeconds"],
        }
        manifest["tracks"][slug] = {**evidence[slug], "backends": {}}
        for backend in BACKENDS:
            files = {}
            for stem in STEMS:
                wav_path = source_wav(source_root, ft_root, slug, backend, stem)
                flac_path = references / slug / backend / f"{stem}.flac"
                wav_identity = wav_pcm_identity(wav_path)
                if wav_identity["frameCount"] != frames:
                    raise ValueError(f"Unexpected frame count for {wav_path}")
                if not flac_path.is_file():
                    compress_flac(args.ffmpeg, wav_path, flac_path)
                flac_identity = flac_pcm_identity(args.ffmpeg, flac_path, frames)
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
        write_json(ft_root / "reference-audio-manifest.json", manifest)

    mapping_path = args.mapping_json.resolve()
    mapping = load_or_create_mapping(mapping_path, evidence)
    blind_root = ft_root / "blind"
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
    write_json(ft_root / "reference-audio-manifest.json", manifest)
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
