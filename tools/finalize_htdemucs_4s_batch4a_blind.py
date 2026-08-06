#!/usr/bin/env python3
"""Create and verify the Batch 4A five-way four-stem blind listening set."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
from typing import Any

import finalize_htdemucs_blind_abc as blind_helpers
from finalize_htdemucs_blind_abc import (
    compress_flac,
    flac_pcm_identity,
    sha256_file,
    wav_pcm_identity,
)


TRACKS = (
    "athletics-ii",
    "john-lennon-imagine",
    "josiah-james-chasing-the-wind",
    "kygo-ed-sheeran-i-see-fire-kygo-remix",
)
VARIANTS = (
    "official-base",
    "official-ft-bag",
    "base-plus-vocals",
    "base-plus-drums",
    "psytrance-onnx",
)
GROUPS = ("A", "B", "C", "D", "E")
STEMS = ("drums", "bass", "other", "vocals")
EXPECTED_FRAMES = 1_323_000
ALGORITHM = "batch4a-v1-hmac-sha256-sort"
FIXED_MTIME = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc).timestamp()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "byteSize": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def executable_evidence(command: str) -> dict[str, Any]:
    resolved = shutil.which(command)
    if resolved is None:
        raise FileNotFoundError(f"Executable not found: {command}")
    result = subprocess.run(
        [resolved, "-version"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "requestedCommand": command,
        "resolvedPath": str(Path(resolved).resolve()),
        "versionFirstLine": result.stdout.splitlines()[0],
    }


def inside(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    resolved_root = root.resolve()
    return resolved == resolved_root or resolved_root in resolved.parents


def ffprobe_contract(ffprobe: str, path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_name,codec_type,sample_rate,channels,sample_fmt,duration_ts,time_base:stream_disposition=attached_pic:stream_tags:format_tags",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    value = json.loads(result.stdout)
    streams = value.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected one FLAC audio stream: {path}")
    stream = streams[0]
    if (
        stream.get("codec_name") != "flac"
        or stream.get("codec_type") != "audio"
        or int(stream.get("sample_rate", 0)) != 44_100
        or int(stream.get("channels", 0)) != 2
        or stream.get("sample_fmt") != "s16"
        or int(stream.get("duration_ts", -1)) != EXPECTED_FRAMES
        or stream.get("time_base") != "1/44100"
        or int(stream.get("disposition", {}).get("attached_pic", 0)) != 0
    ):
        raise ValueError(f"Unexpected FLAC stream contract for {path}: {stream}")
    if stream.get("tags") or value.get("format", {}).get("tags"):
        raise ValueError(f"Blind FLAC unexpectedly contains metadata tags: {path}")
    return stream


def permutation(seed: bytes, slug: str) -> dict[str, str]:
    def score(variant: str) -> bytes:
        message = f"batch4a-v1\0{slug}\0{variant}".encode("utf-8")
        return hmac.new(seed, message, hashlib.sha256).digest()

    ordered = sorted(VARIANTS, key=score)
    return dict(zip(GROUPS, ordered, strict=True))


def load_variant_inputs(output_root: Path) -> dict[str, Any]:
    tracks: dict[str, Any] = {}
    for slug in TRACKS:
        variants: dict[str, Any] = {}
        shared_pcm_sha = None
        shared_wav_sha = None
        for variant in VARIANTS:
            report_path = output_root / "tracks" / slug / variant / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("status") != "complete" or report.get("variantId") != variant:
                raise ValueError(f"Incomplete or mismatched render report: {report_path}")
            source = report.get("source", {})
            if source.get("selectedFrames") != EXPECTED_FRAMES:
                raise ValueError(f"Unexpected selected frame count: {report_path}")
            pcm_sha = source.get("selectedPcmSha256")
            wav_sha = source.get("canonicalWavFileSha256")
            if not is_sha256(pcm_sha) or not is_sha256(wav_sha):
                raise ValueError(f"Missing or invalid input SHA in {report_path}")
            if variant == VARIANTS[0]:
                shared_pcm_sha = pcm_sha
                shared_wav_sha = wav_sha
            elif pcm_sha != shared_pcm_sha or wav_sha != shared_wav_sha:
                raise ValueError(f"Variant input identity mismatch for {slug}")
            files = {}
            for stem in STEMS:
                path = output_root / "tracks" / slug / variant / f"{stem}.wav"
                wav_identity = wav_pcm_identity(path)
                if wav_identity["frameCount"] != EXPECTED_FRAMES:
                    raise ValueError(f"Unexpected WAV frame count: {path}")
                expected_pcm = report["stems"][stem]["outputWav"]["pcmSha256"]
                if wav_identity["pcmSha256"] != expected_pcm:
                    raise ValueError(f"WAV PCM/report identity mismatch: {path}")
                files[stem] = {
                    "path": str(path.resolve()),
                    "wavFile": identity(path),
                    "wavPcm": wav_identity,
                }
            variants[variant] = {
                "report": identity(report_path),
                "files": files,
            }
        tracks[slug] = {
            "selectedPcmSha256": shared_pcm_sha,
            "canonicalWavFileSha256": shared_wav_sha,
            "selectedFrames": EXPECTED_FRAMES,
            "variants": variants,
        }
    return tracks


def validate_mapping(mapping: dict[str, Any], inputs: dict[str, Any]) -> bytes:
    if mapping.get("algorithm") != ALGORITHM:
        raise ValueError("Unexpected blind mapping algorithm")
    try:
        seed = bytes.fromhex(mapping["seedHex"])
    except (KeyError, ValueError) as exc:
        raise ValueError("Invalid blind mapping seed") from exc
    if len(seed) != 32 or hashlib.sha256(seed).hexdigest() != mapping.get("seedSha256"):
        raise ValueError("Blind mapping seed identity mismatch")
    if set(mapping.get("tracks", {})) != set(TRACKS):
        raise ValueError("Blind mapping track set mismatch")
    for slug in TRACKS:
        item = mapping["tracks"][slug]
        groups = item.get("groups", {})
        if set(groups) != set(GROUPS) or set(groups.values()) != set(VARIANTS):
            raise ValueError(f"Blind mapping is not a five-way permutation: {slug}")
        if groups != permutation(seed, slug):
            raise ValueError(f"Blind mapping does not reproduce from its seed: {slug}")
        if item.get("selectedPcmSha256") != inputs[slug]["selectedPcmSha256"]:
            raise ValueError(f"Blind mapping input PCM changed: {slug}")
        if item.get("canonicalWavFileSha256") != inputs[slug]["canonicalWavFileSha256"]:
            raise ValueError(f"Blind mapping input WAV changed: {slug}")
        if item.get("selectedFrames") != EXPECTED_FRAMES:
            raise ValueError(f"Blind mapping frame count changed: {slug}")
    return seed


def load_or_create_mapping(
    path: Path,
    blind_root: Path,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    if path.is_file():
        mapping = json.loads(path.read_text(encoding="utf-8"))
        validate_mapping(mapping, inputs)
        return mapping
    if blind_root.exists():
        raise FileNotFoundError("Blind directory exists without its private mapping; refusing to re-randomize")
    seed = secrets.token_bytes(32)
    mapping = {
        "schemaVersion": 1,
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
        "scope": "Private five-way Batch 4A blind key; keep outside blind-ae",
        "algorithm": ALGORITHM,
        "seedHex": seed.hex(),
        "seedSha256": hashlib.sha256(seed).hexdigest(),
        "groups": list(GROUPS),
        "variants": list(VARIANTS),
        "tracks": {
            slug: {
                "selectedPcmSha256": inputs[slug]["selectedPcmSha256"],
                "canonicalWavFileSha256": inputs[slug]["canonicalWavFileSha256"],
                "selectedFrames": EXPECTED_FRAMES,
                "groups": permutation(seed, slug),
            }
            for slug in TRACKS
        },
    }
    write_json(path, mapping)
    return mapping


def write_mapping_markdown(path: Path, mapping: dict[str, Any], json_path: Path) -> None:
    lines = [
        "# HTDemucs four-stem Batch 4A blind mapping",
        "",
        f"Generated UTC: `{mapping['generatedUtc']}`",
        "",
        "This is the private A-E key. It is intentionally outside the blind audio directory.",
        "",
        f"Machine-readable key: `{json_path.resolve()}`",
        "",
        "| Track | A | B | C | D | E | Selected PCM SHA-256 |",
        "|---|---|---|---|---|---|---|",
    ]
    for slug in TRACKS:
        item = mapping["tracks"][slug]
        groups = item["groups"]
        lines.append(
            f"| {slug} | {groups['A']} | {groups['B']} | {groups['C']} | "
            f"{groups['D']} | {groups['E']} | `{item['selectedPcmSha256']}` |"
        )
    lines.extend(
        [
            "",
            "Variant semantics:",
            "",
            "- `official-base`: official `955717e8` four-stem model.",
            "- `official-ft-bag`: one documented target stem from each official FT specialist.",
            "- `base-plus-vocals`: base drums/bass/other plus FT vocals.",
            "- `base-plus-drums`: FT drums plus base bass/other/vocals.",
            "- `psytrance-onnx`: research-only third-party ONNX under the common 25% OLA contract.",
            "",
            "The Kygo excerpt is an electronic non-Psytrance control, not in-domain Psytrance evidence.",
            "No per-variant loudness matching or residual redistribution was applied.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text("\n".join(lines), encoding="utf-8")
    partial.replace(path)


def build_references(
    output_root: Path,
    inputs: dict[str, Any],
    ffmpeg: str,
    ffprobe: str,
) -> dict[str, Any]:
    root = output_root / "reference-flac"
    evidence: dict[str, Any] = {}
    for slug in TRACKS:
        evidence[slug] = {}
        for variant in VARIANTS:
            evidence[slug][variant] = {}
            for stem in STEMS:
                source = Path(inputs[slug]["variants"][variant]["files"][stem]["path"])
                target = root / slug / variant / f"{stem}.flac"
                if not target.is_file():
                    compress_flac(ffmpeg, source, target)
                wav_pcm = inputs[slug]["variants"][variant]["files"][stem]["wavPcm"]
                flac_pcm = flac_pcm_identity(ffmpeg, target, EXPECTED_FRAMES)
                if flac_pcm["pcmSha256"] != wav_pcm["pcmSha256"]:
                    raise ValueError(f"Reference FLAC PCM mismatch: {target}")
                stream = ffprobe_contract(ffprobe, target)
                os.utime(target, (FIXED_MTIME, FIXED_MTIME))
                evidence[slug][variant][stem] = {
                    "sourceWav": inputs[slug]["variants"][variant]["files"][stem],
                    "referenceFlac": identity(target),
                    "decodedPcm": flac_pcm,
                    "ffprobe": stream,
                }
    return evidence


def validate_blind_inventory(
    blind_root: Path,
    reference_root: Path,
    mapping: dict[str, Any],
    ffmpeg: str,
    ffprobe: str,
) -> dict[str, Any]:
    expected = {
        Path(slug) / group / f"{stem}.flac"
        for slug in TRACKS
        for group in GROUPS
        for stem in STEMS
    }
    actual = {path.relative_to(blind_root) for path in blind_root.rglob("*") if path.is_file()}
    if actual != expected:
        raise ValueError(f"Blind inventory mismatch: missing={expected - actual}, extra={actual - expected}")
    if any(path.is_symlink() for path in blind_root.rglob("*")):
        raise ValueError("Blind directory contains a symbolic link")
    for value in actual:
        lowered = value.as_posix().lower()
        if any(variant in lowered for variant in VARIANTS):
            raise ValueError(f"Blind path leaks a variant identity: {value}")

    files: dict[str, Any] = {}
    for slug in TRACKS:
        files[slug] = {}
        for group in GROUPS:
            variant = mapping["tracks"][slug]["groups"][group]
            files[slug][group] = {}
            for stem in STEMS:
                source = reference_root / slug / variant / f"{stem}.flac"
                target = blind_root / slug / group / f"{stem}.flac"
                if not os.path.samefile(source, target):
                    raise ValueError(f"Blind file is not the expected hard link: {target}")
                source_pcm = flac_pcm_identity(ffmpeg, source, EXPECTED_FRAMES)
                target_pcm = flac_pcm_identity(ffmpeg, target, EXPECTED_FRAMES)
                if target_pcm["pcmSha256"] != source_pcm["pcmSha256"]:
                    raise ValueError(f"Blind FLAC PCM mismatch: {target}")
                stream = ffprobe_contract(ffprobe, target)
                files[slug][group][stem] = {
                    "blindFlac": identity(target),
                    "decodedPcm": target_pcm,
                    "ffprobe": stream,
                    "sameFileAsPrivateReference": True,
                    "hardLinkVerified": True,
                }
    return {
        "fileCount": len(actual),
        "expectedFileCount": len(TRACKS) * len(GROUPS) * len(STEMS),
        "exactInventoryGatePassed": True,
        "noSymlinkGatePassed": True,
        "noVariantNameInPathGatePassed": True,
        "files": files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/htdemucs4-batch4a-host-20260805"),
    )
    parser.add_argument(
        "--mapping-json",
        type=Path,
        default=Path("docs/htdemucs4-batch4a-blind-map-2026-08-05.json"),
    )
    parser.add_argument(
        "--mapping-markdown",
        type=Path,
        default=Path("docs/htdemucs4-batch4a-blind-map-2026-08-05.md"),
    )
    parser.add_argument(
        "--evidence-json",
        type=Path,
        default=Path("docs/htdemucs4-batch4a-audio-evidence-2026-08-05.json"),
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path.cwd().resolve()
    output_root = (repo / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root.resolve()
    mapping_json = (repo / args.mapping_json).resolve() if not args.mapping_json.is_absolute() else args.mapping_json.resolve()
    mapping_markdown = (repo / args.mapping_markdown).resolve() if not args.mapping_markdown.is_absolute() else args.mapping_markdown.resolve()
    evidence_json = (repo / args.evidence_json).resolve() if not args.evidence_json.is_absolute() else args.evidence_json.resolve()
    blind_root = output_root / "blind-ae"
    partial = blind_root.with_name(blind_root.name + ".partial")
    private_paths = (mapping_json, mapping_markdown, evidence_json)
    if len(set(private_paths)) != len(private_paths):
        raise ValueError("Mapping JSON, mapping Markdown, and evidence JSON paths must be distinct")
    for private_path in private_paths:
        if inside(private_path, blind_root) or inside(private_path, partial):
            raise ValueError(f"Private evidence must be outside blind and staging roots: {private_path}")
    ffmpeg_evidence = executable_evidence(args.ffmpeg)
    ffprobe_evidence = executable_evidence(args.ffprobe)
    ffmpeg = ffmpeg_evidence["resolvedPath"]
    ffprobe = ffprobe_evidence["resolvedPath"]

    inputs = load_variant_inputs(output_root)
    mapping = load_or_create_mapping(mapping_json, blind_root, inputs)
    validate_mapping(mapping, inputs)
    references = build_references(output_root, inputs, ffmpeg, ffprobe)
    reference_root = output_root / "reference-flac"
    if blind_root.exists():
        precommit_validation = validate_blind_inventory(
            blind_root, reference_root, mapping, ffmpeg, ffprobe
        )
    else:
        if partial.exists():
            if partial.resolve().parent != output_root.resolve():
                raise ValueError("Unexpected blind staging path")
            shutil.rmtree(partial)
        for slug in TRACKS:
            for group in GROUPS:
                variant = mapping["tracks"][slug]["groups"][group]
                for stem in STEMS:
                    source = reference_root / slug / variant / f"{stem}.flac"
                    target = partial / slug / group / f"{stem}.flac"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.link(source, target)
                    os.utime(target, (FIXED_MTIME, FIXED_MTIME))
        precommit_validation = validate_blind_inventory(
            partial, reference_root, mapping, ffmpeg, ffprobe
        )
        partial.replace(blind_root)
    final_validation = validate_blind_inventory(
        blind_root, reference_root, mapping, ffmpeg, ffprobe
    )
    if precommit_validation["fileCount"] != final_validation["fileCount"]:
        raise ValueError("Blind inventory changed during atomic publication")

    tool_path = Path(__file__).resolve()
    evidence = {
        "schemaVersion": 1,
        "status": "complete",
        "scope": "Private lossless-audio identity evidence for Batch 4A A-E blind set",
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
        "blindRoot": str(blind_root),
        "mapping": identity(mapping_json),
        "mappingSeedSha256": mapping["seedSha256"],
        "inputContracts": inputs,
        "privateReferences": references,
        "publicationValidation": final_validation,
        "encoding": {
            "format": "FLAC",
            "sampleRate": 44_100,
            "channelCount": 2,
            "sampleFormat": "signed PCM16",
            "losslessPcmIdentityVerified": True,
            "metadataRemoved": True,
            "fixedMtimeUtc": datetime.fromtimestamp(FIXED_MTIME, timezone.utc).isoformat(),
        },
        "tools": {
            "finalizer": identity(tool_path),
            "importedFlacHelper": identity(Path(blind_helpers.__file__).resolve()),
            "ffmpeg": ffmpeg_evidence,
            "ffprobe": ffprobe_evidence,
        },
    }
    write_json(evidence_json, evidence)
    write_mapping_markdown(mapping_markdown, mapping, mapping_json)
    print(json.dumps({
        "status": "complete",
        "blindRoot": str(blind_root),
        "fileCount": final_validation["fileCount"],
        "mappingJson": str(mapping_json),
        "mappingMarkdown": str(mapping_markdown),
        "evidenceJson": str(evidence_json),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
