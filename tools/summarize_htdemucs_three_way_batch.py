#!/usr/bin/env python3
"""Summarize the completed S25/ONNX/original-weight HTDemucs batch."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


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
STEMS = ("drums", "bass", "other", "vocals", "guitar", "piano")
PAIR_KEYS = (
    "s25-cpu_vs_original-safetensors-torch",
    "desktop-onnx_vs_original-safetensors-torch",
    "desktop-onnx_vs_s25-cpu",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_source_name_overrides(root: Path) -> dict[str, str]:
    """Load original host filenames keyed by the source MP3 SHA-256.

    S25 transport names are deliberately normalized for ADB.  The progress
    sidecar is the authoritative record of the names supplied by the user.
    """
    progress_path = root / "s25-batch-progress.json"
    if not progress_path.is_file():
        return {}
    progress = load_json(progress_path)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "maximum": max(values),
        "mean": statistics.fmean(values),
    }


def backend_memory(report: dict[str, Any], field: str) -> int | None:
    values = [
        item.get("process", {}).get(field)
        for item in report.get("windows", [])
    ]
    numbers = [int(value) for value in values if value is not None]
    return max(numbers) if numbers else None


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--blind-root", type=Path)
    parser.add_argument("--mapping-markdown", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.output_root.resolve()
    source_name_overrides = load_source_name_overrides(root)
    tracks: dict[str, Any] = {}
    for slug in TRACKS:
        track_root = root / "tracks" / slug
        paths = {
            "s25": track_root / "s25-cpu" / "report.json",
            "onnx": track_root / "desktop-onnx" / "report.json",
            "torch": track_root / "original-safetensors-torch" / "report.json",
            "comparison": track_root / "comparison.json",
        }
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        s25 = load_json(paths["s25"])
        onnx = load_json(paths["onnx"])
        torch = load_json(paths["torch"])
        comparison = load_json(paths["comparison"])
        pcm_hashes = {
            s25["source"]["selectedPcmSha256"],
            onnx["source"]["selectedPcmSha256"],
            torch["source"]["selectedPcmSha256"],
            comparison["sharedSelectedPcmSha256"],
        }
        if len(pcm_hashes) != 1:
            raise ValueError(f"Canonical PCM mismatch for {slug}: {pcm_hashes}")
        attempt = s25["attempts"][0]
        windows = attempt["windows"]
        thermal_counts = Counter(item["thermalStatus"] for item in windows)
        envelope_path = track_root / "s25-device-envelope.json"
        envelope = load_json(envelope_path) if envelope_path.is_file() else None
        tracks[slug] = {
            "sourceFileName": source_name_overrides.get(
                s25["source"]["fileSha256"], s25["source"]["fileName"]
            ),
            "deviceSourceFileName": s25["source"]["fileName"],
            "sourceMp3ByteSize": s25["source"]["fileBytes"],
            "sourceMp3Sha256": s25["source"]["fileSha256"],
            "decodedFrames": s25["source"]["decoded"]["frames"],
            "canonicalFrames": s25["source"]["selectedFrames"],
            "canonicalPcmSha256": next(iter(pcm_hashes)),
            "durationSeconds": s25["source"]["durationSeconds"],
            "windowCount": attempt["expectedWindows"],
            "inputPreparation": s25["inputPreparation"],
            "s25": {
                "report": str(paths["s25"].resolve()),
                "reportSha256": sha256_file(paths["s25"]),
                "totalWallMs": attempt["total"]["wallMs"],
                "realtimeFactor": attempt["realtimeFactor"],
                "prepareWallMs": attempt["prepare"]["total"]["wallMs"],
                "stageSummary": attempt["stageSummary"],
                "peakPssKiB": max(item["process"]["pssKb"] for item in windows),
                "peakNativeHeapBytes": max(
                    item["process"]["nativeHeapAllocatedBytes"] for item in windows
                ),
                "maximumThermalStatus": max(item["thermalStatus"] for item in windows),
                "thermalStatusWindowCounts": {
                    str(status): count for status, count in sorted(thermal_counts.items())
                },
                "outputs": {
                    stem: {
                        key: value
                        for key, value in evidence.items()
                        if key
                        in {
                            "sampleCount",
                            "nonFiniteCount",
                            "clippedSampleCount",
                            "absoluteMean",
                            "rms",
                            "min",
                            "max",
                            "byteSize",
                            "sha256",
                        }
                    }
                    for stem, evidence in attempt["outputs"].items()
                },
                "batteryBefore": envelope.get("before", {}).get("battery") if envelope else None,
                "batteryAfter": envelope.get("after", {}).get("battery") if envelope else None,
            },
            "desktopOnnx": {
                "report": str(paths["onnx"].resolve()),
                "reportSha256": sha256_file(paths["onnx"]),
                "separationWallMs": onnx["separationWallMs"],
                "separationRealtimeFactor": onnx["separationRealtimeFactor"],
                "totalWallMs": onnx["totalWallMs"],
                "peakObservedRssBytes": backend_memory(onnx, "rssBytes"),
            },
            "originalSafetensorsTorch": {
                "report": str(paths["torch"].resolve()),
                "reportSha256": sha256_file(paths["torch"]),
                "separationWallMs": torch["separationWallMs"],
                "separationRealtimeFactor": torch["separationRealtimeFactor"],
                "totalWallMs": torch["totalWallMs"],
                "peakObservedVmHwmKiB": backend_memory(torch, "VmHWMKiB"),
            },
            "comparison": comparison,
        }

    durations = [item["durationSeconds"] for item in tracks.values()]
    total_duration = sum(durations)
    s25_wall = sum(item["s25"]["totalWallMs"] for item in tracks.values())
    onnx_wall = sum(item["desktopOnnx"]["separationWallMs"] for item in tracks.values())
    torch_wall = sum(
        item["originalSafetensorsTorch"]["separationWallMs"] for item in tracks.values()
    )
    pair_summary = {}
    for pair in PAIR_KEYS:
        aggregate_snr = [
            item["comparison"]["pairs"][pair]["aggregate"]["snrDb"]
            for item in tracks.values()
        ]
        pair_summary[pair] = {
            "trackAggregateSnrDb": distribution(aggregate_snr),
            "maximumAbsoluteErrorLsb": max(
                item["comparison"]["pairs"][pair]["aggregate"]["maximumAbsoluteErrorLsb"]
                for item in tracks.values()
            ),
            "minimumCorrelation": min(
                item["comparison"]["pairs"][pair]["aggregate"]["correlation"]
                for item in tracks.values()
            ),
            "perStemSnrDb": {
                stem: distribution(
                    [
                        item["comparison"]["pairs"][pair]["perStem"][stem]["snrDb"]
                        for item in tracks.values()
                    ]
                )
                for stem in STEMS
            },
        }

    first = tracks[TRACKS[0]]
    summary = {
        "schemaVersion": 1,
        "status": "complete",
        "trackCount": len(TRACKS),
        "totalDurationSeconds": total_duration,
        "totalCanonicalFrames": sum(item["canonicalFrames"] for item in tracks.values()),
        "contract": {
            "sampleRate": 44_100,
            "channels": 2,
            "windowSamples": 343_980,
            "strideSamples": 257_985,
            "overlapSamples": 85_995,
            "stems": list(STEMS),
            "inputDecode": "S25 MediaCodec, exported canonical PCM reused by all backends",
        },
        "artifacts": {
            "s25LiteRt": {
                "runtime": first["comparison"]["reports"]["s25-cpu"],
                "modelId": load_json(Path(first["s25"]["report"]))["model"]["modelId"],
                "modelSha256": load_json(Path(first["s25"]["report"]))["model"]["sha256"],
            },
            "desktopOnnx": load_json(Path(first["desktopOnnx"]["report"]))["provenance"][
                "onnxArtifact"
            ],
            "originalSafetensors": load_json(
                Path(first["originalSafetensorsTorch"]["report"])
            )["provenance"]["weight"],
        },
        "performance": {
            "s25": {
                "totalWallMs": s25_wall,
                "aggregateRealtimeFactor": s25_wall / 1000.0 / total_duration,
                "trackRealtimeFactor": distribution(
                    [item["s25"]["realtimeFactor"] for item in tracks.values()]
                ),
                "maximumObservedPssKiB": max(
                    item["s25"]["peakPssKiB"] for item in tracks.values()
                ),
                "maximumThermalStatus": max(
                    item["s25"]["maximumThermalStatus"] for item in tracks.values()
                ),
                "totalClippedSamples": sum(
                    evidence["clippedSampleCount"]
                    for item in tracks.values()
                    for evidence in item["s25"]["outputs"].values()
                ),
                "totalNonFiniteSamples": sum(
                    evidence["nonFiniteCount"]
                    for item in tracks.values()
                    for evidence in item["s25"]["outputs"].values()
                ),
            },
            "desktopOnnx": {
                "totalSeparationWallMs": onnx_wall,
                "aggregateSeparationRealtimeFactor": onnx_wall / 1000.0 / total_duration,
                "maximumObservedRssBytes": max(
                    item["desktopOnnx"]["peakObservedRssBytes"] or 0
                    for item in tracks.values()
                ),
            },
            "originalSafetensorsTorch": {
                "totalSeparationWallMs": torch_wall,
                "aggregateSeparationRealtimeFactor": torch_wall / 1000.0 / total_duration,
                "maximumObservedVmHwmKiB": max(
                    item["originalSafetensorsTorch"]["peakObservedVmHwmKiB"] or 0
                    for item in tracks.values()
                ),
            },
        },
        "numericalComparison": pair_summary,
        "tracks": tracks,
        "blindRoot": str(args.blind_root.resolve()) if args.blind_root else None,
        "blindMappingDocument": (
            str(args.mapping_markdown.resolve()) if args.mapping_markdown else None
        ),
    }
    write_atomic(args.json.resolve(), json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        "# S25 HTDemucs six-stem full-song three-way batch",
        "",
        f"Tracks: **{len(TRACKS)}**; total audio: **{total_duration / 60:.2f} min**.",
        "",
        "All three backends consumed the exact PCM16 canonical decode exported by S25. "
        "The original safetensors/Torch FP32 path is the primary numerical reference.",
        "",
        "## Performance",
        "",
        "| Track | Duration (s) | Windows | S25 RTF | ONNX RTF | Original RTF | Peak S25 PSS (MiB) | Thermal max | Clipped samples |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for slug in TRACKS:
        item = tracks[slug]
        lines.append(
            f"| {item['sourceFileName']} | {item['durationSeconds']:.3f} | {item['windowCount']} | "
            f"{item['s25']['realtimeFactor']:.3f} | "
            f"{item['desktopOnnx']['separationRealtimeFactor']:.3f} | "
            f"{item['originalSafetensorsTorch']['separationRealtimeFactor']:.3f} | "
            f"{item['s25']['peakPssKiB'] / 1024:.1f} | "
            f"{item['s25']['maximumThermalStatus']} | "
            f"{sum(value['clippedSampleCount'] for value in item['s25']['outputs'].values())} |"
        )
    performance = summary["performance"]
    lines.extend(
        [
            "",
            "Aggregate separation RTF:",
            "",
            f"- S25 LiteRT CPU: `{performance['s25']['aggregateRealtimeFactor']:.4f}`",
            f"- Desktop ONNX CPU: `{performance['desktopOnnx']['aggregateSeparationRealtimeFactor']:.4f}`",
            f"- Original safetensors/Torch CPU: `{performance['originalSafetensorsTorch']['aggregateSeparationRealtimeFactor']:.4f}`",
            f"- S25 non-finite samples: `{performance['s25']['totalNonFiniteSamples']}`; clipped before PCM16 write: `{performance['s25']['totalClippedSamples']}`",
            "",
            "## Numerical difference",
            "",
            "| Pair | Track SNR min / median / max (dB) | Max error (LSB) | Min correlation |",
            "|---|---:|---:|---:|",
        ]
    )
    for pair in PAIR_KEYS:
        item = pair_summary[pair]
        snr = item["trackAggregateSnrDb"]
        lines.append(
            f"| {pair} | {snr['minimum']:.2f} / {snr['median']:.2f} / {snr['maximum']:.2f} | "
            f"{item['maximumAbsoluteErrorLsb']} | {item['minimumCorrelation']:.9f} |"
        )
    lines.extend(
        [
            "",
            "The published desktop ONNX stores weights in FP16 and comes from a legacy `.th` "
            "export. Its difference from the original control is therefore not attributable only "
            "to ONNX Runtime. S25 uses the project-owned FP32 neural-core artifact generated from "
            "the official safetensors.",
            "",
            "A track-level aggregate above 80 dB does not imply every low-signal stem passed 80 dB. "
            "Use the JSON report for per-stem SNR, absolute error, and correlation.",
            "",
            "## Outputs",
            "",
            f"- Batch JSON: `{args.json.resolve()}`",
            f"- Blind audio: `{args.blind_root.resolve() if args.blind_root else 'not finalized'}`",
            f"- Blind key: `{args.mapping_markdown.resolve() if args.mapping_markdown else 'not finalized'}`",
            "",
        ]
    )
    write_atomic(args.markdown.resolve(), "\n".join(lines))
    print(json.dumps({
        "status": "complete",
        "json": str(args.json.resolve()),
        "markdown": str(args.markdown.resolve()),
        "trackCount": len(TRACKS),
        "s25AggregateRtf": performance["s25"]["aggregateRealtimeFactor"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
