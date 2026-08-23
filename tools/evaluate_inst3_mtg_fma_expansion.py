#!/usr/bin/env python3
"""Evaluate MTG/FMA expansion songs against Inst 3 and retain the worst 12.

The current H50-continuation+5 model is compared only to the native Inst 3
instrumental output.  The new 16-song pool is ranked by local 50/100/200 ms
positive-projection severity; the selected 12 remain as a focused listening
set.  All artifacts are local research outputs.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
import torch

import analyze_inst3_private_language_quality as quality
import evaluate_inst3_continuous_baseline as continuous
import render_inst3_objective_listening as listening
import run_inst3_distill_pilot as pilot
import run_inst3_vr_continuous_topk_local as local
from tfc_tdf_short_window import ShortWindowContract


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "inst3-mtg-fma-expansion" / "source-manifest.json"
DEFAULT_CHECKPOINT = ROOT / "data" / "musdb18-inst3-vr-continuation" / "runs" / "H50-continuation-plus5" / "step-1600.pt"
DEFAULT_ARCHITECTURE = pilot.DEFAULT_CHECKPOINT
DEFAULT_EXISTING_REPORT = ROOT / "data" / "inst3-private-language-quality" / "private-language-quality-report.json"
DEFAULT_OUTPUT = ROOT / "data" / "inst3-mtg-fma-expansion-evaluation"
SAMPLE_RATE = 44_100
EVENT_MS = (50, 100, 200)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--architecture-checkpoint", type=Path, default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--existing-report", type=Path, default=DEFAULT_EXISTING_REPORT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--retain-count", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    return pilot.sha256_file(path)


def db(value: float, floor: float = -240.0) -> float:
    if not math.isfinite(value) or value <= 10.0 ** (floor / 20.0):
        return floor
    return 20.0 * math.log10(value)


def write_flac(path: Path, audio: np.ndarray) -> dict[str, Any]:
    if audio.ndim != 2 or audio.shape[1] != 2 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio for {path}: {audio.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(audio, -1.0, 1.0), SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    return {
        "file": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "frames": int(audio.shape[0]),
        "sampleRate": SAMPLE_RATE,
        "channels": 2,
    }


def severity_score(short_events: dict[str, dict[str, Any]]) -> float:
    """Higher means a more salient coherent Inst 3-directed leak."""
    values = [
        float(short_events[str(ms)]["positiveProjectionMaxDbfs"])
        for ms in EVENT_MS
    ]
    p95 = [
        float(short_events[str(ms)]["positiveProjectionP95Dbfs"])
        for ms in EVENT_MS
    ]
    return 0.65 * max(values) + 0.35 * float(np.mean(p95))


def load_teacher(path: str) -> np.ndarray:
    with np.load(path) as values:
        result = np.ascontiguousarray(values["instrumental"], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"Non-finite teacher output: {path}")
    return result


def evaluate_song(
    source: np.ndarray,
    teacher: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    whole = quality.metric_row(source, teacher, candidate)
    short = {
        str(ms): quality.short_event_row(source, teacher, candidate, ms)
        for ms in EVENT_MS
    }
    return {
        "wholeSong": whole,
        "shortEvents": short,
        "severityScoreDbfs": severity_score(short),
    }


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    # Track names and attribution are not necessarily ASCII on Windows.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if args.retain_count <= 0 or args.retain_count > 16:
        raise ValueError("retain-count must be between 1 and 16")
    if args.threads <= 0 or args.inference_batch_size <= 0:
        raise ValueError("threads and inference-batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or len(manifest.get("records", [])) != 16:
        raise ValueError("Expected completed 16-song expansion manifest")
    model, model_source = local.load_h50_model(
        args.architecture_checkpoint.resolve(), args.checkpoint.resolve(), device
    )
    contract = ShortWindowContract(num_frames=128, left_context_hops=5, right_context_hops=5)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = sha256_file(args.checkpoint.resolve())
    manifest_hash = sha256_file(args.manifest.resolve())
    report: dict[str, Any] = {
        "schema": "local-inst3-mtg-fma-expansion-evaluation@1",
        "status": "running",
        "reference": "native Inst 3 instrumental output",
        "model": {
            "checkpoint": {"file": str(args.checkpoint.resolve()), "sha256": checkpoint_hash},
            "source": model_source,
            "assembly": "continuous-context-overlap-save",
        },
        "manifest": {"file": str(args.manifest.resolve()), "sha256": manifest_hash},
        "songs": {},
    }
    report_path = output_root / "expansion-evaluation-report.json"
    if not args.force and report_path.is_file():
        try:
            previous = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = None
        if (
            isinstance(previous, dict)
            and previous.get("manifest", {}).get("sha256") == manifest_hash
            and previous.get("model", {}).get("checkpoint", {}).get("sha256") == checkpoint_hash
            and isinstance(previous.get("songs"), dict)
        ):
            # Keep completed songs from an interrupted run and fill in the rest.
            report["songs"] = previous["songs"]
            print(f"resuming {len(report['songs'])} completed song(s)", flush=True)
    json_write(report_path, report)
    try:
        for index, record in enumerate(manifest["records"], start=1):
            slug = record["slug"]
            previous_song = report["songs"].get(slug)
            if not args.force and isinstance(previous_song, dict):
                outputs = previous_song.get("outputs", {})
                output_files = [outputs.get(kind, {}).get("file") for kind in (
                    "h50ContinuationPlus5", "h50Residual", "inst3Instrumental"
                )]
                if all(isinstance(path, str) and Path(path).is_file() and Path(path).stat().st_size > 0 for path in output_files):
                    print(f"skip {index}/{len(manifest['records'])}: {record['artistName']} - {record['trackName']}", flush=True)
                    continue
            raw_path = Path(record["download"]["file"])
            teacher_path = Path(record["teacher"]["file"])
            source, sample_rate = listening.load_audio(raw_path)
            if sample_rate != SAMPLE_RATE:
                raise ValueError(f"Unexpected source rate for {slug}")
            teacher = load_teacher(str(teacher_path))
            if teacher.shape != source.shape:
                raise ValueError(f"Teacher/source shape mismatch for {slug}")
            print(f"evaluate {index}/{len(manifest['records'])}: {record['artistName']} - {record['trackName']}", flush=True)
            residual, timing = continuous.render_student(
                model,
                source,
                contract,
                device,
                args.inference_batch_size,
            )
            candidate = np.ascontiguousarray(source - residual, dtype=np.float32)
            metrics = evaluate_song(source, teacher, candidate)
            instrumental_path = output_root / "all-candidates" / f"{slug}-instrumental.flac"
            residual_path = output_root / "all-candidates" / f"{slug}-residual.flac"
            teacher_path_out = output_root / "all-candidates" / f"{slug}-inst3.flac"
            metrics["outputs"] = {
                "h50ContinuationPlus5": write_flac(instrumental_path, candidate),
                "h50Residual": write_flac(residual_path, residual),
                "inst3Instrumental": write_flac(teacher_path_out, teacher),
            }
            metrics["render"] = timing
            metrics["source"] = {
                "source": record["source"],
                "sourceId": record["sourceId"],
                "artistName": record["artistName"],
                "trackName": record["trackName"],
                "durationSeconds": record["durationSeconds"],
                "genreTags": record.get("genreTags"),
                "languageCode": record.get("languageCode"),
                "license": record["license"],
                "licenseUrl": record["licenseUrl"],
                "sourceUrl": record["sourceUrl"],
                "downloadUrl": record["downloadUrl"],
                "rawSha256": record["download"]["sha256"],
            }
            report["songs"][slug] = metrics
            json_write(report_path, report)
            del source, teacher, residual, candidate
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    rows = []
    for slug, value in report["songs"].items():
        rows.append({"slug": slug, **value["source"], **value["wholeSong"], "severityScoreDbfs": value["severityScoreDbfs"], "shortEvents": value["shortEvents"]})
    rows.sort(key=lambda row: (-float(row["severityScoreDbfs"]), -float(row["shortEvents"]["100"]["positiveProjectionMaxDbfs"])))
    for rank, row in enumerate(rows, start=1):
        row["severityRank"] = rank
        row["retained"] = rank <= args.retain_count
    retained = [row for row in rows if row["retained"]]
    retained_root = output_root / "retained-12"
    for row in retained:
        slug = row["slug"]
        for kind in ("h50ContinuationPlus5", "h50Residual", "inst3Instrumental"):
            source_path = Path(report["songs"][slug]["outputs"][kind]["file"])
            destination = retained_root / kind / source_path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if args.force or not destination.is_file():
                destination.write_bytes(source_path.read_bytes())
    existing = json.loads(args.existing_report.resolve().read_text(encoding="utf-8"))
    existing_rows = [
        row for row in existing["songs"]
        if row["variant"] == "H50-continuation+5"
    ]
    combined = []
    for row in existing_rows:
        combined.append({"sourceSet": "existing-private-12", **row})
    combined.extend({"sourceSet": "mtg-fma-expansion", **row} for row in rows)
    combined.sort(key=lambda row: (-float(row.get("shortEvents", {}).get("100", {}).get("positiveProjectionMaxDbfs", -240.0)), row["song"] if "song" in row else row["slug"]))
    report["status"] = "completed"
    report["ranking"] = {
        "newPoolCount": len(rows),
        "retainedCount": len(retained),
        "retained": retained,
        "allNewRows": rows,
        "combinedExistingAndNewRows": combined,
        "selectionDefinition": "severityScore = 0.65 * max(50/100/200 ms positive projection max) + 0.35 * mean(50/100/200 ms positive projection p95); higher is more salient",
    }
    report["notes"] = {
        "reference": "Inst 3 is the practical reference, not isolated ground-truth stems.",
        "selection": "New 16-song pool ranked before retaining 12; raw miss and positive projection are both reported because target-like coherent residue is the primary listening proxy.",
        "license": "Every retained record preserves source-specific license and attribution fields; artifacts remain local non-commercial research only.",
    }
    json_write(report_path, report)
    print(json.dumps({"status": report["status"], "newSongs": len(rows), "retained": [row["slug"] for row in retained], "report": str(report_path)}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
