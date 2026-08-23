#!/usr/bin/env python3
"""Build evenly covered two-second MTG-Jamendo/FMA leak candidates.

The candidate pool combines the 16-song expansion evaluation and the 12-song
supplement manifest.  Each event is rendered as a matching pair of
``H50-continuation+5`` and native Inst 3 instrumental audio, with the source
mixture included only as optional orientation.  Candidate selection is based
on local residual content that is coherent with the content Inst 3 removes.

All source audio, teacher outputs, model outputs, and listening artifacts are
local non-commercial research files.  No candidate is automatically assigned
to training or validation.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
import torch

import evaluate_inst3_continuous_baseline as continuous
import render_inst3_objective_listening as listening
import run_inst3_distill_pilot as pilot
import run_inst3_vr_continuous_topk_local as local
from tfc_tdf_short_window import ShortWindowContract


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPANSION_MANIFEST = ROOT / "data" / "inst3-mtg-fma-expansion" / "source-manifest.json"
DEFAULT_SUPPLEMENT_MANIFEST = ROOT / "data" / "inst3-mtg-fma-supplement" / "source-manifest.json"
DEFAULT_EXPANSION_OUTPUT = ROOT / "data" / "inst3-mtg-fma-expansion-evaluation" / "all-candidates"
DEFAULT_CHECKPOINT = ROOT / "data" / "musdb18-inst3-vr-continuation" / "runs" / "H50-continuation-plus5" / "step-1600.pt"
DEFAULT_ARCHITECTURE = pilot.DEFAULT_CHECKPOINT
DEFAULT_OUTPUT = ROOT / "data" / "inst3-mtg-fma-event-listening"
SAMPLE_RATE = 44_100
EVENT_MS = (50, 100, 200)
SNIPPET_SAMPLES = SAMPLE_RATE * 2
PAIR_GAP_SAMPLES = round(SAMPLE_RATE * 0.3)
PAIR_SAMPLES = SNIPPET_SAMPLES * 2 + PAIR_GAP_SAMPLES
DEFAULT_CANDIDATES_PER_SONG = 16
DEFAULT_COVERAGE_BINS = 8
ACTIVE_FLOOR_DBFS = -60.0
MIN_CENTER_SEPARATION_SECONDS = 1.5


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expansion-manifest", type=Path, default=DEFAULT_EXPANSION_MANIFEST)
    parser.add_argument("--supplement-manifest", type=Path, default=DEFAULT_SUPPLEMENT_MANIFEST)
    parser.add_argument("--expansion-output", type=Path, default=DEFAULT_EXPANSION_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--architecture-checkpoint", type=Path, default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidates-per-song", type=int, default=DEFAULT_CANDIDATES_PER_SONG)
    parser.add_argument("--coverage-bins", type=int, default=DEFAULT_COVERAGE_BINS)
    parser.add_argument("--scan-hop-ms", type=int, default=50)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
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


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "song"


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
        "peak": float(np.max(np.abs(audio))),
    }


def make_listening_pair(h50: np.ndarray, inst3: np.ndarray) -> np.ndarray:
    """Place H50 first, 300 ms silence, then Inst 3."""
    if h50.shape != (SNIPPET_SAMPLES, 2) or inst3.shape != (SNIPPET_SAMPLES, 2):
        raise ValueError(f"Expected two 2-second stereo clips, got {h50.shape}, {inst3.shape}")
    silence = np.zeros((PAIR_GAP_SAMPLES, 2), dtype=np.float32)
    pair = np.concatenate((h50, silence, inst3), axis=0)
    if pair.shape != (PAIR_SAMPLES, 2) or not np.isfinite(pair).all():
        raise ValueError(f"Invalid listening pair shape: {pair.shape}")
    return np.ascontiguousarray(pair, dtype=np.float32)


def load_audio(path: Path) -> np.ndarray:
    audio, sample_rate = listening.load_audio(path)
    if sample_rate != SAMPLE_RATE or audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError(f"Unexpected source contract for {path}: {audio.shape}, {sample_rate}")
    return np.ascontiguousarray(audio, dtype=np.float32)


def load_teacher(path: Path) -> np.ndarray:
    with np.load(path) as values:
        if "instrumental" not in values:
            raise ValueError(f"Teacher cache lacks instrumental output: {path}")
        audio = np.ascontiguousarray(values["instrumental"], dtype=np.float32)
    if audio.ndim != 2 or audio.shape[1] != 2 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid teacher output: {path}")
    return audio


def load_manifest_records(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if value.get("status") != "completed":
        raise ValueError(f"Manifest is not completed: {path}")
    records = value.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Manifest has no records: {path}")
    return records


def teacher_path(record: dict[str, Any]) -> Path:
    path = Path(record["teacher"]["file"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def source_path(record: dict[str, Any]) -> Path:
    path = Path(record["download"]["file"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def expansion_output_path(root: Path, slug: str, suffix: str) -> Path:
    path = root / f"{slug}-{suffix}.flac"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def block_metric(
    source: np.ndarray,
    teacher: np.ndarray,
    candidate: np.ndarray,
    center: int,
    milliseconds: int,
) -> dict[str, float]:
    length = max(1, round(SAMPLE_RATE * milliseconds / 1000.0))
    start = center - length // 2
    end = start + length
    if start < 0 or end > source.shape[0]:
        raise ValueError("block must be fully inside the source")
    source_block = source[start:end].astype(np.float64, copy=False)
    teacher_block = teacher[start:end].astype(np.float64, copy=False)
    candidate_block = candidate[start:end].astype(np.float64, copy=False)
    removed = source_block - teacher_block
    miss = candidate_block - teacher_block
    removed_power = float(np.sum(removed * removed))
    miss_power = float(np.sum(miss * miss))
    dot = float(np.sum(miss * removed))
    denominator = source_block.size
    removed_rms = math.sqrt(removed_power / max(denominator, 1))
    miss_rms = math.sqrt(miss_power / max(denominator, 1))
    projection = max(dot / max(removed_power, 1.0e-30), 0.0) * removed_rms
    return {
        "removedRmsDbfs": db(removed_rms),
        "missRmsDbfs": db(miss_rms),
        "positiveProjectionDbfs": db(projection),
    }


def event_score(metrics: dict[int, dict[str, float]]) -> float:
    # Prefer coherent content that Inst 3 removes, while retaining raw miss as
    # a secondary signal for cases where the projection is unstable.
    projection = max(10.0 ** (metrics[ms]["positiveProjectionDbfs"] / 20.0) for ms in EVENT_MS)
    miss = max(10.0 ** (metrics[ms]["missRmsDbfs"] / 20.0) for ms in EVENT_MS)
    return db(0.75 * projection + 0.25 * miss)


def scan_candidates(
    source: np.ndarray,
    teacher: np.ndarray,
    candidate: np.ndarray,
    song_slug: str,
    scan_hop_ms: int,
    candidates_per_song: int,
    coverage_bins: int,
) -> list[dict[str, Any]]:
    if source.shape != teacher.shape or source.shape != candidate.shape:
        raise ValueError(f"Shape mismatch for {song_slug}: {source.shape}, {teacher.shape}, {candidate.shape}")
    hop = max(1, round(SAMPLE_RATE * scan_hop_ms / 1000.0))
    half_snippet = SNIPPET_SAMPLES // 2
    first_center = half_snippet
    last_center = source.shape[0] - half_snippet
    if last_center < first_center:
        raise ValueError(f"Song is shorter than two seconds: {song_slug}")
    rows: list[dict[str, Any]] = []
    for center in range(first_center, last_center + 1, hop):
        metrics = {
            ms: block_metric(source, teacher, candidate, center, ms)
            for ms in EVENT_MS
        }
        active = any(metrics[ms]["removedRmsDbfs"] >= ACTIVE_FLOOR_DBFS for ms in EVENT_MS)
        if not active:
            continue
        rows.append(
            {
                "song": song_slug,
                "centerSamples": center,
                "centerSeconds": center / SAMPLE_RATE,
                "snippetStartSamples": center - half_snippet,
                "snippetEndSamples": center + half_snippet,
                "eventScoreDbfs": event_score(metrics),
                "metrics": {str(ms): metrics[ms] for ms in EVENT_MS},
            }
        )
    if not rows:
        raise ValueError(f"No active candidate blocks for {song_slug}")
    rows.sort(key=lambda row: (-float(row["eventScoreDbfs"]), int(row["centerSamples"])))
    selected: list[dict[str, Any]] = []
    selected_centers: list[int] = []
    min_separation = round(MIN_CENTER_SEPARATION_SECONDS * SAMPLE_RATE)

    def acceptable(row: dict[str, Any], separation: int = min_separation) -> bool:
        center = int(row["centerSamples"])
        return all(abs(center - other) >= separation for other in selected_centers)

    # First take one strongest event from each temporal bin, which prevents a
    # long song's candidate set from collapsing onto one chorus or verse.
    bin_count = max(1, min(coverage_bins, candidates_per_song))
    for bin_index in range(bin_count):
        start = first_center + (last_center - first_center) * bin_index // bin_count
        end = first_center + (last_center - first_center) * (bin_index + 1) // bin_count
        bin_rows = [row for row in rows if start <= int(row["centerSamples"]) < max(end, start + 1)]
        for row in bin_rows:
            if acceptable(row):
                selected.append(row)
                selected_centers.append(int(row["centerSamples"]))
                break
    for separation in (min_separation, round(0.75 * SAMPLE_RATE), round(0.5 * SAMPLE_RATE), 0):
        for row in rows:
            if len(selected) >= candidates_per_song:
                break
            if row in selected or not acceptable(row, separation):
                continue
            selected.append(row)
            selected_centers.append(int(row["centerSamples"]))
        if len(selected) >= candidates_per_song:
            break
    selected.sort(key=lambda row: int(row["centerSamples"]))
    for index, row in enumerate(selected, start=1):
        row["songCandidateIndex"] = index
        row["coverageFraction"] = float(row["centerSamples"]) / max(source.shape[0] - 1, 1)
    return selected


def render_h50(
    model: torch.nn.Module,
    source: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    contract = ShortWindowContract(num_frames=128, left_context_hops=5, right_context_hops=5)
    residual, timing = continuous.render_student(model, source, contract, device, batch_size)
    candidate = np.ascontiguousarray(source - residual, dtype=np.float32)
    if candidate.shape != source.shape or not np.isfinite(candidate).all():
        raise ValueError("Invalid H50 output")
    return candidate, timing


def ensure_full_outputs(
    record: dict[str, Any],
    expansion_output: Path,
    output_root: Path,
    model: torch.nn.Module,
    device: torch.device,
    inference_batch_size: int,
    force: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    slug = record["slug"]
    source = load_audio(source_path(record))
    teacher = load_teacher(teacher_path(record))
    if teacher.shape != source.shape:
        raise ValueError(f"Teacher/source shape mismatch for {slug}")
    h50_path = expansion_output_path(expansion_output, slug, "instrumental") if record.get("manifestKind") == "expansion" else None
    # Expansion outputs are already rendered by the preceding full-song
    # evaluation. Supplement records are rendered here under the new report.
    full_h50 = output_root / "full-song" / "h50ContinuationPlus5" / f"{slug}-instrumental.flac"
    full_inst3 = output_root / "full-song" / "inst3Instrumental" / f"{slug}-instrumental.flac"
    timing: dict[str, Any] = {"reusedH50": False}
    if record.get("manifestKind") == "expansion":
        h50 = load_audio(h50_path)
        timing["reusedH50"] = True
    elif not force and full_h50.is_file():
        h50 = load_audio(full_h50)
        timing["reusedH50"] = True
    else:
        h50, render_timing = render_h50(model, source, device, inference_batch_size)
        timing.update(render_timing)
        write_flac(full_h50, h50)
    if h50.shape != source.shape:
        raise ValueError(f"H50/source shape mismatch for {slug}")
    if not force and full_inst3.is_file():
        inst3 = load_audio(full_inst3)
    else:
        inst3 = teacher
        write_flac(full_inst3, inst3)
    return source, h50, inst3, timing


def write_review_csv(path: Path, events: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, str]] = {}
    if path.is_file():
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                existing = {
                    row["eventId"]: {
                        "mark": (row.get("mark") or "").strip().upper(),
                        "notes": row.get("notes") or "",
                    }
                    for row in csv.DictReader(handle)
                    if row.get("eventId")
                }
        except (OSError, KeyError, csv.Error):
            existing = {}
    fields = (
        "mark", "eventId", "sourceSet", "role", "source", "sourceId", "artistName",
        "trackName", "license", "centerSeconds", "eventScoreDbfs",
        "positiveProjection50Dbfs", "positiveProjection100Dbfs",
        "positiveProjection200Dbfs", "h50File", "inst3File", "mixtureFile",
        "pairedFile", "notes",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in events:
            metrics = event["metrics"]
            previous = existing.get(event["eventId"], {})
            writer.writerow(
                {
                    "mark": previous.get("mark", ""),
                    "eventId": event["eventId"],
                    "sourceSet": event["sourceSet"],
                    "role": event["role"],
                    "source": event["source"],
                    "sourceId": event["sourceId"],
                    "artistName": event["artistName"],
                    "trackName": event["trackName"],
                    "license": event["license"],
                    "centerSeconds": f"{event['centerSeconds']:.3f}",
                    "eventScoreDbfs": f"{event['eventScoreDbfs']:.2f}",
                    "positiveProjection50Dbfs": f"{metrics['50']['positiveProjectionDbfs']:.2f}",
                    "positiveProjection100Dbfs": f"{metrics['100']['positiveProjectionDbfs']:.2f}",
                    "positiveProjection200Dbfs": f"{metrics['200']['positiveProjectionDbfs']:.2f}",
                    "h50File": event["outputs"]["h50ContinuationPlus5"]["file"],
                    "inst3File": event["outputs"]["inst3Instrumental"]["file"],
                    "mixtureFile": event["outputs"]["mixture"]["file"],
                    "pairedFile": event["outputs"]["pairedListening"]["file"],
                    "notes": previous.get("notes", ""),
                }
            )


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.candidates_per_song <= 0 or args.coverage_bins <= 0 or args.scan_hop_ms <= 0:
        raise ValueError("candidate, coverage-bin, and scan-hop values must be positive")
    if args.threads <= 0 or args.inference_batch_size <= 0:
        raise ValueError("threads and inference batch size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    expansion = load_manifest_records(args.expansion_manifest.resolve())
    supplement = load_manifest_records(args.supplement_manifest.resolve())
    records: list[dict[str, Any]] = []
    for source_set, values in (("expansion", expansion), ("supplement", supplement)):
        for value in values:
            record = dict(value)
            record["sourceSet"] = source_set
            record["manifestKind"] = source_set
            records.append(record)
    if len({record["slug"] for record in records}) != len(records):
        raise ValueError("Duplicate song slug across MTG/FMA manifests")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "mtg-fma-event-listening-report.json"
    model = local.load_h50_model(args.architecture_checkpoint.resolve(), args.checkpoint.resolve(), device)[0]
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    all_events: list[dict[str, Any]] = []
    song_reports: list[dict[str, Any]] = []
    try:
        for song_index, record in enumerate(records, start=1):
            slug = record["slug"]
            print(f"prepare {song_index}/{len(records)}: {record['artistName']} - {record['trackName']}", flush=True)
            source, h50, inst3, timing = ensure_full_outputs(
                record,
                args.expansion_output.resolve(),
                output_root,
                model,
                device,
                args.inference_batch_size,
                args.force,
            )
            selected = scan_candidates(
                source,
                inst3,
                h50,
                slug,
                args.scan_hop_ms,
                args.candidates_per_song,
                args.coverage_bins,
            )
            song_event_ids: list[str] = []
            for event in selected:
                event_id = f"{record['sourceSet']}-{safe_name(slug)}-{int(event['songCandidateIndex']):02d}"
                start = int(event["snippetStartSamples"])
                end = int(event["snippetEndSamples"])
                prefix = f"{event_id}-{int(round(event['centerSeconds'] * 1000)):09d}ms"
                outputs = {
                    "mixture": write_flac(output_root / "events" / "mixture" / f"{prefix}.flac", source[start:end]),
                    "h50ContinuationPlus5": write_flac(
                        output_root / "events" / "h50ContinuationPlus5" / f"{prefix}.flac", h50[start:end]
                    ),
                    "inst3Instrumental": write_flac(
                        output_root / "events" / "inst3Instrumental" / f"{prefix}.flac", inst3[start:end]
                    ),
                }
                outputs["pairedListening"] = write_flac(
                    output_root / "events" / "pairedListening" / f"{len(all_events) + 1:04d}-{event_id}.flac",
                    make_listening_pair(h50[start:end], inst3[start:end]),
                )
                event_report = {
                    "eventId": event_id,
                    "sourceSet": record["sourceSet"],
                    "role": record.get("role"),
                    "source": record["source"],
                    "sourceId": record["sourceId"],
                    "artistName": record["artistName"],
                    "trackName": record["trackName"],
                    "slug": slug,
                    "license": record["license"],
                    "licenseUrl": record["licenseUrl"],
                    "sourceUrl": record["sourceUrl"],
                    "downloadUrl": record["downloadUrl"],
                    "rawSha256": record["download"]["sha256"],
                    "centerSamples": int(event["centerSamples"]),
                    "centerSeconds": float(event["centerSeconds"]),
                    "snippetStartSamples": start,
                    "snippetEndSamples": end,
                    "snippetDurationSeconds": (end - start) / SAMPLE_RATE,
                    "coverageFraction": event["coverageFraction"],
                    "eventScoreDbfs": event["eventScoreDbfs"],
                    "metrics": event["metrics"],
                    "outputs": outputs,
                    "humanReview": {
                        "mark": None,
                        "notes": "",
                    },
                }
                all_events.append(event_report)
                song_event_ids.append(event_id)
            song_reports.append(
                {
                    "sourceSet": record["sourceSet"],
                    "role": record.get("role"),
                    "source": record["source"],
                    "sourceId": record["sourceId"],
                    "artistName": record["artistName"],
                    "trackName": record["trackName"],
                    "slug": slug,
                    "durationSeconds": source.shape[0] / SAMPLE_RATE,
                    "candidateCount": len(selected),
                    "eventIds": song_event_ids,
                    "render": timing,
                }
            )
            json_write(
                report_path,
                {
                    "schema": "local-inst3-mtg-fma-event-listening@1",
                    "status": "running",
                    "events": all_events,
                    "songs": song_reports,
                },
            )
            del source, h50, inst3
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "schema": "local-inst3-mtg-fma-event-listening@1",
        "status": "completed",
        "objective": "Human screening of local MTG-Jamendo/FMA short vocal-like leaks before assigning songs to training or holdout.",
        "sources": {
            "expansionManifest": {"file": str(args.expansion_manifest.resolve()), "sha256": sha256_file(args.expansion_manifest.resolve()), "songCount": len(expansion)},
            "supplementManifest": {"file": str(args.supplement_manifest.resolve()), "sha256": sha256_file(args.supplement_manifest.resolve()), "songCount": len(supplement)},
        },
        "model": {
            "h50ContinuationPlus5": {"file": str(args.checkpoint.resolve()), "sha256": sha256_file(args.checkpoint.resolve()), "assembly": "continuous-context-overlap-save"},
            "architecture": {"file": str(args.architecture_checkpoint.resolve()), "sha256": sha256_file(args.architecture_checkpoint.resolve())},
            "reference": "native Inst 3 instrumental output",
        },
        "selection": {
            "candidatesPerSong": args.candidates_per_song,
            "coverageBins": args.coverage_bins,
            "scanHopMs": args.scan_hop_ms,
            "snippetDurationSeconds": 2.0,
            "pairedDurationSeconds": 4.3,
            "pairedOrder": "H50-continuation+5, 300 ms silence, Inst 3",
            "edgePolicy": "only full two-second snippets; no zero padding or song-edge candidates",
            "activeFloorDbfs": ACTIVE_FLOOR_DBFS,
            "eventRanking": "0.75 * max local positive projection + 0.25 * max local raw miss, with temporal coverage bins and non-maximum suppression",
            "eventResolutionsMs": list(EVENT_MS),
        },
        "songs": song_reports,
        "events": all_events,
        "review": {
            "status": "awaiting-human-listening",
            "markKey": {
                "R": "需要进一步消除人声、和声、念白或人声效果",
                "K": "当前结果即可，保持现状",
                "I": "H50 相较 Inst 3 额外移除了应保留的非人声内容",
                "blank": "尚未试听",
            },
            "instructions": "Compare the pairedListening file for each eventId (H50, 300 ms silence, Inst 3), then enter exactly R, K, or I in the mark column. Use notes only for details.",
            "csv": str((output_root / "human-review-template.csv").resolve()),
            "keyFile": str((output_root / "human-review-key.txt").resolve()),
        },
        "license": {
            "scope": "local non-commercial research only",
            "audio": "Do not redistribute source recordings or derived teacher caches; preserve per-track attribution and license URL.",
            "weights": "Do not publish derived weights or pseudo-targets without separate rights review.",
        },
    }
    json_write(report_path, report)
    write_review_csv(output_root / "human-review-template.csv", all_events)
    (output_root / "human-review-key.txt").write_text(
        "MTG/FMA event listening mark key\n"
        "\n"
        "R = needs further vocal/harmony/vocal-effect removal\n"
        "K = current result is acceptable; keep as-is\n"
        "I = H50 removed extra non-vocal content that should be retained\n"
        "blank = not reviewed yet\n"
        "\n"
        "The pairedListening file is H50 (2 s), 300 ms silence, Inst 3 (2 s).\n"
        "Compare the paired file for each eventId and enter one letter in the\n"
        "CSV mark column.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"status": report["status"], "songs": len(song_reports), "events": len(all_events), "report": str(report_path)},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
