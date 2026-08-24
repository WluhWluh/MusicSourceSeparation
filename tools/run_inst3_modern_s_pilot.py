#!/usr/bin/env python3
"""Run an equal-budget S-focused modern-song pilot.

S is a perceptual severity subset of R.  This runner keeps the Inst 3 target
and C1 loss contract unchanged, and changes only the external sampling pool:
S0 is a MUSDB-only control; S1 adds 32 cluster-deduplicated S windows from 16
artist/song-disjoint training songs, repeated to 64 external records per pass.
Twelve S-containing songs are held out by a deterministic category-stratified
split before event selection.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import platform
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

import evaluate_inst3_continuous_baseline as continuous
import render_inst3_mtg_fma_event_listening as event_tools
import render_inst3_objective_listening as listening
import run_inst3_distill_pilot as pilot
import run_inst3_mtg_fma_c1 as c1
import run_inst3_vr_continuous_topk_local as local
from tfc_tdf_short_window import ShortWindowContract


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTED_MANIFEST = ROOT / "data" / "modern-song-inst3-event-listening" / "selected-source-manifest.json"
DEFAULT_REVIEW_CSV = ROOT / "data" / "modern-song-inst3-event-listening" / "human-review-template.csv"
DEFAULT_EVENT_REPORT = ROOT / "data" / "modern-song-inst3-event-listening" / "event-listening-report.json"
DEFAULT_MUSDB_CACHE = ROOT / "data" / "inst3-mtg-fma-c1" / "unused-musdb-cache-placeholder"
DEFAULT_MUSDB_CACHE = ROOT / "data" / "inst3-mtg-fma-c1" / ".." / "musdb18-inst3-vr-continuous-topk-local"
DEFAULT_MUSDB_MANIFEST = ROOT / "data" / "musdb18-inst3-oracle" / "musdb18-inst3-oracle-manifest.json"
DEFAULT_ORACLE_ROOT = ROOT / "data" / "musdb18-inst3-oracle"
DEFAULT_ARCHITECTURE = pilot.DEFAULT_CHECKPOINT
DEFAULT_SOURCE = ROOT / "data" / "musdb18-inst3-vr-continuation" / "runs" / "H50-continuation-plus5" / "step-1600.pt"
DEFAULT_BASELINE_REPORT = ROOT / "data" / "musdb18-inst3-continuous-baseline-evaluation" / "continuous-baseline-report.json"
DEFAULT_OUTPUT = ROOT / "data" / "modern-song-s-pilot"

SAMPLE_RATE = 44_100
SOURCE_STEP = 1600
PASSES = 5
MUSDB_RECORDS_PER_PASS = 640
EXTRA_RECORDS_PER_PASS = 64
RECORDS_PER_PASS = MUSDB_RECORDS_PER_PASS + EXTRA_RECORDS_PER_PASS
EVENTS_PER_SONG = 2
EXTERNAL_REPEAT = 2
CLUSTER_GAP_SECONDS = 3.0
CORE_MS = 100
GUARD_MS = 25
ARMS = ("S0-control", "S1-focused")
CHECKPOINT_FORMAT = "local-inst3-modern-s-pilot-checkpoint@1"

TRAIN_QUOTAS = {
    "hiphop-rnb": 5,
    "pop-synth": 5,
    "latin-modern": 2,
    "singer-songwriter": 2,
    "dance-electronic": 1,
    "pop-rock": 1,
}
HOLDOUT_QUOTAS = {
    "hiphop-rnb": 3,
    "pop-synth": 3,
    "latin-modern": 2,
    "singer-songwriter": 2,
    "dance-electronic": 1,
    "pop-rock": 1,
}


def configure_base_runner() -> None:
    """Reuse the tested C1 trainer with S-specific schedule constants."""
    c1.ARMS = ARMS
    c1.CHECKPOINT_FORMAT = CHECKPOINT_FORMAT
    c1.EVENTS_PER_EXTERNAL_SONG = EVENTS_PER_SONG
    c1.EXTERNAL_REPEAT = EXTERNAL_REPEAT
    c1.EXTRA_RECORDS_PER_PASS = EXTRA_RECORDS_PER_PASS
    c1.RECORDS_PER_PASS = RECORDS_PER_PASS
    c1.MUSDB_RECORDS_PER_PASS = MUSDB_RECORDS_PER_PASS
    c1.SOURCE_STEP = SOURCE_STEP


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-manifest", type=Path, default=DEFAULT_SELECTED_MANIFEST)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--event-report", type=Path, default=DEFAULT_EVENT_REPORT)
    parser.add_argument("--musdb-cache-root", type=Path, default=DEFAULT_MUSDB_CACHE)
    parser.add_argument("--musdb-manifest", type=Path, default=DEFAULT_MUSDB_MANIFEST)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--passes", type=int, default=PASSES)
    parser.add_argument("--milestones", default="1,3,5")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--anchor-beta", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=891)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument("--state-interval", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--force-prep", action="store_true")
    parser.add_argument("--force-listening", action="store_true")
    parser.add_argument("--skip-musdb-evaluation", action="store_true")
    parser.add_argument("--skip-modern-evaluation", action="store_true")
    parser.add_argument("--skip-listening", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-updates", type=int, default=4)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args(argv)


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    return pilot.sha256_file(path)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    return {"file": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def load_review(path: Path) -> dict[str, dict[str, Any]]:
    with path.resolve().open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        mark = (row.get("mark") or "").strip().upper()
        if mark not in {"S", "R", "K", "I"}:
            raise ValueError(f"Invalid mark {mark!r} for {row.get('eventId')}")
        row["mark"] = mark
        row["eventScoreDbfs"] = float(row["eventScoreDbfs"])
        result[row["eventId"]] = row
    if len(result) != 912:
        raise ValueError(f"Expected 912 modern event labels, got {len(result)}")
    return result


def load_selected(path: Path) -> dict[str, dict[str, Any]]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if value.get("status") != "frozen" or int(value.get("songCount", -1)) != 57:
        raise ValueError("Expected frozen 57-song selected manifest")
    return {record["slug"]: record for record in value["records"]}


def load_events(path: Path, review: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in value.get("events", []):
        copy = dict(event)
        copy["mark"] = review[event["eventId"]]["mark"]
        result[event["slug"]].append(copy)
    if sum(len(items) for items in result.values()) != 912:
        raise ValueError("Modern event report does not contain all 912 events")
    return result


def stable_key(category: str, slug: str) -> str:
    return hashlib.sha256(f"{category}\0{slug}".encode()).hexdigest()


def s_clusters(events: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    selected = sorted((event for event in events if event["mark"] == "S"), key=lambda event: event["centerSeconds"])
    groups: list[list[dict[str, Any]]] = []
    for event in selected:
        if not groups or float(event["centerSeconds"]) - float(groups[-1][-1]["centerSeconds"]) >= CLUSTER_GAP_SECONDS:
            groups.append([event])
        else:
            groups[-1].append(event)
    return groups


def build_song_split(
    records: dict[str, dict[str, Any]], events: dict[str, list[dict[str, Any]]]
) -> tuple[list[str], list[str], dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for slug, record in records.items():
        groups = s_clusters(events.get(slug, []))
        if len(groups) < EVENTS_PER_SONG:
            continue
        candidates[slug] = {
            "slug": slug,
            "category": record["category"],
            "artistName": record["artistName"],
            "trackName": record["trackName"],
            "sEventCount": sum(len(group) for group in groups),
            "sClusterCount": len(groups),
            "stableKey": stable_key(record["category"], slug),
        }
    train: list[str] = []
    holdout: list[str] = []
    for category, quota in TRAIN_QUOTAS.items():
        values = sorted((item for item in candidates.values() if item["category"] == category), key=lambda item: item["stableKey"])
        hold_quota = HOLDOUT_QUOTAS[category]
        if len(values) < quota + hold_quota:
            raise ValueError(f"Not enough S-rich songs in {category}: {len(values)}")
        train.extend(item["slug"] for item in values[:quota])
        holdout.extend(item["slug"] for item in values[quota : quota + hold_quota])
    if set(train) & set(holdout) or len(train) != 16 or len(holdout) != 12:
        raise AssertionError("Invalid S song split")
    payload = {
        "algorithm": "category quota, stable SHA-256 order, S cluster count >= 2",
        "trainQuotas": TRAIN_QUOTAS,
        "holdoutQuotas": HOLDOUT_QUOTAS,
        "trainSongs": sorted(train),
        "holdoutSongs": sorted(holdout),
        "candidateSongCount": len(candidates),
        "candidateSongs": sorted(candidates.values(), key=lambda item: item["slug"]),
    }
    return train, holdout, payload


def choose_s_events(slug: str, events: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    groups = s_clusters(events[slug])
    representatives = []
    for group in groups:
        representatives.append(max(group, key=lambda item: (float(item["eventScoreDbfs"]), -float(item["centerSeconds"]))))
    representatives.sort(key=lambda item: (-float(item["eventScoreDbfs"]), float(item["centerSeconds"])))
    if len(representatives) < EVENTS_PER_SONG:
        raise ValueError(f"Not enough S clusters for {slug}")
    return representatives[:EVENTS_PER_SONG]


def load_flac(path: Path) -> np.ndarray:
    audio, rate = event_tools.listening.load_audio(path)
    if rate != SAMPLE_RATE or audio.shape[1] != 2 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid cached FLAC {path}")
    return np.ascontiguousarray(audio, dtype=np.float32)


def external_cache(
    args: argparse.Namespace,
    records: dict[str, dict[str, Any]],
    events: dict[str, list[dict[str, Any]]],
    train_songs: list[str],
    contract: ShortWindowContract,
    output_root: Path,
) -> tuple[dict[str, Path], dict[str, Any]]:
    root = output_root / "cache" / "external"
    paths: dict[str, Path] = {}
    selections: dict[str, Any] = {}
    for slug in sorted(train_songs):
        record = records[slug]
        selected = choose_s_events(slug, events)
        key = f"external::{slug}"
        path = root / f"{slug}.npz"
        metadata_path = root / f"{slug}.json"
        source_path = Path(record["download"]["file"])
        teacher_path = output_root.parent / "modern-song-inst3-event-listening" / "full-song" / "inst3Instrumental" / f"{record['order']:03d}-{slug}.flac"
        h50_path = output_root.parent / "modern-song-inst3-event-listening" / "full-song" / "h50ContinuationPlus5" / f"{record['order']:03d}-{slug}.flac"
        source = event_tools.load_audio(source_path)
        teacher = load_flac(teacher_path)
        h50 = load_flac(h50_path)
        if source.shape != teacher.shape or source.shape != h50.shape:
            raise ValueError(f"Source/cache shape mismatch {slug}")
        contract_payload = {
            "schema": "local-modern-s-external-cache@1",
            "rawSha256": record["download"]["sha256"],
            "teacherSha256": sha256_file(teacher_path),
            "h50Sha256": sha256_file(h50_path),
            "sourceCheckpointSha256": sha256_file(args.source_checkpoint.resolve()),
            "reviewSha256": sha256_file(args.review_csv.resolve()),
            "eventIds": [item["eventId"] for item in selected],
            "coreMs": c1.CORE_MS,
            "guardMs": c1.GUARD_MS,
            "contract": contract.as_dict(assembly="continuous-context-overlap-save"),
        }
        contract_id = canonical_sha256(contract_payload)
        reused = False
        if not args.force_prep and path.is_file() and metadata_path.is_file():
            try:
                previous = json.loads(metadata_path.read_text(encoding="utf-8"))
                with np.load(path) as values:
                    reused = previous.get("contractId") == contract_id and values["inputSpec"].shape[0] == EVENTS_PER_SONG
            except (OSError, KeyError, ValueError, json.JSONDecodeError):
                reused = False
        if not reused:
            inputs, targets, anchors, masks = [], [], [], []
            event_meta = []
            for event in selected:
                center = int(event["centerSamples"])
                start = center - contract.useful_samples // 2
                end = start + contract.useful_samples
                if start < 0 or end > source.shape[0]:
                    raise ValueError(f"Event context out of bounds {event['eventId']}")
                inputs.append(local.stft_centered(local.assemble_input(source, start, contract.useful_samples, contract, mode="continuous"), contract)[0])
                targets.append(np.ascontiguousarray(source[start:end] - teacher[start:end], dtype=np.float32))
                anchors.append(np.ascontiguousarray(source[start:end] - h50[start:end], dtype=np.float32))
                masks.append(c1.event_mask(contract.useful_samples, center - start, c1.CORE_MS, c1.GUARD_MS))
                event_meta.append({"eventId": event["eventId"], "centerSamples": center, "startSamples": start, "eventScoreDbfs": event["eventScoreDbfs"]})
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".tmp.npz")
            np.savez_compressed(temporary, inputSpec=np.stack(inputs).astype(np.float32), targetAudio=np.stack(targets).astype(np.float32), anchorAudio=np.stack(anchors).astype(np.float32), eventMask=np.stack(masks).astype(np.float32))
            temporary.replace(path)
            metadata = {"schema": contract_payload["schema"], "contractId": contract_id, "contract": contract_payload, "slug": slug, "trackName": record["trackName"], "recordCount": EVENTS_PER_SONG, "selectedEvents": event_meta, "cache": checkpoint_metadata(path)}
            json_write(metadata_path, metadata)
        else:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        paths[key] = path
        selections[key] = metadata
        print(f"S cache {'reuse' if reused else 'write'}: {record['artistName']} - {record['trackName']}", flush=True)
        del source, teacher, h50
        gc.collect()
    payload = {"schema": "local-modern-s-selection@1", "songs": selections, "selectionSha256": canonical_sha256(selections)}
    json_write(output_root / "external-selection.json", payload)
    return paths, payload


def rms(value: np.ndarray) -> float:
    return math.sqrt(float(np.mean(value.astype(np.float64) ** 2)))


def db_ratio(value: float, reference: float) -> float:
    return 20.0 * math.log10(max(value, 1.0e-30) / max(reference, 1.0e-30))


def movement_coefficient(base: np.ndarray, target: np.ndarray, candidate: np.ndarray) -> float:
    desired = target.astype(np.float64) - base.astype(np.float64)
    movement = candidate.astype(np.float64) - base.astype(np.float64)
    return float(np.sum(movement * desired) / max(float(np.sum(desired * desired)), 1.0e-30))


def evaluate_modern_holdout(
    args: argparse.Namespace,
    records: dict[str, dict[str, Any]],
    events: dict[str, list[dict[str, Any]]],
    holdout_songs: list[str],
    checkpoint_paths: dict[str, Path],
    contract: ShortWindowContract,
    device: torch.device,
) -> dict[str, Any]:
    review = {event["eventId"]: event["mark"] for slug in events for event in events[slug]}
    models = {name: local.load_h50_model(args.checkpoint.resolve(), path.resolve(), device)[0] for name, path in checkpoint_paths.items() if name != "Source-H50-continuation+5"}
    rows = []
    try:
        for index, slug in enumerate(sorted(holdout_songs), start=1):
            record = records[slug]
            source = event_tools.load_audio(Path(record["download"]["file"]))
            teacher = load_flac(ROOT / "data" / "modern-song-inst3-event-listening" / "full-song" / "inst3Instrumental" / f"{record['order']:03d}-{slug}.flac")
            base = load_flac(ROOT / "data" / "modern-song-inst3-event-listening" / "full-song" / "h50ContinuationPlus5" / f"{record['order']:03d}-{slug}.flac")
            candidates = {"Source-H50-continuation+5": base}
            for name, model in models.items():
                residual, _timing = continuous.render_student(model, source, contract, device, args.inference_batch_size)
                candidates[name] = np.ascontiguousarray(source - residual, dtype=np.float32)
            for event in events[slug]:
                center = int(event["centerSamples"])
                for name, candidate in candidates.items():
                    values = {}
                    for milliseconds in (50, 100, 200):
                        length = round(SAMPLE_RATE * milliseconds / 1000.0)
                        start = center - length // 2
                        end = start + length
                        values[str(milliseconds)] = db_ratio(rms(candidate[start:end] - teacher[start:end]), rms(base[start:end] - teacher[start:end]))
                    rows.append({"eventId": event["eventId"], "slug": slug, "trackName": record["trackName"], "category": record["category"], "mark": event["mark"], "variant": name, "localTargetErrorDeltaDbVsSource": values})
            print(f"modern holdout {index}/{len(holdout_songs)}: {record['artistName']} - {record['trackName']}", flush=True)
            del source, teacher, base, candidates
            gc.collect()
    finally:
        for model in models.values():
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    aggregate = {}
    for variant in checkpoint_paths:
        aggregate[variant] = {}
        for mark in ("S", "R", "K", "I"):
            values = [row for row in rows if row["variant"] == variant and row["mark"] == mark]
            aggregate[variant][mark] = {str(ms): {"count": len(values), "meanDeltaDb": statistics.fmean(row["localTargetErrorDeltaDbVsSource"][str(ms)] for row in values), "medianDeltaDb": statistics.median(row["localTargetErrorDeltaDbVsSource"][str(ms)] for row in values), "improved": sum(row["localTargetErrorDeltaDbVsSource"][str(ms)] < 0 for row in values)} for ms in (50, 100, 200)} if values else {}
    return {"holdoutSongs": holdout_songs, "rows": rows, "aggregate": aggregate}


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    configure_base_runner()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.batch_size <= 0 or c1.RECORDS_PER_PASS % args.batch_size:
        raise ValueError("invalid batch size")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for path in (args.selected_manifest, args.review_csv, args.event_report, args.musdb_cache_root / "selection.json", args.source_checkpoint, args.checkpoint, args.baseline_report):
        if not path.resolve().is_file():
            raise FileNotFoundError(path)
    review = load_review(args.review_csv)
    records = load_selected(args.selected_manifest)
    events = load_events(args.event_report, review)
    train_songs, holdout_songs, split = build_song_split(records, events)
    json_write(args.output_root / "song-split.json", split)
    contract = ShortWindowContract(num_frames=128, left_context_hops=5, right_context_hops=5)
    external_paths, external_selection = external_cache(args, records, events, train_songs, contract, args.output_root)
    musdb_paths, records_per_song, musdb_selection = c1.musdb_cache_paths(args.musdb_cache_root)
    schedules = c1.build_schedules(list(musdb_paths), records_per_song, list(external_paths), args.passes, args.seed)
    all_paths = {**musdb_paths, **external_paths}
    common_contract = {
        "schema": "local-modern-s-pilot@1",
        "sourceCheckpoint": c1.checkpoint_metadata(args.source_checkpoint.resolve()),
        "songSplitSha256": sha256_file(args.output_root / "song-split.json"),
        "externalSelectionSha256": external_selection["selectionSha256"],
        "musdbSelectionSha256": c1.canonical_sha256(musdb_selection),
        "recordsPerPass": c1.RECORDS_PER_PASS,
        "musdbRecordsPerPass": c1.MUSDB_RECORDS_PER_PASS,
        "externalRecordsPerPass": c1.EXTRA_RECORDS_PER_PASS,
        "externalFraction": c1.EXTRA_RECORDS_PER_PASS / c1.RECORDS_PER_PASS,
        "passes": args.passes,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "anchorBeta": args.anchor_beta,
        "seed": args.seed,
        "coreMs": c1.CORE_MS,
        "guardMs": c1.GUARD_MS,
        "clusterGapSeconds": CLUSTER_GAP_SECONDS,
        "assembly": "continuous-context-overlap-save",
        "officialFinalTestUsed": False,
    }
    milestones = c1.parse_milestones(args.milestones, args.passes)
    if args.smoke_only:
        smoke = {arm: c1.train_arm(arm, all_paths, schedules[arm], args, device, milestones, common_contract, args.smoke_updates) for arm in ARMS}
        report = {"schema": "local-modern-s-pilot@1", "status": "smoke-completed", "songSplit": split, "contract": common_contract, "smoke": smoke}
        json_write(args.output_root / "reports" / "s-smoke.json", report)
        print(json.dumps({"status": report["status"], "arms": list(smoke)}, indent=2))
        return 0
    training = {arm: c1.train_arm(arm, all_paths, schedules[arm], args, device, milestones, common_contract) for arm in ARMS}
    final_paths = {arm: Path(training[arm]["milestoneCheckpoints"][str(args.passes)]["file"]) for arm in ARMS}
    checkpoint_paths = {"Source-H50-continuation+5": args.source_checkpoint.resolve(), **final_paths}
    musdb_eval = None
    if not args.skip_musdb_evaluation:
        manifest = json.loads(args.musdb_manifest.resolve().read_text(encoding="utf-8"))
        entries = sorted([entry for entry in manifest["entries"] if entry["role"] in {"calibration", "internal-test"}], key=lambda item: (item["role"], item["member"]))
        eval_args = argparse.Namespace(baseline_report=args.baseline_report.resolve(), output_root=args.output_root, checkpoint=args.checkpoint.resolve(), oracle_root=args.oracle_root.resolve(), inference_batch_size=args.inference_batch_size)
        musdb_eval = local.evaluate_trained_models(eval_args, contract, checkpoint_paths, entries, device)
    modern_eval = None
    if not args.skip_modern_evaluation:
        modern_eval = evaluate_modern_holdout(args, records, events, holdout_songs, checkpoint_paths, contract, device)
        json_write(args.output_root / "reports" / "modern-holdout-evaluation.json", modern_eval)
    listening_report = None
    private_analysis = None
    if not args.skip_listening:
        listening_args = argparse.Namespace(samples_root=ROOT / "data" / "samples", output_root=args.output_root, checkpoint=args.checkpoint.resolve(), force=args.force_listening, inference_batch_size=args.inference_batch_size)
        listening_report = local.render_listening(listening_args, contract, checkpoint_paths, device)
        private_analysis = c1.analyze_private(listening_report, args.baseline_report)
        json_write(args.output_root / "reports" / "private-inst3-analysis.json", private_analysis)
    report = {
        "schema": "local-modern-s-pilot@1",
        "status": "completed",
        "songSplit": split,
        "contract": common_contract,
        "schedules": {arm: {"recordCount": len(schedules[arm]), "sha256": c1.canonical_sha256(schedules[arm]), "sourceCounts": dict(Counter("external" if key.startswith("external::") else "musdb" for key, _ in schedules[arm]))} for arm in ARMS},
        "training": training,
        "checkpoints": {name: c1.checkpoint_metadata(path) for name, path in checkpoint_paths.items()},
        "musdbEvaluation": musdb_eval,
        "modernHoldoutEvaluation": modern_eval,
        "listening": listening_report,
        "privateInst3Analysis": private_analysis,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "torchCuda": torch.version.cuda, "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "runner": c1.checkpoint_metadata(Path(__file__).resolve())},
        "licenseDisposition": {"scope": "local non-commercial research only", "originals": "source audio and derived caches remain local", "weights": "teacher-derived checkpoints remain local pending rights review"},
    }
    report_path = args.output_root / "reports" / "s-report.json"
    json_write(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path), "checkpoints": {name: str(path) for name, path in final_paths.items()}, "listening": 0 if listening_report is None else listening_report.get("outputCount", 0)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
