#!/usr/bin/env python3
"""Continue the residual-vocals model on all reviewed FMA S/R events.

The input pool is assembled from every valid batch below
``data/fma-s-leakage-survey``.  Selection is event-level: K and I rows are
excluded, while S/R rows remain eligible even when another event in the same
song was marked I.  The run uses continuous 128-frame overlap-save windows,
the Inst 3 residual target on a 100 ms core plus a 25 ms guard, and the source
model output as an anchor outside the event mask.

This is a local research continuation.  The event metrics are measured on the
same reviewed FMA pool used for training and are therefore a stopping signal,
not an unbiased generalization estimate.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import platform
import shutil
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

import evaluate_inst3_continuous_baseline as continuous
import render_inst3_mtg_fma_event_listening as external
import run_inst3_distill_pilot as pilot
import run_inst3_distill_stability_sweep as sweep
import run_inst3_mtg_fma_c1 as c1
import run_inst3_vr_continuous_topk_local as local
import run_inst3_vr_continuation as continuation
from tfc_tdf_short_window import ShortWindowContract


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SURVEY_ROOT = ROOT / "data" / "fma-s-leakage-survey"
DEFAULT_SOURCE = (
    ROOT
    / "data"
    / "modern-song-fma-sr-event-only-continuation"
    / "extensions"
    / "from-step-5120"
    / "runs"
    / "step-5648.pt"
)
DEFAULT_ARCHITECTURE = pilot.DEFAULT_CHECKPOINT
DEFAULT_OUTPUT = ROOT / "data" / "modern-song-fma-s-leakage-survey-continuation"

SAMPLE_RATE = 44_100
SOURCE_STEP = 5_648
RECORDS_PER_PASS = 704
BATCH_SIZE = 4
CORE_MS = 100
GUARD_MS = 25
DEFAULT_MAX_PASSES = 10
DEFAULT_MIN_PASSES = 3
DEFAULT_PATIENCE = 2
DEFAULT_MIN_IMPROVEMENT_DB = 0.10
CONTRACT = ShortWindowContract(num_frames=128, left_context_hops=5, right_context_hops=5)
CHECKPOINT_FORMAT = "local-inst3-fma-s-leakage-survey-continuation-checkpoint@1"
SCHEMA = "local-inst3-fma-s-leakage-survey-continuation@1"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey-root", type=Path, default=DEFAULT_SURVEY_ROOT)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--architecture-checkpoint", type=Path, default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-passes", type=int, default=DEFAULT_MAX_PASSES)
    parser.add_argument("--min-passes", type=int, default=DEFAULT_MIN_PASSES)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-improvement-db", type=float, default=DEFAULT_MIN_IMPROVEMENT_DB)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--anchor-beta", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=891)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-updates", type=int, default=8)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args(argv)


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"file": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in ".-_" else "-" for char in value).strip("-") or "song"


def cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            if isinstance(value, torch.Tensor) and value.device != device:
                state[key] = value.to(device=device)


def _valid_batch_paths(batch: Path) -> tuple[Path, Path] | None:
    report = batch / "survey-report.json"
    review = batch / "retained" / "human-review-template.csv"
    manifest = batch / "source-manifest.json"
    if not (report.is_file() and review.is_file() and manifest.is_file()):
        return None
    return report, review


def load_pool(survey_root: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Read all valid batch CSVs, filtering marks per event rather than song."""
    survey_root = survey_root.resolve()
    groups: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    batch_audit: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()

    for batch in sorted(survey_root.glob("batch-*")):
        paths = _valid_batch_paths(batch)
        if paths is None:
            batch_audit.append({"batch": batch.name, "status": "skipped", "reason": "missing manifest/report/review"})
            continue
        report_path, review_path = paths
        report = read_json(report_path)
        if report.get("status") != "completed":
            batch_audit.append({"batch": batch.name, "status": "skipped", "reason": "report not completed"})
            continue
        events = report.get("retainedEvents")
        songs = report.get("songs")
        if not isinstance(events, list) or not isinstance(songs, dict):
            batch_audit.append({"batch": batch.name, "status": "skipped", "reason": "invalid report schema"})
            continue

        events_by_id = {str(event.get("eventId")): event for event in events if event.get("eventId")}
        song_by_event: dict[str, tuple[str, dict[str, Any]]] = {}
        for slug, song in songs.items():
            if not isinstance(song, dict):
                continue
            for event_id in song.get("retainedEventIds", []) or []:
                event_id = str(event_id)
                if event_id in song_by_event:
                    raise ValueError(f"Event belongs to multiple songs: {event_id}")
                song_by_event[event_id] = (str(slug), song)

        total_rows = 0
        selected_rows = 0
        selected_s = 0
        selected_r = 0
        with review_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = csv.DictReader(handle)
            for row in rows:
                total_rows += 1
                mark = (row.get("mark") or "").strip().upper()
                if mark not in {"S", "R"}:
                    continue
                event_id = str(row.get("eventId") or "")
                if not event_id or event_id in seen_event_ids:
                    raise ValueError(f"Duplicate or empty selected event ID: {event_id!r}")
                event = events_by_id.get(event_id)
                song_info = song_by_event.get(event_id)
                if event is None or song_info is None:
                    raise ValueError(f"Selected event is absent from report: {event_id}")
                slug, song = song_info
                source_info = song.get("source") or {}
                outputs = song.get("retainedOutputs") or {}
                source_path = Path(str(source_info.get("rawFile", ""))).resolve()
                teacher_path = Path(str(outputs.get("inst3Instrumental", ""))).resolve()
                h50_path = Path(str(outputs.get("h50Residual", ""))).resolve()
                if not source_path.is_file() or not teacher_path.is_file() or not h50_path.is_file():
                    raise FileNotFoundError(f"Missing source/teacher/H50 file for {event_id}")
                center_seconds = float(event.get("centerSeconds", row.get("centerSeconds", 0.0)))
                center_samples = int(event.get("centerSamples", round(center_seconds * SAMPLE_RATE)))
                key = f"{batch.name}::{slug}"
                group = groups.setdefault(
                    key,
                    {
                        "key": key,
                        "sourceBatch": batch.name,
                        "slug": slug,
                        "sourceOrder": int(source_info.get("sourceOrder", row.get("sourceOrder", 0))),
                        "artistName": str(source_info.get("artistName", row.get("artistName", ""))),
                        "trackName": str(source_info.get("trackName", row.get("trackName", ""))),
                        "category": str(source_info.get("category", row.get("category", ""))),
                        "languageCode": str(source_info.get("languageCode", "")),
                        "sourcePath": str(source_path),
                        "teacherPath": str(teacher_path),
                        "h50Path": str(h50_path),
                        "sourceSha256": str(source_info.get("rawSha256", "")).lower(),
                        "license": str(source_info.get("license", row.get("license", ""))),
                        "licenseUrl": str(source_info.get("licenseUrl", row.get("licenseUrl", ""))),
                        "events": [],
                    },
                )
                group["events"].append(
                    {
                        "eventId": event_id,
                        "mark": mark,
                        "centerSamples": center_samples,
                        "centerSeconds": center_seconds,
                        "eventScoreDbfs": float(event.get("eventScoreDbfs", row.get("eventScoreDbfs", -240.0))),
                        "sProxyDbfs": float(event.get("sProxyDbfs", row.get("sProxyDbfs", -240.0))),
                        "serial": int(event.get("serial", row.get("serial", 0))),
                        "category": str(event.get("category", row.get("category", ""))),
                        "batch": batch.name,
                        "reportPath": str(report_path.resolve()),
                        "reviewPath": str(review_path.resolve()),
                    }
                )
                seen_event_ids.add(event_id)
                selected_rows += 1
                if mark == "S":
                    selected_s += 1
                else:
                    selected_r += 1

        batch_audit.append(
            {
                "batch": batch.name,
                "status": "valid",
                "totalReviewRows": total_rows,
                "selectedRows": selected_rows,
                "sRows": selected_s,
                "rRows": selected_r,
                "report": str(report_path.resolve()),
                "review": str(review_path.resolve()),
            }
        )

    for group in groups.values():
        group["events"].sort(key=lambda event: (int(event["centerSamples"]), event["eventId"]))
        for index, event in enumerate(group["events"]):
            records.append({"key": group["key"], "index": index, **event})
    records.sort(key=lambda record: (record["key"], int(record["index"])))
    s_records = [record for record in records if record["mark"] == "S"]
    r_records = [record for record in records if record["mark"] == "R"]
    if not s_records or not r_records:
        raise ValueError(f"Expected both S and R events, got S={len(s_records)} R={len(r_records)}")

    selection: dict[str, Any] = {
        "schema": SCHEMA + ".selection",
        "surveyRoot": str(survey_root),
        "validBatchNames": [item["batch"] for item in batch_audit if item["status"] == "valid"],
        "batchAudit": batch_audit,
        "selectedSongCount": len(groups),
        "selectedEventCount": len(records),
        "sRecordCount": len(s_records),
        "rRecordCount": len(r_records),
        "songKeys": sorted(groups),
        "eventIds": [record["eventId"] for record in records],
        "markSemantics": {
            "S": "especially conspicuous residual vocal; subset of R",
            "R": "further vocal/harmony/spoken/vocal-effect removal desired",
            "K": "excluded per event",
            "I": "excluded per event; does not exclude S/R rows in the same song",
        },
        "trainingBoundary": "all S/R rows from all valid batches; no MUSDB records",
    }
    selection["selectionSha256"] = canonical_sha256(selection)
    return {key: groups[key] for key in sorted(groups)}, records, selection


def validate_cache(path: Path, expected_records: int) -> None:
    required = ("inputSpec", "targetAudio", "anchorAudio", "eventMask")
    with np.load(path) as values:
        if any(name not in values for name in required):
            raise ValueError(f"Missing cache arrays in {path}")
        if values["inputSpec"].shape != (expected_records, 4, 1025, 128):
            raise ValueError(f"Unexpected input shape in {path}: {values['inputSpec'].shape}")
        if values["targetAudio"].shape != (expected_records, CONTRACT.useful_samples, 2):
            raise ValueError(f"Unexpected target shape in {path}: {values['targetAudio'].shape}")
        if values["anchorAudio"].shape != values["targetAudio"].shape:
            raise ValueError(f"Anchor shape mismatch in {path}")
        if values["eventMask"].shape != (expected_records, CONTRACT.useful_samples):
            raise ValueError(f"Unexpected event mask shape in {path}: {values['eventMask'].shape}")
        if any(not np.isfinite(values[name]).all() for name in required):
            raise ValueError(f"Non-finite cache value in {path}")


def load_source_model(
    args: argparse.Namespace, device: torch.device
) -> tuple[torch.nn.Module, torch.optim.Optimizer, dict[str, Any], dict[str, Any]]:
    source_path = args.source_checkpoint.resolve()
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if payload.get("format") != "local-inst3-fma-sr-event-only-continuation-checkpoint@1":
        raise ValueError(f"Unexpected source format: {payload.get('format')}")
    if int(payload.get("step", -1)) != SOURCE_STEP:
        raise ValueError(f"Expected source step {SOURCE_STEP}, got {payload.get('step')}")
    if not isinstance(payload.get("stateDict"), dict) or not isinstance(payload.get("optimizerStateDict"), dict):
        raise ValueError("Source checkpoint lacks state or optimizer state")
    model, architecture = pilot.make_model(args.architecture_checkpoint.resolve(), device)
    model.load_state_dict(payload["stateDict"], strict=True)
    model.train()
    sweep.freeze_batchnorm_running_statistics(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    optimizer.load_state_dict(payload["optimizerStateDict"])
    move_optimizer_state(optimizer, device)
    for group in optimizer.param_groups:
        group["lr"] = args.learning_rate
    source = {
        "checkpoint": checkpoint_metadata(source_path),
        "architecture": architecture["checkpoint"],
        "sourceStep": int(payload["step"]),
        "sourcePass": int(payload.get("continuationPass", -1)),
        "sourceVariant": payload.get("variant"),
        "optimizerRestored": True,
    }
    return model, optimizer, source, payload


def infer_anchor(
    model: torch.nn.Module,
    input_specs: np.ndarray,
    device: torch.device,
    batch_size: int,
    window: torch.Tensor,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for begin in range(0, input_specs.shape[0], batch_size):
            tensor = torch.from_numpy(input_specs[begin : begin + batch_size]).to(device)
            full = local.torch_packed_istft(model(tensor), window)
            trim = pilot.DEFAULT_CONFIG.trim_samples
            useful = full[:, trim : trim + CONTRACT.useful_samples]
            outputs.append(np.ascontiguousarray(useful.detach().cpu().numpy(), dtype=np.float32))
    result = np.concatenate(outputs, axis=0)
    if result.shape != (input_specs.shape[0], CONTRACT.useful_samples, 2) or not np.isfinite(result).all():
        raise ValueError("Invalid inferred anchor")
    return result


def prepare_caches(
    args: argparse.Namespace,
    groups: dict[str, dict[str, Any]],
    selection: dict[str, Any],
    device: torch.device,
) -> dict[str, Path]:
    output_root = args.output_root.resolve()
    cache_root = output_root / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    reusable = not args.force_cache
    for key, group in groups.items():
        cache_path = cache_root / f"{safe_name(key)}.npz"
        metadata_path = cache_path.with_suffix(".json")
        if reusable and cache_path.is_file() and metadata_path.is_file():
            try:
                metadata = read_json(metadata_path)
                if (
                    metadata.get("selectionSha256") == selection["selectionSha256"]
                    and metadata.get("sourceCheckpointSha256") == selection["sourceCheckpoint"]["sha256"]
                ):
                    validate_cache(cache_path, len(group["events"]))
                    paths[key] = cache_path
                    continue
            except (OSError, ValueError, json.JSONDecodeError):
                pass

    missing = [key for key in groups if key not in paths]
    if not missing:
        print(json.dumps({"event": "cache-reuse", "groups": len(paths)}, sort_keys=True), flush=True)
        return paths

    anchor_model, anchor_optimizer, anchor_source, _ = load_source_model(args, device)
    del anchor_optimizer
    window = torch.hann_window(pilot.DEFAULT_CONFIG.n_fft, periodic=True, device=device)
    try:
        for number, key in enumerate(missing, start=1):
            group = groups[key]
            source_path = Path(group["sourcePath"])
            teacher_path = Path(group["teacherPath"])
            source = external.load_audio(source_path)
            teacher = external.load_audio(teacher_path)
            if source.shape != teacher.shape:
                raise ValueError(f"Source/teacher shape mismatch for {key}: {source.shape} vs {teacher.shape}")
            expected_hash = group.get("sourceSha256", "")
            if expected_hash and sha256_file(source_path).lower() != expected_hash.lower():
                raise ValueError(f"Source hash mismatch for {key}")
            input_rows: list[np.ndarray] = []
            target_rows: list[np.ndarray] = []
            mask_rows: list[np.ndarray] = []
            for event in group["events"]:
                center = int(event["centerSamples"])
                start = center - CONTRACT.useful_samples // 2
                end = start + CONTRACT.useful_samples
                if start < 0 or end > source.shape[0]:
                    raise ValueError(f"Event context out of bounds: {event['eventId']}")
                assembled = local.assemble_input(source, start, CONTRACT.useful_samples, CONTRACT, mode="continuous")
                input_rows.append(np.ascontiguousarray(local.stft_centered(assembled, CONTRACT)[0], dtype=np.float32))
                target_rows.append(np.ascontiguousarray(source[start:end] - teacher[start:end], dtype=np.float32))
                mask_rows.append(
                    np.ascontiguousarray(
                        c1.event_mask(CONTRACT.useful_samples, center - start, CORE_MS, GUARD_MS), dtype=np.float32
                    )
                )
            input_array = np.stack(input_rows).astype(np.float32)
            target_array = np.stack(target_rows).astype(np.float32)
            mask_array = np.stack(mask_rows).astype(np.float32)
            anchor_array = infer_anchor(anchor_model, input_array, device, args.inference_batch_size, window)
            cache_path = cache_root / f"{safe_name(key)}.npz"
            temporary = cache_path.with_name(cache_path.name + ".tmp.npz")
            np.savez_compressed(
                temporary,
                inputSpec=input_array,
                targetAudio=target_array,
                anchorAudio=anchor_array,
                eventMask=mask_array,
            )
            temporary.replace(cache_path)
            metadata = {
                "schema": SCHEMA + ".cache",
                "selectionSha256": selection["selectionSha256"],
                "sourceCheckpointSha256": selection["sourceCheckpoint"]["sha256"],
                "anchorCheckpoint": anchor_source["checkpoint"],
                "key": key,
                "sourceBatch": group["sourceBatch"],
                "slug": group["slug"],
                "sourcePath": str(source_path),
                "teacherPath": str(teacher_path),
                "h50Path": group["h50Path"],
                "sourceSha256": expected_hash,
                "teacherSha256": sha256_file(teacher_path),
                "recordCount": len(group["events"]),
                "events": group["events"],
                "targetSemantic": "mixture - Inst3 instrumental",
                "anchorSemantic": "step-5648 residual-vocals output",
                "eventMask": "100ms core + 25ms linear guard",
                "contract": CONTRACT.as_dict(assembly="continuous-context-overlap-save"),
                "cache": checkpoint_metadata(cache_path),
            }
            json_write(cache_path.with_suffix(".json"), metadata)
            validate_cache(cache_path, len(group["events"]))
            paths[key] = cache_path
            print(
                json.dumps(
                    {"event": "cache-prepared", "index": number, "total": len(missing), "key": key, "records": len(group["events"])},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
            del source, teacher, input_array, target_array, anchor_array, mask_array
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        del anchor_model, window
        if device.type == "cuda":
            torch.cuda.empty_cache()
    json_write(output_root / "selection-with-cache.json", {**selection, "cacheCount": len(paths)})
    return paths


class GpuCacheStore:
    """Keep the small reviewed event pool on the selected device.

    The pool is only 78 records, so one-time GPU residency removes repeated
    host-to-device copies from the update loop and makes utilization measurable.
    """

    def __init__(self, paths: dict[str, Path], groups: dict[str, dict[str, Any]], device: torch.device) -> None:
        self.device = device
        self.lookup: dict[tuple[str, int], int] = {}
        input_parts: list[torch.Tensor] = []
        target_parts: list[torch.Tensor] = []
        anchor_parts: list[torch.Tensor] = []
        mask_parts: list[torch.Tensor] = []
        offset = 0
        for key in sorted(paths):
            group = groups[key]
            validate_cache(paths[key], len(group["events"]))
            with np.load(paths[key]) as values:
                arrays = {
                    name: np.ascontiguousarray(values[name], dtype=np.float32)
                    for name in ("inputSpec", "targetAudio", "anchorAudio", "eventMask")
                }
            count = len(group["events"])
            for index in range(count):
                self.lookup[(key, index)] = offset + index
            offset += count
            input_parts.append(torch.from_numpy(arrays["inputSpec"]).to(device))
            target_parts.append(torch.from_numpy(arrays["targetAudio"]).to(device))
            anchor_parts.append(torch.from_numpy(arrays["anchorAudio"]).to(device))
            mask_parts.append(torch.from_numpy(arrays["eventMask"]).to(device))
        if not input_parts:
            raise ValueError("No cache arrays available")
        self.input_spec = torch.cat(input_parts, dim=0)
        self.target_audio = torch.cat(target_parts, dim=0)
        self.anchor_audio = torch.cat(anchor_parts, dim=0)
        self.event_mask = torch.cat(mask_parts, dim=0)
        self.record_count = offset
        self.bytes = sum(t.numel() * t.element_size() for t in (self.input_spec, self.target_audio, self.anchor_audio, self.event_mask))

    def batch(self, items: Sequence[tuple[str, int]]) -> tuple[torch.Tensor, ...]:
        indices = torch.tensor([self.lookup[item] for item in items], dtype=torch.long, device=self.device)
        return (
            self.input_spec.index_select(0, indices),
            self.target_audio.index_select(0, indices),
            self.anchor_audio.index_select(0, indices),
            self.event_mask.index_select(0, indices),
        )


class GpuMonitor:
    def __init__(self, device: torch.device, interval_seconds: float = 0.5) -> None:
        self.enabled = device.type == "cuda" and shutil.which("nvidia-smi") is not None
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, float]] = []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def _read(self) -> None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=1.0,
                check=True,
            )
            fields = [item.strip() for item in result.stdout.splitlines()[0].split(",")]
            if len(fields) >= 4:
                self.samples.append(
                    {
                        "gpuUtilizationPercent": float(fields[0]),
                        "memoryUtilizationPercent": float(fields[1]),
                        "memoryUsedMiB": float(fields[2]),
                        "powerWatts": float(fields[3]),
                    }
                )
        except (OSError, IndexError, ValueError, subprocess.SubprocessError):
            return

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self._read()
            self.stop_event.wait(self.interval_seconds)

    def start(self) -> None:
        if not self.enabled:
            return
        self.thread = threading.Thread(target=self._run, name="gpu-monitor", daemon=True)
        self.thread.start()

    def stop(self) -> dict[str, Any]:
        if self.thread is not None:
            self.stop_event.set()
            self.thread.join(timeout=2.0)
            self._read()
        if not self.samples:
            return {"enabled": self.enabled, "sampleCount": 0}
        values = [item["gpuUtilizationPercent"] for item in self.samples]
        return {
            "enabled": self.enabled,
            "sampleCount": len(self.samples),
            "gpuUtilizationPercent": {
                "mean": float(np.mean(values)),
                "p10": float(np.percentile(values, 10)),
                "p50": float(np.percentile(values, 50)),
                "p90": float(np.percentile(values, 90)),
                "max": float(np.max(values)),
            },
            "last": self.samples[-1],
        }


def build_schedule(
    records: Sequence[dict[str, Any]], passes: int, seed: int
) -> tuple[list[tuple[str, int]], dict[str, Any]]:
    s_items = [(record["key"], int(record["index"])) for record in records if record["mark"] == "S"]
    r_items = [(record["key"], int(record["index"])) for record in records if record["mark"] == "R"]
    if not s_items or not r_items:
        raise ValueError("Schedule requires both S and R records")
    weighted_units = s_items * 2 + r_items
    base_repeat, remainder = divmod(RECORDS_PER_PASS, len(weighted_units))
    result: list[tuple[str, int]] = []
    pass_details: list[dict[str, Any]] = []
    labels = {(record["key"], int(record["index"])): record["mark"] for record in records}
    for pass_index in range(passes):
        rng = np.random.default_rng(seed + pass_index * 1_000_003 + 97)
        values = weighted_units * base_repeat
        extra_order = rng.permutation(len(weighted_units))[:remainder]
        values.extend(weighted_units[int(index)] for index in extra_order)
        order = rng.permutation(len(values))
        ordered = [values[int(index)] for index in order]
        result.extend(ordered)
        counts = Counter(ordered)
        s_counts = [counts[item] for item in s_items]
        r_counts = [counts[item] for item in r_items]
        pass_details.append(
            {
                "pass": pass_index + 1,
                "recordCount": len(ordered),
                "sRecords": sum(labels[item] == "S" for item in ordered),
                "rRecords": sum(labels[item] == "R" for item in ordered),
                "sUniqueRecords": len(s_items),
                "rUniqueRecords": len(r_items),
                "baseRepeat": base_repeat,
                "weightedUnitCount": len(weighted_units),
                "remainder": remainder,
                "sCountMin": min(s_counts),
                "sCountMax": max(s_counts),
                "rCountMin": min(r_counts),
                "rCountMax": max(r_counts),
                "eventFrequencyRatio": (sum(labels[item] == "S" for item in ordered) / len(s_items))
                / (sum(labels[item] == "R" for item in ordered) / len(r_items)),
            }
        )
    summary = {
        "recordsPerPass": RECORDS_PER_PASS,
        "passes": pass_details,
        "sUniqueRecords": len(s_items),
        "rUniqueRecords": len(r_items),
        "weightedUnitCount": len(weighted_units),
        "baseRepeat": base_repeat,
        "remainder": remainder,
        "weightedInterpretation": "each S event is represented twice per R event in the repeated pool; no song-level balancing",
        "recordCount": len(result),
        "updatesPerPass": RECORDS_PER_PASS // BATCH_SIZE,
        "scheduleSha256": canonical_sha256(result),
    }
    return result, summary


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"count": len(rows)}
    if not rows:
        return output
    for milliseconds in (50, 100, 200):
        projection = np.asarray(
            [row["metrics"][str(milliseconds)]["positiveProjectionDbfs"] for row in rows], dtype=np.float64
        )
        miss = np.asarray([row["metrics"][str(milliseconds)]["missRmsDbfs"] for row in rows], dtype=np.float64)
        output[str(milliseconds)] = {
            "positiveProjectionP50Dbfs": float(np.percentile(projection, 50)),
            "positiveProjectionP90Dbfs": float(np.percentile(projection, 90)),
            "positiveProjectionP95Dbfs": float(np.percentile(projection, 95)),
            "positiveProjectionMaxDbfs": float(np.max(projection)),
            "missRmsP95Dbfs": float(np.percentile(miss, 95)),
            "missRmsMaxDbfs": float(np.max(miss)),
        }
    return output


def evaluate_model(
    model: torch.nn.Module,
    groups: dict[str, dict[str, Any]],
    device: torch.device,
    inference_batch_size: int,
) -> dict[str, Any]:
    model.eval()
    all_rows: list[dict[str, Any]] = []
    by_mark: dict[str, list[dict[str, Any]]] = {"S": [], "R": []}
    by_batch: dict[str, list[dict[str, Any]]] = {}
    songs: list[dict[str, Any]] = []
    started = time.perf_counter()
    for key, group in groups.items():
        source = external.load_audio(Path(group["sourcePath"]))
        teacher = external.load_audio(Path(group["teacherPath"]))
        if source.shape != teacher.shape:
            raise ValueError(f"Evaluation source/teacher mismatch for {key}")
        residual, timing = continuous.render_student(model, source, CONTRACT, device, inference_batch_size)
        candidate = np.ascontiguousarray(source - residual, dtype=np.float32)
        song_rows: list[dict[str, Any]] = []
        for event in group["events"]:
            metrics = {
                str(milliseconds): external.block_metric(
                    source,
                    teacher,
                    candidate,
                    int(event["centerSamples"]),
                    milliseconds,
                )
                for milliseconds in (50, 100, 200)
            }
            row = {
                "eventId": event["eventId"],
                "mark": event["mark"],
                "batch": event["batch"],
                "song": key,
                "metrics": metrics,
            }
            all_rows.append(row)
            song_rows.append(row)
            by_mark[event["mark"]].append(row)
            by_batch.setdefault(event["batch"], []).append(row)
        songs.append(
            {
                "key": key,
                "song": f"{group['artistName']} - {group['trackName']}",
                "batch": group["sourceBatch"],
                "eventCount": len(song_rows),
                "render": timing,
                "metrics": summarize_rows(song_rows),
            }
        )
        del source, teacher, residual, candidate
    result = {
        "elapsedSeconds": time.perf_counter() - started,
        "poolRole": "training-pool-event-evaluation",
        "all": summarize_rows(all_rows),
        "byMark": {mark: summarize_rows(rows) for mark, rows in by_mark.items()},
        "byBatch": {batch: summarize_rows(rows) for batch, rows in sorted(by_batch.items())},
        "songs": songs,
        "eventRows": all_rows,
    }
    model.train()
    sweep.freeze_batchnorm_running_statistics(model)
    return result


def train_pass(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    store: GpuCacheStore,
    schedule: Sequence[tuple[str, int]],
    pass_index: int,
    global_updates_before: int,
    args: argparse.Namespace,
    device: torch.device,
    max_updates: int | None = None,
) -> list[dict[str, Any]]:
    model.train()
    sweep.freeze_batchnorm_running_statistics(model)
    window = torch.hann_window(pilot.DEFAULT_CONFIG.n_fft, periodic=True, device=device)
    updates_per_pass = RECORDS_PER_PASS // args.batch_size
    updates = updates_per_pass if max_updates is None else min(updates_per_pass, max_updates)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    begin = pass_index * RECORDS_PER_PASS
    for update in range(updates):
        batch_started = time.perf_counter()
        items = schedule[begin + update * args.batch_size : begin + (update + 1) * args.batch_size]
        input_tensor, target_tensor, anchor_tensor, mask_tensor = store.batch(items)
        optimizer.zero_grad(set_to_none=True)
        forward_started = time.perf_counter()
        predicted_full = local.torch_packed_istft(model(input_tensor), window)
        trim = pilot.DEFAULT_CONFIG.trim_samples
        predicted = predicted_full[:, trim : trim + CONTRACT.useful_samples]
        forward_seconds = time.perf_counter() - forward_started
        event_losses = local.charbonnier_per_record(predicted, target_tensor, mask_tensor)
        anchor_losses = local.charbonnier_per_record(predicted, anchor_tensor, 1.0 - mask_tensor)
        loss = (event_losses + args.anchor_beta * anchor_losses).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at pass {pass_index + 1}, update {update + 1}")
        backward_started = time.perf_counter()
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False).item())
        if not math.isfinite(gradient):
            raise FloatingPointError(f"Non-finite gradient at pass {pass_index + 1}, update {update + 1}")
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        backward_seconds = time.perf_counter() - backward_started
        global_step = SOURCE_STEP + global_updates_before + update + 1
        row = {
            "pass": pass_index + 1,
            "update": update + 1,
            "globalStep": global_step,
            "loss": float(loss.detach().cpu()),
            "eventLoss": float(event_losses.mean().detach().cpu()),
            "anchorLoss": float(anchor_losses.mean().detach().cpu()),
            "gradientNormBeforeClip": gradient,
            "forwardSeconds": forward_seconds,
            "backwardSeconds": backward_seconds,
            "batchSeconds": time.perf_counter() - batch_started,
            "cudaAllocatedBytes": int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else None,
            "cudaReservedBytes": int(torch.cuda.memory_reserved(device)) if device.type == "cuda" else None,
        }
        history.append(row)
        if update == 0 or (update + 1) % args.log_every == 0 or update + 1 == updates:
            print(json.dumps({"event": "progress", **row, "updates": updates}, sort_keys=True), flush=True)
    print(
        json.dumps(
            {"event": "pass-complete", "pass": pass_index + 1, "elapsedSeconds": time.perf_counter() - started},
            sort_keys=True,
        ),
        flush=True,
    )
    del window
    return history


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    contract_id: str,
    schedule_hash: str,
    selection: dict[str, Any],
    source: dict[str, Any],
    history: list[dict[str, Any]],
    local_pass: int,
    global_step: int,
) -> dict[str, Any]:
    payload = {
        "format": CHECKPOINT_FORMAT,
        "status": "milestone",
        "variant": "FMA-S-leakage-survey-SR",
        "runContractId": contract_id,
        "sourceCheckpoint": source["checkpoint"]["file"],
        "sourceStep": SOURCE_STEP,
        "sourcePass": source["sourcePass"],
        "localPass": local_pass,
        "step": global_step,
        "globalStep": global_step,
        "continuationPass": int(source["sourcePass"]) + local_pass,
        "recordsPerPass": RECORDS_PER_PASS,
        "updatesPerPass": RECORDS_PER_PASS // args.batch_size,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "anchorBeta": args.anchor_beta,
        "coreMs": CORE_MS,
        "guardMs": GUARD_MS,
        "seed": args.seed,
        "scheduleSha256": schedule_hash,
        "selectionSha256": selection["selectionSha256"],
        "lossContract": "Inst3 residual target on 100ms+25ms event mask plus step-5648 anchor outside mask",
        "assembly": CONTRACT.as_dict(assembly="continuous-context-overlap-save"),
        "stateDict": cpu_tree(model.state_dict()),
        "optimizerStateDict": cpu_tree(optimizer.state_dict()),
        "history": history,
    }
    atomic_torch_save(path, payload)
    return checkpoint_metadata(path)


def load_latest_checkpoint(
    run_root: Path, contract_id: str, selection_hash: str, schedule_hash: str
) -> tuple[Path, dict[str, Any]] | None:
    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    for path in run_root.glob("step-*.pt"):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if (
                payload.get("format") == CHECKPOINT_FORMAT
                and payload.get("runContractId") == contract_id
                and payload.get("selectionSha256") == selection_hash
                and payload.get("scheduleSha256") == schedule_hash
            ):
                candidates.append((int(payload.get("localPass", -1)), path, payload))
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
    if not candidates:
        return None
    _, path, payload = max(candidates, key=lambda item: item[0])
    return path, payload


def make_report(
    args: argparse.Namespace,
    selection: dict[str, Any],
    schedule_summary: dict[str, Any],
    contract_id: str,
    contract_payload: dict[str, Any],
    source: dict[str, Any],
    history: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    status: str,
    gpu: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": status,
        "selection": selection,
        "schedule": schedule_summary,
        "contract": {"id": contract_id, "payload": contract_payload},
        "source": source,
        "training": {"maxPasses": args.max_passes, "completedPasses": len(history) // (RECORDS_PER_PASS // args.batch_size), "history": history},
        "evaluations": evaluations,
        "stop": {"patience": args.patience, "minImprovementDb": args.min_improvement_db},
        "environment": {
            "device": str(gpu.get("device", "")) if gpu else None,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "gpuName": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpuCacheBytes": gpu.get("cacheBytes") if gpu else None,
            "gpuMonitor": gpu.get("monitor") if gpu else None,
        },
    }


def run_training(
    args: argparse.Namespace,
    groups: dict[str, dict[str, Any]],
    selection: dict[str, Any],
    paths: dict[str, Path],
    schedule: list[tuple[str, int]],
    schedule_summary: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    contract_payload = {
        "schema": SCHEMA,
        "sourceCheckpointSha256": selection["sourceCheckpoint"]["sha256"],
        "selectionSha256": selection["selectionSha256"],
        "scheduleSha256": schedule_summary["scheduleSha256"],
        "recordsPerPass": RECORDS_PER_PASS,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "anchorBeta": args.anchor_beta,
        "coreMs": CORE_MS,
        "guardMs": GUARD_MS,
        "assembly": "continuous-context-overlap-save",
        "loss": "Inst3 residual target on event mask plus source-checkpoint anchor outside mask",
        "officialFinalTestUsed": False,
    }
    contract_id = canonical_sha256(contract_payload)
    output_root = args.output_root.resolve()
    run_root = output_root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "reports" / "training-report.json"
    set_seed(args.seed)
    model, optimizer, source, _source_payload = load_source_model(args, device)
    store = GpuCacheStore(paths, groups, device)
    gpu_info: dict[str, Any] = {"device": str(device), "cacheBytes": store.bytes}
    history: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    completed_passes = 0

    if args.resume and not args.smoke_only:
        found = load_latest_checkpoint(run_root, contract_id, selection["selectionSha256"], schedule_summary["scheduleSha256"])
        if found is not None:
            checkpoint_path, payload = found
            model.load_state_dict(payload["stateDict"], strict=True)
            optimizer.load_state_dict(payload["optimizerStateDict"])
            move_optimizer_state(optimizer, device)
            for group_state in optimizer.param_groups:
                group_state["lr"] = args.learning_rate
            completed_passes = int(payload.get("localPass", 0))
            history = list(payload.get("history", []))
            if report_path.is_file():
                old_report = read_json(report_path)
                evaluations = list(old_report.get("evaluations", []))
            print(json.dumps({"event": "resume", "pass": completed_passes, "checkpoint": str(checkpoint_path)}, sort_keys=True), flush=True)

    if args.smoke_only:
        monitor = GpuMonitor(device)
        monitor.start()
        smoke_history = train_pass(model, optimizer, store, schedule, 0, 0, args, device, args.smoke_updates)
        gpu_info["monitor"] = monitor.stop()
        result = {
            "schema": SCHEMA + ".smoke",
            "status": "smoke-completed",
            "updates": len(smoke_history),
            "history": smoke_history,
            "selection": selection,
            "schedule": schedule_summary,
            "gpu": gpu_info,
        }
        del model, optimizer, store
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return result

    if not evaluations:
        baseline = evaluate_model(model, groups, device, args.inference_batch_size)
        evaluations.append({"pass": 0, "globalStep": SOURCE_STEP, "metrics": baseline, "improvementDb": None, "role": "source-baseline"})
        print(json.dumps({"event": "baseline", "pass": 0, "p95_100ms": baseline["all"]["100"]["positiveProjectionP95Dbfs"]}, sort_keys=True), flush=True)
        json_write(
            report_path,
            make_report(args, selection, schedule_summary, contract_id, contract_payload, source, history, evaluations, "running", gpu_info),
        )

    previous_p95 = float(evaluations[-1]["metrics"]["all"]["100"]["positiveProjectionP95Dbfs"])
    no_improvement = 0
    monitor = GpuMonitor(device)
    monitor.start()
    try:
        for pass_index in range(completed_passes, args.max_passes):
            pass_history = train_pass(
                model,
                optimizer,
                store,
                schedule,
                pass_index,
                pass_index * (RECORDS_PER_PASS // args.batch_size),
                args,
                device,
            )
            history.extend(pass_history)
            local_pass = pass_index + 1
            global_step = SOURCE_STEP + local_pass * (RECORDS_PER_PASS // args.batch_size)
            checkpoint_path = run_root / f"step-{global_step}.pt"
            checkpoint_info = save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                args,
                contract_id,
                schedule_summary["scheduleSha256"],
                selection,
                source,
                history,
                local_pass,
                global_step,
            )
            metrics = evaluate_model(model, groups, device, args.inference_batch_size)
            current_p95 = float(metrics["all"]["100"]["positiveProjectionP95Dbfs"])
            improvement = previous_p95 - current_p95
            evaluations.append(
                {
                    "pass": local_pass,
                    "globalStep": global_step,
                    "checkpoint": checkpoint_info,
                    "metrics": metrics,
                    "improvementDb": improvement,
                    "meaningfulImprovement": improvement >= args.min_improvement_db,
                    "role": "survey-continuation",
                }
            )
            gpu_info["monitor"] = monitor.stop()
            monitor = GpuMonitor(device)
            monitor.start()
            report = make_report(
                args, selection, schedule_summary, contract_id, contract_payload, source, history, evaluations, "running", gpu_info
            )
            json_write(report_path, report)
            print(
                json.dumps(
                    {"event": "evaluation", "pass": local_pass, "globalStep": global_step, "p95_100ms": current_p95, "improvementDb": improvement},
                    sort_keys=True,
                ),
                flush=True,
            )
            if improvement >= args.min_improvement_db:
                no_improvement = 0
            else:
                no_improvement += 1
            previous_p95 = current_p95
            if local_pass >= args.min_passes and no_improvement >= args.patience:
                report["status"] = "completed"
                report["stop"] = {
                    "reason": "100ms projection p95 plateau",
                    "stoppedAfterPass": local_pass,
                    "consecutiveNonMeaningfulPasses": no_improvement,
                    "minImprovementDb": args.min_improvement_db,
                    "patience": args.patience,
                }
                json_write(report_path, report)
                break
        else:
            report = read_json(report_path) if report_path.is_file() else make_report(
                args, selection, schedule_summary, contract_id, contract_payload, source, history, evaluations, "running", gpu_info
            )
            report["status"] = "completed"
            report["stop"] = {
                "reason": "max-passes guard",
                "maxPasses": args.max_passes,
                "minImprovementDb": args.min_improvement_db,
                "patience": args.patience,
            }
            json_write(report_path, report)
    finally:
        gpu_info["monitor"] = monitor.stop()
        del model, optimizer, store
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return read_json(report_path)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_passes <= 0 or args.min_passes <= 0 or args.min_passes > args.max_passes:
        raise ValueError("Invalid pass limits")
    if args.patience <= 0 or args.min_improvement_db < 0 or args.batch_size <= 0:
        raise ValueError("Invalid stopping or batch settings")
    if RECORDS_PER_PASS % args.batch_size:
        raise ValueError("records per pass must be divisible by batch size")
    if args.smoke_updates <= 0:
        raise ValueError("smoke-updates must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    groups, records, selection = load_pool(args.survey_root)
    selection["sourceCheckpoint"] = checkpoint_metadata(args.source_checkpoint)
    selection["sourceStep"] = SOURCE_STEP
    selection["selectionSha256"] = canonical_sha256({key: value for key, value in selection.items() if key != "selectionSha256"})
    json_write(args.output_root / "pool-selection.json", selection)
    paths = prepare_caches(args, groups, selection, device)
    schedule, schedule_summary = build_schedule(records, args.max_passes, args.seed)
    json_write(args.output_root / "schedule.json", {"summary": schedule_summary, "schedule": schedule})
    result = run_training(args, groups, selection, paths, schedule, schedule_summary, device)
    if args.smoke_only:
        json_write(args.output_root / "reports" / "smoke-report.json", result)
        print(json.dumps({"status": result["status"], "report": str((args.output_root / "reports" / "smoke-report.json").resolve())}, indent=2), flush=True)
    else:
        print(
            json.dumps(
                {
                    "status": result.get("status"),
                    "report": str((args.output_root / "reports" / "training-report.json").resolve()),
                    "completedPasses": result.get("training", {}).get("completedPasses"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
