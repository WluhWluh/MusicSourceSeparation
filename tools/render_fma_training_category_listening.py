#!/usr/bin/env python3
"""Render selected FMA songs used to train step 7408.

The training pool is reconstructed from the three authoritative event-pool
artifacts in the checkpoint lineage.  The tool can render either the stable
category-balanced sample or its exact training-pool complement.  Songs use the
continuous 128-frame overlap-save contract, preserving both the residual-vocal
output and the derived instrumental (mixture minus residual).
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import platform
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
import torch

import evaluate_inst3_continuous_baseline as continuous
import prepare_all_fma_s86_event_listening as fma_audio
import render_fma_sr_continuation_private_listening as private_renderer
import run_inst3_distill_pilot as pilot
from tfc_tdf_short_window import ShortWindowContract


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    ROOT
    / "data"
    / "modern-song-fma-s-leakage-survey-continuation"
    / "runs"
    / "step-7408.pt"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "data"
    / "modern-song-fma-s-leakage-survey-continuation"
    / "listening-fma-training-category-sample-20"
    / "step-7408"
)
DEFAULT_SAMPLE_SELECTION = DEFAULT_OUTPUT_ROOT / "selection.json"
S86_CACHE_ROOT = ROOT / "data" / "modern-song-s-r-continuation" / "cache" / "external" / "S"
ALL_FMA_SELECTION = ROOT / "data" / "modern-song-fma-all-s86-event-pass5" / "selected-songs.json"
FMA16_SELECTION = ROOT / "data" / "modern-song-fma-train-sr-s86-event-pass5-16" / "selected-songs.json"
FMA16_POOL = ROOT / "data" / "modern-song-fma-sr-event-only-continuation" / "selected-pool.json"
SURVEY_POOL = ROOT / "data" / "modern-song-fma-s-leakage-survey-continuation" / "pool-selection.json"
SURVEY_ROOT = ROOT / "data" / "fma-s-leakage-survey"

SAMPLE_RATE = 44_100
CONTRACT = ShortWindowContract(num_frames=128, left_context_hops=5, right_context_hops=5)
STAGES = ("S86", "FMA16", "survey")
SELECTION_SEED = "fma-step-7408-category-listening-v1"
CATEGORY_QUOTAS = {
    "hiphop-rnb": 4,
    "singer-songwriter": 4,
    "pop-synth": 4,
    "pop-rock": 3,
    "latin-modern": 3,
    "dance-electronic": 2,
}
EXPECTED_POOL_CATEGORIES = {
    "hiphop-rnb": 22,
    "singer-songwriter": 19,
    "pop-synth": 18,
    "pop-rock": 5,
    "latin-modern": 5,
    "dance-electronic": 2,
}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--architecture-checkpoint", type=Path, default=pilot.DEFAULT_CHECKPOINT)
    parser.add_argument("--expected-step", type=int, default=7408)
    parser.add_argument("--selection-seed", default=SELECTION_SEED)
    parser.add_argument(
        "--exclude-selection",
        type=Path,
        help="Render the exact training-pool complement of this frozen selection JSON",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_key(seed: str, category: str, stage: str, slug: str) -> str:
    return hashlib.sha256(f"{seed}\0{category}\0{stage}\0{slug}".encode("utf-8")).hexdigest()


def stage_sort(stages: Iterable[str]) -> list[str]:
    values = set(stages)
    unknown = values.difference(STAGES)
    if unknown:
        raise ValueError(f"Unknown training stages: {sorted(unknown)}")
    return [stage for stage in STAGES if stage in values]


def normalized_song(metadata: dict[str, Any], stage: str, provenance: str) -> dict[str, Any]:
    source_path = metadata.get("sourcePath") or metadata.get("rawFile")
    if not source_path:
        raise ValueError(f"Missing source path for {metadata.get('slug')}")
    slug = str(metadata.get("slug", "")).strip()
    category = str(metadata.get("category", "")).strip()
    if not slug or not category:
        raise ValueError(f"Missing slug/category in {provenance}")
    return {
        "slug": slug,
        "category": category,
        "artistName": str(metadata.get("artistName", "")),
        "trackName": str(metadata.get("trackName", "")),
        "languageCode": str(metadata.get("languageCode", "")),
        "sourceId": str(metadata.get("sourceId", "")),
        "sourceBatch": str(metadata.get("sourceBatch", "")),
        "sourceOrder": metadata.get("sourceOrder"),
        "sourcePath": str(Path(source_path).resolve()),
        "sourceSha256Declared": str(metadata.get("sourceSha256", "")),
        "license": str(metadata.get("license", "")),
        "licenseUrl": str(metadata.get("licenseUrl", "")),
        "trainingStages": [stage],
        "trainingProvenance": [{"stage": stage, "source": provenance}],
    }


def merge_song(pool: dict[str, dict[str, Any]], candidate: dict[str, Any]) -> None:
    slug = candidate["slug"]
    existing = pool.get(slug)
    if existing is None:
        pool[slug] = candidate
        return
    if existing["category"] != candidate["category"]:
        raise ValueError(
            f"Category conflict for {slug}: {existing['category']} versus {candidate['category']}"
        )
    existing["trainingStages"] = stage_sort(
        [*existing["trainingStages"], *candidate["trainingStages"]]
    )
    existing["trainingProvenance"].extend(candidate["trainingProvenance"])
    if existing["sourcePath"] != candidate["sourcePath"]:
        existing.setdefault("alternateSourcePaths", []).append(candidate["sourcePath"])


def load_s86_pool() -> list[dict[str, Any]]:
    metadata = read_json(ALL_FMA_SELECTION).get("songs", [])
    by_slug = {str(song["slug"]): song for song in metadata}
    songs: list[dict[str, Any]] = []
    for cache_path in sorted(S86_CACHE_ROOT.glob("*.json")):
        cache = read_json(cache_path)
        slug = str(cache.get("slug", ""))
        if slug not in by_slug:
            raise ValueError(f"S86 cache slug is absent from the all-FMA selection: {slug}")
        songs.append(normalized_song(by_slug[slug], "S86", str(cache_path.resolve())))
    if len(songs) != 38:
        raise ValueError(f"Expected 38 S86 training songs, got {len(songs)}")
    return songs


def fma16_song_key(song: dict[str, Any]) -> str:
    return (
        f"external::{song['sourceBatch']}::"
        f"{int(song['sourceOrder']):03d}-{song['slug']}"
    )


def load_fma16_pool() -> list[dict[str, Any]]:
    selected_keys = set(read_json(FMA16_POOL).get("songKeys", []))
    songs = read_json(FMA16_SELECTION).get("songs", [])
    selected = [song for song in songs if fma16_song_key(song) in selected_keys]
    actual_keys = {fma16_song_key(song) for song in selected}
    if actual_keys != selected_keys:
        missing = sorted(selected_keys.difference(actual_keys))
        raise ValueError(f"FMA16 selected songs are incomplete: {missing}")
    if len(selected) != 13:
        raise ValueError(f"Expected 13 FMA16 training songs, got {len(selected)}")
    return [
        normalized_song(song, "FMA16", str(FMA16_POOL.resolve()))
        for song in selected
    ]


def source_order_from_path(path: str) -> int | None:
    match = re.match(r"(\d+)-", Path(path).name)
    return int(match.group(1)) if match else None


def load_survey_pool() -> list[dict[str, Any]]:
    selection = read_json(SURVEY_POOL)
    selected_keys = set(selection.get("songKeys", []))
    songs: list[dict[str, Any]] = []
    actual_keys: set[str] = set()
    for batch in selection.get("validBatchNames", []):
        csv_path = SURVEY_ROOT / batch / "retained" / "retained-songs.csv"
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = f"{batch}::{row['slug']}"
                if key not in selected_keys:
                    continue
                row = dict(row)
                row["sourceBatch"] = batch
                row["sourcePath"] = row["rawFile"]
                row["sourceOrder"] = source_order_from_path(row["rawFile"])
                songs.append(normalized_song(row, "survey", str(csv_path.resolve())))
                actual_keys.add(key)
    if actual_keys != selected_keys:
        missing = sorted(selected_keys.difference(actual_keys))
        raise ValueError(f"Survey selected songs are incomplete: {missing}")
    if len(songs) != 30:
        raise ValueError(f"Expected 30 survey training songs, got {len(songs)}")
    return songs


def reconstruct_training_pool() -> list[dict[str, Any]]:
    pool: dict[str, dict[str, Any]] = {}
    for candidate in [*load_s86_pool(), *load_fma16_pool(), *load_survey_pool()]:
        merge_song(pool, candidate)
    songs = sorted(pool.values(), key=lambda song: song["slug"])
    for song in songs:
        song["trainingStages"] = stage_sort(song["trainingStages"])
        source_path = Path(song["sourcePath"])
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
    categories = dict(sorted(Counter(song["category"] for song in songs).items()))
    if len(songs) != 71 or categories != dict(sorted(EXPECTED_POOL_CATEGORIES.items())):
        raise ValueError(f"Unexpected reconstructed pool: songs={len(songs)}, categories={categories}")
    return songs


def select_category_sample(
    songs: list[dict[str, Any]],
    quotas: dict[str, int],
    seed: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_slugs: set[str] = set()
    selected_artists: set[str] = set()
    for category_index, (category, quota) in enumerate(quotas.items()):
        available = [song for song in songs if song["category"] == category]
        if len(available) < quota:
            raise ValueError(f"Category {category} has only {len(available)} songs for quota {quota}")
        for slot in range(quota):
            requested_stage = STAGES[(category_index + slot) % len(STAGES)]
            candidates = [
                song
                for song in available
                if song["slug"] not in selected_slugs and requested_stage in song["trainingStages"]
            ]
            fallback = not candidates
            if fallback:
                candidates = [song for song in available if song["slug"] not in selected_slugs]
            candidates.sort(
                key=lambda song: (
                    song["artistName"].casefold() in selected_artists,
                    stable_key(seed, category, requested_stage, song["slug"]),
                )
            )
            chosen = dict(candidates[0])
            chosen["requestedStage"] = requested_stage
            chosen["selectedForStage"] = (
                requested_stage if requested_stage in chosen["trainingStages"] else chosen["trainingStages"][0]
            )
            chosen["stageFallback"] = fallback
            selected.append(chosen)
            selected_slugs.add(chosen["slug"])
            selected_artists.add(chosen["artistName"].casefold())
    for index, song in enumerate(selected, start=1):
        song["index"] = index
    counts = Counter(song["category"] for song in selected)
    if counts != Counter(quotas):
        raise AssertionError(f"Selection quota mismatch: {dict(counts)}")
    if not set(STAGES).issubset({stage for song in selected for stage in song["trainingStages"]}):
        raise AssertionError("The selection does not cover every checkpoint training stage")
    return selected


def select_complement(
    songs: list[dict[str, Any]],
    excluded_selection: dict[str, Any],
) -> list[dict[str, Any]]:
    excluded_rows = excluded_selection.get("songs")
    if not isinstance(excluded_rows, list) or not excluded_rows:
        raise ValueError("Excluded selection has no songs")
    excluded_slugs = [str(song.get("slug", "")) for song in excluded_rows]
    if any(not slug for slug in excluded_slugs) or len(set(excluded_slugs)) != len(excluded_slugs):
        raise ValueError("Excluded selection contains blank or duplicate slugs")
    pool_slugs = {song["slug"] for song in songs}
    missing = sorted(set(excluded_slugs).difference(pool_slugs))
    if missing:
        raise ValueError(f"Excluded songs are absent from the reconstructed training pool: {missing}")

    category_order = {category: index for index, category in enumerate(EXPECTED_POOL_CATEGORIES)}
    selected = [dict(song) for song in songs if song["slug"] not in set(excluded_slugs)]
    selected.sort(
        key=lambda song: (
            category_order[song["category"]],
            song["artistName"].casefold(),
            song["trackName"].casefold(),
            song["slug"],
        )
    )
    for index, song in enumerate(selected, start=1):
        song["index"] = index
        song["requestedStage"] = ""
        song["selectedForStage"] = song["trainingStages"][-1]
        song["stageFallback"] = False
    if len(selected) + len(excluded_slugs) != len(songs):
        raise AssertionError("Complement does not partition the reconstructed training pool")
    return selected


def enrich_selection(songs: list[dict[str, Any]]) -> None:
    for song in songs:
        source_path = Path(song["sourcePath"])
        source_sha = private_renderer.sha256_file(source_path)
        declared = song.get("sourceSha256Declared")
        if declared and declared != source_sha:
            raise ValueError(f"Source SHA mismatch for {song['slug']}")
        song["sourceSha256"] = source_sha
        song["sourceBytes"] = source_path.stat().st_size


def selection_document(
    pool: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    seed: str,
    *,
    selection_kind: str = "category-sample",
    algorithm: str = "fixed category quotas; rotating stage request; stable SHA-256 order; artist-repeat penalty",
    excluded_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    digest_rows = [
        {
            "index": song["index"],
            "slug": song["slug"],
            "category": song["category"],
            "selectedForStage": song["selectedForStage"],
            "trainingStages": song["trainingStages"],
            "sourceSha256": song["sourceSha256"],
        }
        for song in selected
    ]
    document = {
        "schema": "local-fma-training-category-listening-selection@1",
        "status": "frozen",
        "selectionSha256": canonical_sha256(digest_rows),
        "selectionKind": selection_kind,
        "algorithm": algorithm,
        "seed": seed,
        "checkpointLineageSources": {
            "S86": str(S86_CACHE_ROOT.resolve()),
            "FMA16": str(FMA16_POOL.resolve()),
            "survey": str(SURVEY_POOL.resolve()),
        },
        "trainingPool": {
            "songCount": len(pool),
            "categoryCounts": dict(sorted(Counter(song["category"] for song in pool).items())),
            "stageMembershipCounts": {
                stage: sum(stage in song["trainingStages"] for song in pool) for stage in STAGES
            },
        },
        "selectedSongCount": len(selected),
        "selectedCategoryCounts": dict(sorted(Counter(song["category"] for song in selected).items())),
        "selectedStageMembershipCounts": {
            stage: sum(stage in song["trainingStages"] for song in selected) for stage in STAGES
        },
        "songs": selected,
    }
    if selection_kind == "category-sample":
        document["quota"] = CATEGORY_QUOTAS
    if excluded_selection is not None:
        document["excludedSelection"] = excluded_selection
    return document


def write_selection_csv(path: Path, songs: list[dict[str, Any]]) -> None:
    fields = (
        "index",
        "category",
        "selectedForStage",
        "requestedStage",
        "stageFallback",
        "trainingStages",
        "artistName",
        "trackName",
        "languageCode",
        "slug",
        "sourceBatch",
        "sourceOrder",
        "sourceId",
        "sourcePath",
        "sourceSha256",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for song in songs:
            row = {field: song.get(field, "") for field in fields}
            row["trainingStages"] = "+".join(song["trainingStages"])
            writer.writerow(row)


def compact_output_name(song: dict[str, Any]) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "-", song["slug"]).strip("-._")
    digest = hashlib.sha256(song["slug"].encode("utf-8")).hexdigest()[:8]
    return f"{song['index']:02d}-{readable[:36]}-{digest}.flac"


def output_path(output_root: Path, kind: str, song: dict[str, Any]) -> Path:
    return output_root / kind / song["category"] / compact_output_name(song)


def valid_completed_song(item: dict[str, Any], source_frames: int) -> bool:
    try:
        for kind in ("instrumental", "residual-vocals"):
            path = Path(item[kind]["file"])
            info = sf.info(path)
            if info.samplerate != SAMPLE_RATE or info.channels != 2 or info.frames != source_frames:
                return False
        return True
    except (KeyError, OSError, RuntimeError):
        return False


def write_playlist(path: Path, songs: list[dict[str, Any]], kind: str) -> None:
    lines = ["#EXTM3U"]
    for song in songs:
        audio_path = output_path(path.parent, kind, song)
        relative = audio_path.relative_to(path.parent).as_posix()
        title = f"{song['index']:02d} {song['artistName']} - {song['trackName']} [{song['category']}]"
        lines.extend((f"#EXTINF:-1,{title}", relative))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def clipping_summary(songs: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for kind in ("instrumental", "residual-vocals"):
        items = [item[kind] for item in songs.values()]
        result[kind] = {
            "samplesAboveFullScaleBeforeWrite": sum(
                int(item["samplesAboveFullScaleBeforeWrite"]) for item in items
            ),
            "songsAboveFullScaleBeforeWrite": sum(
                int(item["samplesAboveFullScaleBeforeWrite"]) > 0 for item in items
            ),
            "maxPeakBeforePcm16Clip": max(float(item["peakBeforePcm16Clip"]) for item in items),
        }
    return result


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.threads <= 0 or args.inference_batch_size <= 0:
        raise ValueError("threads and inference-batch-size must be positive")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    pool = reconstruct_training_pool()
    excluded_meta: dict[str, Any] | None = None
    if args.exclude_selection is None:
        selected = select_category_sample(pool, CATEGORY_QUOTAS, args.selection_seed)
        selection_kind = "category-sample"
        selection_algorithm = (
            "fixed category quotas; rotating stage request; stable SHA-256 order; artist-repeat penalty"
        )
    else:
        excluded_path = args.exclude_selection.resolve()
        excluded_document = read_json(excluded_path)
        selected = select_complement(pool, excluded_document)
        selection_kind = "training-pool-complement"
        selection_algorithm = "exact reconstructed training-pool complement of excluded selection slugs"
        excluded_meta = {
            "file": str(excluded_path),
            "sha256": private_renderer.sha256_file(excluded_path),
            "selectionSha256": excluded_document.get("selectionSha256"),
            "songCount": len(excluded_document.get("songs", [])),
        }
    enrich_selection(selected)
    selection = selection_document(
        pool,
        selected,
        args.selection_seed,
        selection_kind=selection_kind,
        algorithm=selection_algorithm,
        excluded_selection=excluded_meta,
    )
    json_write(output_root / "selection.json", selection)
    write_selection_csv(output_root / "selection.csv", selected)
    if args.selection_only:
        print(
            json.dumps(
                {
                    "status": "selection-only",
                    "trainingPoolSongs": len(pool),
                    "selectedSongs": len(selected),
                    "categories": selection["selectedCategoryCounts"],
                    "selection": str((output_root / "selection.json").resolve()),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return 0

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    checkpoint_path = args.checkpoint.resolve()
    model, checkpoint_meta = private_renderer.load_model(
        checkpoint_path,
        args.architecture_checkpoint.resolve(),
        device,
    )
    if checkpoint_meta["step"] != args.expected_step:
        raise ValueError(
            f"Expected checkpoint step {args.expected_step}, got {checkpoint_meta['step']}"
        )

    report_path = output_root / "render-report.json"
    old_report: dict[str, Any] = {}
    if report_path.is_file() and not args.force:
        old_report = read_json(report_path)
    can_resume = (
        old_report.get("selectionSha256") == selection["selectionSha256"]
        and old_report.get("checkpoint", {}).get("sha256") == checkpoint_meta["sha256"]
    )
    report: dict[str, Any] = {
        "schema": "local-fma-training-category-continuous-listening@1",
        "status": "running",
        "semantic": "residual-vocals-to-instrumental",
        "format": "PCM16 FLAC with clipping, no normalization",
        "assembly": CONTRACT.as_dict(assembly="continuous-context-overlap-save"),
        "selection": str((output_root / "selection.json").resolve()),
        "selectionSha256": selection["selectionSha256"],
        "checkpoint": checkpoint_meta,
        "runtime": {
            "device": str(device),
            "threads": args.threads,
            "inferenceBatchSize": args.inference_batch_size,
            "torch": torch.__version__,
            "torchCuda": torch.version.cuda,
            "cudaAvailable": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "platform": platform.platform(),
        },
        "songs": {},
    }
    started = time.perf_counter()
    try:
        for position, song in enumerate(selected, start=1):
            source_path = Path(song["sourcePath"])
            source = fma_audio.load_audio(source_path)
            old_item = old_report.get("songs", {}).get(song["slug"], {}) if can_resume else {}
            if not args.force and old_item and valid_completed_song(old_item, source.shape[0]):
                report["songs"][song["slug"]] = old_item
                print(f"reuse {position}/{len(selected)}: {song['category']} / {song['slug']}", flush=True)
                continue

            print(f"render {position}/{len(selected)}: {song['category']} / {song['slug']}", flush=True)
            residual, timing = continuous.render_student(
                model,
                source,
                CONTRACT,
                device,
                args.inference_batch_size,
            )
            instrumental = np.ascontiguousarray(source - residual, dtype=np.float32)
            if residual.shape != source.shape or instrumental.shape != source.shape:
                raise ValueError(f"Output shape mismatch for {song['slug']}")
            item = {
                "index": song["index"],
                "slug": song["slug"],
                "category": song["category"],
                "selectedForStage": song["selectedForStage"],
                "trainingStages": song["trainingStages"],
                "artistName": song["artistName"],
                "trackName": song["trackName"],
                "source": {
                    "file": str(source_path.resolve()),
                    "sha256": song["sourceSha256"],
                    "bytes": song["sourceBytes"],
                    "sampleRate": SAMPLE_RATE,
                    "channels": 2,
                    "frames": int(source.shape[0]),
                    "durationSeconds": source.shape[0] / SAMPLE_RATE,
                },
                "render": timing,
                "residual-vocals": private_renderer.write_candidate(
                    output_path(output_root, "residual-vocals", song), residual
                ),
                "instrumental": private_renderer.write_candidate(
                    output_path(output_root, "instrumental", song), instrumental
                ),
            }
            report["songs"][song["slug"]] = item
            report["completedSongs"] = len(report["songs"])
            report["elapsedSeconds"] = time.perf_counter() - started
            json_write(report_path, report)
            del source, residual, instrumental
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        report["status"] = "completed"
        report["songCount"] = len(report["songs"])
        report["outputFileCount"] = len(report["songs"]) * 2
        report["categoryCounts"] = selection["selectedCategoryCounts"]
        report["clipping"] = clipping_summary(report["songs"])
        report["elapsedSeconds"] = time.perf_counter() - started
        write_playlist(output_root / "instrumental-playlist.m3u8", selected, "instrumental")
        write_playlist(output_root / "residual-vocals-playlist.m3u8", selected, "residual-vocals")
        report["playlists"] = {
            "instrumental": str((output_root / "instrumental-playlist.m3u8").resolve()),
            "residualVocals": str((output_root / "residual-vocals-playlist.m3u8").resolve()),
        }
        json_write(report_path, report)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["elapsedSeconds"] = time.perf_counter() - started
        json_write(report_path, report)
        raise
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(
        json.dumps(
            {
                "status": report["status"],
                "songs": report["songCount"],
                "files": report["outputFileCount"],
                "categories": report["categoryCounts"],
                "clipping": report["clipping"],
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
