#!/usr/bin/env python3
"""Run the fixed ten-song Inst 3 aggressive-target generalization pilot.

The experiment compares two otherwise identical 128-frame students:

    alpha=0: MUSDB18 instrumental target
    alpha=1: direct Inst 3 instrumental target

Ten train songs are sampled across their complete duration.  All ten
calibration and ten internal-test songs are held out at song level.  Official
MUSDB18 final-test songs are rejected by construction.  Outputs are local,
non-commercial research artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

import render_inst3_objective_listening as extra_listening
import run_inst3_aggressive_target as aggressive
import run_inst3_distill_pilot as pilot
import run_inst3_distill_stability_sweep as sweep


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "musdb18-inst3-scale10"
DEFAULT_EVAL_ROOT = ROOT / "data" / "musdb18-inst3-oracle"
DEFAULT_EXPERIMENT_ROOT = ROOT / "data" / "musdb18-inst3-aggressive-scale10"
DEFAULT_MANIFEST = ROOT / "data" / "musdb18-inst3-oracle" / "musdb18-inst3-oracle-manifest.json"
DEFAULT_ALPHAS = (0.0, 1.0)
DEFAULT_MILESTONES = (0, 2048, 8192)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=pilot.DEFAULT_CHECKPOINT)
    parser.add_argument("--teacher", type=Path, default=pilot.DEFAULT_TEACHER)
    parser.add_argument("--contract", type=Path, default=pilot.DEFAULT_CONTRACT)
    parser.add_argument("--teacher-tflite", type=Path, default=pilot.DEFAULT_TEACHER_TFLITE)
    parser.add_argument("--train-windows-per-song", type=int, default=32)
    parser.add_argument("--holdout-windows-per-song", type=int, default=16)
    parser.add_argument("--eval-windows-per-song", type=int, default=16)
    parser.add_argument("--steps", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--alphas", default="0.0,1.0")
    parser.add_argument("--milestones", default="0,2048,8192")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=891)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--force-listening", action="store_true")
    parser.add_argument("--skip-extra-listening", action="store_true")
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_teacher_cache(directory: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    metadata_paths = sorted(directory.glob("mixture-gt-*.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"No mixture-gt teacher cache in {directory}")
    metadata_path = metadata_paths[-1]
    metadata = read_json(metadata_path)
    npz_path = Path(metadata["output"]["npz"]["file"])
    if not npz_path.is_file():
        npz_path = metadata_path.with_suffix(".npz")
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    with np.load(npz_path) as values:
        instrumental = np.ascontiguousarray(values["instrumental"], dtype=np.float32)
        vocals = np.ascontiguousarray(values["vocals"], dtype=np.float32)
    return instrumental, vocals, metadata


def load_song(root: Path, entry: dict[str, Any]) -> pilot.SongBundle:
    role = str(entry["role"])
    slug = aggressive.pilot.slugify(entry["fileName"])
    decoded_dir = root / "decoded" / role / slug
    decoded_path = decoded_dir / "decoded.npz"
    metadata_path = decoded_dir / "decoded.json"
    if not decoded_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Missing decoded cache for {role}/{slug}")
    metadata = read_json(metadata_path)
    with np.load(decoded_path) as values:
        # The student objective only needs the exact mixture, vocal reference,
        # and instrumental reference.  The complete five-stream decode remains
        # on disk, but auxiliary stems are not copied into the training heap.
        arrays = {
            name: np.ascontiguousarray(values[name], dtype=np.float32)
            for name in ("mixtureGt", "vocals", "instrumentalGt")
        }
    teacher_instrumental, teacher_vocals, teacher_metadata = find_teacher_cache(
        root / "teacher" / role / slug
    )
    if arrays["mixtureGt"].shape != teacher_instrumental.shape:
        raise ValueError(f"Teacher length mismatch for {slug}")
    return pilot.SongBundle(
        role=role,
        member=str(entry["member"]),
        source_sha256=str(entry["sourceSha256"]),
        source_path=Path(metadata["sourceFile"]),
        slug=slug,
        mixture_encoded=arrays["mixtureGt"],
        mixture_gt=arrays["mixtureGt"],
        vocals=arrays["vocals"],
        instrumental=arrays["instrumentalGt"],
        drums=np.empty((0, 2), dtype=np.float32),
        bass=np.empty((0, 2), dtype=np.float32),
        other=np.empty((0, 2), dtype=np.float32),
        sample_rate=int(metadata["sampleRate"]),
        teacher_instrumental=teacher_instrumental,
        teacher_vocals=teacher_vocals,
        teacher_metadata=teacher_metadata,
    )


def select_coverage_windows(
    candidates: list[aggressive.pilot.WindowRecord], count: int, seed: int
) -> list[aggressive.pilot.WindowRecord]:
    """Select windows with both temporal coverage and activity stratification."""
    candidates = [
        item
        for item in candidates
        if item.length >= aggressive.pilot.DEFAULT_CONFIG.useful_samples // 3
    ]
    if count <= 0:
        raise ValueError("window count must be positive")
    if count >= len(candidates):
        return list(candidates)
    selected_indices: set[int] = set()
    coverage_count = min(count // 2, len(candidates))
    for raw in np.linspace(0, len(candidates) - 1, coverage_count, dtype=int):
        selected_indices.add(int(raw))
    remaining = count - len(selected_indices)
    if remaining > 0:
        available = [item for index, item in enumerate(candidates) if index not in selected_indices]
        stratified, _ = sweep.select_stratified_windows(available, remaining)
        index_by_identity = {id(item): index for index, item in enumerate(candidates)}
        selected_indices.update(index_by_identity[id(item)] for item in stratified)
    if len(selected_indices) < count:
        rng = np.random.default_rng(seed)
        available = [index for index in range(len(candidates)) if index not in selected_indices]
        selected_indices.update(rng.choice(available, size=count - len(selected_indices), replace=False).tolist())
    return [candidates[index] for index in sorted(selected_indices)]


def compact_song(
    song: pilot.SongBundle,
    groups: list[tuple[str, list[pilot.WindowRecord]]],
) -> tuple[pilot.SongBundle, dict[str, list[pilot.WindowRecord]]]:
    """Retain only selected windows while preserving each window's samples."""
    if song.teacher_instrumental is None:
        raise ValueError(f"Teacher output missing for {song.slug}")
    selected = [record for _, records in groups for record in records]
    if not selected:
        raise ValueError(f"No selected windows for {song.slug}")

    def concatenate(source: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(
            np.concatenate(
                [source[record.start : record.start + record.length] for record in selected],
                axis=0,
            ),
            dtype=np.float32,
        )

    mixture = concatenate(song.mixture_gt)
    compact = pilot.SongBundle(
        role=song.role,
        member=song.member,
        source_sha256=song.source_sha256,
        source_path=song.source_path,
        slug=song.slug,
        mixture_encoded=mixture,
        mixture_gt=mixture,
        vocals=concatenate(song.vocals),
        instrumental=concatenate(song.instrumental),
        drums=np.empty((0, 2), dtype=np.float32),
        bass=np.empty((0, 2), dtype=np.float32),
        other=np.empty((0, 2), dtype=np.float32),
        sample_rate=song.sample_rate,
        teacher_instrumental=concatenate(song.teacher_instrumental),
        teacher_vocals=None,
        teacher_metadata=song.teacher_metadata,
    )
    compact_records: dict[str, list[pilot.WindowRecord]] = {
        name: [] for name, _ in groups
    }
    offset = 0
    for name, records in groups:
        for original in records:
            compact_records[name].append(
                pilot.WindowRecord(
                    compact,
                    offset,
                    original.length,
                    original.rms,
                    original.vocal_rms,
                )
            )
            offset += original.length
    if offset != compact.mixture_gt.shape[0]:
        raise AssertionError(f"Compact length mismatch for {song.slug}")
    return compact, compact_records


def prepare_compact_data(
    data_root: Path,
    eval_root: Path,
    selected_train_entries: list[dict[str, Any]],
    eval_entries: list[dict[str, Any]],
    *,
    train_count: int,
    holdout_count: int,
    eval_count: int,
    seed: int,
) -> tuple[
    list[pilot.SongBundle],
    list[pilot.SongBundle],
    list[pilot.WindowRecord],
    dict[str, list[pilot.WindowRecord]],
    dict[str, list[pilot.WindowRecord]],
    dict[str, list[pilot.WindowRecord]],
    dict[str, Any],
]:
    train_songs: list[pilot.SongBundle] = []
    eval_songs: list[pilot.SongBundle] = []
    train_records: list[pilot.WindowRecord] = []
    holdout_records: dict[str, list[pilot.WindowRecord]] = {}
    eval_records: dict[str, list[pilot.WindowRecord]] = {}
    all_records: dict[str, list[pilot.WindowRecord]] = {}
    report: dict[str, Any] = {}

    for song_index, entry in enumerate(selected_train_entries):
        print(f"compact train {song_index + 1}/{len(selected_train_entries)}: {entry['member']}", flush=True)
        full_song = load_song(data_root, entry)
        candidates = sweep.candidate_windows(full_song)
        selected = select_coverage_windows(candidates, train_count, seed + song_index * 2)
        selected_ids = {id(item) for item in selected}
        remaining = [item for item in candidates if id(item) not in selected_ids]
        holdout = select_coverage_windows(
            remaining, min(holdout_count, len(remaining)), seed + song_index * 2 + 1
        )
        compact, groups = compact_song(
            full_song, [("train", selected), ("holdout", holdout)]
        )
        train_songs.append(compact)
        train_records.extend(groups["train"])
        holdout_records[compact.slug] = groups["holdout"]
        all_records[compact.slug] = groups["holdout"]
        report[compact.slug] = {
            "role": compact.role,
            "totalWindowCount": len(candidates),
            "trainOriginalTimeline": sweep.window_summary(selected),
            "holdoutOriginalTimeline": sweep.window_summary(holdout),
        }
        del full_song

    for song_index, entry in enumerate(eval_entries):
        print(f"compact held-out {song_index + 1}/{len(eval_entries)}: {entry['member']}", flush=True)
        full_song = load_song(eval_root, entry)
        candidates = sweep.candidate_windows(full_song)
        selected = select_coverage_windows(
            candidates, min(eval_count, len(candidates)), seed + 1000 + song_index * 2
        )
        compact, groups = compact_song(full_song, [("evaluation", selected)])
        eval_songs.append(compact)
        eval_records[compact.slug] = groups["evaluation"]
        all_records[compact.slug] = groups["evaluation"]
        report[compact.slug] = {
            "role": compact.role,
            "totalWindowCount": len(candidates),
            "evaluationOriginalTimeline": sweep.window_summary(selected),
        }
        del full_song

    return (
        train_songs,
        eval_songs,
        train_records,
        eval_records,
        holdout_records,
        all_records,
        report,
    )


def validate_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if manifest.get("manifestId") != "musdb18-inst3-oracle-split@1":
        raise ValueError("Unexpected frozen manifest")
    counts = {
        role: sum(1 for entry in manifest["entries"] if entry["role"] == role)
        for role in ("train", "calibration", "internal-test", "final-test")
    }
    if counts != {"train": 80, "calibration": 10, "internal-test": 10, "final-test": 50}:
        raise ValueError(f"Unexpected split counts: {counts}")
    return manifest


def select_entries(manifest: dict[str, Any], train_slugs: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [entry for entry in manifest["entries"] if entry["role"] == "train"]
    train.sort(key=lambda entry: (entry["rank"], entry["member"]))
    selected_train = train[: len(train_slugs)]
    if {pilot.slugify(entry["fileName"]) for entry in selected_train} != train_slugs:
        raise ValueError("Scale10 selection does not match the first rank-ordered train songs")
    held_out = [entry for entry in manifest["entries"] if entry["role"] in {"calibration", "internal-test"}]
    held_out.sort(key=lambda entry: (entry["role"], entry["member"]))
    if any(entry["role"] == "final-test" for entry in selected_train + held_out):
        raise AssertionError("final-test entry selected")
    return selected_train, held_out


def parse_numbers(raw: str, integer: bool = False) -> tuple[float | int, ...]:
    values: list[float | int] = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item) if integer else float(item))
    if not values:
        raise ValueError("empty number list")
    return tuple(values)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.train_windows_per_song <= 0 or args.holdout_windows_per_song <= 0 or args.eval_windows_per_song <= 0:
        raise ValueError("window counts must be positive")
    if args.steps <= 0 or args.learning_rate <= 0 or args.threads <= 0:
        raise ValueError("steps, learning-rate, and threads must be positive")
    alphas = tuple(float(value) for value in parse_numbers(args.alphas))
    milestones = tuple(int(value) for value in parse_numbers(args.milestones, integer=True))
    if alphas != (0.0, 1.0):
        raise ValueError("This controlled run must compare exactly alpha=0 and alpha=1")
    if milestones[0] != 0 or milestones[-1] != args.steps or tuple(sorted(set(milestones))) != milestones:
        raise ValueError("milestones must be sorted, unique, start at 0, and end at steps")
    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    manifest_path = args.manifest.resolve()
    manifest = validate_manifest(manifest_path)
    selection = read_json(args.data_root.resolve() / "scale10-selection.json")
    selected_train_entries, eval_entries = select_entries(
        manifest, {str(item["slug"]) for item in selection["entries"]}
    )
    if selection.get("trainCount") != 10 or len(selected_train_entries) != 10:
        raise ValueError("This run requires exactly ten selected train songs")

    data_root = args.data_root.resolve()
    eval_root = args.eval_root.resolve()
    (
        train_songs,
        eval_songs,
        train_records,
        eval_records,
        holdout_records,
        all_records,
        window_report,
    ) = prepare_compact_data(
        data_root,
        eval_root,
        selected_train_entries,
        eval_entries,
        train_count=args.train_windows_per_song,
        holdout_count=args.holdout_windows_per_song,
        eval_count=args.eval_windows_per_song,
        seed=args.seed,
    )
    all_songs = train_songs + eval_songs
    if any(song.role == "final-test" for song in all_songs):
        raise AssertionError("final-test data entered the run")
    # The final aggregate contains the selected held-out train windows and all
    # selected calibration/internal-test windows, never the training windows.
    all_eval_songs = eval_songs + train_songs
    eval_plus_holdout = dict(all_records)
    initial_model, checkpoint_metadata = aggressive.pilot.make_model(args.checkpoint.resolve(), device)
    baseline_probe = sweep.predicted_probe_specs(
        initial_model,
        [record for records in eval_records.values() for record in records],
        device,
    )
    baseline_by_alpha: dict[str, dict[str, Any]] = {}
    for alpha in alphas:
        print(f"baseline evaluation: alpha={alpha:.2f}", flush=True)
        targets = aggressive.AudioDomainTargetCache(alpha)
        baseline_by_alpha[aggressive.alpha_name(alpha)] = {
            "calibrationInternal": aggressive.evaluate_model(
                f"initial-alpha-{alpha}", initial_model, eval_songs, eval_records, targets, device
            ),
            "trainHoldout": aggressive.evaluate_model(
                f"initial-alpha-{alpha}-holdout", initial_model, train_songs, holdout_records, targets, device
            ),
            "selectedEvaluation": aggressive.evaluate_model(
                f"initial-alpha-{alpha}-selected", initial_model, train_songs + eval_songs, eval_plus_holdout, targets, device
            ),
        }
    del initial_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    trajectories: dict[str, dict[str, Any]] = {}
    experiment_root = args.experiment_root.resolve()
    for alpha in alphas:
        key = aggressive.alpha_name(alpha)
        baseline = baseline_by_alpha[key]
        trajectories[key] = aggressive.train_alpha(
            alpha=alpha,
            initial_checkpoint=args.checkpoint.resolve(),
            train_records=train_records,
            eval_records=eval_records,
            holdout_records=holdout_records,
            all_records=eval_plus_holdout,
            eval_songs=eval_songs,
            all_songs=all_eval_songs,
            baseline_probe=baseline_probe,
            baseline_eval=baseline["calibrationInternal"],
            baseline_holdout=baseline["trainHoldout"],
            baseline_full=baseline["selectedEvaluation"],
            device=device,
            steps=args.steps,
            learning_rate=args.learning_rate,
            milestones=milestones,
            seed=args.seed,
            experiment_root=experiment_root,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    extra_report: dict[str, Any] | None = None
    if not args.skip_extra_listening:
        extra_report = aggressive.render_extra_listening(
            initial_checkpoint=args.checkpoint.resolve(),
            trajectories=trajectories,
            start_seconds=15.0,
            duration_seconds=30.0,
            output_root=experiment_root / "listening-extra",
            teacher_path=args.teacher.resolve(),
            teacher_contract=args.contract.resolve(),
            teacher_tflite=args.teacher_tflite.resolve(),
            threads=args.threads,
            device=device,
            require_teacher_cuda=False,
            force=args.force_listening,
        )

    report = {
        "schemaVersion": 1,
        "status": "completed",
        "experimentId": "inst3-aggressive-scale10@1",
        "licenseDisposition": {
            "scope": "local non-commercial research only",
            "musdb18": "educational/non-commercial source audio; not redistributed",
            "inst3": "source weight redistribution permission not established; not redistributed",
            "derivedWeights": "local aggressive-target artifacts only; do not publish",
        },
        "manifest": {
            "file": str(manifest_path),
            "sha256": pilot.sha256_file(manifest_path),
            "manifestId": manifest["manifestId"],
            "selectionFile": str((data_root / "scale10-selection.json").resolve()),
            "selectedTrain": [entry["member"] for entry in selected_train_entries],
            "calibration": [entry["member"] for entry in eval_entries if entry["role"] == "calibration"],
            "internalTest": [entry["member"] for entry in eval_entries if entry["role"] == "internal-test"],
            "finalTestUsed": False,
        },
        "dataContract": {
            "inputSemantic": "mixture-gt",
            "mixtureGtDefinition": "vocals + drums + bass + other",
            "studentOutputSemantic": "instrumental",
            "neuralCoreSemantic": "vocals-residual",
            "targetDefinition": "(1 - alpha) * MUSDB18 instrumental + alpha * Inst 3 instrumental in audio domain before student STFT",
        },
        "split": {
            "trainSongCount": len(train_songs),
            "calibrationSongCount": sum(song.role == "calibration" for song in eval_songs),
            "internalTestSongCount": sum(song.role == "internal-test" for song in eval_songs),
            "train": [song.slug for song in train_songs],
            "calibration": [song.slug for song in eval_songs if song.role == "calibration"],
            "internalTest": [song.slug for song in eval_songs if song.role == "internal-test"],
            "officialFinalTestUsed": False,
        },
        "student": {
            "checkpoint": checkpoint_metadata["checkpoint"],
            "expectedCheckpointSha256": pilot.EXPECTED_CHECKPOINT_SHA256,
            "device": str(device),
            "cuda": {
                "available": torch.cuda.is_available(),
                "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "torchVersion": torch.__version__,
                "cudaVersion": torch.version.cuda,
            },
            "batchNormTrainingPolicy": "running-statistics-frozen; model kept in eval mode during gradient updates",
        },
        "training": {
            "seed": args.seed,
            "alphas": alphas,
            "trainWindowsPerSong": args.train_windows_per_song,
            "holdoutWindowsPerSong": args.holdout_windows_per_song,
            "evalWindowsPerSong": args.eval_windows_per_song,
            "trainWindowCount": len(train_records),
            "steps": args.steps,
            "milestones": milestones,
            "learningRate": args.learning_rate,
            "optimizer": "AdamW(weight_decay=0)",
            "gradientClipNorm": 1.0,
            "lossDefinition": "L1 packed student spectrum against STFT(audio-domain mixed target)",
            "trajectories": trajectories,
        },
        "baseline": baseline_by_alpha,
        "windowSelection": window_report,
        "extraListening": extra_report,
        "notes": {
            "qualityIntent": "Aggressive karaoke/vocal removal; standard instrumental fidelity is a safety diagnostic, not the sole quality gate.",
            "fullSongSampling": "Candidates cover each complete decoded song contiguously; selected train/evaluation windows combine temporal coverage with vocal/mixture RMS stratification.",
            "finalTestIsolation": "Official final-test songs were not extracted, loaded, or evaluated.",
            "publication": "Do not publish checkpoints, teacher-derived audio, or MUSDB18 audio.",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "runner": {
                "file": str(Path(__file__).resolve()),
                "sha256": pilot.sha256_file(Path(__file__).resolve()),
            },
        },
    }
    report_path = experiment_root / "reports" / "inst3-aggressive-scale10-report.json"
    pilot.json_write(report_path, report)
    print(json.dumps({"status": report["status"], "device": str(device), "report": str(report_path), "variants": list(trajectories)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
