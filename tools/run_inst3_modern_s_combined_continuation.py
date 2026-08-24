#!/usr/bin/env python3
"""Continue the combined modern-S pilot from both final arm checkpoints.

This runner keeps the combined pilot's schedules and target caches fixed.  It
only changes the initialization point: C0 and C1 each restore their own model
and AdamW state from step 2480, then receive five additional passes.  The
external caches remain tied to the original H50 target checkpoint because
those caches describe the frozen training target, not the student initializer.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Iterable

import torch

import run_inst3_distill_pilot as pilot
import run_inst3_modern_s_combined_pilot as combined
import run_inst3_mtg_fma_c1 as c1
import run_inst3_vr_continuation as continuation
import run_inst3_vr_continuous_topk_local as local
from tfc_tdf_short_window import ShortWindowContract


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "modern-song-s-combined-continuation"
DEFAULT_SOURCE_C0 = (
    ROOT
    / "data"
    / "modern-song-s-combined-pilot"
    / "runs"
    / "C0-combined-control"
    / "step-2480.pt"
)
DEFAULT_SOURCE_C1 = (
    ROOT
    / "data"
    / "modern-song-s-combined-pilot"
    / "runs"
    / "C1-combined-stable-S"
    / "step-2480.pt"
)

SOURCE_STEP = 2480
PASSES = 5
MUSDB_RECORDS_PER_PASS = combined.MUSDB_RECORDS_PER_PASS
EXTERNAL_RECORDS_PER_PASS = combined.EXTERNAL_RECORDS_PER_PASS
RECORDS_PER_PASS = combined.RECORDS_PER_PASS
PREVIOUS_EXTERNAL_PER_PASS = combined.PREVIOUS_EXTERNAL_PER_PASS
CURRENT_EXTERNAL_PER_PASS = combined.CURRENT_EXTERNAL_PER_PASS
ARMS = ("C0-continuation", "C1-continuation")
SOURCE_ARM_FILES = {ARMS[0]: DEFAULT_SOURCE_C0, ARMS[1]: DEFAULT_SOURCE_C1}
CHECKPOINT_FORMAT = "local-inst3-modern-s-combined-continuation-checkpoint@1"
SCHEMA = "local-inst3-modern-s-combined-continuation@1"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-root", type=Path, default=combined.DEFAULT_PREVIOUS_ROOT)
    parser.add_argument("--current-pool", type=Path, default=combined.DEFAULT_CURRENT_POOL)
    parser.add_argument("--current-report", type=Path, default=combined.DEFAULT_CURRENT_REPORT)
    parser.add_argument("--current-manifest", type=Path, default=combined.DEFAULT_CURRENT_MANIFEST)
    parser.add_argument("--current-full-root", type=Path, default=combined.DEFAULT_CURRENT_FULL_ROOT)
    parser.add_argument("--musdb-cache-root", type=Path, default=combined.DEFAULT_MUSDB_CACHE)
    parser.add_argument("--musdb-manifest", type=Path, default=combined.DEFAULT_MUSDB_MANIFEST)
    parser.add_argument("--oracle-root", type=Path, default=combined.DEFAULT_ORACLE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=combined.DEFAULT_ARCHITECTURE)
    parser.add_argument("--base-source-checkpoint", type=Path, default=combined.DEFAULT_SOURCE)
    parser.add_argument("--source-c0", type=Path, default=DEFAULT_SOURCE_C0)
    parser.add_argument("--source-c1", type=Path, default=DEFAULT_SOURCE_C1)
    parser.add_argument("--baseline-report", type=Path, default=combined.DEFAULT_BASELINE_REPORT)
    parser.add_argument("--cache-root", type=Path, default=combined.DEFAULT_OUTPUT)
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
    return {"file": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def configure_modules() -> None:
    """Configure the existing C1 loop for the new source-step contract."""

    combined.ARMS = ARMS
    c1.ARMS = ARMS
    c1.CHECKPOINT_FORMAT = CHECKPOINT_FORMAT
    c1.SOURCE_STEP = SOURCE_STEP
    c1.RECORDS_PER_PASS = RECORDS_PER_PASS
    c1.MUSDB_RECORDS_PER_PASS = MUSDB_RECORDS_PER_PASS
    c1.EXTRA_RECORDS_PER_PASS = EXTERNAL_RECORDS_PER_PASS


def load_source_continuation(
    args: argparse.Namespace, device: torch.device
) -> tuple[torch.nn.Module, torch.optim.Optimizer, dict[str, Any]]:
    payload = torch.load(args.source_checkpoint.resolve(), map_location="cpu", weights_only=False)
    if payload.get("format") not in {
        "local-inst3-modern-s-combined-pilot-checkpoint@1",
        CHECKPOINT_FORMAT,
    }:
        raise ValueError(f"Unexpected continuation source format: {args.source_checkpoint}")
    if int(payload.get("step", -1)) != SOURCE_STEP:
        raise ValueError(f"Expected source step {SOURCE_STEP}, got {payload.get('step')}")
    if not isinstance(payload.get("optimizerStateDict"), dict):
        raise ValueError("Continuation source has no optimizer state")
    model, architecture = pilot.make_model(args.checkpoint.resolve(), device)
    model.load_state_dict(payload["stateDict"], strict=True)
    model.train()
    import run_inst3_distill_stability_sweep as sweep

    sweep.freeze_batchnorm_running_statistics(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    optimizer.load_state_dict(payload["optimizerStateDict"])
    continuation.move_optimizer_state(optimizer, device)
    for group in optimizer.param_groups:
        group["lr"] = args.learning_rate
    return model, optimizer, {
        "checkpoint": checkpoint_metadata(args.source_checkpoint.resolve()),
        "sourceVariant": payload.get("variant"),
        "sourceStep": int(payload["step"]),
        "optimizerRestored": True,
        "architecture": architecture["checkpoint"],
    }


def prepare_data(args: argparse.Namespace, contract: ShortWindowContract) -> tuple[
    dict[str, Path], dict[str, list[tuple[str, int]]], dict[str, Any], dict[str, Any]
]:
    # The external caches are frozen targets generated from the original H50
    # checkpoint. Reuse them from the completed pilot output rather than
    # rebuilding equivalent arrays under the continuation output directory.
    prep_args = copy.copy(args)
    prep_args.output_root = args.cache_root.resolve()
    previous_paths, previous_records = combined.load_previous_pool(
        args.previous_root, args.base_source_checkpoint
    )
    current_paths, current_records, current_metadata = combined.prepare_current_pool(
        prep_args, contract, args.base_source_checkpoint
    )
    external_paths = {**previous_paths, **current_paths}
    musdb_paths, records_per_song, musdb_selection = c1.musdb_cache_paths(args.musdb_cache_root)
    schedules, schedule_details = combined.build_combined_schedules(
        list(musdb_paths), records_per_song, previous_records, current_records, args.passes, args.seed
    )
    expected_report = args.cache_root / "reports" / "combined-s-report.json"
    if expected_report.is_file():
        prior = json.loads(expected_report.read_text(encoding="utf-8"))
        prior_schedules = prior.get("schedules", {})
        for arm, old_name in zip(ARMS, ("C0-combined-control", "C1-combined-stable-S")):
            old_hash = prior_schedules.get(old_name, {}).get("sha256")
            new_hash = combined.schedule_summary(schedules[arm])["sha256"]
            if old_hash and old_hash != new_hash:
                raise ValueError(f"Schedule changed for {arm}: {old_hash} != {new_hash}")
    all_paths = {**musdb_paths, **external_paths}
    selection = {
        "schema": "local-inst3-modern-s-combined-continuation-selection@1",
        "targetSource": checkpoint_metadata(args.base_source_checkpoint.resolve()),
        "previousRecordCount": len(previous_records),
        "currentRecordCount": len(current_records),
        "previousSongCount": len(previous_paths),
        "currentSongCount": len(current_paths),
        "currentSongs": current_metadata,
        "contract": contract.as_dict(assembly="continuous-context-overlap-save"),
    }
    selection["selectionSha256"] = canonical_sha256(selection)
    json_write(args.output_root / "external-selection.json", selection)
    return all_paths, schedules, schedule_details, {
        "selection": selection,
        "musdbSelectionSha256": canonical_sha256(musdb_selection),
    }


def run_training(
    args: argparse.Namespace,
    device: torch.device,
    all_paths: dict[str, Path],
    schedules: dict[str, list[tuple[str, int]]],
    contract: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    original_loader = c1.load_source
    c1.load_source = load_source_continuation
    try:
        results: dict[str, dict[str, Any]] = {}
        for arm, source_path in (
            (ARMS[0], args.source_c0.resolve()),
            (ARMS[1], args.source_c1.resolve()),
        ):
            arm_args = copy.copy(args)
            arm_args.source_checkpoint = source_path
            arm_contract = {**contract, "arm": arm, "sourceCheckpoint": checkpoint_metadata(source_path)}
            milestones = c1.parse_milestones(args.milestones, args.passes)
            results[arm] = c1.train_arm(
                arm,
                all_paths,
                schedules[arm],
                arm_args,
                device,
                milestones,
                arm_contract,
                args.smoke_updates if args.smoke_only else None,
            )
        return results
    finally:
        c1.load_source = original_loader


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    configure_modules()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.passes <= 0 or args.batch_size <= 0 or RECORDS_PER_PASS % args.batch_size:
        raise ValueError("invalid passes or batch size")
    if args.learning_rate <= 0 or args.anchor_beta < 0 or args.threads <= 0:
        raise ValueError("invalid optimizer/runtime arguments")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    required = (
        args.previous_root / "external-selection.json",
        args.current_pool,
        args.current_report,
        args.current_manifest,
        args.musdb_cache_root / "selection.json",
        args.musdb_manifest,
        args.base_source_checkpoint,
        args.source_c0,
        args.source_c1,
        args.checkpoint,
        args.baseline_report,
    )
    for path in required:
        if not path.resolve().is_file():
            raise FileNotFoundError(path)
    contract_obj = ShortWindowContract(num_frames=128, left_context_hops=5, right_context_hops=5)
    all_paths, schedules, schedule_details, selection_info = prepare_data(args, contract_obj)
    schedule_summaries = {arm: combined.schedule_summary(schedules[arm]) for arm in ARMS}
    common_contract = {
        "schema": SCHEMA,
        "targetSource": checkpoint_metadata(args.base_source_checkpoint.resolve()),
        "passes": args.passes,
        "recordsPerPass": RECORDS_PER_PASS,
        "musdbRecordsPerPass": MUSDB_RECORDS_PER_PASS,
        "externalRecordsPerPass": EXTERNAL_RECORDS_PER_PASS,
        "previousExternalPerPass": PREVIOUS_EXTERNAL_PER_PASS,
        "currentExternalPerPass": CURRENT_EXTERNAL_PER_PASS,
        "externalFraction": EXTERNAL_RECORDS_PER_PASS / RECORDS_PER_PASS,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "anchorBeta": args.anchor_beta,
        "seed": args.seed,
        "coreMs": combined.CORE_MS,
        "guardMs": combined.GUARD_MS,
        "assembly": "continuous-context-overlap-save",
        "sourceStep": SOURCE_STEP,
        "optimizerRestored": True,
        "officialFinalTestUsed": False,
    }
    if args.smoke_only:
        training = run_training(args, device, all_paths, schedules, common_contract)
        report = {
            "schema": SCHEMA,
            "status": "smoke-completed",
            "contract": common_contract,
            "schedules": schedule_summaries,
            "scheduleDetails": schedule_details,
            "training": training,
        }
        json_write(args.output_root / "reports" / "continuation-smoke.json", report)
        print(json.dumps({"status": report["status"], "arms": list(training)}, indent=2))
        return 0
    training = run_training(args, device, all_paths, schedules, common_contract)
    final_paths = {
        arm: Path(training[arm]["milestoneCheckpoints"][str(args.passes)]["file"])
        for arm in ARMS
    }
    checkpoint_paths = {
        "Source-H50-continuation+5": args.base_source_checkpoint.resolve(),
        **final_paths,
    }
    musdb_evaluation = None
    if not args.skip_musdb_evaluation:
        manifest = json.loads(args.musdb_manifest.resolve().read_text(encoding="utf-8"))
        entries = sorted(
            [entry for entry in manifest["entries"] if entry["role"] in {"calibration", "internal-test"}],
            key=lambda item: (item["role"], item["member"]),
        )
        eval_args = argparse.Namespace(
            baseline_report=args.baseline_report.resolve(),
            output_root=args.output_root,
            checkpoint=args.checkpoint.resolve(),
            oracle_root=args.oracle_root.resolve(),
            inference_batch_size=args.inference_batch_size,
        )
        musdb_evaluation = local.evaluate_trained_models(
            eval_args, contract_obj, checkpoint_paths, entries, device
        )
        json_write(args.output_root / "reports" / "musdb-evaluation.json", musdb_evaluation)
    listening_report = None
    private_analysis = None
    if not args.skip_listening:
        listening_args = argparse.Namespace(
            samples_root=ROOT / "data" / "samples",
            output_root=args.output_root,
            checkpoint=args.checkpoint.resolve(),
            force=args.force_listening,
            inference_batch_size=args.inference_batch_size,
        )
        listening_report = local.render_listening(listening_args, contract_obj, checkpoint_paths, device)
        private_analysis = c1.analyze_private(listening_report, args.baseline_report)
        json_write(args.output_root / "reports" / "private-inst3-analysis.json", private_analysis)
    report = {
        "schema": SCHEMA,
        "status": "completed",
        "contract": common_contract,
        "selection": selection_info["selection"],
        "scheduleDetails": schedule_details,
        "schedules": schedule_summaries,
        "training": training,
        "checkpoints": {name: checkpoint_metadata(path) for name, path in checkpoint_paths.items()},
        "musdbEvaluation": musdb_evaluation,
        "listening": listening_report,
        "privateInst3Analysis": private_analysis,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchCuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "runner": checkpoint_metadata(Path(__file__).resolve()),
        },
        "licenseDisposition": {
            "scope": "local non-commercial research only",
            "originals": "source audio and derived caches remain local",
            "weights": "teacher-derived checkpoints remain local pending rights review",
        },
    }
    report_path = args.output_root / "reports" / "continuation-report.json"
    json_write(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(report_path),
                "checkpoints": {name: str(path) for name, path in final_paths.items()},
                "listening": 0 if listening_report is None else listening_report.get("outputCount", 0),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
