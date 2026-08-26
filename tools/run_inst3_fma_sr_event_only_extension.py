#!/usr/bin/env python3
"""Extend an existing FMA S/R event-only continuation run.

The original runner's pass limit is part of its run contract.  This small
extension runner deliberately starts from its latest checkpoint and reuses
the original cache, whose anchor remains S86-event-only@pass-5.  It avoids
retraining the completed passes while preserving the same weighted S/R loss.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, Iterable

import torch

import run_inst3_fma_sr_event_only_continuation as base
import run_inst3_distill_stability_sweep as sweep
import run_inst3_vr_continuous_topk_local as local
import run_inst3_s_r_continuation as prior


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = ROOT / "data" / "modern-song-fma-sr-event-only-continuation"
DEFAULT_ADDITIONAL_PASSES = 10


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--additional-passes", type=int, default=DEFAULT_ADDITIONAL_PASSES)
    parser.add_argument("--min-passes", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-improvement-db", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=891)
    parser.add_argument("--batch-size", type=int, default=base.BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--anchor-beta", type=float, default=0.25)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-updates", type=int, default=8)
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def checkpoint_path(run_root: Path) -> Path:
    candidates = []
    for path in (run_root / "runs").glob("step-*.pt"):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        if payload.get("format") != base.CHECKPOINT_FORMAT:
            continue
        candidates.append((int(payload.get("step", -1)), path, payload))
    if not candidates:
        raise FileNotFoundError(f"No continuation checkpoint in {run_root / 'runs'}")
    return max(candidates, key=lambda item: item[0])[1]


def load_checkpoint_model(
    path: Path,
    architecture_checkpoint: Path,
    device: torch.device,
    learning_rate: float,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != base.CHECKPOINT_FORMAT:
        raise ValueError(f"Unexpected checkpoint format: {payload.get('format')}")
    model, architecture = base.pilot.make_model(architecture_checkpoint.resolve(), device)
    model.load_state_dict(payload["stateDict"], strict=True)
    model.train()
    sweep.freeze_batchnorm_running_statistics(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    optimizer.load_state_dict(payload["optimizerStateDict"])
    prior.continuation.move_optimizer_state(optimizer, device)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return model, optimizer, payload | {"architecture": architecture["checkpoint"]}


def save_extension_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    payload: dict[str, Any],
    source_path: Path,
    selection: dict[str, Any],
    schedule_hash: str,
    history: list[dict[str, Any]],
    pass_number: int,
    global_step: int,
) -> dict[str, Any]:
    checkpoint = {
        "format": base.CHECKPOINT_FORMAT,
        "status": "milestone",
        "variant": "FMA-SR-event-only-extension",
        "sourceCheckpoint": str(source_path.resolve()),
        "baseCheckpoint": str(source_path.resolve()),
        "baseStep": int(payload["step"]),
        "basePass": int(payload.get("continuationPass", 0)),
        "step": global_step,
        "globalStep": global_step,
        "continuationPass": pass_number,
        "recordsPerPass": base.RECORDS_PER_PASS,
        "updatesPerPass": base.RECORDS_PER_PASS // args.batch_size,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "anchorBeta": 0.25,
        "coreMs": base.CORE_MS,
        "guardMs": base.GUARD_MS,
        "seed": args.seed,
        "scheduleSha256": schedule_hash,
        "selectionSha256": selection["selectionSha256"],
        "lossContract": "S86-event-only masked Inst3 target plus S86 anchor outside 100ms+25ms event mask",
        "stateDict": local.hard.cpu_tree(model.state_dict()),
        "optimizerStateDict": local.hard.cpu_tree(optimizer.state_dict()),
        "history": history,
    }
    local.hard.atomic_torch_save(path, checkpoint)
    return checkpoint


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.additional_passes <= 0 or args.min_passes <= 0 or args.min_passes > args.additional_passes:
        raise ValueError("invalid pass limits")
    if args.patience <= 0 or args.min_improvement_db < 0:
        raise ValueError("invalid stopping settings")
    run_root = args.run_root.resolve()
    old_report_path = run_root / "reports" / "training-report.json"
    old_report = read_json(old_report_path)
    if old_report.get("status") != "completed":
        raise ValueError("The base run is not completed")
    source_path = checkpoint_path(run_root)
    source_payload = torch.load(source_path, map_location="cpu", weights_only=False)
    start_step = int(source_payload["step"])
    start_pass = int(source_payload.get("continuationPass", 0))
    if start_pass <= 0:
        raise ValueError("Base checkpoint has no continuation pass")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    base_args = base.parse_args(
        [
            "--output-root",
            str(run_root),
            "--device",
            str(args.device),
            "--threads",
            str(args.threads),
            "--inference-batch-size",
            str(args.inference_batch_size),
        ]
    )
    groups, records, selection = base.load_pool(base_args)
    selection["sourceCheckpoint"] = base.checkpoint_metadata(base_args.source_checkpoint)
    cache_root = run_root / "cache"
    paths = {
        key: cache_root / f"{base.safe_name(key)}.npz"
        for key in groups
    }
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing reusable cache for {key}: {path}")
        base.validate_arrays(path, len(groups[key]["events"]))

    schedule, schedule_summary = base.build_schedule(records, args.additional_passes, args.seed + start_pass * 1_000_003)
    model, optimizer, loaded_payload = load_checkpoint_model(
        source_path,
        base_args.architecture_checkpoint,
        device,
        args.learning_rate,
    )
    store = base.load_cache_store(paths)
    if args.smoke_only:
        history = base.train_pass(
            model,
            optimizer,
            store,
            schedule,
            0,
            args,
            device,
            max_updates=args.smoke_updates,
        )
        print(json.dumps({"status": "smoke-completed", "updates": len(history)}, indent=2), flush=True)
        del model, optimizer, store
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return 0
    history: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    baseline = base.evaluate_model(model, groups, device, args.inference_batch_size)
    previous_p95 = float(baseline["all"]["100"]["positiveProjectionP95Dbfs"])
    evaluations.append(
        {
            "pass": start_pass,
            "globalStep": start_step,
            "checkpoint": base.checkpoint_metadata(source_path),
            "metrics": baseline,
            "improvementDb": None,
            "role": "extension-baseline",
        }
    )
    print(
        json.dumps(
            {"event": "extension-baseline", "pass": start_pass, "globalStep": start_step, "p95_100ms": previous_p95},
            sort_keys=True,
        ),
        flush=True,
    )
    no_improvement = 0
    extension_root = run_root / "extensions" / f"from-step-{start_step}"
    extension_run_root = extension_root / "runs"
    extension_run_root.mkdir(parents=True, exist_ok=True)
    for extension_index in range(args.additional_passes):
        pass_number = start_pass + extension_index + 1
        pass_history = base.train_pass(
            model,
            optimizer,
            store,
            schedule,
            extension_index,
            args,
            device,
        )
        history.extend(pass_history)
        global_step = start_step + (extension_index + 1) * (base.RECORDS_PER_PASS // args.batch_size)
        checkpoint_file = extension_run_root / f"step-{global_step}.pt"
        saved = save_extension_checkpoint(
            checkpoint_file,
            model,
            optimizer,
            args,
            loaded_payload,
            source_path,
            selection,
            schedule_summary["scheduleSha256"],
            history,
            pass_number,
            global_step,
        )
        metrics = base.evaluate_model(model, groups, device, args.inference_batch_size)
        current_p95 = float(metrics["all"]["100"]["positiveProjectionP95Dbfs"])
        improvement = previous_p95 - current_p95
        evaluations.append(
            {
                "pass": pass_number,
                "globalStep": global_step,
                "checkpoint": base.checkpoint_metadata(checkpoint_file),
                "metrics": metrics,
                "improvementDb": improvement,
                "meaningfulImprovement": improvement >= args.min_improvement_db,
                "role": "extension",
            }
        )
        report = {
            "schema": "local-inst3-fma-sr-event-only-extension@1",
            "status": "running",
            "baseRunRoot": str(run_root),
            "baseReport": str(old_report_path),
            "baseCheckpoint": base.checkpoint_metadata(source_path),
            "selection": selection,
            "schedule": schedule_summary,
            "training": {
                "baseCompletedPasses": start_pass,
                "completedExtensionPasses": extension_index + 1,
                "history": history,
            },
            "evaluations": evaluations,
            "stop": {"patience": args.patience, "minImprovementDb": args.min_improvement_db},
            "environment": {
                "device": str(device),
                "torch": torch.__version__,
                "python": platform.python_version(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
        }
        base.json_write(extension_root / "reports" / "training-report.json", report)
        print(
            json.dumps(
                {
                    "event": "extension-evaluation",
                    "pass": pass_number,
                    "globalStep": global_step,
                    "p95_100ms": current_p95,
                    "improvementDb": improvement,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if improvement >= args.min_improvement_db:
            no_improvement = 0
        else:
            no_improvement += 1
        previous_p95 = current_p95
        if extension_index + 1 >= args.min_passes and no_improvement >= args.patience:
            report["status"] = "completed"
            report["stop"] = {
                "reason": "100ms projection p95 plateau",
                "stoppedAfterPass": pass_number,
                "consecutiveNonMeaningfulPasses": no_improvement,
                "minImprovementDb": args.min_improvement_db,
                "patience": args.patience,
            }
            base.json_write(extension_root / "reports" / "training-report.json", report)
            break
    else:
        report["status"] = "completed"
        report["stop"] = {
            "reason": "extension max-passes guard",
            "additionalPasses": args.additional_passes,
            "minImprovementDb": args.min_improvement_db,
            "patience": args.patience,
        }
        base.json_write(extension_root / "reports" / "training-report.json", report)
    del model, optimizer, store
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str((extension_root / "reports" / "training-report.json").resolve()),
                "completedExtensionPasses": report["training"]["completedExtensionPasses"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
