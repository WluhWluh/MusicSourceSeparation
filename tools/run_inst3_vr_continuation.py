#!/usr/bin/env python3
"""Continue the continuous H50 full-window target for five more passes.

This runner starts from the completed ``H50-continuation@step-800`` checkpoint,
restores its AdamW state, and trains the same 640 continuous central-event
records for five additional passes.  It deliberately does not refresh event
selection or add a local anchor; the only question is whether the observed
listening improvement has more useful learning headroom.

All artifacts are local non-commercial MUSDB18 research outputs.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

import evaluate_inst3_continuous_baseline as baseline
import run_inst3_distill_pilot as pilot
import run_inst3_vr_continuous_topk_local as local
from tfc_tdf_short_window import ShortWindowContract


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "data" / "musdb18-inst3-vr-continuous-topk-local"
DEFAULT_SOURCE = DEFAULT_ROOT / "runs" / "H50-continuation" / "step-800.pt"
DEFAULT_OUTPUT = ROOT / "data" / "musdb18-inst3-vr-continuation"
DEFAULT_CACHE_ROOT = DEFAULT_ROOT
DEFAULT_ORACLE_ROOT = ROOT / "data" / "musdb18-inst3-oracle"
DEFAULT_MANIFEST = DEFAULT_ORACLE_ROOT / "musdb18-inst3-oracle-manifest.json"
DEFAULT_CHECKPOINT = pilot.DEFAULT_CHECKPOINT
DEFAULT_BASELINE_REPORT = (
    ROOT
    / "data"
    / "musdb18-inst3-continuous-baseline-evaluation"
    / "continuous-baseline-report.json"
)
DEFAULT_H50 = (
    ROOT
    / "data"
    / "musdb18-inst3-vr-hard-sampling-h50"
    / "runs"
    / "V-R-H50"
    / "step-8000.pt"
)
MILESTONES = (1, 3, 5)
SOURCE_STEP = 800
RECORDS_PER_PASS = 640
ADDITIONAL_PASSES = 5
SCHEMA = "local-inst3-vr-continuation@1"


def parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or tuple(sorted(set(values))) != values:
        raise ValueError(f"Expected sorted unique integer list, got {raw!r}")
    return values


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--h50-checkpoint", type=Path, default=DEFAULT_H50)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--additional-passes", type=int, default=ADDITIONAL_PASSES)
    parser.add_argument("--milestones", default="1,3,5")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--state-interval", type=int, default=100)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=891)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--inference-batch-size", type=int, default=1)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--skip-listening", action="store_true")
    parser.add_argument("--force-listening", action="store_true")
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


def sha256_file(path: Path) -> str:
    return pilot.sha256_file(path)


def canonical_sha256(value: Any) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    return {"file": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            if isinstance(value, torch.Tensor) and value.device != device:
                state[key] = value.to(device=device)


def load_source(
    architecture_checkpoint: Path,
    source_checkpoint: Path,
    device: torch.device,
    learning_rate: float,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, dict[str, Any]]:
    payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != "local-inst3-vr-continuous-topk-local-checkpoint@1":
        raise ValueError(f"Unexpected source format: {source_checkpoint}")
    if payload.get("variant") != "H50-continuation" or int(payload.get("step", -1)) != SOURCE_STEP:
        raise ValueError(
            f"Expected H50-continuation step {SOURCE_STEP}, got "
            f"{payload.get('variant')} step {payload.get('step')}"
        )
    if not isinstance(payload.get("optimizerStateDict"), dict):
        raise ValueError("Source checkpoint has no optimizer state")
    model, architecture = pilot.make_model(architecture_checkpoint, device)
    model.load_state_dict(payload["stateDict"], strict=True)
    model.train()
    import run_inst3_distill_stability_sweep as sweep

    sweep.freeze_batchnorm_running_statistics(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    optimizer.load_state_dict(payload["optimizerStateDict"])
    move_optimizer_state(optimizer, device)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return model, optimizer, {
        "source": checkpoint_metadata(source_checkpoint),
        "sourceVariant": payload.get("variant"),
        "sourceStep": int(payload["step"]),
        "sourcePasses": int(payload.get("passes", -1)),
        "sourceOptimizerPreserved": True,
        "architecture": architecture["checkpoint"],
    }


def build_offset_schedule(
    slugs: list[str],
    records_per_song: int,
    additional_passes: int,
    seed: int,
    offset_pass: int,
) -> list[tuple[str, int]]:
    schedule: list[tuple[str, int]] = []
    for local_pass in range(additional_passes):
        items = [
            (slug, record_index)
            for slug in sorted(slugs)
            for record_index in range(records_per_song)
        ]
        rng = np.random.default_rng(seed + (offset_pass + local_pass) * 1_000_003)
        schedule.extend(items[int(index)] for index in rng.permutation(len(items)))
    return schedule


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    local_pass: int,
    contract_id: str,
    args: argparse.Namespace,
    history: list[dict[str, Any]],
    source: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "format": "local-inst3-vr-continuation-checkpoint@1",
        "status": "milestone",
        "variant": "H50-continuation-plus5",
        "runContractId": contract_id,
        "step": global_step,
        "globalStep": global_step,
        "sourceStep": SOURCE_STEP,
        "additionalPass": local_pass,
        "passes": args.additional_passes,
        "recordsPerPass": RECORDS_PER_PASS,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "seed": args.seed,
        "stateDict": local.hard.cpu_tree(model.state_dict()),
        "optimizerStateDict": local.hard.cpu_tree(optimizer.state_dict()),
        "history": history,
        "source": source,
    }
    return local.hard.atomic_torch_save(path, payload)


def train(
    args: argparse.Namespace,
    device: torch.device,
    cache_paths: dict[str, Path],
    slugs: list[str],
    schedule: list[tuple[str, int]],
    milestones: tuple[int, ...],
) -> dict[str, Any]:
    if len(schedule) % args.batch_size:
        raise ValueError("schedule is not divisible by batch size")
    records_per_pass = len(schedule) // args.additional_passes
    updates_per_pass = records_per_pass // args.batch_size
    total_updates = len(schedule) // args.batch_size
    output_root = args.output_root.resolve()
    run_root = output_root / "runs" / "H50-continuation-plus5"
    run_root.mkdir(parents=True, exist_ok=True)
    contract_payload = {
        "schema": SCHEMA,
        "sourceCheckpointSha256": sha256_file(args.source_checkpoint.resolve()),
        "cacheRoot": str(args.cache_root.resolve()),
        "cacheSelectionSha256": sha256_file(args.cache_root.resolve() / "selection.json"),
        "recordsPerPass": records_per_pass,
        "additionalPasses": args.additional_passes,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "seed": args.seed,
        "assembly": "continuous-context-overlap-save",
        "target": "full useful-window V_T audio Charbonnier",
        "officialFinalTestUsed": False,
    }
    contract_id = canonical_sha256(contract_payload)
    model, optimizer, source = load_source(
        args.checkpoint.resolve(),
        args.source_checkpoint.resolve(),
        device,
        args.learning_rate,
    )
    store = local.CacheStore(cache_paths, max_open=len(cache_paths))
    store.preload()
    window = torch.hann_window(pilot.DEFAULT_CONFIG.n_fft, periodic=True, device=device)
    history: list[dict[str, Any]] = []
    checkpoint_files: dict[str, dict[str, Any]] = {}
    current_update = 0
    started = time.perf_counter()
    if args.resume:
        existing: list[tuple[int, Path, dict[str, Any]]] = []
        for path in run_root.glob("step-*.pt"):
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                if payload.get("runContractId") != contract_id:
                    continue
                existing.append((int(payload["step"]), path, payload))
            except (OSError, KeyError, TypeError, ValueError, RuntimeError):
                continue
        if existing:
            step, path, payload = max(existing, key=lambda item: item[0])
            model.load_state_dict(payload["stateDict"], strict=True)
            optimizer.load_state_dict(payload["optimizerStateDict"])
            move_optimizer_state(optimizer, device)
            current_update = step - SOURCE_STEP
            history = list(payload.get("history", []))
            print(json.dumps({"event": "resume", "step": step, "file": str(path)}), flush=True)
    import run_inst3_distill_stability_sweep as sweep

    sweep.freeze_batchnorm_running_statistics(model)
    model.train()
    sweep.freeze_batchnorm_running_statistics(model)
    milestone_updates = {value * updates_per_pass: value for value in milestones}
    for local_step, pass_count in milestone_updates.items():
        global_step = SOURCE_STEP + local_step
        path = run_root / f"step-{global_step}.pt"
        if path.is_file():
            checkpoint_files[str(pass_count)] = checkpoint_metadata(path)
    if current_update == 0 and 0 in milestone_updates:
        metadata = save_checkpoint(
            run_root / f"step-{SOURCE_STEP}.pt",
            model,
            optimizer,
            SOURCE_STEP,
            0,
            contract_id,
            args,
            history,
            source,
        )
        checkpoint_files["0"] = metadata
    while current_update < total_updates:
        items = schedule[current_update * args.batch_size : (current_update + 1) * args.batch_size]
        input_array, target_array, _anchor_array, _mask_array = store.batch(items)
        input_tensor = torch.from_numpy(input_array).to(device)
        target_tensor = torch.from_numpy(target_array).to(device)
        optimizer.zero_grad(set_to_none=True)
        predicted_full = local.torch_packed_istft(model(input_tensor), window)
        trim = pilot.DEFAULT_CONFIG.trim_samples
        predicted = predicted_full[:, trim : trim + pilot.DEFAULT_CONFIG.useful_samples]
        loss = local.charbonnier_per_record(
            predicted,
            target_tensor,
            torch.ones(
                (predicted.shape[0], predicted.shape[1]),
                dtype=predicted.dtype,
                device=device,
            ),
        ).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite continuation loss at {current_update + 1}")
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False).item())
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        current_update += 1
        global_step = SOURCE_STEP + current_update
        local_pass = current_update / updates_per_pass
        history.append({
            "update": current_update,
            "globalStep": global_step,
            "pass": local_pass,
            "loss": float(loss.detach().cpu()),
            "gradientNormBeforeClip": gradient_norm,
        })
        if current_update == 1 or current_update % 100 == 0:
            print(json.dumps({"event": "progress", "update": current_update, "globalStep": global_step, "totalUpdates": total_updates, "loss": history[-1]["loss"]}, sort_keys=True), flush=True)
        if current_update in milestone_updates and current_update != 0:
            pass_count = milestone_updates[current_update]
            metadata = save_checkpoint(
                run_root / f"step-{global_step}.pt",
                model,
                optimizer,
                global_step,
                pass_count,
                contract_id,
                args,
                history,
                source,
            )
            checkpoint_files[str(pass_count)] = metadata
            print(json.dumps({"event": "milestone", "pass": pass_count, "globalStep": global_step}, sort_keys=True), flush=True)
        elif args.state_interval > 0 and current_update % args.state_interval == 0:
            save_checkpoint(
                run_root / f"step-{global_step}.pt",
                model,
                optimizer,
                global_step,
                int(math.floor(local_pass)),
                contract_id,
                args,
                history,
                source,
            )
    result = {
        "status": "completed",
        "source": source,
        "sourceStep": SOURCE_STEP,
        "additionalPasses": args.additional_passes,
        "recordsPerPass": records_per_pass,
        "updatesPerPass": updates_per_pass,
        "updates": total_updates,
        "globalFinalStep": SOURCE_STEP + total_updates,
        "learningRate": args.learning_rate,
        "batchSize": args.batch_size,
        "optimizerRestored": True,
        "milestoneCheckpoints": checkpoint_files,
        "contract": {"id": contract_id, "payload": contract_payload},
        "history": history,
        "elapsedSeconds": time.perf_counter() - started,
    }
    del model, optimizer, store, window
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def validate_args(args: argparse.Namespace, milestones: tuple[int, ...]) -> None:
    if args.additional_passes <= 0 or args.batch_size <= 0 or args.threads <= 0:
        raise ValueError("additional-passes, batch-size, and threads must be positive")
    if args.learning_rate <= 0 or args.state_interval <= 0:
        raise ValueError("learning-rate and state-interval must be positive")
    if milestones[0] != 1 or milestones[-1] != args.additional_passes:
        raise ValueError("milestones must start at 1 and end at additional-passes")
    if tuple(sorted(set(milestones))) != milestones:
        raise ValueError("milestones must be sorted and unique")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    milestones = parse_int_list(args.milestones)
    validate_args(args, milestones)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for path in (args.source_checkpoint, args.h50_checkpoint, args.checkpoint, args.manifest, args.baseline_report):
        if not path.resolve().is_file():
            raise FileNotFoundError(path)
    selection_path = args.cache_root.resolve() / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    slugs = sorted(selection["songs"])
    cache_paths = {
        slug: args.cache_root.resolve() / "cache" / "train" / f"{slug}.npz"
        for slug in slugs
    }
    if any(not path.is_file() for path in cache_paths.values()):
        raise FileNotFoundError("One or more continuous training caches are missing")
    schedule = build_offset_schedule(
        slugs,
        int(selection["contract"]["recordsPerSong"]),
        args.additional_passes,
        args.seed,
        offset_pass=5,
    )
    if args.smoke_only:
        # Reuse the normal trainer with a shortened schedule only for a finite
        # backward/finite-value check; no checkpoint is written.
        store = local.CacheStore(cache_paths, max_open=len(cache_paths))
        store.preload()
        model, optimizer, _source = load_source(
            args.checkpoint.resolve(), args.source_checkpoint.resolve(), device, args.learning_rate
        )
        window = torch.hann_window(pilot.DEFAULT_CONFIG.n_fft, periodic=True, device=device)
        losses: list[float] = []
        for index in range(args.smoke_updates):
            items = [schedule[index % len(schedule)]]
            input_array, target_array, _a, _m = store.batch(items)
            input_tensor = torch.from_numpy(input_array).to(device)
            target_tensor = torch.from_numpy(target_array).to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted_full = local.torch_packed_istft(model(input_tensor), window)
            trim = pilot.DEFAULT_CONFIG.trim_samples
            predicted = predicted_full[:, trim : trim + pilot.DEFAULT_CONFIG.useful_samples]
            weights = torch.ones((1, predicted.shape[1]), dtype=predicted.dtype, device=device)
            loss = local.charbonnier_per_record(predicted, target_tensor, weights).mean()
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
            if not torch.isfinite(loss) or not torch.isfinite(gradient):
                raise FloatingPointError("Non-finite continuation smoke")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        result = {"status": "smoke-completed", "updates": args.smoke_updates, "lossFirst": losses[0], "lossLast": losses[-1], "device": str(device)}
        print(json.dumps(result, indent=2), flush=True)
        return 0
    args.additional_passes = int(args.additional_passes)
    training = train(args, device, cache_paths, slugs, schedule, milestones)
    checkpoint_paths = {
        f"H50-continuation+{pass_count}": Path(metadata["file"])
        for pass_count, metadata in training["milestoneCheckpoints"].items()
        if str(pass_count) != "0"
    }
    evaluation = None
    if not args.skip_evaluation:
        eval_entries = sorted(
            [entry for entry in json.loads(args.manifest.resolve().read_text(encoding="utf-8"))["entries"] if entry["role"] in {"calibration", "internal-test"}],
            key=lambda item: (item["role"], item["member"]),
        )
        eval_args = argparse.Namespace(
            baseline_report=args.baseline_report.resolve(),
            output_root=args.output_root.resolve(),
            checkpoint=args.checkpoint.resolve(),
            oracle_root=args.oracle_root.resolve(),
            inference_batch_size=args.inference_batch_size,
        )
        evaluation = local.evaluate_trained_models(
            eval_args,
            ShortWindowContract(num_frames=128, left_context_hops=5, right_context_hops=5),
            checkpoint_paths,
            eval_entries,
            device,
        )
    listening_report = None
    if not args.skip_listening:
        listening_args = argparse.Namespace(
            samples_root=ROOT / "data" / "samples",
            output_root=args.output_root.resolve(),
            checkpoint=args.checkpoint.resolve(),
            force=args.force_listening,
            inference_batch_size=args.inference_batch_size,
        )
        all_paths = {
            "H50-pass-50": args.h50_checkpoint.resolve(),
            "H50-continuation@step-800": args.source_checkpoint.resolve(),
            **checkpoint_paths,
        }
        listening_report = local.render_listening(
            listening_args,
            ShortWindowContract(num_frames=128, left_context_hops=5, right_context_hops=5),
            all_paths,
            device,
        )
    report = {
        "schema": SCHEMA,
        "status": "completed",
        "contract": {
            "sourceCheckpoint": checkpoint_metadata(args.source_checkpoint.resolve()),
            "h50BaseCheckpoint": checkpoint_metadata(args.h50_checkpoint.resolve()),
            "cacheRoot": str(args.cache_root.resolve()),
            "cacheSelectionSha256": sha256_file(selection_path),
            "recordsPerPass": RECORDS_PER_PASS,
            "additionalPasses": args.additional_passes,
            "milestones": list(milestones),
            "batchSize": args.batch_size,
            "learningRate": args.learning_rate,
            "optimizerRestored": True,
            "assembly": "continuous-context-overlap-save",
            "target": "full useful-window V_T audio Charbonnier",
            "officialFinalTestUsed": False,
        },
        "training": training,
        "evaluation": evaluation,
        "listening": listening_report,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchCuda": torch.version.cuda,
            "cudaAvailable": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device": str(device),
            "runner": {"file": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        },
        "notes": {
            "selection": "Same continuous central-event records as H50-continuation; no dynamic refresh in this round.",
            "nextGate": "50 ms positive-projection maximum, top-10 hotspot direction, raw miss maximum, and 12-song listening.",
            "publication": "Local non-commercial research only; do not publish MUSDB18-derived checkpoints or audio.",
        },
    }
    report_path = args.output_root.resolve() / "reports" / "continuation-report.json"
    json_write(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path), "evaluation": list(evaluation or {}), "listening": 0 if listening_report is None else listening_report.get("outputCount", 0)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
