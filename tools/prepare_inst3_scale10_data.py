#!/usr/bin/env python3
"""Prepare the fixed ten-song Inst 3 generalization data set.

This runner extracts and decodes ten deterministic MUSDB18 train songs in
full, then renders the frozen Inst 3 teacher on their exact stem-sum mixture.
It is deliberately separate from the existing ten-song calibration and
internal-test oracle.  All outputs are local, ignored, non-commercial
research artifacts.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import run_inst3_distill_pilot as pilot
import run_inst3_teacher_oracle as oracle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT / "data" / "musdb18-inst3-oracle" / "musdb18-inst3-oracle-manifest.json"
)
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "musdb18-inst3-scale10"
DEFAULT_TRAIN_COUNT = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=oracle.DEFAULT_ARCHIVE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--contract", type=Path, default=oracle.DEFAULT_CONTRACT)
    parser.add_argument("--teacher", type=Path, default=oracle.DEFAULT_TEACHER)
    parser.add_argument(
        "--teacher-tflite", type=Path, default=oracle.DEFAULT_TEACHER_TFLITE
    )
    parser.add_argument("--train-count", type=int, default=DEFAULT_TRAIN_COUNT)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--force-decode", action="store_true")
    parser.add_argument("--force-teacher", action="store_true")
    parser.add_argument(
        "--keep-decoded-wav",
        action="store_true",
        help="Keep intermediate float32 WAV files after NPZ creation",
    )
    parser.add_argument(
        "--require-teacher-cuda",
        action="store_true",
        help="Fail instead of silently falling back to CPU",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("manifestId") != "musdb18-inst3-oracle-split@1":
        raise ValueError(f"Unexpected frozen manifest: {manifest.get('manifestId')}")
    counts = {
        role: sum(1 for entry in manifest.get("entries", []) if entry["role"] == role)
        for role in ("train", "calibration", "internal-test", "final-test")
    }
    if counts != {"train": 80, "calibration": 10, "internal-test": 10, "final-test": 50}:
        raise ValueError(f"Unexpected split counts: {counts}")
    return manifest


def select_train_entries(manifest: dict[str, Any], count: int) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("train-count must be positive")
    entries = [entry for entry in manifest["entries"] if entry["role"] == "train"]
    entries.sort(key=lambda entry: (entry["rank"], entry["member"]))
    if count > len(entries):
        raise ValueError(f"Requested {count} train songs but only {len(entries)} exist")
    selected = entries[:count]
    if any(entry["role"] != "train" for entry in selected):
        raise AssertionError("non-train entry selected")
    return selected


def main() -> int:
    args = parse_args()
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    if args.require_teacher_cuda and not args.teacher.is_file():
        raise FileNotFoundError(args.teacher)

    archive = args.archive.resolve()
    manifest_path = args.manifest.resolve()
    output_root = args.output_root.resolve()
    manifest = read_manifest(manifest_path)
    selected = select_train_entries(manifest, args.train_count)

    output_root.mkdir(parents=True, exist_ok=True)
    selection_path = output_root / "scale10-selection.json"
    selection = {
        "schemaVersion": 1,
        "selectionId": "musdb18-inst3-scale10-train@1",
        "selectionRule": "frozen manifest train entries sorted by rank then member; take first train-count",
        "manifestId": manifest["manifestId"],
        "manifestFile": str(manifest_path),
        "manifestSha256": oracle.sha256_file(manifest_path),
        "trainCount": len(selected),
        "entries": [
            {
                "member": entry["member"],
                "fileName": entry["fileName"],
                "slug": oracle.slugify(entry["fileName"]),
                "role": entry["role"],
                "rank": entry["rank"],
                "sourceSha256": entry["sourceSha256"],
            }
            for entry in selected
        ],
        "excludedRoles": ["calibration", "internal-test", "final-test"],
    }
    oracle.json_write(selection_path, selection)

    contract = oracle.verify_teacher_contract(
        args.contract.resolve(), args.teacher.resolve(), args.teacher_tflite.resolve()
    )
    pilot.inspect_archive(archive, [entry["member"] for entry in selected])
    raw_root = output_root / "raw"
    decoded_root = output_root / "decoded"
    teacher_root = output_root / "teacher"
    session, providers = oracle.make_teacher_session(
        args.teacher.resolve(), args.threads, args.require_teacher_cuda
    )

    songs: list[dict[str, Any]] = []
    try:
        for index, entry in enumerate(selected, start=1):
            print(f"prepare {index}/{len(selected)}: {entry['member']}", flush=True)
            source = oracle.extract_member(
                archive, entry, raw_root, args.force_extract
            )
            song = oracle.decode_song(
                archive,
                entry,
                raw_root,
                decoded_root,
                args.force_extract,
                args.force_decode,
                args.keep_decoded_wav,
            )
            _, _, teacher_metadata = oracle.render_or_load_cached(
                output_root,
                song,
                "mixture-gt",
                song.mixture_gt,
                session,
                contract,
                oracle.sha256_file(Path(__file__).resolve()),
                args.force_teacher,
            )
            songs.append(
                {
                    "slug": song.slug,
                    "member": entry["member"],
                    "role": entry["role"],
                    "sourceSha256": entry["sourceSha256"],
                    "sourceFile": str(source.resolve()),
                    "decoded": song.metadata,
                    "teacher": teacher_metadata,
                }
            )
            del song
    finally:
        del session

    report = {
        "schemaVersion": 1,
        "status": "prepared",
        "experimentId": "inst3-scale10-data@1",
        "licenseDisposition": {
            "scope": "local non-commercial research only",
            "musdb18": "educational/non-commercial source audio; not redistributed",
            "inst3": "source weight redistribution permission not established; not redistributed",
            "derivedOutputs": "ignored local caches only; do not publish",
        },
        "archive": {
            "file": str(archive),
            "bytes": archive.stat().st_size,
        },
        "selection": selection,
        "contract": contract,
        "teacher": {
            "providers": providers,
            "cudaRequired": args.require_teacher_cuda,
            "songs": songs,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "runner": {
                "file": str(Path(__file__).resolve()),
                "sha256": oracle.sha256_file(Path(__file__).resolve()),
            },
        },
    }
    report_path = output_root / "reports" / "inst3-scale10-data-report.json"
    oracle.json_write(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "trainSongs": [item["slug"] for item in songs],
                "providers": providers,
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
