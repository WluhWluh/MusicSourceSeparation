#!/usr/bin/env python3
"""Merge verified App Live DSP per-run summaries into one analysis table."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)


def discover(inputs: list[Path]) -> list[Path]:
    discovered: set[Path] = set()
    for input_path in inputs:
        resolved = input_path.expanduser().resolve()
        if resolved.is_file():
            if resolved.name != "dsp-matrix-summary.json":
                raise ValueError(f"Unsupported summary file name: {resolved}")
            discovered.add(resolved)
        elif resolved.is_dir():
            discovered.update(resolved.rglob("dsp-matrix-summary.json"))
        else:
            raise FileNotFoundError(f"Input does not exist: {resolved}")
    return sorted(discovered)


def validate(
    path: Path,
    summary: dict[str, Any],
    contract_version: int | None,
) -> tuple[int, list[str], list[dict[str, Any]]]:
    if summary.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported summary schema in {path}")
    current_contract = summary.get("contractVersion")
    if not isinstance(current_contract, int) or current_contract not in (2, 3):
        raise ValueError(f"Unsupported DSP contract in {path}")
    if contract_version is not None and current_contract != contract_version:
        raise ValueError(f"Mixed DSP contracts in {path}")
    if summary.get("status") != "complete" or not summary.get("allRowsQualified"):
        raise ValueError(f"Unqualified DSP run: {path}")
    fields = summary.get("csvFields")
    rows = summary.get("rows")
    if not isinstance(fields, list) or not all(isinstance(value, str) for value in fields):
        raise ValueError(f"Invalid CSV fields in {path}")
    if not isinstance(rows, list) or not rows or len(rows) != summary.get("rowCount"):
        raise ValueError(f"Invalid DSP row count in {path}")
    run_id = summary.get("runId")
    for row in rows:
        if not isinstance(row, dict) or list(row.keys()) != fields:
            raise ValueError(f"DSP row schema mismatch in {path}")
        if row.get("run_id") != run_id or not row.get("qualified"):
            raise ValueError(f"Invalid DSP row identity or gate in {path}")
        if row.get("sample_count") != row.get("measured_runs"):
            raise ValueError(f"Unexpected measured sample count in {path}")
    return current_contract, fields, rows


def parse_args(repository: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[repository.parent / "BSSUploadRelay/results/app-live-mdx-dsp-matrix-v2"],
        help="Summary JSON file or directory containing downloaded relay runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository.parent / "BSSUploadRelay/results/app-live-mdx-dsp-matrix-v2",
    )
    parser.add_argument("--contract-version", type=int, choices=(2, 3))
    return parser.parse_args()


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    args = parse_args(repository)
    paths = discover(args.inputs)
    if not paths:
        raise FileNotFoundError("No dsp-matrix-summary.json files found")
    fields: list[str] | None = None
    contract_version = args.contract_version
    rows_by_key: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    runs: list[dict[str, Any]] = []
    for path in paths:
        summary = read_json(path)
        current_contract, current_fields, rows = validate(path, summary, contract_version)
        contract_version = current_contract
        if fields is None:
            fields = current_fields
        elif current_fields != fields:
            raise ValueError(f"CSV field contract changed in {path}")
        for row in rows:
            key = (
                str(row["run_id"]),
                str(row["shape_id"]),
                int(row["worker_count"]),
                str(row["profile"]),
            )
            if key in rows_by_key and rows_by_key[key] != row:
                raise ValueError(f"Conflicting duplicate DSP row: {key}")
            rows_by_key[key] = row
        first = rows[0]
        runs.append(
            {
                "runId": summary["runId"],
                "model": first["model"],
                "socModel": first["soc_model"],
                "androidRelease": first["android_release"],
                "sdk": first["sdk"],
                "bundleId": summary["bundleId"],
                "sourceCommit": summary["sourceCommit"],
                "uniformNativeWinner": summary["uniformNativeWinner"],
                "nativeWinnerCounts": summary["nativeWinnerCounts"],
                "matrixElapsedMs": first["matrix_elapsed_ms"],
                "thermalStart": first["thermal_start"],
                "thermalEnd": first["thermal_end"],
                "summaryPath": str(path),
            }
        )
    ordered_rows = sorted(
        rows_by_key.values(),
        key=lambda row: (str(row["soc_model"]), str(row["model"]), str(row["run_id"]),
                         str(row["shape_id"]), int(row["worker_count"]), str(row["profile"])),
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "dsp-matrix-batch-summary.csv"
    partial_csv = csv_path.with_name(csv_path.name + ".part")
    with partial_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered_rows)
    os.replace(partial_csv, csv_path)
    json_path = output_dir / "dsp-matrix-batch-summary.json"
    write_json_atomic(
        json_path,
        {
            "schemaVersion": SCHEMA_VERSION,
            "contractVersion": contract_version,
            "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "runCount": len(runs),
            "rowCount": len(ordered_rows),
            "csvFields": fields,
            "runs": sorted(runs, key=lambda value: (value["socModel"], value["model"], value["runId"])),
            "rows": ordered_rows,
        },
    )
    print(f"Merged {len(runs)} run(s), {len(ordered_rows)} row(s)")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
