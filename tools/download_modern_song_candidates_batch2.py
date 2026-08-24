#!/usr/bin/env python3
"""Download a surplus FMA-only modern-song candidate batch.

This is an original-audio acquisition step only. It uses FMA metadata already
present locally, excludes all artists from the previous MTG/FMA and modern
candidate pools, and downloads enough distinct artists to fill category quotas.
Failed source URLs are skipped in favor of the next candidate in the same
category. No audio is decoded, probed, resampled, or analyzed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import download_modern_song_candidates as base


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FMA_ZIP = ROOT / ".tmp" / "fma-metadata" / "fma_metadata.zip"
DEFAULT_OUTPUT = ROOT / "data" / "modern-song-original-candidates-batch2"

# Surplus batch. Dance/Electronic has only 12 eligible new artists and
# Latin/Modern only 18 under the one-track-per-artist rule. The first batch
# supplies additional accepted tracks in both categories; do not duplicate
# artists merely to force a nominal quota.
BATCH2_QUOTAS = {
    "pop-synth": 22,
    "hiphop-rnb": 22,
    "dance-electronic": 12,
    "latin-modern": 18,
    "pop-rock": 21,
    "singer-songwriter": 23,
}
BATCH2_COUNT = sum(BATCH2_QUOTAS.values())


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fma-metadata-zip", type=Path, default=DEFAULT_FMA_ZIP)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def current_artists() -> set[str]:
    artists = base.existing_artists()
    for path in (
        ROOT / "data" / "modern-song-original-candidates" / "source-manifest.json",
        ROOT / "data" / "modern-song-original-candidates-batch2" / "source-manifest.json",
    ):
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            artists.update(record["artistName"].strip().casefold() for record in value.get("records", []))
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return artists


def ordered_candidates(zip_path: Path, excluded: set[str]) -> dict[str, list[dict[str, Any]]]:
    values = base.load_fma_candidates(zip_path, excluded)
    result: dict[str, list[dict[str, Any]]] = {category: [] for category in BATCH2_QUOTAS}
    for value in values:
        if not value.get("sourceUrl") or not value.get("licenseUrl"):
            continue
        result[value["category"]].append(value)
    for category in result:
        result[category].sort(
            key=lambda item: (-float(item["rankingScore"]), item["artistName"].casefold(), item["sourceId"])
        )
    return result


def output_record(record: dict[str, Any], order: int, destination: Path) -> dict[str, Any]:
    value = dict(record)
    value.update(
        {
            "batch": "modern-candidates-batch2",
            "order": order,
            "role": "pending-style-review",
            "slug": base.safe_name(f"fma-{record['artistName']}-{record['trackName']}") ,
            "download": base.download_raw(record["downloadUrl"], destination, False),
        }
    )
    return value


def write_review_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = (
        "keep", "order", "category", "artistName", "trackName", "sourceId",
        "languageCode", "genreTags", "durationSecondsMetadata", "license",
        "licenseUrl", "sourceUrl", "file", "notes",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "keep": "",
                    "order": record["order"],
                    "category": record["category"],
                    "artistName": record["artistName"],
                    "trackName": record["trackName"],
                    "sourceId": record["sourceId"],
                    "languageCode": record.get("languageCode", ""),
                    "genreTags": json.dumps(record["genreTags"], ensure_ascii=False),
                    "durationSecondsMetadata": record["durationSeconds"],
                    "license": record["license"],
                    "licenseUrl": record["licenseUrl"],
                    "sourceUrl": record["sourceUrl"],
                    "file": record["download"]["file"],
                    "notes": "",
                }
            )


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not args.fma_metadata_zip.resolve().is_file():
        raise FileNotFoundError(args.fma_metadata_zip)
    excluded = current_artists()
    candidates = ordered_candidates(args.fma_metadata_zip.resolve(), excluded)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(json.dumps({"status": "selected", "quota": BATCH2_QUOTAS, "available": {key: len(value) for key, value in candidates.items()}, "excludedArtists": len(excluded), "output": str(output_root)}, ensure_ascii=False, indent=2))
        return 0
    raw_root = output_root / "raw-originals"
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    used_artists: set[str] = set(excluded)
    order = 0
    for category, quota in BATCH2_QUOTAS.items():
        selected = 0
        for candidate in candidates[category]:
            artist_key = candidate["artistName"].strip().casefold()
            if artist_key in used_artists:
                continue
            order += 1
            destination = raw_root / f"{order:03d}-fma-{base.safe_name(candidate['artistName'] + '-' + candidate['trackName'])}.mp3"
            try:
                record = dict(candidate)
                record["batch"] = "modern-candidates-batch2"
                record["order"] = order
                record["role"] = "pending-style-review"
                record["slug"] = base.safe_name(f"fma-{candidate['artistName']}-{candidate['trackName']}")
                record["download"] = base.download_raw(candidate["downloadUrl"], destination, args.force)
            except Exception as error:  # noqa: BLE001 - preserve failed source URL and continue
                failures.append({"sourceId": candidate["sourceId"], "artistName": candidate["artistName"], "trackName": candidate["trackName"], "url": candidate["downloadUrl"], "error": repr(error)})
                order -= 1
                continue
            records.append(record)
            used_artists.add(artist_key)
            selected += 1
            print(f"downloaded {category} {selected}/{quota}: {candidate['artistName']} - {candidate['trackName']}", flush=True)
            if selected == quota:
                break
        if selected != quota:
            raise RuntimeError(f"Could not download quota for {category}: {selected}/{quota}")
    manifest = {
        "schema": "local-modern-song-original-candidates-batch2@1",
        "status": "completed",
        "selectionPolicy": {
            "batch": "second modern FMA surplus pool",
            "count": BATCH2_COUNT,
            "categoryQuotas": BATCH2_QUOTAS,
            "artistPolicy": "one track per artist; all previous candidate artists excluded",
            "license": "explicit Attribution-family license; NoDerivatives excluded",
            "role": "pending-style-review; no train/validation assignment",
        },
        "audioProcessing": {
            "decoded": False,
            "probed": False,
            "resampled": False,
            "teacherRun": False,
            "studentRun": False,
            "snippetsGenerated": False,
            "metricsComputed": False,
        },
        "excludedArtistCount": len(excluded),
        "failures": failures,
        "records": records,
        "downloadSummary": {
            "fileCount": len(records),
            "categoryCounts": dict(Counter(record["category"] for record in records)),
            "totalBytes": sum(record["download"]["bytes"] for record in records),
            "audioDecodedOrProcessed": False,
        },
    }
    json_write(output_root / "source-manifest.json", manifest)
    write_review_csv(output_root / "original-style-review.csv", records)
    (output_root / "README.txt").write_text(
        "Modern FMA candidate batch 2\n\n"
        "Listen to raw-originals in numeric order. Mark keep=Y/N in\n"
        "original-style-review.csv. Blank means undecided.\n\n"
        "Files are original downloads only. The preparation tool did not decode,\n"
        "probe, resample, separate, or analyze audio.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": manifest["status"], "records": len(records), "bytes": manifest["downloadSummary"]["totalBytes"], "failures": len(failures), "manifest": str((output_root / "source-manifest.json").resolve()), "reviewCsv": str((output_root / "original-style-review.csv").resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
