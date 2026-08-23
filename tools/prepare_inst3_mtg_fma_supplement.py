#!/usr/bin/env python3
"""Select, download, and teacher-cache a small MTG-Jamendo/FMA supplement.

The selected records are intentionally fixed and auditable.  They are chosen
for vocal presence, genre/language diversity, artist diversity, and explicit
non-commercial licenses without a NoDerivatives restriction.  The resulting
audio and Inst 3 pseudo-targets remain local research artifacts.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import render_inst3_objective_listening as listening
import run_inst3_distill_pilot as pilot
import run_inst3_teacher_oracle as oracle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MTG_METADATA = ROOT / ".tmp" / "mtg-metadata"
DEFAULT_FMA_METADATA = ROOT / ".tmp" / "fma-metadata" / "fma_metadata.zip"
DEFAULT_OUTPUT = ROOT / "data" / "inst3-mtg-fma-supplement"
SCHEMA = "local-inst3-mtg-fma-supplement@1"

# The first eight are intended for a supplemental training pilot.  The last
# four are held out at source level for a later supplemental validation pass.
SELECTED = (
    {
        "source": "mtg-jamendo",
        "id": "track_0072836",
        "role": "supplement-train",
        "rationale": "French chanson/blues vocal; explicit voice tag",
    },
    {
        "source": "mtg-jamendo",
        "id": "track_0200050",
        "role": "supplement-train",
        "rationale": "metal vocal onset and dense accompaniment; explicit voice tag",
    },
    {
        "source": "mtg-jamendo",
        "id": "track_1254619",
        "role": "supplement-train",
        "rationale": "Latin vocal with featured singer; explicit voice tag",
    },
    {
        "source": "mtg-jamendo",
        "id": "track_1158501",
        "role": "supplement-train",
        "rationale": "short blues vocal phrasing; explicit voice tag",
    },
    {
        "source": "fma",
        "id": "134",
        "role": "supplement-train",
        "rationale": "English hip-hop consonant and rap onsets; BY-NC-SA",
    },
    {
        "source": "fma",
        "id": "67733",
        "role": "supplement-train",
        "rationale": "Portuguese pop/electronic vocal; BY-NC-SA",
    },
    {
        "source": "fma",
        "id": "41502",
        "role": "supplement-train",
        "rationale": "Spanish pop/Latin vocal; BY-NC",
    },
    {
        "source": "fma",
        "id": "44807",
        "role": "supplement-train",
        "rationale": "English country/pop singer-songwriter phrasing; BY-NC",
    },
    {
        "source": "mtg-jamendo",
        "id": "track_1082696",
        "role": "supplement-holdout",
        "rationale": "electronic vocal-effect event; explicit voice tag",
    },
    {
        "source": "mtg-jamendo",
        "id": "track_1131292",
        "role": "supplement-holdout",
        "rationale": "ambient/choral vocal texture; explicit voice tag",
    },
    {
        "source": "fma",
        "id": "36373",
        "role": "supplement-holdout",
        "rationale": "German rock vocal; BY-NC-SA",
    },
    {
        "source": "fma",
        "id": "67605",
        "role": "supplement-holdout",
        "rationale": "international/folk vocal; BY-NC-SA",
    },
)

SELECTED_EXPANSION = (
    {
        "source": "mtg-jamendo",
        "id": "track_0000382",
        "role": "candidate-pool",
        "rationale": "French/European choral vocal and harmony texture",
    },
    {
        "source": "mtg-jamendo",
        "id": "track_0076656",
        "role": "candidate-pool",
        "rationale": "French folk vocal phrasing",
    },
    {
        "source": "mtg-jamendo",
        "id": "track_0340016",
        "role": "candidate-pool",
        "rationale": "short classical vocal/choir event",
    },
    {
        "source": "mtg-jamendo",
        "id": "track_1054098",
        "role": "candidate-pool",
        "rationale": "Russian-titled electronic vocal context",
    },
    {
        "source": "mtg-jamendo",
        "id": "track_1119469",
        "role": "candidate-pool",
        "rationale": "Eastern-European-titled electronic vocal context",
    },
    {
        "source": "mtg-jamendo",
        "id": "track_1159658",
        "role": "candidate-pool",
        "rationale": "folk vocal and consonant phrasing",
    },
    {
        "source": "mtg-jamendo",
        "id": "track_1159861",
        "role": "candidate-pool",
        "rationale": "indie vocal texture",
    },
    {
        "source": "mtg-jamendo",
        "id": "track_1253254",
        "role": "candidate-pool",
        "rationale": "short ambient vocal/effect context",
    },
    {
        "source": "fma",
        "id": "461",
        "role": "candidate-pool",
        "rationale": "explicit Jazz: Vocal tag",
    },
    {
        "source": "fma",
        "id": "9382",
        "role": "candidate-pool",
        "rationale": "experimental electronic singer-songwriter",
    },
    {
        "source": "fma",
        "id": "17623",
        "role": "candidate-pool",
        "rationale": "Spanish electronic vocal",
    },
    {
        "source": "fma",
        "id": "41491",
        "role": "candidate-pool",
        "rationale": "Spanish metal/hardcore vocal",
    },
    {
        "source": "fma",
        "id": "67732",
        "role": "candidate-pool",
        "rationale": "Portuguese pop/electronic vocal",
    },
    {
        "source": "fma",
        "id": "78833",
        "role": "candidate-pool",
        "rationale": "Portuguese hip-hop vocal onset",
    },
    {
        "source": "fma",
        "id": "81337",
        "role": "candidate-pool",
        "rationale": "French classical/experimental vocal",
    },
    {
        "source": "fma",
        "id": "100482",
        "role": "candidate-pool",
        "rationale": "English country/folk singer-songwriter",
    },
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mtg-metadata-root", type=Path, default=DEFAULT_MTG_METADATA)
    parser.add_argument("--fma-metadata-zip", type=Path, default=DEFAULT_FMA_METADATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selection-set", choices=("initial", "expansion"), default="initial")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-teacher", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--require-teacher-cuda", action="store_true")
    return parser.parse_args(argv)


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-").lower() or "track"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def parse_mtg_licenses(path: Path) -> dict[str, dict[str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    result: dict[str, dict[str, str]] = {}
    for block in (item for item in text.split("\n\n") if item.strip()):
        lines = block.splitlines()
        if len(lines) < 3:
            continue
        license_line = lines[-1]
        match = re.search(r"Available under a (.+?) license:\s*(\S+)", license_line, re.I)
        if not match:
            continue
        result[lines[0].strip()] = {
            "license": match.group(1).strip(),
            "licenseUrl": match.group(2).strip(),
            "attribution": lines[1].strip(),
        }
    return result


def load_mtg(root: Path, track_id: str) -> dict[str, Any]:
    genre_rows = {row["TRACK_ID"]: row for row in read_tsv(root / "autotagging_genre.tsv")}
    instrument_rows = {row["TRACK_ID"]: row for row in read_tsv(root / "autotagging_instrument.tsv")}
    metadata_rows = {row["TRACK_ID"]: row for row in read_tsv(root / "raw.meta.tsv")}
    licenses = parse_mtg_licenses(root / "audio_licenses.txt")
    genre = genre_rows.get(track_id)
    instrument = instrument_rows.get(track_id)
    metadata = metadata_rows.get(track_id)
    if not genre or not instrument or not metadata:
        raise ValueError(f"MTG metadata missing for {track_id}")
    if "instrument---voice" not in instrument.get("TAGS", ""):
        raise ValueError(f"MTG track lacks explicit voice tag: {track_id}")
    path = genre["PATH"]
    license_info = licenses.get(path)
    if not license_info:
        raise ValueError(f"MTG license missing for {track_id}: {path}")
    if "No Derivatives" in license_info["license"] or "NoDerivatives" in license_info["license"]:
        raise ValueError(f"MTG license not accepted for {track_id}: {license_info['license']}")
    numeric_id = int(track_id.removeprefix("track_"))
    return {
        "source": "mtg-jamendo",
        "sourceId": track_id,
        "trackPath": path,
        "trackName": metadata["TRACK_NAME"],
        "artistName": metadata["ARTIST_NAME"],
        "albumName": metadata["ALBUM_NAME"],
        "durationSeconds": float(genre["DURATION"]),
        "genreTags": genre.get("TAGS", ""),
        "instrumentTags": instrument.get("TAGS", ""),
        "sourceUrl": metadata["URL"],
        "downloadUrl": f"https://mp3d.jamendo.com/download/track/{numeric_id}/mp31",
        **license_info,
        "metadataSource": "https://github.com/MTG/mtg-jamendo-dataset",
        "metadataRevision": "master",
    }


def duration_seconds(value: str) -> float:
    parts = [int(item) for item in value.split(":")]
    result = 0.0
    for item in parts:
        result = result * 60.0 + item
    return result


def load_fma(zip_path: Path, track_id: str) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path) as archive, archive.open("fma_metadata/raw_tracks.csv") as stream:
        rows = csv.DictReader((line.decode("utf-8", "replace") for line in stream))
        row = next((item for item in rows if item.get("track_id") == track_id), None)
    if row is None:
        raise ValueError(f"FMA metadata missing for {track_id}")
    license_title = row.get("license_title", "") or ""
    if not license_title.startswith("Attribution-NonCommercial") or "NoDerivatives" in license_title or "NoDerivs" in license_title:
        raise ValueError(f"FMA license not accepted for {track_id}: {license_title}")
    if row.get("track_instrumental") == "1":
        raise ValueError(f"FMA track is marked instrumental: {track_id}")
    track_file = row.get("track_file", "")
    if not track_file:
        raise ValueError(f"FMA track file missing: {track_id}")
    try:
        genres = [item.get("genre_title", "") for item in ast.literal_eval(row.get("track_genres", "[]"))]
    except (ValueError, SyntaxError):
        genres = []
    return {
        "source": "fma",
        "sourceId": track_id,
        "trackPath": track_file,
        "trackName": row.get("track_title", ""),
        "artistName": row.get("artist_name", ""),
        "albumName": row.get("album_title", ""),
        "durationSeconds": duration_seconds(row.get("track_duration", "0:0")),
        "genreTags": genres,
        "languageCode": row.get("track_language_code", ""),
        "sourceUrl": row.get("track_url", ""),
        "downloadUrl": "https://files.freemusicarchive.org/storage-freemusicarchive-org/" + track_file,
        "license": license_title,
        "licenseUrl": row.get("license_url", ""),
        "metadataSource": "https://github.com/mdeff/fma",
        "metadataRevision": "master metadata archive",
    }


def download(url: str, destination: Path, force: bool) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and not force:
        return {
            "file": str(destination.resolve()),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "reused": True,
        }
    temporary = destination.with_name(destination.name + ".part")
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "MusicSourceSeparation-research/1",
                    "Range": "bytes=0-",
                },
            )
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
            temporary.replace(destination)
            last_error = None
            break
        except Exception as error:  # noqa: BLE001 - retry transient mirrors
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt < 4:
                time.sleep(float(attempt * 2))
    if last_error is not None:
        raise last_error
    return {
        "file": str(destination.resolve()),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "reused": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    if args.device == "cuda" and not __import__("torch").cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = __import__("torch").device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and __import__("torch").cuda.is_available()) else "cpu"
    )
    mtg_root = args.mtg_metadata_root.resolve()
    fma_zip = args.fma_metadata_zip.resolve()
    if not mtg_root.is_dir() or not fma_zip.is_file():
        raise FileNotFoundError("MTG metadata directory or FMA metadata archive is missing")
    output_root = args.output_root.resolve()
    raw_root = output_root / "raw"
    decoded_root = output_root / "decoded"
    teacher_root = output_root / "teacher"
    selected_records = SELECTED if args.selection_set == "initial" else SELECTED_EXPANSION
    records: list[dict[str, Any]] = []
    for selected in selected_records:
        record = (
            load_mtg(mtg_root, selected["id"])
            if selected["source"] == "mtg-jamendo"
            else load_fma(fma_zip, selected["id"])
        )
        record.update(selected)
        record["slug"] = safe_name(f"{record['source']}-{record['artistName']}-{record['trackName']}")
        records.append(record)
    # Artist and role checks make the pilot split explicit.
    if len({(r["source"], r["sourceId"]) for r in records}) != len(records):
        raise AssertionError("Duplicate selected source record")
    if len({r["artistName"] for r in records}) != len(records):
        raise AssertionError("Selected pilot must use unique artists")
    for record in records:
        destination = raw_root / f"{record['slug']}.mp3"
        if not args.skip_download:
            record["download"] = download(record["downloadUrl"], destination, args.force)
        elif not destination.is_file():
            raise FileNotFoundError(destination)
        else:
            record["download"] = {"file": str(destination.resolve()), "bytes": destination.stat().st_size, "sha256": sha256_file(destination), "reused": True}
        audio, sample_rate = listening.load_audio(destination)
        record["decoded"] = {
            "sampleRate": sample_rate,
            "frames": int(audio.shape[0]),
            "channels": int(audio.shape[1]),
            "durationSeconds": audio.shape[0] / sample_rate,
            "float32Sha256": pilot.sha256_array(audio),
            "rmsDbfs": 20.0 * math.log10(max(float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))), 1.0e-12)),
        }
        if not args.skip_teacher:
            session, providers = pilot.make_teacher_session(
                pilot.DEFAULT_TEACHER,
                args.threads,
                args.require_teacher_cuda,
            )
            try:
                instrumental, vocals, timing = oracle.render_teacher_audio(
                    session,
                    audio,
                    progress_label=f"teacher/{record['slug']}",
                )
            finally:
                del session
            teacher_dir = teacher_root / record["slug"]
            teacher_dir.mkdir(parents=True, exist_ok=True)
            teacher_path = teacher_dir / "teacher.npz"
            np.savez_compressed(teacher_path, instrumental=instrumental, residual=vocals)
            record["teacher"] = {
                "providers": list(providers),
                "contractId": "uvr_mdxnet_inst_3@2",
                "timing": timing,
                "file": str(teacher_path.resolve()),
                "bytes": teacher_path.stat().st_size,
                "sha256": sha256_file(teacher_path),
                "instrumentalRmsDbfs": 20.0 * math.log10(max(float(np.sqrt(np.mean(instrumental.astype(np.float64) ** 2))), 1.0e-12)),
                "residualRmsDbfs": 20.0 * math.log10(max(float(np.sqrt(np.mean(vocals.astype(np.float64) ** 2))), 1.0e-12)),
            }
        json_write(output_root / "source-manifest.json", {"schema": SCHEMA, "status": "running", "records": records})
        print(f"prepared {record['source']} {record['artistName']} - {record['trackName']}", flush=True)
    manifest = {
        "schema": SCHEMA,
        "status": "completed",
        "licenseDisposition": {
            "scope": "local non-commercial research only",
            "derivedTargets": "Inst 3 pseudo-targets remain local and are not publishable",
            "audio": "retain per-track attribution and license URL; do not redistribute from this repository",
        },
        "selectionPolicy": {
            "accept": "explicit Creative Commons Attribution-family license; no NoDerivatives; voice/non-instrumental metadata; unique artists",
            "roles": {"supplement-train": 8, "supplement-holdout": 4},
        },
        "metadataInputs": {
            "mtgRoot": str(mtg_root),
            "mtgFiles": {
                name: {
                    "file": str((mtg_root / name).resolve()),
                    "sha256": sha256_file(mtg_root / name),
                }
                for name in (
                    "raw_30s_cleantags_50artists.tsv",
                    "autotagging_genre.tsv",
                    "autotagging_instrument.tsv",
                    "raw.meta.tsv",
                    "audio_licenses.txt",
                )
            },
            "fmaMetadataArchive": {
                "file": str(fma_zip),
                "sha256": sha256_file(fma_zip),
            },
        },
        "records": records,
        "runtime": {"device": str(device), "threads": args.threads, "teacher": "uvr_mdxnet_inst_3@2" if not args.skip_teacher else None},
    }
    json_write(output_root / "source-manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "records": len(records), "output": str((output_root / "source-manifest.json").resolve())}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
