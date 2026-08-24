#!/usr/bin/env python3
"""Select and download raw modern-song candidates without audio processing.

The pool is selected from the already downloaded MTG-Jamendo and FMA metadata.
Only explicit Creative Commons Attribution-family licenses without an ND term
are accepted.  Files are downloaded byte-for-byte and hashed; this tool never
decodes, probes, resamples, analyzes, separates, or excerpts audio.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MTG_ROOT = ROOT / ".tmp" / "mtg-metadata"
DEFAULT_FMA_ZIP = ROOT / ".tmp" / "fma-metadata" / "fma_metadata.zip"
DEFAULT_OUTPUT = ROOT / "data" / "modern-song-original-candidates"
DEFAULT_COUNT = 72
MIN_SECONDS = 150
MAX_SECONDS = 330

# Metadata entries whose canonical source URL was confirmed unavailable during
# raw-byte download. They are excluded before deterministic reselection.
UNAVAILABLE_SOURCE_IDS = {"136940"}
UNAVAILABLE_ARTISTS = {"lobo loco"}

CATEGORY_QUOTAS = {
    "pop-synth": 14,
    "hiphop-rnb": 14,
    "dance-electronic": 12,
    "latin-modern": 10,
    "pop-rock": 12,
    "singer-songwriter": 10,
}

EXCLUDED_GENRE_FRAGMENTS = (
    "ambient",
    "avant-garde",
    "breakcore",
    "chip music",
    "chiptune",
    "classical",
    "drone",
    "electroacoustic",
    "experimental",
    "field recording",
    "free-folk",
    "free-jazz",
    "freak-folk",
    "hardcore",
    "improv",
    "instrumental",
    "metal",
    "musique concrete",
    "noise",
    "no wave",
    "old-time",
    "post-rock",
    "psych-folk",
    "sound art",
    "sound collage",
    "sound poetry",
    "soundtrack",
    "spoken weird",
    "unclassifiable",
)

CATEGORY_GENRES = {
    "pop-synth": {"Pop", "Synth Pop", "Power-Pop"},
    "hiphop-rnb": {
        "Hip-Hop",
        "Alternative Hip-Hop",
        "Abstract Hip-Hop",
        "Rap",
        "Soul-RnB",
        "Trip-Hop",
    },
    "dance-electronic": {
        "Dance",
        "House",
        "Disco",
        "Synth Pop",
        "Techno",
        "Dubstep",
        "Drum & Bass",
        "Breakbeat",
        "Downtempo",
    },
    "latin-modern": {"Latin America", "Latin", "Brazilian", "Spanish"},
    "pop-rock": {"Indie-Rock", "Power-Pop", "Rock", "Pop"},
    "singer-songwriter": {"Singer-Songwriter"},
}

CATEGORY_PRIORITY = (
    "latin-modern",
    "hiphop-rnb",
    "pop-synth",
    "dance-electronic",
    "pop-rock",
    "singer-songwriter",
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mtg-metadata-root", type=Path, default=DEFAULT_MTG_ROOT)
    parser.add_argument("--fma-metadata-zip", type=Path, default=DEFAULT_FMA_ZIP)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
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
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def duration_seconds(raw: str) -> float:
    try:
        result = 0.0
        for part in raw.split(":"):
            result = result * 60.0 + float(part)
        return result
    except (TypeError, ValueError):
        return 0.0


def accepted_license(title: str) -> bool:
    normalized = (title or "").casefold()
    return (
        normalized.startswith("attribution")
        or normalized.startswith("creative commons attribution")
    ) and "no deriv" not in normalized and "noderiv" not in normalized


def existing_artists() -> set[str]:
    result: set[str] = set()
    for path in ROOT.glob("data/inst3-mtg-fma-*/source-manifest.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            result.update(str(record["artistName"]).strip().casefold() for record in value["records"])
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return result


def parse_mtg_licenses(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    for block in (part for part in text.split("\n\n") if part.strip()):
        lines = block.splitlines()
        if len(lines) < 3:
            continue
        match = re.search(r"Available under a (.+?) license:\s*(\S+)", lines[-1], re.I)
        if match:
            result[lines[0].strip()] = {
                "license": match.group(1).strip(),
                "licenseUrl": match.group(2).strip(),
                "attribution": lines[1].strip(),
            }
    return result


def category_for(genres: list[str], language: str) -> str | None:
    genre_set = set(genres)
    if any(fragment in genre.casefold() for genre in genres for fragment in EXCLUDED_GENRE_FRAGMENTS):
        return None
    for category in CATEGORY_PRIORITY:
        if genre_set & CATEGORY_GENRES[category]:
            if category == "latin-modern" and not (
                language in {"es", "pt"}
                or genre_set & {"Latin America", "Latin", "Brazilian", "Spanish"}
            ):
                continue
            if category == "pop-rock" and "Rock" in genre_set and not (
                genre_set & {"Pop", "Power-Pop", "Indie-Rock", "Singer-Songwriter"}
            ):
                continue
            return category
    return None


def ranking_score(row: dict[str, str], genres: list[str], category: str) -> float:
    def number(key: str) -> float:
        try:
            return float(row.get(key) or 0.0)
        except ValueError:
            return 0.0

    score = math.log1p(number("track_interest")) * 3.0
    score += math.log1p(number("track_listens"))
    score += math.log1p(number("track_favorites")) * 2.0
    if row.get("track_language_code") in {"en", "es", "pt", "ja"}:
        score += 2.0
    if row.get("track_language_code") in {"ja", "zh", "cmn"}:
        # The metadata contains almost no cleared East-Asian modern tracks;
        # reserve any strict match for human style review.
        score += 1.0e6
    if category in {"pop-synth", "hiphop-rnb", "dance-electronic", "latin-modern"}:
        score += 3.0
    if "Pop" in genres or "Synth Pop" in genres:
        score += 2.0
    return score


def likely_vocal_track(row: dict[str, str], genres: list[str], category: str) -> bool:
    language = (row.get("track_language_code") or "").strip().lower()
    lyricist = (row.get("track_lyricist") or "").strip().casefold()
    title = (row.get("track_title") or "").casefold()
    album = (row.get("album_title") or "").casefold()
    information = (row.get("track_information") or "").casefold()
    tags = (row.get("tags") or "").casefold()
    negative = ("instrumental", "karaoke", "acapella stems", "backing track")
    if any(token in title or token in album or token in tags for token in negative):
        return False
    explicit_genre = bool(
        set(genres)
        & {"Singer-Songwriter", "Pop", "Soul-RnB", "Rap", "Jazz: Vocal", "Easy Listening: Vocal"}
    )
    textual_signal = any(
        token in information
        for token in (" vocal", "vocals", "voice", " vox", "singer", "lyrics", "lyricist")
    )
    basic_signal = bool(language or (lyricist and lyricist != "null") or explicit_genre or textual_signal)
    if not basic_signal:
        return False
    if category in {"dance-electronic", "hiphop-rnb", "pop-rock", "latin-modern"}:
        # These genres contain many beat tapes and instrumental acts despite
        # track_instrumental=0; require a stronger signal than the genre alone.
        return bool(language or (lyricist and lyricist != "null") or textual_signal or explicit_genre)
    return True


def load_fma_candidates(zip_path: Path, excluded_artists: set[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as archive, archive.open("fma_metadata/raw_tracks.csv") as stream:
        rows = csv.DictReader((line.decode("utf-8", "replace") for line in stream))
        for row in rows:
            artist = (row.get("artist_name") or "").strip()
            license_title = row.get("license_title") or ""
            duration = duration_seconds(row.get("track_duration") or "")
            if (
                not artist
                or row.get("track_id") in UNAVAILABLE_SOURCE_IDS
                or artist.casefold() in UNAVAILABLE_ARTISTS
                or artist.casefold() in excluded_artists
                or not accepted_license(license_title)
                or row.get("track_instrumental") == "1"
                or not row.get("track_file")
                or not MIN_SECONDS <= duration <= MAX_SECONDS
            ):
                continue
            try:
                genres = [str(item.get("genre_title", "")) for item in ast.literal_eval(row.get("track_genres") or "[]")]
            except (SyntaxError, ValueError):
                continue
            language = (row.get("track_language_code") or "").strip().lower()
            category = category_for(genres, language)
            if category is None or not likely_vocal_track(row, genres, category):
                continue
            track_file = row["track_file"]
            result.append(
                {
                    "source": "fma",
                    "sourceId": row["track_id"],
                    "artistName": artist,
                    "trackName": row.get("track_title") or "",
                    "albumName": row.get("album_title") or "",
                    "durationSeconds": duration,
                    "genreTags": genres,
                    "languageCode": language,
                    "category": category,
                    "sourceUrl": row.get("track_url") or "",
                    "downloadUrl": "https://files.freemusicarchive.org/storage-freemusicarchive-org/"
                    + urllib.parse.quote(track_file, safe="/"),
                    "trackPath": track_file,
                    "license": license_title,
                    "licenseUrl": row.get("license_url") or "",
                    "metadataSource": "https://github.com/mdeff/fma",
                    "rankingScore": ranking_score(row, genres, category),
                }
            )
    return result


def load_mtg_candidates(root: Path, excluded_artists: set[str]) -> list[dict[str, Any]]:
    genres = {row["TRACK_ID"]: row for row in read_tsv(root / "autotagging_genre.tsv")}
    instruments = {row["TRACK_ID"]: row for row in read_tsv(root / "autotagging_instrument.tsv")}
    metadata = {row["TRACK_ID"]: row for row in read_tsv(root / "raw.meta.tsv")}
    licenses = parse_mtg_licenses(root / "audio_licenses.txt")
    tag_to_category = {"genre---pop": "pop-synth", "genre---rock": "pop-rock"}
    result: list[dict[str, Any]] = []
    for track_id, genre in genres.items():
        category = tag_to_category.get(genre.get("TAGS", ""))
        meta = metadata.get(track_id)
        instrument = instruments.get(track_id, {})
        if category is None or meta is None or "instrument---voice" not in instrument.get("TAGS", ""):
            continue
        artist = meta["ARTIST_NAME"].strip()
        duration = float(genre["DURATION"])
        license_info = licenses.get(genre["PATH"])
        if (
            not license_info
            or not accepted_license(license_info["license"])
            or artist.casefold() in excluded_artists
            or not MIN_SECONDS <= duration <= MAX_SECONDS
        ):
            continue
        numeric_id = int(track_id.removeprefix("track_"))
        result.append(
            {
                "source": "mtg-jamendo",
                "sourceId": track_id,
                "artistName": artist,
                "trackName": meta["TRACK_NAME"],
                "albumName": meta["ALBUM_NAME"],
                "durationSeconds": duration,
                "genreTags": genre["TAGS"],
                "instrumentTags": instrument.get("TAGS", ""),
                "languageCode": "",
                "category": category,
                "sourceUrl": meta["URL"],
                "downloadUrl": f"https://mp3d.jamendo.com/download/track/{numeric_id}/mp31",
                "trackPath": genre["PATH"],
                **license_info,
                "metadataSource": "https://github.com/MTG/mtg-jamendo-dataset",
                "rankingScore": 1.0e9,
            }
        )
    return result


def choose_candidates(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count != sum(CATEGORY_QUOTAS.values()):
        raise ValueError(f"This frozen selection contract requires {sum(CATEGORY_QUOTAS.values())} tracks")
    by_category = {
        category: sorted(
            [item for item in candidates if item["category"] == category],
            key=lambda item: (-float(item["rankingScore"]), item["artistName"].casefold(), item["sourceId"]),
        )
        for category in CATEGORY_QUOTAS
    }
    selected: list[dict[str, Any]] = []
    artists: set[str] = set()
    for category, quota in CATEGORY_QUOTAS.items():
        for candidate in by_category[category]:
            artist = candidate["artistName"].casefold()
            if artist in artists:
                continue
            selected.append(candidate)
            artists.add(artist)
            if sum(item["category"] == category for item in selected) == quota:
                break
        actual = sum(item["category"] == category for item in selected)
        if actual != quota:
            raise ValueError(f"Only selected {actual}/{quota} for {category}")
    for index, record in enumerate(selected, start=1):
        record["order"] = index
        record["role"] = "pending-style-review"
        record["slug"] = safe_name(f"{record['source']}-{record['artistName']}-{record['trackName']}")
    return selected


def download_raw(url: str, destination: Path, force: bool) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and not force:
        return {
            "file": str(destination.resolve()),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "reused": True,
            "contentType": None,
        }
    temporary = destination.with_name(destination.name + ".part")
    last_error: Exception | None = None
    content_type = None
    for attempt in range(1, 5):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "MusicSourceSeparation-research/1"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                content_type = response.headers.get_content_type()
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
            if temporary.stat().st_size < 16_384:
                raise ValueError(f"Downloaded file is unexpectedly small: {temporary.stat().st_size}")
            with temporary.open("rb") as stream:
                prefix = stream.read(512).lower()
            if b"<html" in prefix or b"<!doctype" in prefix:
                raise ValueError("Download returned HTML")
            temporary.replace(destination)
            last_error = None
            break
        except Exception as error:  # noqa: BLE001 - retry mirrors and transient HTTP failures
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt < 4:
                time.sleep(attempt * 2.0)
    if last_error is not None:
        raise last_error
    return {
        "file": str(destination.resolve()),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "reused": False,
        "contentType": content_type,
    }


def write_review_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = (
        "keep",
        "order",
        "category",
        "artistName",
        "trackName",
        "source",
        "languageCode",
        "genreTags",
        "durationSecondsMetadata",
        "license",
        "licenseUrl",
        "sourceUrl",
        "file",
        "notes",
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
                    "source": record["source"],
                    "languageCode": record.get("languageCode", ""),
                    "genreTags": json.dumps(record["genreTags"], ensure_ascii=False),
                    "durationSecondsMetadata": record["durationSeconds"],
                    "license": record["license"],
                    "licenseUrl": record["licenseUrl"],
                    "sourceUrl": record["sourceUrl"],
                    "file": record.get("download", {}).get("file", ""),
                    "notes": "",
                }
            )


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.count != DEFAULT_COUNT:
        raise ValueError(f"Frozen pool size is {DEFAULT_COUNT}")
    mtg_root = args.mtg_metadata_root.resolve()
    fma_zip = args.fma_metadata_zip.resolve()
    if not mtg_root.is_dir() or not fma_zip.is_file():
        raise FileNotFoundError("MTG/FMA metadata is missing")
    excluded = existing_artists()
    candidates = load_mtg_candidates(mtg_root, excluded) + load_fma_candidates(fma_zip, excluded)
    selected = choose_candidates(candidates, args.count)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "source-manifest.json"
    manifest = {
        "schema": "local-modern-song-original-candidates@1",
        "status": "selected" if args.dry_run else "downloading",
        "selectionPolicy": {
            "count": args.count,
            "categoryQuotas": CATEGORY_QUOTAS,
            "durationSecondsMetadata": [MIN_SECONDS, MAX_SECONDS],
            "license": "explicit Creative Commons Attribution-family; NoDerivatives excluded",
            "artistPolicy": "one track per artist; all existing 28-pool artists excluded",
            "mainPoolExclusions": list(EXCLUDED_GENRE_FRAGMENTS),
            "unavailableSourceIds": sorted(UNAVAILABLE_SOURCE_IDS),
            "unavailableArtists": sorted(UNAVAILABLE_ARTISTS),
            "role": "pending-style-review; no train/validation assignment before original-song review",
        },
        "languageCoverage": {
            "verifiedJapaneseModernCandidates": sum(record.get("languageCode") == "ja" for record in selected),
            "verifiedMandarinModernCandidates": sum(record.get("languageCode") in {"zh", "cmn"} for record in selected),
            "note": "No language is inferred from artist/title. Mandarin remains an explicit acquisition gap if zero.",
        },
        "audioProcessing": {
            "decoded": False,
            "resampled": False,
            "probed": False,
            "teacherRun": False,
            "studentRun": False,
            "snippetsGenerated": False,
            "metricsComputed": False,
        },
        "records": selected,
    }
    json_write(manifest_path, manifest)
    if args.dry_run:
        print(json.dumps({"status": manifest["status"], "count": len(selected), "categories": Counter(record["category"] for record in selected), "sources": Counter(record["source"] for record in selected), "languages": Counter(record.get("languageCode") or "<blank>" for record in selected), "manifest": str(manifest_path)}, ensure_ascii=False, indent=2, default=dict))
        return 0
    raw_root = output_root / "raw-originals"
    for index, record in enumerate(selected, start=1):
        destination = raw_root / f"{index:03d}-{record['slug']}.mp3"
        record["download"] = download_raw(record["downloadUrl"], destination, args.force)
        json_write(manifest_path, manifest)
        print(f"downloaded {index}/{len(selected)}: {record['artistName']} - {record['trackName']}", flush=True)
    manifest["status"] = "completed"
    manifest["downloadSummary"] = {
        "fileCount": len(selected),
        "totalBytes": sum(record["download"]["bytes"] for record in selected),
        "sha256Recorded": True,
        "audioDecodedOrProcessed": False,
    }
    json_write(manifest_path, manifest)
    write_review_csv(output_root / "original-style-review.csv", selected)
    (output_root / "README.txt").write_text(
        "Modern song original-candidate review\n"
        "\n"
        "Listen to raw-originals in numeric order.\n"
        "In original-style-review.csv, mark keep=Y for a suitable modern-production\n"
        "candidate, keep=N for reject, or leave blank if undecided.\n"
        "\n"
        "No file in this directory has been decoded, resampled, separated, clipped,\n"
        "or otherwise audio-processed by the preparation tool.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": manifest["status"], "count": len(selected), "bytes": manifest["downloadSummary"]["totalBytes"], "manifest": str(manifest_path), "reviewCsv": str((output_root / "original-style-review.csv").resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
