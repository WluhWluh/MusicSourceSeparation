#!/usr/bin/env python3
"""Package gain-sweep renders as a blind listening set.

The listening directory intentionally contains no gain names. The answer key
is written to a separate sibling directory so the listener can inspect only
the blind set during evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path


GAINS = (1.00, 1.10, 1.20)
STEMS = ("accompaniment", "vocals")
LETTERS = ("A", "B", "C")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--source",
        action="append",
        nargs=2,
        metavar=("ID", "RAW_DIR"),
        required=True,
        help="Song id and directory containing gain-1p00x/gain-1p10x/gain-1p20x",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gain_dir(gain: float) -> str:
    return f"gain-{gain:.2f}".replace(".", "p") + "x"


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    if args.key_dir.exists():
        shutil.rmtree(args.key_dir)
    args.output_dir.mkdir(parents=True)
    args.key_dir.mkdir(parents=True)

    rng = random.Random(args.seed)
    key: dict[str, object] = {
        "schemaVersion": 1,
        "purpose": "tfc-tdf-vocal-gain-blind-listening",
        "seed": args.seed,
        "songs": {},
    }
    public_manifest: dict[str, object] = {
        "schemaVersion": 1,
        "purpose": "tfc-tdf-vocal-gain-blind-listening",
        "songs": {},
    }

    for song_id, raw_dir_text in args.source:
        raw_dir = Path(raw_dir_text).resolve()
        song_output = args.output_dir / song_id
        song_key_dir = args.key_dir / song_id
        for stem in STEMS:
            (song_output / stem).mkdir(parents=True, exist_ok=True)
        song_key_dir.mkdir(parents=True, exist_ok=True)

        shuffled = list(GAINS)
        rng.shuffle(shuffled)
        mapping = dict(zip(LETTERS, shuffled, strict=True))
        public_song: dict[str, object] = {"files": {}}
        key_song: dict[str, object] = {"mapping": mapping, "files": {}}

        for letter, gain in mapping.items():
            source_dir = raw_dir / gain_dir(gain)
            for stem in STEMS:
                source = source_dir / f"{stem}.flac"
                if not source.is_file():
                    raise FileNotFoundError(source)
                destination = song_output / stem / f"{letter}.flac"
                shutil.copy2(source, destination)
                public_song["files"][f"{stem}-{letter}"] = {
                    "path": str(destination.relative_to(args.output_dir)),
                    "sha256": sha256_file(destination),
                }
                key_song["files"][f"{stem}-{letter}"] = {
                    "sourceGain": gain,
                    "source": str(source),
                }

        key["songs"][song_id] = key_song
        public_manifest["songs"][song_id] = public_song

    (args.output_dir / "manifest.json").write_text(
        json.dumps(public_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.key_dir / "answer-key.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"blindDir": str(args.output_dir), "keyFile": str(args.key_dir / "answer-key.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
