#!/usr/bin/env python3
"""Generate the frozen Booming SS Phase 7 source-format corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SCHEMA_VERSION = "phase7-source-format-corpus-v1"
SAMPLE_RATE = 44_100
CHANNELS = 2
DURATION_SECONDS = 15
FRAME_COUNT = SAMPLE_RATE * DURATION_SECONDS
MDX_GENERATION_SIZE = 254_976
FFMPEG_VERSION_PREFIX = "ffmpeg version 8.1.1-full_build-www.gyan.dev"


@dataclass(frozen=True)
class CorpusArtifact:
    fixture_id: str
    file_name: str
    decode_class: str
    command: tuple[str, ...] | None
    post_process: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/phase7/source-formats-v1"),
        help="Generated corpus directory (default: %(default)s)",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable")
    parser.add_argument("--ffprobe", default="ffprobe", help="ffprobe executable")
    parser.add_argument(
        "--allow-unpinned-ffmpeg",
        action="store_true",
        help="Allow a version other than the corpus-pinned ffmpeg build",
    )
    return parser.parse_args()


def run(command: Sequence[str]) -> str:
    completed = subprocess.run(
        list(command),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip()


def first_version_line(executable: str) -> str:
    output = run((executable, "-version"))
    return output.splitlines()[0]


def triangle(frame: int, period: int, amplitude: int) -> int:
    phase = frame % period
    half = period // 2
    if phase < half:
        return -amplitude + (2 * amplitude * phase) // half
    return amplitude - (2 * amplitude * (phase - half)) // (period - half)


def marker(frame: int) -> tuple[int, int]:
    marker_frames = (0, MDX_GENERATION_SIZE, MDX_GENERATION_SIZE * 2)
    for index, start in enumerate(marker_frames):
        offset = frame - start
        if 0 <= offset < 768:
            sign = 1 if ((offset // 24) + index) % 2 == 0 else -1
            amplitude = 9_000 - (offset * 7)
            return sign * amplitude, -sign * (amplitude // 2)
    return 0, 0


def clamp_pcm16(value: int) -> int:
    return max(-32_768, min(32_767, value))


def generate_source_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    noise_state = 0x13579BDF
    with wave.open(str(path), "wb") as output:
        output.setnchannels(CHANNELS)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        pending = bytearray()
        for frame in range(FRAME_COUNT):
            noise_state = (1_664_525 * noise_state + 1_013_904_223) & 0xFFFFFFFF
            noise = (((noise_state >> 16) & 0xFFFF) - 32_768) // 32
            marker_left, marker_right = marker(frame)
            left = (
                triangle(frame, 101, 7_200)
                + triangle(frame, 337, 4_100)
                + noise
                + marker_left
            )
            right = (
                triangle(frame, 149, 6_700)
                + triangle(frame, 431, 3_800)
                - noise
                + marker_right
            )
            pending.extend(struct.pack("<hh", clamp_pcm16(left), clamp_pcm16(right)))
            if len(pending) >= 256 * 1024:
                output.writeframesraw(pending)
                pending.clear()
        if pending:
            output.writeframesraw(pending)


def ffmpeg_command(
    ffmpeg: str,
    source: Path,
    output: Path,
    codec_arguments: Sequence[str],
) -> tuple[str, ...]:
    return (
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-map_metadata",
        "-1",
        "-vn",
        *codec_arguments,
        str(output),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ogg_crc_table() -> tuple[int, ...]:
    values: list[int] = []
    for index in range(256):
        value = index << 24
        for _ in range(8):
            value = ((value << 1) ^ 0x04C11DB7) if value & 0x80000000 else value << 1
        values.append(value & 0xFFFFFFFF)
    return tuple(values)


def normalize_ogg_serial(path: Path, serial: int) -> None:
    data = bytearray(path.read_bytes())
    table = ogg_crc_table()
    offset = 0
    while offset < len(data):
        if data[offset : offset + 4] != b"OggS":
            raise RuntimeError(f"Invalid Ogg page at byte {offset}: {path}")
        segment_count = data[offset + 26]
        header_size = 27 + segment_count
        body_size = sum(data[offset + 27 : offset + header_size])
        page_end = offset + header_size + body_size
        if page_end > len(data):
            raise RuntimeError(f"Truncated Ogg page at byte {offset}: {path}")

        data[offset + 14 : offset + 18] = serial.to_bytes(4, "little")
        data[offset + 22 : offset + 26] = b"\0\0\0\0"
        checksum = 0
        for value in data[offset:page_end]:
            checksum = ((checksum << 8) & 0xFFFFFFFF) ^ table[(checksum >> 24) ^ value]
        data[offset + 22 : offset + 26] = checksum.to_bytes(4, "little")
        offset = page_end
    path.write_bytes(data)


def probe(ffprobe: str, path: Path) -> dict[str, object]:
    raw = run((
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,sample_fmt,sample_rate,channels,bits_per_sample,"
        "duration,duration_ts,time_base,initial_padding:"
        "format=format_name,duration,size",
        "-of",
        "json",
        str(path),
    ))
    parsed = json.loads(raw)
    stream = parsed["streams"][0]
    container = parsed["format"]
    duration = stream.get("duration") or container.get("duration")
    if duration is None:
        raise RuntimeError(f"ffprobe returned no duration for {path}")
    return {
        "codec": stream["codec_name"],
        "container": container["format_name"],
        "sampleFormat": stream.get("sample_fmt"),
        "sampleRate": int(stream["sample_rate"]),
        "channels": int(stream["channels"]),
        "bitsPerSample": int(stream.get("bits_per_sample", 0)),
        "durationUs": round(float(duration) * 1_000_000),
        "durationTs": int(stream["duration_ts"]) if "duration_ts" in stream else None,
        "timeBase": stream.get("time_base"),
        "initialPadding": int(stream.get("initial_padding", 0)),
    }


def normalized_command(command: Sequence[str] | None) -> list[str] | None:
    if command is None:
        return None
    result: list[str] = []
    for token in command:
        name = Path(token).name
        if name == "phase7_format_source.wav":
            result.append("{source}")
        elif name.startswith("phase7_format_"):
            result.append(f"{{output}}/{name}")
        else:
            result.append(token)
    return result


def generate(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = output_dir / "phase7_format_source.wav"
    generate_source_wav(source)

    artifacts: list[CorpusArtifact] = [
        CorpusArtifact("format_wav_44100", source.name, "wav-window", None),
    ]

    conversions = (
        (
            "format_flac_44100",
            "phase7_format_flac_44100.flac",
            "flac-44100-window",
            ("-c:a", "flac", "-compression_level", "8", "-sample_fmt", "s16", "-ar", "44100", "-ac", "2"),
        ),
        (
            "format_vorbis_44100",
            "phase7_format_vorbis_44100.ogg",
            "ogg-vorbis-window",
            ("-c:a", "libvorbis", "-q:a", "5", "-ar", "44100", "-ac", "2"),
        ),
        (
            "format_mp3_gapless_44100",
            "phase7_format_mp3_gapless_44100.mp3",
            "mp3-44100-gapless-window",
            ("-c:a", "libmp3lame", "-b:a", "192k", "-write_xing", "1", "-id3v2_version", "3", "-ar", "44100", "-ac", "2"),
        ),
        (
            "format_mp3_no_xing_44100",
            "phase7_format_mp3_no_xing_44100.mp3",
            "mp3-44100-no-gapless-window",
            ("-c:a", "libmp3lame", "-b:a", "192k", "-write_xing", "0", "-id3v2_version", "0", "-write_id3v1", "0", "-ar", "44100", "-ac", "2"),
        ),
        (
            "format_aac_44100",
            "phase7_format_aac_44100.m4a",
            "unsupported-mime-full-song",
            ("-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-ar", "44100", "-ac", "2"),
        ),
        (
            "format_flac_48000",
            "phase7_format_flac_48000.flac",
            "flac-non-44100-full-song",
            ("-c:a", "flac", "-compression_level", "8", "-sample_fmt", "s16", "-ar", "48000", "-ac", "2"),
        ),
        (
            "format_mp3_48000",
            "phase7_format_mp3_48000.mp3",
            "mp3-non-44100-full-song",
            ("-c:a", "libmp3lame", "-b:a", "192k", "-write_xing", "1", "-id3v2_version", "3", "-ar", "48000", "-ac", "2"),
        ),
    )
    for fixture_id, file_name, decode_class, codec_arguments in conversions:
        destination = output_dir / file_name
        command = ffmpeg_command(args.ffmpeg, source, destination, codec_arguments)
        run(command)
        post_process = None
        if fixture_id == "format_vorbis_44100":
            normalize_ogg_serial(destination, serial=0x42535331)
            post_process = "normalize-ogg-serial-v1:0x42535331"
        artifacts.append(CorpusArtifact(
            fixture_id,
            file_name,
            decode_class,
            command,
            post_process,
        ))

    wrong_extension = output_dir / "phase7_format_wav_wrong_extension.wave"
    shutil.copyfile(source, wrong_extension)
    artifacts.append(CorpusArtifact(
        "format_wav_wrong_extension",
        wrong_extension.name,
        "wav-name-full-song",
        ("copy", str(source), str(wrong_extension)),
    ))

    return {
        "schemaVersion": SCHEMA_VERSION,
        "generator": "tools/generate_phase7_format_corpus.py",
        "ffmpegVersion": first_version_line(args.ffmpeg),
        "ffprobeVersion": first_version_line(args.ffprobe),
        "source": {
            "sampleRate": SAMPLE_RATE,
            "channels": CHANNELS,
            "durationSeconds": DURATION_SECONDS,
            "frameCount": FRAME_COUNT,
            "mdxGenerationSize": MDX_GENERATION_SIZE,
            "signal": "integer-triangle-noise-markers-v1",
        },
        "artifacts": [
            {
                "fixtureId": artifact.fixture_id,
                "fileName": artifact.file_name,
                "relativePath": f"data/phase7/source-formats-v1/{artifact.file_name}",
                "byteSize": (output_dir / artifact.file_name).stat().st_size,
                "sha256": sha256(output_dir / artifact.file_name),
                "decodeClass": artifact.decode_class,
                "redistribution": "generated-apache-2.0",
                "command": normalized_command(artifact.command),
                "postProcess": artifact.post_process,
                **probe(args.ffprobe, output_dir / artifact.file_name),
            }
            for artifact in artifacts
        ],
    }


def main() -> int:
    args = parse_args()
    ffmpeg_version = first_version_line(args.ffmpeg)
    if not args.allow_unpinned_ffmpeg and not ffmpeg_version.startswith(FFMPEG_VERSION_PREFIX):
        print(
            "Refusing to generate the frozen corpus with an unpinned ffmpeg build:\n"
            f"  expected prefix: {FFMPEG_VERSION_PREFIX}\n"
            f"  actual:          {ffmpeg_version}\n"
            "Pass --allow-unpinned-ffmpeg only for exploratory output.",
            file=sys.stderr,
        )
        return 2

    manifest = generate(args)
    manifest_path = args.output_dir.resolve() / "manifest.generated.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"Generated {len(manifest['artifacts'])} fixtures in {args.output_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
