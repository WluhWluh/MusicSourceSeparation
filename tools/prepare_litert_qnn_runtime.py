#!/usr/bin/env python3
"""Materialize a pinned LiteRT 2.1.5 / QAIRT Qualcomm HTP JIT runtime.

The QAIRT SDK archive is about 1.56 GB. By default this tool uses HTTP range
requests to extract only the selected Android HTP libraries and required notices.
Nothing produced by this tool is intended to be committed to Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol


LITERT_VERSION = "2.1.5"
LITERT_ARCHIVE_URL = (
    "https://github.com/google-ai-edge/LiteRT/releases/download/v2.1.5/"
    "litert_npu_runtime_libraries_jit.zip"
)
LITERT_ARCHIVE_SHA256 = (
    "5ba2faaa66b0fc35b5d68de26175f2762107f5fd9e962fdd303a9d554a2ea0b8"
)
QAIRT_VERSION = "2.44.0.260225"
QAIRT_ARCHIVE_URL = (
    "https://softwarecenter.qualcomm.com/api/download/software/sdks/"
    "Qualcomm_AI_Runtime_Community/All/2.44.0.260225/"
    "v2.44.0.260225.zip"
)
QAIRT_ARCHIVE_ETAG = "ac2ec666fa18ff0b0a9b1e09d8e70a13"
QAIRT_ARCHIVE_BYTES = 1_560_450_667
USER_AGENT = "MusicSourceSeparation-QNN-Harness/1"
SUPPORTED_HTP_VERSIONS = (69, 73, 75, 79, 81)


@dataclass(frozen=True)
class RuntimeEntry:
    source: str
    destination: str
    size: int
    sha256: str
    component: str


HTP_LIBRARY_METADATA = {
    69: {
        "stub": (672_648, "5703b806359918f81165a31a7eff3abcff9e724a2c2931b1a2f18bfd703bd92d"),
        "skel": (10_754_252, "220414e7ced270840350eed9399020d81eb2d3a3d0ac627976435b29cd5ae43c"),
    },
    73: {
        "stub": (679_168, "ef0d0cbbdd423a81c19fb0f60d1451b602bcd60c08b3564f1c82e632fb59f157"),
        "skel": (10_761_464, "11c76e1d9bcc8a3168e47f73e48baba0da8e1ca03b7c429e4a5777eab5d20c5a"),
    },
    75: {
        "stub": (679_168, "07eb7cd2e68f6fb82bfbcd6e2cf09455670fb5ca31537e9b6725d044d378c306"),
        "skel": (10_794_236, "3997e70b0409060756d8c4e45068ca3f5b8dc405f1978db0b2b03a74b79f4b6a"),
    },
    79: {
        "stub": (679_168, "005bd3de462851ce3dde55260d7d8560d6d07dbc309f554780b1f6412e6d9df1"),
        "skel": (10_975_268, "41f83395ed4b1bcfc43417a1b82f3f137c825747711c9ab4c9d50034ed198f98"),
    },
    81: {
        "stub": (755_464, "f85cc467bda23b8253f085c4fa2680a1428e5b94c6c5a3e572abaa20c5b13ba1"),
        "skel": (11_797_220, "66719325303f22562679edfed509d540cbff53ac59127b6e4ead84b62f1dc655"),
    },
}


COMMON_QAIRT_ENTRIES = (
    RuntimeEntry(
        f"qairt/{QAIRT_VERSION}/lib/aarch64-android/libQnnHtp.so",
        "jni/arm64-v8a/libQnnHtp.so",
        2_778_176,
        "090e993822564851eab1405aff171643b21e644e3f696c95c96f2732aaed813a",
        "qairt",
    ),
    RuntimeEntry(
        f"qairt/{QAIRT_VERSION}/lib/aarch64-android/libQnnSystem.so",
        "jni/arm64-v8a/libQnnSystem.so",
        2_983_560,
        "7e69258e1278cc9b2bb62dbc6e2a52c227a100d6505a13fd6324a87993d0bba8",
        "qairt",
    ),
    RuntimeEntry(
        f"qairt/{QAIRT_VERSION}/lib/aarch64-android/libQnnHtpPrepare.so",
        "jni/arm64-v8a/libQnnHtpPrepare.so",
        85_539_184,
        "09b1c15c62b6875af49ffd3d841961c098b85c367f584fee370f986c62511298",
        "qairt",
    ),
    RuntimeEntry(
        f"qairt/{QAIRT_VERSION}/lib/aarch64-android/libQnnIr.so",
        "jni/arm64-v8a/libQnnIr.so",
        1_741_288,
        "982d7e403eec3de800219bf8de7039e7fa020749618e0505f831fcecfd2bd85d",
        "qairt",
    ),
    RuntimeEntry(
        f"qairt/{QAIRT_VERSION}/lib/aarch64-android/libQnnSaver.so",
        "jni/arm64-v8a/libQnnSaver.so",
        788_048,
        "5dbe2eb7f17c217d035ce288b75b6cc9445df551739dba787ccb55097af48b0e",
        "qairt",
    ),
    RuntimeEntry(
        f"qairt/{QAIRT_VERSION}/LICENSE.pdf",
        "licenses/QAIRT-LICENSE.pdf",
        147_577,
        "ec1dccfdcba5c6e64126e84199b8362bf4999107bfa567ebe831dbb4c461692b",
        "qairt",
    ),
    RuntimeEntry(
        f"qairt/{QAIRT_VERSION}/NOTICE.txt",
        "licenses/QAIRT-NOTICE.txt",
        129_238,
        "c8a22fa8b9ed3e9266c4a3366965474cef5cb147ee2a1f33e488bbb24a071c09",
        "qairt",
    ),
    RuntimeEntry(
        f"qairt/{QAIRT_VERSION}/QNN_NOTICE.txt",
        "licenses/QNN-NOTICE.txt",
        129_238,
        "c8a22fa8b9ed3e9266c4a3366965474cef5cb147ee2a1f33e488bbb24a071c09",
        "qairt",
    ),
)


def runtime_entries(htp_version: int) -> tuple[RuntimeEntry, ...]:
    metadata = HTP_LIBRARY_METADATA[htp_version]
    stub_size, stub_sha256 = metadata["stub"]
    skel_size, skel_sha256 = metadata["skel"]
    return (
        RuntimeEntry(
            f"qualcomm_runtime_v{htp_version}/src/main/jni/arm64-v8a/"
            "libLiteRtCompilerPlugin_Qualcomm.so",
            "jni/arm64-v8a/libLiteRtCompilerPlugin_Qualcomm.so",
            691_744,
            "c7fe5ee3ac5b89b9e903989a90d1584158b3db889f6c85330495b338951f735d",
            "litert",
        ),
        RuntimeEntry(
            f"qualcomm_runtime_v{htp_version}/src/main/jni/arm64-v8a/"
            "libLiteRtDispatch_Qualcomm.so",
            "jni/arm64-v8a/libLiteRtDispatch_Qualcomm.so",
            462_528,
            "f8ee14eb9cad99a1fc8522478d1db6712417dffeced287e180a0f84d17b6acfb",
            "litert",
        ),
        RuntimeEntry(
            f"qairt/{QAIRT_VERSION}/lib/aarch64-android/libQnnHtpV{htp_version}Stub.so",
            f"jni/arm64-v8a/libQnnHtpV{htp_version}Stub.so",
            stub_size,
            stub_sha256,
            "qairt",
        ),
        RuntimeEntry(
            f"qairt/{QAIRT_VERSION}/lib/hexagon-v{htp_version}/unsigned/"
            f"libQnnHtpV{htp_version}Skel.so",
            f"jni/arm64-v8a/libQnnHtpV{htp_version}Skel.so",
            skel_size,
            skel_sha256,
            "qairt",
        ),
        *COMMON_QAIRT_ENTRIES,
    )


class ZipReader(Protocol):
    def read(self, name: str) -> bytes: ...


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_litert_archive(destination: Path, attempts: int) -> None:
    for attempt in range(1, attempts + 1):
        destination.unlink(missing_ok=True)
        try:
            request = urllib.request.Request(
                LITERT_ARCHIVE_URL,
                headers={"User-Agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                with destination.open("wb") as output:
                    shutil.copyfileobj(response, output)
            actual = sha256_file(destination)
            if actual != LITERT_ARCHIVE_SHA256:
                raise ValueError(
                    "LiteRT archive SHA-256 mismatch: "
                    f"expected {LITERT_ARCHIVE_SHA256}, got {actual}"
                )
            return
        except Exception as error:
            destination.unlink(missing_ok=True)
            if attempt == attempts:
                raise RuntimeError(
                    f"Failed to download the LiteRT archive after {attempts} attempts"
                ) from error
            delay_seconds = min(30, 2 ** (attempt - 1))
            print(
                f"LiteRT archive download failed (attempt {attempt}/{attempts}): "
                f"{error}; retrying in {delay_seconds}s",
                file=sys.stderr,
            )
            time.sleep(delay_seconds)


def write_entry_data(
    data: bytes,
    entry: RuntimeEntry,
    output_dir: Path,
) -> dict[str, object]:
    actual_hash = sha256_bytes(data)
    if len(data) != entry.size or actual_hash != entry.sha256:
        raise ValueError(
            f"Entry mismatch for {entry.source}: expected {entry.size}/{entry.sha256}, "
            f"got {len(data)}/{actual_hash}"
        )
    destination = output_dir / PurePosixPath(entry.destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return {
        "component": entry.component,
        "sourcePath": entry.source,
        "path": entry.destination,
        "bytes": entry.size,
        "sha256": entry.sha256,
    }


def write_entry(reader: ZipReader, entry: RuntimeEntry, output_dir: Path) -> dict[str, object]:
    return write_entry_data(reader.read(entry.source), entry, output_dir)


def read_prepared_entry(entry: RuntimeEntry, prepared_dir: Path) -> bytes:
    source_name = PurePosixPath(entry.source).name
    candidates = (
        prepared_dir / PurePosixPath(entry.destination),
        prepared_dir / "jni" / "arm64-v8a" / source_name,
        prepared_dir / "licenses" / source_name,
    )
    source = next((candidate for candidate in candidates if candidate.is_file()), None)
    if source is None:
        tried = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"Prepared QAIRT entry is missing; tried: {tried}")
    return source.read_bytes()


def open_qairt_archive(local_archive: Path | None):
    if local_archive is not None:
        return zipfile.ZipFile(local_archive)
    try:
        from remotezip import RemoteZip
    except ImportError as error:
        raise RuntimeError(
            "remotezip is required for range extraction; install requirements-qnn.txt"
        ) from error
    return RemoteZip(
        QAIRT_ARCHIVE_URL,
        initial_buffer_size=1024 * 1024,
        headers={"User-Agent": USER_AGENT},
    )


def read_remote_qairt_entry(entry: RuntimeEntry, attempts: int) -> bytes:
    for attempt in range(1, attempts + 1):
        try:
            with open_qairt_archive(None) as reader:
                return reader.read(entry.source)
        except Exception as error:
            if attempt == attempts:
                raise RuntimeError(
                    f"Failed to range-extract {entry.source} after {attempts} attempts"
                ) from error
            delay_seconds = min(30, 2 ** (attempt - 1))
            print(
                f"Range extraction failed for {entry.source} "
                f"(attempt {attempt}/{attempts}): {error}; retrying in "
                f"{delay_seconds}s",
                file=sys.stderr,
            )
            time.sleep(delay_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--htp-version",
        type=int,
        choices=SUPPORTED_HTP_VERSIONS,
        default=79,
        help="Qualcomm HTP runtime generation to package (default: 79).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: .tmp/litert-qnn-v<version>-runtime).",
    )
    parser.add_argument(
        "--qairt-sdk-zip",
        type=Path,
        help="Use a previously downloaded QAIRT 2.44.0.260225 archive.",
    )
    parser.add_argument(
        "--qairt-extracted-dir",
        type=Path,
        help=(
            "Reuse an already extracted and hash-verified QAIRT runtime directory. "
            "This is mutually exclusive with --qairt-sdk-zip."
        ),
    )
    parser.add_argument(
        "--remote-attempts",
        type=int,
        default=5,
        help="Attempts per QAIRT entry when using HTTP range extraction (default: 5).",
    )
    parser.add_argument(
        "--accept-qairt-license",
        action="store_true",
        help="Confirm acceptance of the AI Stack License before extracting QAIRT files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.accept_qairt_license:
        raise SystemExit(
            "QAIRT extraction requires --accept-qairt-license after reviewing "
            "the AI Stack License included in the SDK."
        )
    if args.qairt_sdk_zip is not None and not args.qairt_sdk_zip.is_file():
        raise SystemExit(f"QAIRT archive not found: {args.qairt_sdk_zip}")
    if args.qairt_extracted_dir is not None and not args.qairt_extracted_dir.is_dir():
        raise SystemExit(f"QAIRT extracted directory not found: {args.qairt_extracted_dir}")
    if args.qairt_sdk_zip is not None and args.qairt_extracted_dir is not None:
        raise SystemExit("Use only one of --qairt-sdk-zip and --qairt-extracted-dir.")
    if args.remote_attempts < 1:
        raise SystemExit("--remote-attempts must be at least 1.")

    entries = runtime_entries(args.htp_version)
    output_dir = (
        args.output_dir or Path(f".tmp/litert-qnn-v{args.htp_version}-runtime")
    ).resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-",
        dir=output_dir.parent,
    ) as temporary:
        staging_dir = Path(temporary) / "runtime"
        staging_dir.mkdir()
        litert_archive = Path(temporary) / "litert_npu_runtime_libraries_jit.zip"
        download_litert_archive(litert_archive, args.remote_attempts)
        with zipfile.ZipFile(litert_archive) as reader:
            for entry in entries:
                if entry.component == "litert":
                    records.append(write_entry(reader, entry, staging_dir))

        qairt_entries = [entry for entry in entries if entry.component == "qairt"]
        if args.qairt_extracted_dir is not None:
            prepared_dir = args.qairt_extracted_dir.resolve()
            for entry in qairt_entries:
                records.append(
                    write_entry_data(read_prepared_entry(entry, prepared_dir), entry, staging_dir)
                )
        elif args.qairt_sdk_zip is not None:
            with open_qairt_archive(args.qairt_sdk_zip) as reader:
                for entry in qairt_entries:
                    records.append(write_entry(reader, entry, staging_dir))
        else:
            for entry in qairt_entries:
                records.append(
                    write_entry_data(
                        read_remote_qairt_entry(entry, args.remote_attempts),
                        entry,
                        staging_dir,
                    )
                )

        manifest = {
            "schemaVersion": 1,
            "runtime": {
                "abi": "arm64-v8a",
                "accelerator": f"Qualcomm HTP v{args.htp_version}",
                "htpVersion": args.htp_version,
                "mode": "jit",
                "jniBytes": sum(
                    int(record["bytes"])
                    for record in records
                    if str(record["path"]).startswith("jni/")
                ),
            },
            "sources": {
                "litert": {
                    "version": LITERT_VERSION,
                    "url": LITERT_ARCHIVE_URL,
                    "archiveSha256": LITERT_ARCHIVE_SHA256,
                },
                "qairt": {
                    "version": QAIRT_VERSION,
                    "url": QAIRT_ARCHIVE_URL,
                    "archiveBytes": QAIRT_ARCHIVE_BYTES,
                    "archiveEtag": QAIRT_ARCHIVE_ETAG,
                    "licenseAcceptedByInvoker": True,
                    "rangeExtracted": (
                        args.qairt_sdk_zip is None and args.qairt_extracted_dir is None
                    ),
                    "preparedDirectoryReused": args.qairt_extracted_dir is not None,
                },
            },
            "files": sorted(records, key=lambda record: str(record["path"])),
        }
        manifest_path = staging_dir / "runtime-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.move(str(staging_dir), output_dir)

    manifest_path = output_dir / "runtime-manifest.json"
    print(f"Prepared {manifest['runtime']['jniBytes']} bytes of JNI runtime at {output_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
