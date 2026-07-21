#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
source_dir="${1:-/opt/litert-v2.1.5}"
ndk_dir="${2:-${ANDROID_NDK_HOME:-/opt/android-sdk/ndk/25.1.8937393}}"
output_so="${3:-${repo_root}/.tmp/litert-2.1.5-x86/libLiteRt.so}"
output_user_root="${BAZEL_OUTPUT_USER_ROOT:-/opt/bazel-output/litert215-x86}"
bazel_bin="${BAZEL:-bazel}"
patch_file="${script_dir}/litert-2.1.5-x86.patch"

if [[ ! -f "${source_dir}/WORKSPACE" ]]; then
    echo "LiteRT source tree not found: ${source_dir}" >&2
    exit 1
fi
if [[ ! -f "${ndk_dir}/source.properties" ]]; then
    echo "Android NDK not found: ${ndk_dir}" >&2
    exit 1
fi
if ! grep -q 'Pkg.Revision = 25.1.8937393' "${ndk_dir}/source.properties"; then
    echo "This build is pinned to Android NDK 25.1.8937393 (r25b)." >&2
    exit 1
fi
if [[ "$(${bazel_bin} --version)" != "bazel 7.7.0" ]]; then
    echo "This build is pinned to Bazel 7.7.0." >&2
    exit 1
fi

# Bazel embeds multiline genrule commands verbatim, so CRLF BUILD files fail
# under Linux before compilation starts.
find "${source_dir}" -type f \
    \( -name BUILD -o -name 'BUILD.*' -o -name '*.bzl' -o -name '*.bazel' \
       -o -name WORKSPACE -o -name 'WORKSPACE.*' -o -name '*.sh' \) \
    -exec sed -i 's/\r$//' {} +

if patch --batch --reverse --dry-run --silent -d "${source_dir}" -p1 \
    < "${patch_file}" >/dev/null 2>&1; then
    :
elif patch --batch --forward --dry-run --silent -d "${source_dir}" -p1 \
    < "${patch_file}" >/dev/null 2>&1; then
    patch --batch --forward --silent -d "${source_dir}" -p1 < "${patch_file}"
else
    echo "LiteRT source does not match the expected 2.1.5 tree or patch state." >&2
    exit 1
fi

export ANDROID_HOME="${ANDROID_HOME:-$(dirname "$(dirname "${ndk_dir}")")}"
export ANDROID_NDK_HOME="${ndk_dir}"
export ANDROID_NDK_ROOT="${ndk_dir}"

cd "${source_dir}"
"${bazel_bin}" --output_user_root="${output_user_root}" build \
    --config=android_x86 \
    --incompatible_enable_cc_toolchain_resolution \
    --incompatible_enable_android_toolchain_resolution \
    --repo_env=HERMETIC_PYTHON_VERSION=3.11 \
    --python_path=/usr/bin/python3 \
    --//litert/build_common:build_include=cpu_only \
    --define=xnn_enable_avxvnni=false \
    --define=xnn_enable_avxvnniint8=false \
    --define=xnn_enable_avx512fp16=false \
    --define=xnn_enable_avx512amx=false \
    //litert/kotlin:LiteRt

built_so="$(readlink -f bazel-bin/litert/kotlin/libLiteRt.so)"
readelf_bin="${ndk_dir}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-readelf"
if ! "${readelf_bin}" -h "${built_so}" | grep -q 'Class:.*ELF32'; then
    echo "Built library is not ELF32: ${built_so}" >&2
    exit 1
fi
if ! "${readelf_bin}" -h "${built_so}" | grep -q 'Machine:.*Intel 80386'; then
    echo "Built library is not 32-bit x86: ${built_so}" >&2
    exit 1
fi
if "${readelf_bin}" -d "${built_so}" | grep -Eq 'lib(EGL|GLES|OpenCL|vulkan)'; then
    echo "CPU-only library unexpectedly has a GPU runtime dependency." >&2
    exit 1
fi

mkdir -p "$(dirname "${output_so}")"
if [[ -e "${output_so}" ]]; then
    chmod u+w "${output_so}" || true
    rm -f "${output_so}"
fi
install -m 0644 "${built_so}" "${output_so}"
ls -lh "${output_so}"
sha256sum "${output_so}"
"${readelf_bin}" -d "${output_so}" | grep -E 'NEEDED|SONAME'
