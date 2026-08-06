from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = ROOT / "models" / "demucs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one schema v3 tensor-only candidate and its pinned artifacts.",
    )
    parser.add_argument("contract", type=Path)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--run-zero", action="store_true")
    parser.add_argument("--threads", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def resolve_artifact(root: Path, local_path: str) -> Path:
    candidate = (root / Path(*local_path.split("/"))).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Artifact path escapes root: {local_path}") from error
    return candidate


def artifact_descriptors(contract: dict[str, Any]) -> list[dict[str, Any]]:
    upstream = contract["upstream"]
    return [
        contract["conversionSource"],
        contract["conversion"]["exporterInput"],
        contract["conversion"]["repositoryLicense"],
        contract["conversion"]["modelCard"],
        upstream["weight"],
        upstream["metadata"],
        upstream["bagManifest"],
    ]


def validate_artifact(root: Path, descriptor: dict[str, Any]) -> dict[str, Any]:
    path = resolve_artifact(root, descriptor["localPath"])
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_size = path.stat().st_size
    actual_sha256 = sha256(path)
    if actual_size != descriptor["byteSize"]:
        raise ValueError(
            f"Artifact size mismatch for {path}: {actual_size} != {descriptor['byteSize']}",
        )
    if actual_sha256 != descriptor["sha256"]:
        raise ValueError(
            f"Artifact SHA-256 mismatch for {path}: {actual_sha256} != {descriptor['sha256']}",
        )
    return {
        "localPath": descriptor["localPath"],
        "byteSize": actual_size,
        "sha256": actual_sha256,
        "status": "verified",
    }


def tensor_shape(value: onnx.ValueInfoProto) -> list[int | str]:
    result: list[int | str] = []
    for dimension in value.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(int(dimension.dim_value))
        elif dimension.dim_param:
            result.append(dimension.dim_param)
        else:
            result.append("dynamic")
    return result


def validate_graph(contract: dict[str, Any], model_path: Path) -> dict[str, Any]:
    model = onnx.load(model_path, load_external_data=False)
    onnx.checker.check_model(model)
    expected_inputs = contract["tensorContract"]["inputs"]
    expected_outputs = contract["tensorContract"]["outputs"]
    if len(model.graph.input) != len(expected_inputs):
        raise ValueError("ONNX input count does not match the contract")
    if len(model.graph.output) != len(expected_outputs):
        raise ValueError("ONNX output count does not match the contract")
    for value, expected in zip(model.graph.input, expected_inputs, strict=True):
        if value.name != expected["name"]:
            raise ValueError(f"ONNX input name mismatch: {value.name} != {expected['name']}")
        if value.type.tensor_type.elem_type != TensorProto.FLOAT:
            raise ValueError(f"ONNX input is not float32: {value.name}")
        if tensor_shape(value) != expected["shape"]:
            raise ValueError(
                f"ONNX input shape mismatch for {value.name}: {tensor_shape(value)} != {expected['shape']}",
            )
    for value, expected in zip(model.graph.output, expected_outputs, strict=True):
        if value.name != expected["name"]:
            raise ValueError(f"ONNX output name mismatch: {value.name} != {expected['name']}")
        if value.type.tensor_type.elem_type != TensorProto.FLOAT:
            raise ValueError(f"ONNX output is not float32: {value.name}")
    opsets = {item.domain or "ai.onnx": item.version for item in model.opset_import}
    if opsets.get("ai.onnx") != contract["conversion"]["opset"]:
        raise ValueError("ONNX opset does not match the contract")
    if model.producer_name != contract["conversion"]["producerName"]:
        raise ValueError("ONNX producer name does not match the contract")
    if model.producer_version != contract["conversion"]["producerVersion"]:
        raise ValueError("ONNX producer version does not match the contract")
    histogram = Counter(node.op_type for node in model.graph.node)
    precision = validate_precision_contract(contract["precisionContract"], model)
    return {
        "checker": "passed",
        "irVersion": model.ir_version,
        "opsets": opsets,
        "producer": {
            "name": model.producer_name,
            "version": model.producer_version,
        },
        "inputs": [
            {"name": item.name, "shape": tensor_shape(item), "dtype": "float32"}
            for item in model.graph.input
        ],
        "declaredOutputs": [
            {"name": item.name, "shape": tensor_shape(item), "dtype": "float32"}
            for item in model.graph.output
        ],
        "nodeCount": len(model.graph.node),
        "initializerCount": len(model.graph.initializer),
        "opHistogram": dict(histogram.most_common()),
        "precision": precision,
    }


def _tensor_inventory(tensors: list[onnx.TensorProto]) -> dict[str, dict[str, int]]:
    byte_width = {
        TensorProto.FLOAT16: 2,
        TensorProto.FLOAT: 4,
        TensorProto.DOUBLE: 8,
        TensorProto.INT64: 8,
    }
    dtype_name = {
        TensorProto.FLOAT16: "float16",
        TensorProto.FLOAT: "float32",
        TensorProto.DOUBLE: "float64",
        TensorProto.INT64: "int64",
    }
    result: dict[str, dict[str, int]] = {}
    for tensor in tensors:
        if tensor.data_type not in byte_width:
            raise ValueError(f"Unsupported tensor storage dtype in precision inventory: {tensor.data_type}")
        dtype = dtype_name[tensor.data_type]
        elements = int(np.prod(tensor.dims, dtype=np.int64)) if tensor.dims else 1
        entry = result.setdefault(dtype, {"tensorCount": 0, "elementCount": 0, "byteSize": 0})
        entry["tensorCount"] += 1
        entry["elementCount"] += elements
        entry["byteSize"] += elements * byte_width[tensor.data_type]
    return result


def validate_precision_contract(
    expected: dict[str, Any],
    model: onnx.ModelProto,
) -> dict[str, Any]:
    if expected["scope"] != "conversion-source-storage":
        raise ValueError("Unsupported precision contract scope")
    if expected["ioDtype"] != "float32":
        raise ValueError("Candidate I/O precision must be float32")
    if expected["quantization"] != "none":
        raise ValueError("Candidate conversion source must be unquantized")
    if expected["runtimePrecisionStatus"] != "not-established":
        raise ValueError("Candidate runtime precision status must remain not-established")

    def expected_inventory(items: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for item in items:
            dtype = item["dtype"]
            if dtype in result:
                raise ValueError(f"Duplicate precision inventory dtype: {dtype}")
            result[dtype] = {
                "tensorCount": int(item["tensorCount"]),
                "elementCount": int(item["elementCount"]),
                "byteSize": int(item["byteSize"]),
            }
        return result

    initializers = _tensor_inventory(list(model.graph.initializer))
    constants = _tensor_inventory(
        [
            attribute.t
            for node in model.graph.node
            if node.op_type == "Constant"
            for attribute in node.attribute
            if attribute.type == onnx.AttributeProto.TENSOR
        ],
    )
    expected_initializers = expected_inventory(expected["initializerStorage"])
    expected_constants = expected_inventory(expected["constantStorage"])
    if initializers != expected_initializers:
        raise ValueError(f"Initializer precision inventory mismatch: {initializers} != {expected_initializers}")
    if constants != expected_constants:
        raise ValueError(f"Constant precision inventory mismatch: {constants} != {expected_constants}")
    return {
        "ioDtype": expected["ioDtype"],
        "quantization": expected["quantization"],
        "initializerStorage": initializers,
        "constantStorage": constants,
    }


def run_zero_inference(
    contract: dict[str, Any],
    model_path: Path,
    threads: int,
) -> dict[str, Any]:
    import onnxruntime as ort
    import psutil

    process = psutil.Process()
    samples: list[int] = []
    stop = threading.Event()

    def monitor() -> None:
        while not stop.wait(0.01):
            samples.append(process.memory_info().rss)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    before = process.memory_info().rss
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads > 0:
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
    setup_started = time.perf_counter()
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    setup_seconds = time.perf_counter() - setup_started
    after_setup = process.memory_info().rss
    expected_inputs = contract["tensorContract"]["inputs"]
    expected_outputs = contract["tensorContract"]["outputs"]
    runtime_inputs = session.get_inputs()
    runtime_outputs = session.get_outputs()
    for actual, expected in zip(runtime_inputs, expected_inputs, strict=True):
        if (
            actual.name != expected["name"]
            or actual.shape != expected["shape"]
            or actual.type != "tensor(float)"
        ):
            raise ValueError(
                f"ORT input does not match contract: {actual.name} {actual.shape} {actual.type}",
            )
    for actual, expected in zip(runtime_outputs, expected_outputs, strict=True):
        if (
            actual.name != expected["name"]
            or actual.shape != expected["shape"]
            or actual.type != "tensor(float)"
        ):
            raise ValueError(
                f"ORT output does not match contract: {actual.name} {actual.shape} {actual.type}",
            )
    feeds = {
        item["name"]: np.zeros(item["shape"], dtype=np.float32)
        for item in expected_inputs
    }
    inference_started = time.perf_counter()
    values = session.run([item["name"] for item in expected_outputs], feeds)
    inference_seconds = time.perf_counter() - inference_started
    after_inference = process.memory_info().rss
    output_reports = []
    for expected, value in zip(expected_outputs, values, strict=True):
        if list(value.shape) != expected["shape"]:
            raise ValueError(
                f"ORT result shape does not match contract: {list(value.shape)} != {expected['shape']}",
            )
        finite = bool(np.isfinite(value).all())
        if not finite:
            raise ValueError(f"ORT output contains non-finite values: {expected['name']}")
        output_reports.append(
            {
                "name": expected["name"],
                "shape": list(value.shape),
                "finite": finite,
                "minimum": float(value.min()),
                "maximum": float(value.max()),
                "mean": float(value.mean()),
            },
        )
    stop.set()
    monitor_thread.join()
    return {
        "onnxRuntimeVersion": ort.__version__,
        "providers": session.get_providers(),
        "threadCount": threads if threads > 0 else "runtime-default",
        "setupSeconds": setup_seconds,
        "inferenceSeconds": inference_seconds,
        "rssMiB": {
            "before": before / 1024**2,
            "afterSetup": after_setup / 1024**2,
            "afterInference": after_inference / 1024**2,
            "sampledPeak": max(samples, default=after_inference) / 1024**2,
        },
        "outputs": output_reports,
    }


def main() -> int:
    args = parse_args()
    contract_path = args.contract.resolve()
    artifact_root = args.artifact_root.resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("contractSchemaVersion") != 3:
        raise ValueError("Expected contractSchemaVersion 3")
    if contract.get("contractKind") != "tensor-only-candidate":
        raise ValueError("Expected a tensor-only-candidate contract")
    if (
        contract["conversion"]["verificationStatus"] == "local-structure-and-zero-input"
        and not args.run_zero
    ):
        raise ValueError("This contract's verification status requires --run-zero")
    artifacts = [validate_artifact(artifact_root, item) for item in artifact_descriptors(contract)]
    model_path = resolve_artifact(artifact_root, contract["conversionSource"]["localPath"])
    graph_report = validate_graph(contract, model_path)
    gc.collect()
    report: dict[str, Any] = {
        "contract": {
            "path": str(contract_path),
            "id": contract["contractId"],
            "sha256": sha256(contract_path),
        },
        "artifacts": artifacts,
        "graph": graph_report,
    }
    if args.run_zero:
        report["zeroInference"] = run_zero_inference(contract, model_path, args.threads)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
