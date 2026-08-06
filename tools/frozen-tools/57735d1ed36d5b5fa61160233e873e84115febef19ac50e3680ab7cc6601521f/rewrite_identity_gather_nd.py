#!/usr/bin/env python3
"""Rewrite identity GATHER_ND operators in a fixed TFLite model to RESHAPE.

The rewrite is intentionally narrow. It accepts only rank-one data tensors and
constant INT32 indices whose values are exactly ``0..N-1`` and whose shape is
``output_shape + [1]``. The original index tensors and buffers are retained;
new, deduplicated shape constants are appended to avoid global index remapping.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import flatbuffers
import numpy as np


DEFAULT_INPUT_SHA256 = "a9fcc89e84aa65313e0540b582e710007ed12064969a0d49a3c85e49f1ae4e3d"
DEFAULT_SCHEMA_SHA256 = "b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8"
DEFAULT_MODEL_ID = "bandbuddy_htdemucs_6s_core_gather_nd_reshape_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--schema-module",
        type=Path,
        required=True,
        help="Python schema generated with flatc --gen-object-api",
    )
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--expected-input-sha256", default=DEFAULT_INPUT_SHA256)
    parser.add_argument("--expected-schema-sha256", default=DEFAULT_SCHEMA_SHA256)
    parser.add_argument("--expected-operator-count", type=int, default=3504)
    parser.add_argument("--expected-rewrite-count", type=int, default=132)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_schema(path: Path, expected_sha256: str) -> ModuleType:
    actual_sha256 = sha256(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Schema module SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    spec = importlib.util.spec_from_file_location("pinned_tflite_schema", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load schema module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = (
        "Model",
        "ModelT",
        "BufferT",
        "TensorT",
        "ReshapeOptionsT",
        "BuiltinOperator",
        "BuiltinOptions",
        "TensorType",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"Schema module lacks object API: {', '.join(missing)}")
    return module


def int_list(value: Any) -> list[int]:
    if value is None:
        return []
    return [int(item) for item in value]


def bytes_from_u8(value: Any) -> bytes:
    if value is None:
        return b""
    return np.asarray(value, dtype=np.uint8).tobytes()


def builtin_name_map(schema: ModuleType) -> dict[int, str]:
    return {
        value: name
        for name, value in vars(schema.BuiltinOperator).items()
        if isinstance(value, int) and not name.startswith("_")
    }


def opcode_value(code: Any) -> int:
    value = int(code.builtinCode)
    if value == 0 and int(code.deprecatedBuiltinCode) != 0:
        return int(code.deprecatedBuiltinCode)
    return value


def operator_histogram(schema: ModuleType, model: Any, subgraph: Any) -> dict[str, int]:
    names = builtin_name_map(schema)
    counts = Counter(
        names.get(opcode_value(model.operatorCodes[int(operator.opcodeIndex)]), "UNKNOWN")
        for operator in subgraph.operators
    )
    return dict(sorted(counts.items()))


def tensor_shape(tensor: Any, label: str) -> list[int]:
    shape = int_list(tensor.shape)
    if not shape or any(dimension <= 0 for dimension in shape):
        raise RuntimeError(f"{label} must have a fully static positive shape, got {shape}")
    signature = int_list(tensor.shapeSignature)
    if signature and signature != shape:
        raise RuntimeError(f"{label} has a dynamic or inconsistent shape signature: {signature}")
    if not bool(tensor.hasRank):
        raise RuntimeError(f"{label} does not have a known rank")
    return shape


def tensor_buffer_bytes(model: Any, tensor: Any, label: str) -> bytes:
    if int(tensor.externalBuffer) != 0:
        raise RuntimeError(f"{label} uses an external buffer")
    buffer_index = int(tensor.buffer)
    if buffer_index <= 0 or buffer_index >= len(model.buffers):
        raise RuntimeError(f"{label} has no embedded constant buffer")
    buffer = model.buffers[buffer_index]
    if int(buffer.offset) != 0 or int(buffer.size) != 0:
        raise RuntimeError(f"{label} uses external buffer offset/size fields")
    data = bytes_from_u8(buffer.data)
    if not data:
        raise RuntimeError(f"{label} has an empty constant buffer")
    return data


def product(shape: list[int]) -> int:
    return math.prod(shape)


def signature_fingerprint(model: Any) -> list[dict[str, Any]]:
    return [
        {
            "key": (signature.signatureKey or b"").decode("utf-8", errors="strict"),
            "subgraphIndex": int(signature.subgraphIndex),
            "inputs": [
                {
                    "name": (item.name or b"").decode("utf-8", errors="strict"),
                    "tensorIndex": int(item.tensorIndex),
                }
                for item in signature.inputs
            ],
            "outputs": [
                {
                    "name": (item.name or b"").decode("utf-8", errors="strict"),
                    "tensorIndex": int(item.tensorIndex),
                }
                for item in signature.outputs
            ],
        }
        for signature in model.signatureDefs
    ]


def io_fingerprint(subgraph: Any) -> dict[str, list[dict[str, Any]]]:
    def describe(position: int, tensor_index: int) -> dict[str, Any]:
        tensor = subgraph.tensors[tensor_index]
        return {
            "index": position,
            "tensorIndex": tensor_index,
            "name": (tensor.name or b"").decode("utf-8", errors="replace"),
            "type": int(tensor.type),
            "shape": int_list(tensor.shape),
            "shapeSignature": int_list(tensor.shapeSignature),
        }

    return {
        "inputs": [describe(position, int(index)) for position, index in enumerate(subgraph.inputs)],
        "outputs": [describe(position, int(index)) for position, index in enumerate(subgraph.outputs)],
    }


def validate_operator_fields(operator: Any, node: int) -> None:
    if operator.customOptions is not None and len(operator.customOptions) != 0:
        raise RuntimeError(f"GATHER_ND node {node} has custom options")
    if operator.mutatingVariableInputs is not None and len(operator.mutatingVariableInputs) != 0:
        raise RuntimeError(f"GATHER_ND node {node} has mutating variable inputs")
    if operator.intermediates is not None and len(operator.intermediates) != 0:
        raise RuntimeError(f"GATHER_ND node {node} has intermediates")
    if int(operator.largeCustomOptionsOffset) != 0 or int(operator.largeCustomOptionsSize) != 0:
        raise RuntimeError(f"GATHER_ND node {node} has large custom options")
    if int(operator.builtinOptionsType) != 0 or operator.builtinOptions is not None:
        raise RuntimeError(f"GATHER_ND node {node} has unexpected builtin options")
    if int(operator.builtinOptions2Type) != 0 or operator.builtinOptions2 is not None:
        raise RuntimeError(f"GATHER_ND node {node} has unexpected builtin options 2")


def find_candidates(schema: ModuleType, model: Any, subgraph: Any) -> list[dict[str, Any]]:
    consumer_counts = Counter(
        int(tensor_index)
        for operator in subgraph.operators
        for tensor_index in operator.inputs
        if int(tensor_index) >= 0
    )
    protected_tensors = {int(index) for index in subgraph.inputs} | {
        int(index) for index in subgraph.outputs
    }
    for signature in model.signatureDefs:
        protected_tensors.update(int(item.tensorIndex) for item in signature.inputs)
        protected_tensors.update(int(item.tensorIndex) for item in signature.outputs)

    candidates: list[dict[str, Any]] = []
    for node, operator in enumerate(subgraph.operators):
        code = model.operatorCodes[int(operator.opcodeIndex)]
        if opcode_value(code) != schema.BuiltinOperator.GATHER_ND:
            continue
        validate_operator_fields(operator, node)
        inputs = int_list(operator.inputs)
        outputs = int_list(operator.outputs)
        if len(inputs) != 2 or len(outputs) != 1 or min(inputs + outputs) < 0:
            raise RuntimeError(f"GATHER_ND node {node} must have two inputs and one output")
        data_index, indices_index = inputs
        output_index = outputs[0]
        if max(data_index, indices_index, output_index) >= len(subgraph.tensors):
            raise RuntimeError(f"GATHER_ND node {node} references an invalid tensor")

        data = subgraph.tensors[data_index]
        indices = subgraph.tensors[indices_index]
        output = subgraph.tensors[output_index]
        data_shape = tensor_shape(data, f"node {node} data")
        indices_shape = tensor_shape(indices, f"node {node} indices")
        output_shape = tensor_shape(output, f"node {node} output")
        if len(data_shape) != 1:
            raise RuntimeError(f"GATHER_ND node {node} data rank is not one: {data_shape}")
        if int(data.type) != schema.TensorType.FLOAT32 or int(output.type) != int(data.type):
            raise RuntimeError(f"GATHER_ND node {node} is not FLOAT32-preserving")
        if int(indices.type) != schema.TensorType.INT32:
            raise RuntimeError(f"GATHER_ND node {node} indices are not INT32")
        if indices_shape != output_shape + [1]:
            raise RuntimeError(
                f"GATHER_ND node {node} indices shape {indices_shape} does not equal "
                f"output shape plus one {output_shape + [1]}"
            )
        if product(data_shape) != product(output_shape):
            raise RuntimeError(f"GATHER_ND node {node} changes the element count")
        if indices_index in protected_tensors:
            raise RuntimeError(f"GATHER_ND node {node} indices tensor is externally visible")
        raw_indices = tensor_buffer_bytes(model, indices, f"node {node} indices")
        expected_bytes = product(indices_shape) * np.dtype("<i4").itemsize
        if len(raw_indices) != expected_bytes:
            raise RuntimeError(
                f"GATHER_ND node {node} index buffer has {len(raw_indices)} bytes, "
                f"expected {expected_bytes}"
            )
        values = np.frombuffer(raw_indices, dtype="<i4")
        expected = np.arange(data_shape[0], dtype=np.int32)
        if values.size != expected.size or not np.array_equal(values, expected):
            raise RuntimeError(f"GATHER_ND node {node} indices are not the identity range")

        candidates.append(
            {
                "node": node,
                "dataTensor": data_index,
                "indicesTensor": indices_index,
                "outputTensor": output_index,
                "dataShape": data_shape,
                "indicesShape": indices_shape,
                "outputShape": output_shape,
                "indexCount": int(values.size),
                "indicesTensorConsumerCount": consumer_counts[indices_index],
                "debugMetadataIndex": int(operator.debugMetadataIndex),
            }
        )
    return candidates


def append_shape_tensor(
    schema: ModuleType,
    model: Any,
    subgraph: Any,
    output_shape: tuple[int, ...],
) -> int:
    raw = np.asarray(output_shape, dtype="<i4").tobytes()
    buffer = schema.BufferT()
    buffer.data = np.frombuffer(raw, dtype=np.uint8).copy()
    buffer.offset = 0
    buffer.size = 0
    buffer_index = len(model.buffers)
    model.buffers.append(buffer)

    tensor = schema.TensorT()
    tensor.shape = np.asarray([len(output_shape)], dtype=np.int32)
    tensor.type = schema.TensorType.INT32
    tensor.buffer = buffer_index
    shape_digest = hashlib.sha256(raw).hexdigest()[:12]
    tensor.name = f"mss.identity_gather_nd_shape.{shape_digest}".encode("ascii")
    tensor.quantization = None
    tensor.isVariable = False
    tensor.sparsity = None
    tensor.shapeSignature = None
    tensor.hasRank = True
    tensor.variantTensors = None
    tensor.externalBuffer = 0
    subgraph.tensors.append(tensor)
    return len(subgraph.tensors) - 1


def rewrite_candidates(
    schema: ModuleType,
    model: Any,
    subgraph: Any,
    candidates: list[dict[str, Any]],
) -> dict[tuple[int, ...], int]:
    reshape_codes = [
        index
        for index, code in enumerate(model.operatorCodes)
        if opcode_value(code) == schema.BuiltinOperator.RESHAPE
    ]
    if len(reshape_codes) != 1:
        raise RuntimeError(f"Expected one existing RESHAPE opcode, got {reshape_codes}")
    reshape_opcode_index = reshape_codes[0]
    shape_tensors: dict[tuple[int, ...], int] = {}

    for candidate in candidates:
        shape = tuple(candidate["outputShape"])
        shape_tensor = shape_tensors.get(shape)
        if shape_tensor is None:
            shape_tensor = append_shape_tensor(schema, model, subgraph, shape)
            shape_tensors[shape] = shape_tensor
        operator = subgraph.operators[candidate["node"]]
        operator.opcodeIndex = reshape_opcode_index
        operator.inputs = np.asarray(
            [candidate["dataTensor"], shape_tensor],
            dtype=np.int32,
        )
        options = schema.ReshapeOptionsT()
        options.newShape = np.asarray(shape, dtype=np.int32)
        operator.builtinOptionsType = schema.BuiltinOptions.ReshapeOptions
        operator.builtinOptions = options
        operator.builtinOptions2Type = 0
        operator.builtinOptions2 = None
        candidate["shapeTensor"] = shape_tensor
    return shape_tensors


def validate_repacked(
    schema: ModuleType,
    output_bytes: bytes,
    original_io: dict[str, Any],
    original_signatures: list[dict[str, Any]],
    original_operator_count: int,
    original_tensor_count: int,
    original_buffer_count: int,
    expected_rewrites: int,
    expected_new_shapes: int,
) -> dict[str, Any]:
    if len(output_bytes) < 8 or output_bytes[4:8] != b"TFL3":
        raise RuntimeError("Repacked model lacks the TFL3 file identifier")
    packed = schema.Model.GetRootAsModel(output_bytes, 0)
    model = schema.ModelT.InitFromObj(packed)
    if int(model.version) != 3 or len(model.subgraphs) != 1:
        raise RuntimeError("Repacked model changed the schema version or subgraph count")
    subgraph = model.subgraphs[0]
    if len(subgraph.operators) != original_operator_count:
        raise RuntimeError("Repacked model changed the operator count")
    if len(subgraph.tensors) != original_tensor_count + expected_new_shapes:
        raise RuntimeError("Repacked model has an unexpected tensor count")
    if len(model.buffers) != original_buffer_count + expected_new_shapes:
        raise RuntimeError("Repacked model has an unexpected buffer count")
    if io_fingerprint(subgraph) != original_io:
        raise RuntimeError("Repacked model changed graph inputs or outputs")
    if signature_fingerprint(model) != original_signatures:
        raise RuntimeError("Repacked model changed signature definitions")

    histogram = operator_histogram(schema, model, subgraph)
    if histogram.get("GATHER_ND", 0) != 0:
        raise RuntimeError("Repacked model still contains GATHER_ND")
    rewritten = [
        operator
        for operator in subgraph.operators
        if opcode_value(model.operatorCodes[int(operator.opcodeIndex)])
        == schema.BuiltinOperator.RESHAPE
        and int(operator.builtinOptionsType) == schema.BuiltinOptions.ReshapeOptions
        and len(operator.inputs) == 2
        and int(operator.inputs[1]) >= original_tensor_count
    ]
    if len(rewritten) != expected_rewrites:
        raise RuntimeError(
            f"Repacked model exposes {len(rewritten)} rewritten RESHAPE nodes, "
            f"expected {expected_rewrites}"
        )
    for operator in rewritten:
        data = subgraph.tensors[int(operator.inputs[0])]
        shape_tensor = subgraph.tensors[int(operator.inputs[1])]
        output = subgraph.tensors[int(operator.outputs[0])]
        shape_bytes = tensor_buffer_bytes(model, shape_tensor, "rewritten shape tensor")
        shape_values = np.frombuffer(shape_bytes, dtype="<i4").astype(np.int64).tolist()
        if shape_values != int_list(output.shape):
            raise RuntimeError("Rewritten RESHAPE constant does not match its output shape")
        if product(int_list(data.shape)) != product(shape_values):
            raise RuntimeError("Rewritten RESHAPE changes the element count")
    return {
        "schemaVersion": int(model.version),
        "subgraphCount": len(model.subgraphs),
        "operatorCount": len(subgraph.operators),
        "tensorCount": len(subgraph.tensors),
        "bufferCount": len(model.buffers),
        "operatorHistogram": histogram,
        "inputs": original_io["inputs"],
        "outputs": original_io["outputs"],
        "signatures": original_signatures,
    }


def main() -> int:
    args = parse_args()
    for path, label in (
        (args.input, "input model"),
        (args.schema_module, "schema module"),
        (args.source_contract, "source contract"),
    ):
        if not path.is_file():
            raise RuntimeError(f"Missing {label}: {path}")
    if args.output.exists() and not args.force:
        raise RuntimeError(f"Output already exists; pass --force to replace it: {args.output}")
    if args.report.exists() and not args.force:
        raise RuntimeError(f"Report already exists; pass --force to replace it: {args.report}")

    source_bytes = args.input.read_bytes()
    input_sha256 = sha256_bytes(source_bytes)
    if input_sha256 != args.expected_input_sha256:
        raise RuntimeError(
            f"Input SHA-256 mismatch: expected {args.expected_input_sha256}, got {input_sha256}"
        )
    if len(source_bytes) < 8 or source_bytes[4:8] != b"TFL3":
        raise RuntimeError("Input is not a TFL3 FlatBuffer")

    source_contract_bytes = args.source_contract.read_bytes()
    source_contract = json.loads(source_contract_bytes.decode("utf-8"))
    artifact = source_contract.get("artifact", {})
    if artifact.get("sha256") != input_sha256 or artifact.get("byteSize") != len(source_bytes):
        raise RuntimeError("Source contract does not identify the input model")

    schema = load_schema(args.schema_module, args.expected_schema_sha256)
    packed = schema.Model.GetRootAsModel(source_bytes, 0)
    model = schema.ModelT.InitFromObj(packed)
    if int(model.version) != 3:
        raise RuntimeError(f"Expected TFLite schema version 3, got {model.version}")
    if len(model.subgraphs) != 1:
        raise RuntimeError(f"Expected one subgraph, got {len(model.subgraphs)}")
    subgraph = model.subgraphs[0]
    if len(subgraph.operators) != args.expected_operator_count:
        raise RuntimeError(
            f"Expected {args.expected_operator_count} operators, got {len(subgraph.operators)}"
        )

    original_operator_count = len(subgraph.operators)
    original_tensor_count = len(subgraph.tensors)
    original_buffer_count = len(model.buffers)
    original_histogram = operator_histogram(schema, model, subgraph)
    original_io = io_fingerprint(subgraph)
    original_signatures = signature_fingerprint(model)
    candidates = find_candidates(schema, model, subgraph)
    if len(candidates) != args.expected_rewrite_count:
        raise RuntimeError(
            f"Expected {args.expected_rewrite_count} identity GATHER_ND nodes, got {len(candidates)}"
        )

    shape_tensors = rewrite_candidates(schema, model, subgraph, candidates)
    builder = flatbuffers.Builder(len(source_bytes) + len(shape_tensors) * 256)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    output_bytes = bytes(builder.Output())
    validation = validate_repacked(
        schema=schema,
        output_bytes=output_bytes,
        original_io=original_io,
        original_signatures=original_signatures,
        original_operator_count=original_operator_count,
        original_tensor_count=original_tensor_count,
        original_buffer_count=original_buffer_count,
        expected_rewrites=len(candidates),
        expected_new_shapes=len(shape_tensors),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output_bytes)
    output_sha256 = sha256_bytes(output_bytes)
    tool_path = Path(__file__).resolve()
    report = {
        "schemaVersion": 1,
        "kind": "litert-flatbuffer-rewrite",
        "modelId": args.model_id,
        "source": {
            "modelId": source_contract.get("modelId"),
            "contract": {
                "fileName": args.source_contract.name,
                "byteSize": len(source_contract_bytes),
                "sha256": sha256_bytes(source_contract_bytes),
            },
            "artifact": {
                "fileName": args.input.name,
                "byteSize": len(source_bytes),
                "sha256": input_sha256,
            },
        },
        "artifact": {
            "fileName": args.output.name,
            "byteSize": len(output_bytes),
            "sha256": output_sha256,
            "format": "tflite-flatbuffer",
        },
        "rewrite": {
            "rule": "identity-gather-nd-to-reshape-v1",
            "rewriteCount": len(candidates),
            "uniqueOutputShapeCount": len(shape_tensors),
            "orphanIdentityIndexTensorsRetained": True,
            "tool": {
                "fileName": tool_path.name,
                "sha256": sha256(tool_path),
            },
            "schemaModule": {
                "fileName": args.schema_module.name,
                "sha256": sha256(args.schema_module),
            },
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "flatbuffers": importlib.metadata.version("flatbuffers"),
            },
            "shapeTensors": [
                {"shape": list(shape), "tensorIndex": tensor_index}
                for shape, tensor_index in shape_tensors.items()
            ],
            "operators": candidates,
        },
        "invariants": {
            "ioUnchanged": True,
            "signaturesUnchanged": True,
            "operatorCountUnchanged": True,
            "nonTargetOperatorsUnchangedInObjectGraph": True,
            "identityRewriteExact": True,
        },
        "flatBuffer": {
            **validation,
            "sourceTensorCount": original_tensor_count,
            "sourceBufferCount": original_buffer_count,
            "sourceOperatorHistogram": original_histogram,
        },
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
