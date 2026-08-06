from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
from typing import Any

import tflite


RULE_ID = "identity-gather-nd-to-reshape-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite identity TFLite GATHER_ND operators as two-input RESHAPE "
            "operators without changing FlatBuffer offsets."
        ),
    )
    parser.add_argument("input_model", type=Path)
    parser.add_argument("output_model", type=Path)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-rewrite-count", required=True, type=int)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_bytes(payload: bytes | bytearray) -> str:
    return hashlib.sha256(payload).hexdigest()


def tensor_shape(tensor: Any) -> list[int]:
    return [int(tensor.Shape(index)) for index in range(tensor.ShapeLength())]


def vector_location(table: Any, field: int) -> tuple[int, int]:
    offset = int(table._tab.Offset(field))
    if offset == 0:
        raise ValueError(f"FlatBuffer field {field} has no vector")
    data_start = int(table._tab.Vector(offset))
    return data_start - 4, data_start


def scalar_location(table: Any, field: int) -> int:
    offset = int(table._tab.Offset(field))
    if offset == 0:
        raise ValueError(f"FlatBuffer field {field} has no scalar")
    return int(table._tab.Pos) + offset


def builtin_code(model: Any, opcode_index: int) -> int:
    return int(model.OperatorCodes(opcode_index).BuiltinCode())


def opcode_indices(model: Any, wanted: int) -> list[int]:
    return [
        index
        for index in range(model.OperatorCodesLength())
        if builtin_code(model, index) == wanted
    ]


def buffer_bytes(model: Any, buffer_index: int) -> bytes:
    value = model.Buffers(buffer_index)
    if value.DataLength() == 0:
        return b""
    return bytes(value.DataAsNumpy())


def decode_name(value: bytes | None) -> str:
    return value.decode("utf-8") if value else ""


def analyze(payload: bytes | bytearray, expected_rewrite_count: int) -> dict[str, Any]:
    if len(payload) < 8 or bytes(payload[4:8]) != b"TFL3":
        raise ValueError("Input is not a TFL3 FlatBuffer")
    model = tflite.Model.GetRootAsModel(payload, 0)
    if int(model.Version()) != 3:
        raise ValueError(f"Expected TFLite schema version 3, got {model.Version()}")
    if model.SubgraphsLength() != 1:
        raise ValueError("Rewrite requires exactly one TFLite subgraph")

    gather_opcodes = opcode_indices(model, tflite.BuiltinOperator.GATHER_ND)
    reshape_opcodes = opcode_indices(model, tflite.BuiltinOperator.RESHAPE)
    if len(gather_opcodes) != 1:
        raise ValueError(f"Expected one GATHER_ND OperatorCode, got {gather_opcodes}")
    if len(reshape_opcodes) != 1:
        raise ValueError(f"Expected one RESHAPE OperatorCode, got {reshape_opcodes}")
    gather_opcode = gather_opcodes[0]
    reshape_opcode = reshape_opcodes[0]
    if int(model.OperatorCodes(reshape_opcode).Version()) != 1:
        raise ValueError("Rewrite requires the existing RESHAPE OperatorCode to be version 1")

    subgraph = model.Subgraphs(0)
    consumers: Counter[int] = Counter()
    producers: Counter[int] = Counter()
    tensor_buffers: defaultdict[int, list[int]] = defaultdict(list)
    for tensor_index in range(subgraph.TensorsLength()):
        tensor_buffers[int(subgraph.Tensors(tensor_index).Buffer())].append(tensor_index)
    for operator_index in range(subgraph.OperatorsLength()):
        operator = subgraph.Operators(operator_index)
        for input_index in range(operator.InputsLength()):
            tensor_index = int(operator.Inputs(input_index))
            if tensor_index >= 0:
                consumers[tensor_index] += 1
        for output_index in range(operator.OutputsLength()):
            tensor_index = int(operator.Outputs(output_index))
            if tensor_index >= 0:
                producers[tensor_index] += 1
    graph_io = {
        int(subgraph.Inputs(index)) for index in range(subgraph.InputsLength())
    } | {
        int(subgraph.Outputs(index)) for index in range(subgraph.OutputsLength())
    }

    rewrites: list[dict[str, Any]] = []
    shape_tensor_targets: dict[int, tuple[int, ...]] = {}
    for operator_index in range(subgraph.OperatorsLength()):
        operator = subgraph.Operators(operator_index)
        if int(operator.OpcodeIndex()) != gather_opcode:
            continue
        if operator.InputsLength() != 2 or operator.OutputsLength() != 1:
            raise ValueError(f"GATHER_ND operator {operator_index} does not have 2 inputs and 1 output")
        if operator.BuiltinOptionsType() != tflite.BuiltinOptions.NONE:
            raise ValueError(f"GATHER_ND operator {operator_index} has unexpected builtin options")
        if operator.CustomOptionsLength() != 0:
            raise ValueError(f"GATHER_ND operator {operator_index} has custom options")

        data_index = int(operator.Inputs(0))
        indices_index = int(operator.Inputs(1))
        output_index = int(operator.Outputs(0))
        data_tensor = subgraph.Tensors(data_index)
        indices_tensor = subgraph.Tensors(indices_index)
        output_tensor = subgraph.Tensors(output_index)
        data_shape = tensor_shape(data_tensor)
        indices_shape = tensor_shape(indices_tensor)
        output_shape = tensor_shape(output_tensor)
        element_count = math.prod(output_shape)

        if not output_shape or any(dimension <= 0 for dimension in output_shape):
            raise ValueError(f"GATHER_ND operator {operator_index} has a dynamic or scalar output")
        if data_shape != [element_count]:
            raise ValueError(
                f"GATHER_ND operator {operator_index} data shape {data_shape} is not "
                f"the flattened output shape {output_shape}",
            )
        if indices_shape != [*output_shape, 1]:
            raise ValueError(
                f"GATHER_ND operator {operator_index} indices shape {indices_shape} "
                f"does not equal output shape plus a singleton index depth",
            )
        if indices_tensor.Type() != tflite.TensorType.INT32:
            raise ValueError(f"GATHER_ND operator {operator_index} indices are not INT32")
        if bool(indices_tensor.IsVariable()):
            raise ValueError(f"GATHER_ND operator {operator_index} indices tensor is variable")
        if not bool(indices_tensor.HasRank()):
            raise ValueError(f"GATHER_ND operator {operator_index} indices tensor has no declared rank")
        if indices_tensor.ShapeSignatureLength() != 0:
            raise ValueError(f"GATHER_ND operator {operator_index} indices tensor has a shape signature")
        if indices_tensor.Sparsity() is not None:
            raise ValueError(f"GATHER_ND operator {operator_index} indices tensor is sparse")
        if producers[indices_index] != 0 or indices_index in graph_io:
            raise ValueError(f"GATHER_ND operator {operator_index} indices tensor is not a private constant")
        if data_tensor.Type() != output_tensor.Type():
            raise ValueError(f"GATHER_ND operator {operator_index} changes tensor dtype")

        buffer_index = int(indices_tensor.Buffer())
        if buffer_index == 0 or len(tensor_buffers[buffer_index]) != 1:
            raise ValueError(
                f"GATHER_ND operator {operator_index} indices buffer is absent or shared by another tensor",
            )
        value = model.Buffers(buffer_index)
        if value.Offset() != 0 or value.Size() != 0:
            raise ValueError(f"GATHER_ND operator {operator_index} uses an external indices buffer")
        _, data_position = vector_location(value, 4)
        if data_position % 4 != 0:
            raise ValueError(f"GATHER_ND operator {operator_index} indices buffer is not INT32-aligned")
        raw_indices = buffer_bytes(model, buffer_index)
        if len(raw_indices) != element_count * 4:
            raise ValueError(f"GATHER_ND operator {operator_index} indices byte size is inconsistent")
        indices = struct.unpack(f"<{element_count}i", raw_indices)
        if any(actual != expected for expected, actual in enumerate(indices)):
            raise ValueError(f"GATHER_ND operator {operator_index} indices are not exactly 0..N-1")

        target = tuple(output_shape)
        previous_target = shape_tensor_targets.setdefault(indices_index, target)
        if previous_target != target:
            raise ValueError(f"Shared indices tensor {indices_index} has inconsistent reshape targets")
        rewrites.append(
            {
                "operatorIndex": operator_index,
                "dataTensorIndex": data_index,
                "indicesTensorIndex": indices_index,
                "outputTensorIndex": output_index,
                "outputShape": output_shape,
            },
        )

    if len(rewrites) != expected_rewrite_count:
        raise ValueError(
            f"Expected {expected_rewrite_count} identity GATHER_ND operators, got {len(rewrites)}",
        )
    rewritten_indices = Counter(item["indicesTensorIndex"] for item in rewrites)
    for indices_index, use_count in rewritten_indices.items():
        if consumers[indices_index] != use_count:
            raise ValueError(
                f"Indices tensor {indices_index} has {consumers[indices_index]} consumers, "
                f"but only {use_count} are proven identity GATHER_ND operators",
            )

    return {
        "model": model,
        "subgraph": subgraph,
        "gatherOpcodeIndex": gather_opcode,
        "reshapeOpcodeIndex": reshape_opcode,
        "rewrites": rewrites,
        "shapeTensorTargets": shape_tensor_targets,
    }


def apply_rewrite(payload: bytes, analysis: dict[str, Any]) -> bytearray:
    patched = bytearray(payload)
    subgraph = analysis["subgraph"]
    reshape_opcode = analysis["reshapeOpcodeIndex"]

    for item in analysis["rewrites"]:
        operator = subgraph.Operators(item["operatorIndex"])
        struct.pack_into("<I", patched, scalar_location(operator, 4), reshape_opcode)

    for tensor_index, target_shape in analysis["shapeTensorTargets"].items():
        tensor = subgraph.Tensors(tensor_index)
        shape_length_position, shape_data_position = vector_location(tensor, 4)
        if tensor.ShapeLength() < 1:
            raise ValueError(f"Shape tensor {tensor_index} has no shape-vector capacity")
        struct.pack_into("<I", patched, shape_length_position, 1)
        struct.pack_into("<i", patched, shape_data_position, len(target_shape))

        buffer = analysis["model"].Buffers(int(tensor.Buffer()))
        data_length_position, data_position = vector_location(buffer, 4)
        required_bytes = len(target_shape) * 4
        if buffer.DataLength() < required_bytes:
            raise ValueError(f"Shape tensor {tensor_index} buffer is too small for its target")
        struct.pack_into("<I", patched, data_length_position, required_bytes)
        struct.pack_into(f"<{len(target_shape)}i", patched, data_position, *target_shape)
    return patched


def validate_output(
    payload: bytes | bytearray,
    analysis: dict[str, Any],
    expected_rewrite_count: int,
) -> dict[str, Any]:
    model = tflite.Model.GetRootAsModel(payload, 0)
    subgraph = model.Subgraphs(0)
    reshape_opcode = analysis["reshapeOpcodeIndex"]
    gather_count = 0
    reshape_count = 0
    for operator_index in range(subgraph.OperatorsLength()):
        code = builtin_code(model, int(subgraph.Operators(operator_index).OpcodeIndex()))
        gather_count += int(code == tflite.BuiltinOperator.GATHER_ND)
        reshape_count += int(code == tflite.BuiltinOperator.RESHAPE)
    if gather_count != 0:
        raise ValueError(f"Output still contains {gather_count} GATHER_ND operators")

    for item in analysis["rewrites"]:
        operator = subgraph.Operators(item["operatorIndex"])
        if int(operator.OpcodeIndex()) != reshape_opcode:
            raise ValueError(f"Operator {item['operatorIndex']} was not changed to RESHAPE")
        if operator.InputsLength() != 2 or operator.OutputsLength() != 1:
            raise ValueError(f"Rewritten operator {item['operatorIndex']} has an invalid ABI")
        tensor = subgraph.Tensors(item["indicesTensorIndex"])
        expected_shape = item["outputShape"]
        if tensor_shape(tensor) != [len(expected_shape)]:
            raise ValueError(f"Rewritten shape tensor {item['indicesTensorIndex']} has an invalid shape")
        raw = buffer_bytes(model, int(tensor.Buffer()))
        if list(struct.unpack(f"<{len(expected_shape)}i", raw)) != expected_shape:
            raise ValueError(f"Rewritten shape tensor {item['indicesTensorIndex']} has invalid data")

    original_reshape_count = sum(
        int(
            builtin_code(analysis["model"], int(analysis["subgraph"].Operators(index).OpcodeIndex()))
            == tflite.BuiltinOperator.RESHAPE
        )
        for index in range(analysis["subgraph"].OperatorsLength())
    )
    if reshape_count != original_reshape_count + expected_rewrite_count:
        raise ValueError("Output RESHAPE count does not match the exact rewrite delta")
    return {
        "operatorCount": int(subgraph.OperatorsLength()),
        "tensorCount": int(subgraph.TensorsLength()),
        "gatherNdCount": gather_count,
        "reshapeCount": reshape_count,
        "reshapeDelta": expected_rewrite_count,
    }


def atomic_write(path: Path, payload: bytes | bytearray, force: bool) -> None:
    path = path.resolve()
    if path.exists() and not force:
        raise FileExistsError(f"Output already exists: {path}; pass --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as output:
        temporary = Path(output.name)
        output.write(payload)
        output.flush()
    try:
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    source = args.input_model.resolve()
    destination = args.output_model.resolve()
    report_path = args.report.resolve() if args.report is not None else None
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == destination:
        raise ValueError("Input and output model paths must be different")
    if report_path is not None and report_path in {source, destination}:
        raise ValueError("Report path must differ from input and output model paths")
    targets = [destination] + ([report_path] if report_path is not None else [])
    existing_targets = [path for path in targets if path.exists()]
    if existing_targets and not args.force:
        joined = ", ".join(str(path) for path in existing_targets)
        raise FileExistsError(f"Output already exists: {joined}; pass --force to replace it")
    if args.expected_rewrite_count <= 0:
        raise ValueError("--expected-rewrite-count must be positive")
    expected_sha256 = args.expected_input_sha256.lower()
    if len(expected_sha256) != 64 or any(value not in "0123456789abcdef" for value in expected_sha256):
        raise ValueError("--expected-input-sha256 must be 64 lowercase hexadecimal characters")

    payload = source.read_bytes()
    input_sha256 = sha256_bytes(payload)
    if input_sha256 != expected_sha256:
        raise ValueError(f"Input SHA-256 mismatch: {input_sha256} != {expected_sha256}")
    analysis = analyze(payload, args.expected_rewrite_count)
    patched = apply_rewrite(payload, analysis)
    validation = validate_output(patched, analysis, args.expected_rewrite_count)
    atomic_write(destination, patched, args.force)

    shape_histogram = Counter(
        "x".join(str(value) for value in target)
        for target in analysis["shapeTensorTargets"].values()
    )
    report = {
        "schemaVersion": 1,
        "ruleId": RULE_ID,
        "input": {
            "path": str(source),
            "byteSize": len(payload),
            "sha256": input_sha256,
        },
        "output": {
            "path": str(destination),
            "byteSize": len(patched),
            "sha256": sha256_bytes(patched),
        },
        "proof": {
            "rewrittenOperatorCount": len(analysis["rewrites"]),
            "rewrittenShapeTensorCount": len(analysis["shapeTensorTargets"]),
            "shapeHistogram": dict(sorted(shape_histogram.items())),
            "operatorIndices": [item["operatorIndex"] for item in analysis["rewrites"]],
            "shapeTensorIndices": sorted(analysis["shapeTensorTargets"]),
            "preconditions": [
                "data tensor is a static one-dimensional flattening of the output",
                "INT32 indices shape equals output shape plus singleton index depth",
                "indices values are exactly 0..N-1",
                "indices tensor has no non-rewritten consumers",
                "indices buffer belongs to exactly one tensor and is stored inline",
                "input and output dtypes are identical",
            ],
            "expectedEquivalence": "bitwise",
            "flatBufferByteSizePreserved": len(payload) == len(patched),
        },
        "validation": validation,
    }
    if report_path is not None:
        report_payload = (json.dumps(report, indent=2) + "\n").encode("utf-8")
        atomic_write(report_path, report_payload, args.force)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
