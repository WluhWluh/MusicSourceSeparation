from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import tflite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and optionally validate a TFLite FlatBuffer without TensorFlow.",
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--expected-byte-size", type=int)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def decode(value: bytes | None) -> str:
    return value.decode("utf-8") if value else ""


def enum_names(enum: type[Any]) -> dict[int, str]:
    return {
        value: name
        for name, value in vars(enum).items()
        if isinstance(value, int) and not name.startswith("_")
    }


TENSOR_TYPES = enum_names(tflite.TensorType)
BUILTIN_OPERATORS = enum_names(tflite.BuiltinOperator)
BUILTIN_OPERATORS.update({
    162: "STABLEHLO_LOGISTIC",
    163: "STABLEHLO_ADD",
    164: "STABLEHLO_DIVIDE",
    165: "STABLEHLO_MULTIPLY",
    166: "STABLEHLO_MAXIMUM",
    167: "STABLEHLO_RESHAPE",
    168: "STABLEHLO_CLAMP",
    169: "STABLEHLO_CONCATENATE",
    170: "STABLEHLO_BROADCAST_IN_DIM",
    171: "STABLEHLO_CONVOLUTION",
    172: "STABLEHLO_SLICE",
    173: "STABLEHLO_CUSTOM_CALL",
    174: "STABLEHLO_REDUCE",
    175: "STABLEHLO_ABS",
    176: "STABLEHLO_AND",
    177: "STABLEHLO_COSINE",
    178: "STABLEHLO_EXPONENTIAL",
    179: "STABLEHLO_FLOOR",
    180: "STABLEHLO_LOG",
    181: "STABLEHLO_MINIMUM",
    182: "STABLEHLO_NEGATE",
    183: "STABLEHLO_OR",
    184: "STABLEHLO_POWER",
    185: "STABLEHLO_REMAINDER",
    186: "STABLEHLO_RSQRT",
    187: "STABLEHLO_SELECT",
    188: "STABLEHLO_SUBTRACT",
    189: "STABLEHLO_TANH",
    190: "STABLEHLO_SCATTER",
    191: "STABLEHLO_COMPARE",
    192: "STABLEHLO_CONVERT",
    193: "STABLEHLO_DYNAMIC_SLICE",
    194: "STABLEHLO_DYNAMIC_UPDATE_SLICE",
    195: "STABLEHLO_PAD",
    196: "STABLEHLO_IOTA",
    197: "STABLEHLO_DOT_GENERAL",
    198: "STABLEHLO_REDUCE_WINDOW",
    199: "STABLEHLO_SORT",
    200: "STABLEHLO_WHILE",
    201: "STABLEHLO_GATHER",
    202: "STABLEHLO_TRANSPOSE",
    203: "DILATE",
    204: "STABLEHLO_RNG_BIT_GENERATOR",
    205: "REDUCE_WINDOW",
    206: "STABLEHLO_COMPOSITE",
    207: "STABLEHLO_SHIFT_LEFT",
    208: "STABLEHLO_CBRT",
    209: "STABLEHLO_CASE",
})


def tensor_report(subgraph: Any, tensor_index: int, io_index: int) -> dict[str, Any]:
    tensor = subgraph.Tensors(tensor_index)
    shape = [int(tensor.Shape(index)) for index in range(tensor.ShapeLength())]
    shape_signature = [
        int(tensor.ShapeSignature(index))
        for index in range(tensor.ShapeSignatureLength())
    ]
    quantization = tensor.Quantization()
    scale_count = quantization.ScaleLength() if quantization is not None else 0
    zero_point_count = quantization.ZeroPointLength() if quantization is not None else 0
    return {
        "index": io_index,
        "tensorIndex": int(tensor_index),
        "name": decode(tensor.Name()),
        "dtype": TENSOR_TYPES.get(tensor.Type(), f"UNKNOWN_{tensor.Type()}").lower(),
        "shape": shape,
        "shapeSignature": shape_signature,
        "dynamic": any(value < 0 for value in (shape_signature or shape)),
        "bufferIndex": int(tensor.Buffer()),
        "variable": bool(tensor.IsVariable()),
        "quantization": {
            "scaleCount": int(scale_count),
            "zeroPointCount": int(zero_point_count),
        },
    }


def operator_name(model: Any, opcode_index: int) -> tuple[str, str | None]:
    code = model.OperatorCodes(opcode_index)
    builtin = int(code.BuiltinCode())
    name = BUILTIN_OPERATORS.get(builtin, f"UNKNOWN_{builtin}")
    custom = decode(code.CustomCode()) if name == "CUSTOM" else None
    return name, custom


def metadata_report(model: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for index in range(model.MetadataLength()):
        metadata = model.Metadata(index)
        name = decode(metadata.Name())
        buffer = model.Buffers(metadata.Buffer())
        value = bytes(buffer.Data(item) for item in range(buffer.DataLength()))
        result[name] = value.rstrip(b"\x00").decode("utf-8", errors="replace")
    return result


def inspect_model(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    model = tflite.Model.GetRootAsModel(payload, 0)
    subgraphs: list[dict[str, Any]] = []
    custom_operator_count = 0
    total_operators = 0
    total_tensors = 0
    for subgraph_index in range(model.SubgraphsLength()):
        subgraph = model.Subgraphs(subgraph_index)
        histogram: Counter[str] = Counter()
        custom_histogram: Counter[str] = Counter()
        for operator_index in range(subgraph.OperatorsLength()):
            operator = subgraph.Operators(operator_index)
            name, custom = operator_name(model, operator.OpcodeIndex())
            histogram[name] += 1
            if custom is not None:
                custom_operator_count += 1
                custom_histogram[custom] += 1
        total_operators += subgraph.OperatorsLength()
        total_tensors += subgraph.TensorsLength()
        subgraphs.append(
            {
                "index": subgraph_index,
                "name": decode(subgraph.Name()),
                "operatorCount": int(subgraph.OperatorsLength()),
                "tensorCount": int(subgraph.TensorsLength()),
                "inputs": [
                    tensor_report(subgraph, int(subgraph.Inputs(index)), index)
                    for index in range(subgraph.InputsLength())
                ],
                "outputs": [
                    tensor_report(subgraph, int(subgraph.Outputs(index)), index)
                    for index in range(subgraph.OutputsLength())
                ],
                "operatorHistogram": dict(histogram.most_common()),
                "customOperatorHistogram": dict(custom_histogram.most_common()),
            },
        )
    signatures = []
    for index in range(model.SignatureDefsLength()):
        signature = model.SignatureDefs(index)
        signatures.append(
            {
                "key": decode(signature.SignatureKey()),
                "subgraphIndex": int(signature.SubgraphIndex()),
                "inputs": [
                    {
                        "name": decode(signature.Inputs(item).Name()),
                        "tensorIndex": int(signature.Inputs(item).TensorIndex()),
                    }
                    for item in range(signature.InputsLength())
                ],
                "outputs": [
                    {
                        "name": decode(signature.Outputs(item).Name()),
                        "tensorIndex": int(signature.Outputs(item).TensorIndex()),
                    }
                    for item in range(signature.OutputsLength())
                ],
            },
        )
    return {
        "artifact": {
            "path": str(path),
            "byteSize": len(payload),
            "sha256": sha256(path),
            "fileIdentifier": payload[4:8].decode("ascii", errors="replace"),
        },
        "schemaVersion": int(model.Version()),
        "description": decode(model.Description()),
        "subgraphCount": int(model.SubgraphsLength()),
        "operatorCodeCount": int(model.OperatorCodesLength()),
        "bufferCount": int(model.BuffersLength()),
        "operatorCount": int(total_operators),
        "tensorCount": int(total_tensors),
        "customOperatorCount": custom_operator_count,
        "subgraphs": subgraphs,
        "signatures": signatures,
        "metadata": metadata_report(model),
    }


def comparable_tensor(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in ("index", "tensorIndex", "name", "dtype", "shape")
        if key in value
    }


def validate_contract(report: dict[str, Any], contract: dict[str, Any]) -> None:
    if contract.get("contractSchemaVersion") != 4:
        raise ValueError("Expected contractSchemaVersion 4")
    if contract.get("contractKind") != "external-litert-runtime-candidate":
        raise ValueError("Expected external-litert-runtime-candidate contract")
    artifact = contract["artifact"]
    for key in ("byteSize", "sha256"):
        if report["artifact"][key] != artifact[key]:
            raise ValueError(f"Artifact {key} does not match the contract")
    expected = contract["flatBuffer"]
    if report["artifact"]["fileIdentifier"] != expected["fileIdentifier"]:
        raise ValueError("FlatBuffer fileIdentifier does not match the contract")
    if report["schemaVersion"] != expected["schemaVersion"]:
        raise ValueError("FlatBuffer schemaVersion does not match the contract")
    for key in (
        "subgraphCount",
        "operatorCount",
        "tensorCount",
        "customOperatorCount",
    ):
        if report[key] != expected[key]:
            raise ValueError(f"FlatBuffer {key} does not match the contract")
    if report["subgraphCount"] != 1:
        raise ValueError("Contract validation currently requires one subgraph")
    actual_graph = report["subgraphs"][0]
    for direction in ("inputs", "outputs"):
        actual = [comparable_tensor(item) for item in actual_graph[direction]]
        declared = [comparable_tensor(item) for item in expected[direction]]
        if actual != declared:
            raise ValueError(f"FlatBuffer {direction} do not match the contract")
    if report["signatures"] != expected["signatures"]:
        raise ValueError("FlatBuffer signatures do not match the contract")
    io = actual_graph["inputs"] + actual_graph["outputs"]
    if any(item["dynamic"] for item in io):
        raise ValueError("FlatBuffer contract requires static I/O")
    if any(item["dtype"] != "float32" for item in io):
        raise ValueError("FlatBuffer contract requires float32 I/O")
    if any(
        item["quantization"]["scaleCount"] or item["quantization"]["zeroPointCount"]
        for item in io
    ):
        raise ValueError("FlatBuffer contract requires unquantized I/O")


def main() -> int:
    args = parse_args()
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    report = inspect_model(model_path)
    if args.expected_byte_size is not None:
        if report["artifact"]["byteSize"] != args.expected_byte_size:
            raise ValueError("Model byte size does not match --expected-byte-size")
    if args.expected_sha256 is not None:
        expected_sha256 = args.expected_sha256.lower()
        if report["artifact"]["sha256"] != expected_sha256:
            raise ValueError("Model SHA-256 does not match --expected-sha256")
    if args.contract is not None:
        contract = json.loads(args.contract.resolve().read_text(encoding="utf-8"))
        validate_contract(report, contract)
        report["contract"] = {
            "path": str(args.contract.resolve()),
            "sha256": sha256(args.contract.resolve()),
            "status": "verified",
        }
    payload = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="ascii")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
