#!/usr/bin/env python3
"""Rewrite numerically sensitive HTDemucs ONNX normalization reductions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_SOURCE_SHA256 = "48f8e84945579f8ab340e083339e9221e03785dbe733a52c388200b6d3ca779a"
FREQUENCY_CHANNELS = 4
FREQUENCY_BINS = 2048
FREQUENCY_FRAMES = 336
ELEMENT_COUNT = FREQUENCY_CHANNELS * FREQUENCY_BINS * FREQUENCY_FRAMES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--accumulator-dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Precision used only while summing squared deviations.",
    )
    parser.add_argument(
        "--transformer-group-norm-accumulator-dtype",
        choices=("unchanged", "float32", "float64"),
        default="unchanged",
        help=(
            "Replace transformer group=1 InstanceNormalization statistics "
            "with explicit reductions in the selected precision."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    manifest_path = args.manifest.resolve()
    if sha256_file(source) != EXPECTED_SOURCE_SHA256:
        raise ValueError("Source FP32 ONNX identity mismatch")
    if (output.exists() or manifest_path.exists()) and not args.force:
        raise FileExistsError("Output exists; pass --force")

    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(source), load_external_data=False)
    by_name = {node.name: node for node in model.graph.node}
    required = {
        "/ReduceMean": "frequency mean used for centering",
        "/ReduceMean_1": "duplicate frequency mean used for variance",
        "/Sub": "frequency centering used for variance",
        "/Mul_1": "frequency squared deviations",
        "/ReduceMean_2": "frequency population variance",
        "/Cast_2": "frequency element count",
        "/Mul_2": "frequency Bessel numerator",
        "/Sub_1": "frequency Bessel denominator",
        "/Div": "frequency unbiased variance",
        "/Sqrt": "frequency standard deviation",
        "/Add": "frequency normalization divisor",
        "/Sub_2": "frequency centered input",
        "/Div_1": "frequency normalized input",
    }
    missing = sorted(set(required) - set(by_name))
    if missing:
        raise ValueError(f"Missing expected nodes: {missing}")

    frequency_input = by_name["/ReduceMean"].input[0]
    original_divisor_output = by_name["/Add"].output[0]
    original_normalized_output = by_name["/Div_1"].output[0]
    epsilon_initializer = "mss_freq_norm_epsilon"
    correction_initializer = "mss_freq_norm_correction"
    axes_initializer = "mss_freq_norm_axes"
    stable_nodes = [
        helper.make_node(
            "ReduceMean",
            [frequency_input],
            ["/ReduceMean_output_0"],
            name="/mss_frequency_normalization/ReduceMean",
            axes=[1, 2, 3],
            keepdims=1,
        ),
        helper.make_node(
            "Sub",
            [frequency_input, "/ReduceMean_output_0"],
            ["mss_freq_centered"],
            name="/mss_frequency_normalization/Sub",
        ),
        helper.make_node(
            "Mul",
            ["mss_freq_centered", "mss_freq_centered"],
            ["mss_freq_squared_deviation"],
            name="/mss_frequency_normalization/Mul",
        ),
    ]
    if args.accumulator_dtype == "float32":
        stable_nodes.extend(
            [
                helper.make_node(
                    "ReduceSum",
                    ["mss_freq_squared_deviation", axes_initializer],
                    ["mss_freq_squared_deviation_sum"],
                    name="/mss_frequency_normalization/ReduceSum",
                    keepdims=1,
                ),
                helper.make_node(
                    "Div",
                    ["mss_freq_squared_deviation_sum", correction_initializer],
                    ["/Div_output_0"],
                    name="/mss_frequency_normalization/DivVariance",
                ),
                helper.make_node(
                    "Sqrt",
                    ["/Div_output_0"],
                    ["/Sqrt_output_0"],
                    name="/mss_frequency_normalization/Sqrt",
                ),
            ]
        )
        correction_dtype = np.float32
    else:
        stable_nodes.extend(
            [
                helper.make_node(
                    "Cast",
                    ["mss_freq_squared_deviation"],
                    ["mss_freq_squared_deviation_fp64"],
                    name="/mss_frequency_normalization/CastSquaredDeviationToDouble",
                    to=TensorProto.DOUBLE,
                ),
                helper.make_node(
                    "ReduceSum",
                    ["mss_freq_squared_deviation_fp64", axes_initializer],
                    ["mss_freq_squared_deviation_sum_fp64"],
                    name="/mss_frequency_normalization/ReduceSumDouble",
                    keepdims=1,
                ),
                helper.make_node(
                    "Div",
                    ["mss_freq_squared_deviation_sum_fp64", correction_initializer],
                    ["mss_freq_variance_fp64"],
                    name="/mss_frequency_normalization/DivVarianceDouble",
                ),
                helper.make_node(
                    "Cast",
                    ["mss_freq_variance_fp64"],
                    ["/Div_output_0"],
                    name="/mss_frequency_normalization/CastVarianceToFloat",
                    to=TensorProto.FLOAT,
                ),
                helper.make_node(
                    "Sqrt",
                    ["mss_freq_variance_fp64"],
                    ["mss_freq_std_fp64"],
                    name="/mss_frequency_normalization/SqrtDouble",
                ),
                helper.make_node(
                    "Cast",
                    ["mss_freq_std_fp64"],
                    ["/Sqrt_output_0"],
                    name="/mss_frequency_normalization/CastStdToFloat",
                    to=TensorProto.FLOAT,
                ),
            ]
        )
        correction_dtype = np.float64
    stable_nodes.extend(
        [
            helper.make_node(
                "Add",
                ["/Sqrt_output_0", epsilon_initializer],
                [original_divisor_output],
                name="/mss_frequency_normalization/AddEpsilon",
            ),
            helper.make_node(
                "Div",
                ["mss_freq_centered", original_divisor_output],
                [original_normalized_output],
                name="/mss_frequency_normalization/DivNormalize",
            ),
        ]
    )

    removed_names = {
        "/ReduceMean",
        "/ReduceMean_1",
        "/Sub",
        "/Mul_1",
        "/ReduceMean_2",
        "/Cast_2",
        "/Mul_2",
        "/Sub_1",
        "/Div",
        "/Sqrt",
        "/Add",
        "/Sub_2",
        "/Div_1",
    }
    original_nodes = list(model.graph.node)
    first_index = min(
        index for index, node in enumerate(original_nodes) if node.name in removed_names
    )
    retained = [node for node in original_nodes if node.name not in removed_names]
    insertion_index = sum(
        1
        for node in original_nodes[:first_index]
        if node.name not in removed_names
    )
    rewritten = retained[:insertion_index] + stable_nodes + retained[insertion_index:]

    group_norm_nodes = [
        node
        for node in rewritten
        if node.name.startswith("/crosstransformer/")
        and node.name.endswith("/norm_out/InstanceNormalization")
    ]
    if len(group_norm_nodes) != 10:
        raise ValueError(
            f"Expected ten transformer group-normalization nodes, got "
            f"{len(group_norm_nodes)}"
        )
    group_norm_replacements: dict[str, list[Any]] = {}
    if args.transformer_group_norm_accumulator_dtype != "unchanged":
        group_norm_epsilon_initializer = "mss_transformer_group_norm_epsilon"
        group_norm_epsilon_dtype = (
            np.float32
            if args.transformer_group_norm_accumulator_dtype == "float32"
            else np.float64
        )
        model.graph.initializer.append(
            numpy_helper.from_array(
                np.asarray(1e-5, dtype=group_norm_epsilon_dtype),
                name=group_norm_epsilon_initializer,
            )
        )
        for node in group_norm_nodes:
            source_name = node.input[0]
            output_name = node.output[0]
            prefix = node.name.removesuffix("/InstanceNormalization")
            internal_prefix = prefix + "/mss_explicit_statistics"
            statistics_input = source_name
            replacement: list[Any] = []
            if args.transformer_group_norm_accumulator_dtype == "float64":
                statistics_input = internal_prefix + "/input_fp64"
                replacement.append(
                    helper.make_node(
                        "Cast",
                        [source_name],
                        [statistics_input],
                        name=internal_prefix + "/CastInputToDouble",
                        to=TensorProto.DOUBLE,
                    )
                )
            mean_output = internal_prefix + "/mean"
            centered_output = internal_prefix + "/centered"
            squared_output = internal_prefix + "/squared_deviation"
            variance_output = internal_prefix + "/variance"
            variance_epsilon_output = internal_prefix + "/variance_plus_epsilon"
            standard_deviation_output = internal_prefix + "/standard_deviation"
            normalized_output = (
                output_name
                if args.transformer_group_norm_accumulator_dtype == "float32"
                else internal_prefix + "/normalized_fp64"
            )
            replacement.extend(
                [
                    helper.make_node(
                        "ReduceMean",
                        [statistics_input],
                        [mean_output],
                        name=internal_prefix + "/ReduceMean",
                        axes=[2],
                        keepdims=1,
                    ),
                    helper.make_node(
                        "Sub",
                        [statistics_input, mean_output],
                        [centered_output],
                        name=internal_prefix + "/Sub",
                    ),
                    helper.make_node(
                        "Mul",
                        [centered_output, centered_output],
                        [squared_output],
                        name=internal_prefix + "/Mul",
                    ),
                    helper.make_node(
                        "ReduceMean",
                        [squared_output],
                        [variance_output],
                        name=internal_prefix + "/ReduceMeanVariance",
                        axes=[2],
                        keepdims=1,
                    ),
                    helper.make_node(
                        "Add",
                        [variance_output, group_norm_epsilon_initializer],
                        [variance_epsilon_output],
                        name=internal_prefix + "/AddEpsilon",
                    ),
                    helper.make_node(
                        "Sqrt",
                        [variance_epsilon_output],
                        [standard_deviation_output],
                        name=internal_prefix + "/Sqrt",
                    ),
                    helper.make_node(
                        "Div",
                        [centered_output, standard_deviation_output],
                        [normalized_output],
                        name=internal_prefix + "/DivNormalize",
                    ),
                ]
            )
            if args.transformer_group_norm_accumulator_dtype == "float64":
                replacement.append(
                    helper.make_node(
                        "Cast",
                        [normalized_output],
                        [output_name],
                        name=internal_prefix + "/CastOutputToFloat",
                        to=TensorProto.FLOAT,
                    )
                )
            group_norm_replacements[node.name] = replacement
        rewritten = [
            replacement_node
            for node in rewritten
            for replacement_node in group_norm_replacements.get(node.name, [node])
        ]
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(
                np.asarray(1e-5, dtype=np.float32), name=epsilon_initializer
            ),
            numpy_helper.from_array(
                np.asarray(ELEMENT_COUNT - 1, dtype=correction_dtype),
                name=correction_initializer,
            ),
            numpy_helper.from_array(
                np.asarray([1, 2, 3], dtype=np.int64), name=axes_initializer
            ),
        ]
    )
    onnx.checker.check_model(model)

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    onnx.save(model, str(partial))
    partial.replace(output)
    manifest = {
        "schemaVersion": 1,
        "status": "complete",
        "tool": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "source": {
            "path": str(source),
            "byteSize": source.stat().st_size,
            "sha256": EXPECTED_SOURCE_SHA256,
        },
        "artifact": {
            "path": str(output),
            "byteSize": output.stat().st_size,
            "sha256": sha256_file(output),
        },
        "rewrite": {
            "scope": "frequency-branch global unbiased standard deviation",
            "removedNodes": sorted(removed_names),
            "addedNodes": [node.name for node in stable_nodes],
            "reductionAxes": [1, 2, 3],
            "reductionShape": [FREQUENCY_CHANNELS, FREQUENCY_BINS, FREQUENCY_FRAMES],
            "elementCount": ELEMENT_COUNT,
            "correctionDenominator": ELEMENT_COUNT - 1,
            "accumulatorDtype": args.accumulator_dtype,
            "elementwiseDtype": "float32",
            "formula": "sqrt(reduce_sum((x - reduce_mean(x))^2) / (N - 1))",
            "timeBranchChanged": False,
            "weightsChanged": False,
        },
        "transformerGroupNormalizationRewrite": {
            "scope": "ten transformer MyGroupNorm(num_groups=1) statistics",
            "accumulatorDtype": args.transformer_group_norm_accumulator_dtype,
            "elementwiseDtype": (
                "float32"
                if args.transformer_group_norm_accumulator_dtype != "float64"
                else "float64-statistics-cast-back-to-float32"
            ),
            "removedNodes": (
                [node.name for node in group_norm_nodes]
                if group_norm_replacements
                else []
            ),
            "addedNodes": [
                node.name
                for replacement in group_norm_replacements.values()
                for node in replacement
            ],
            "frequencyElementCount": 384 * 2688,
            "timeElementCount": 384 * 1344,
            "formula": "(x - reduce_mean(x)) / sqrt(reduce_mean((x - mean)^2) + epsilon)",
            "epsilon": 1e-5,
            "learnedAffineChanged": False,
            "weightsChanged": False,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(manifest_path, manifest)
    print(json.dumps({
        "status": "complete",
        "artifact": manifest["artifact"],
        "manifest": str(manifest_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
