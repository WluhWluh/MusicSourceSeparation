#!/usr/bin/env python3
"""Audit downloaded four-stem HTDemucs candidates without unsafe unpickling."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import pickletools
import platform
import zipfile
from typing import Any, Mapping


OFFICIAL_BASE = {
    "bytes": 84_025_440,
    "sha256": "d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd",
}

FT_MEMBERS = {
    "f7e0c4bc": {
        "target": "drums",
        "bytes": 84_025_440,
        "sha256": "2c85ab3c62dd6edd8e0b965e38b16fd1cdde357cc25de6b6bc9ce7c83f60925f",
    },
    "d12395a8": {
        "target": "bass",
        "bytes": 84_025_440,
        "sha256": "5b01a97567ae9a3178a6236fb520251045c03eb8834bc8c24a4eec11d6c8fb56",
    },
    "92cfc3b6": {
        "target": "other",
        "bytes": 84_025_440,
        "sha256": "a241863551f30d01c42bd7b97da40839922ead3acb0f1fcab25682f55b4eeb59",
    },
    "04573f0d": {
        "target": "vocals",
        "bytes": 84_025_440,
        "sha256": "68854b0d7c2b3274723b5761f6fd9f5aec5f1bcd3f0de7c1669546fdb7871b7c",
    },
}

OTHER_ARTIFACTS = {
    "psytrance": {
        "bytes": 174_266_467,
        "sha256": "d7cfcfaf41dc611dd14d35a206fe14a667bdaa6a1910f4c09a8d8ed43606cc78",
    },
    "kaniReference": {
        "bytes": 174_266_467,
        "sha256": "21cdabc8246f5052397647399e48292dc7394475e92335c65c19eb7e90bde6e0",
    },
    "raddhuhaA": {
        "bytes": 107_715_779,
        "sha256": "1bfc71644adba7d2065ec8ab1bb8d71d4e68d2fb15ed2c4d284e816b704dc698",
    },
    "raddhuhaB": {
        "bytes": 107_715_779,
        "sha256": "f9150cd79333a8e8539defbd74073793e9ac85d0851de090371cbf243e4e4028",
    },
    "waspaa": {
        "bytes": 1_635_710_219,
        "sha256": "30d165a32b0a2238210f818c06b4b9f9da0198f2c734a95075ee21077cd596b8",
    },
}


class SpecsDataModule:
    """Inert allowlisted stand-in for checkpoint metadata from sgmsvs."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_identity(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    identity = {
        "path": str(path.resolve()),
        "byteSize": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if identity["byteSize"] != expected["bytes"] or identity["sha256"] != expected["sha256"]:
        raise ValueError(f"Artifact identity mismatch: {identity}")
    return identity


def finite_snr(reference_energy: float, error_energy: float) -> float | None:
    if error_energy == 0.0:
        return None
    return 10.0 * math.log10(reference_energy / error_energy)


def compare_numpy_pairs(pairs: list[tuple[str, Any, Any]]) -> dict[str, Any]:
    import numpy as np

    changed_tensors: list[str] = []
    changed_values = 0
    compared_values = 0
    error_energy = 0.0
    reference_energy = 0.0
    maximum_error = 0.0
    for name, left, right in pairs:
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch for {name}: {left.shape} != {right.shape}")
        left64 = left.astype(np.float64, copy=False)
        right64 = right.astype(np.float64, copy=False)
        delta = left64 - right64
        compared_values += int(left64.size)
        nonzero = int(np.count_nonzero(delta))
        if nonzero:
            changed_tensors.append(name)
            changed_values += nonzero
            maximum_error = max(maximum_error, float(np.max(np.abs(delta))))
        flat_delta = delta.reshape(-1)
        flat_reference = left64.reshape(-1)
        error_energy += float(np.dot(flat_delta, flat_delta))
        reference_energy += float(np.dot(flat_reference, flat_reference))
    return {
        "bitwiseEqual": not changed_tensors,
        "changedTensorCount": len(changed_tensors),
        "changedValueCount": changed_values,
        "comparedValueCount": compared_values,
        "maximumAbsoluteError": maximum_error,
        "rootMeanSquareDelta": (
            math.sqrt(error_energy / compared_values) if compared_values else 0.0
        ),
        "referenceToDeltaSnrDb": finite_snr(reference_energy, error_energy),
        "changedTensorExamples": changed_tensors[:20],
    }


def safetensors_contract(path: Path) -> dict[str, tuple[tuple[int, ...], str]]:
    from safetensors import safe_open

    result: dict[str, tuple[tuple[int, ...], str]] = {}
    with safe_open(str(path), framework="np") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            result[key] = (tuple(int(value) for value in tensor.shape), str(tensor.dtype))
    return result


def compare_contracts(
    reference: Mapping[str, tuple[tuple[int, ...], str]],
    candidate: Mapping[str, tuple[tuple[int, ...], str]],
) -> dict[str, Any]:
    common = set(reference) & set(candidate)
    shape_mismatches = [
        key for key in sorted(common) if reference[key][0] != candidate[key][0]
    ]
    dtype_mismatches = [
        key for key in sorted(common) if reference[key][1] != candidate[key][1]
    ]
    return {
        "referenceTensorCount": len(reference),
        "candidateTensorCount": len(candidate),
        "referenceOnlyCount": len(set(reference) - set(candidate)),
        "candidateOnlyCount": len(set(candidate) - set(reference)),
        "shapeMismatchCount": len(shape_mismatches),
        "dtypeMismatchCount": len(dtype_mismatches),
        "referenceOnlyExamples": sorted(set(reference) - set(candidate))[:20],
        "candidateOnlyExamples": sorted(set(candidate) - set(reference))[:20],
        "shapeMismatchExamples": [
            {
                "key": key,
                "reference": list(reference[key][0]),
                "candidate": list(candidate[key][0]),
            }
            for key in shape_mismatches[:20]
        ],
        "dtypeMismatchExamples": dtype_mismatches[:20],
        "architectureCompatible": not (
            set(reference) - set(candidate)
            or set(candidate) - set(reference)
            or shape_mismatches
        ),
    }


def compare_safetensors(reference: Path, candidate: Path) -> dict[str, Any]:
    from safetensors import safe_open

    pairs = []
    with safe_open(str(reference), framework="np") as left_handle, safe_open(
        str(candidate), framework="np"
    ) as right_handle:
        left_keys = set(left_handle.keys())
        right_keys = set(right_handle.keys())
        if left_keys != right_keys:
            raise ValueError("Safetensors key mismatch")
        for key in sorted(left_keys):
            pairs.append((key, left_handle.get_tensor(key), right_handle.get_tensor(key)))
    return compare_numpy_pairs(pairs)


def official_ft_report(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from safetensors import safe_open

    base_path = repo_root / "models/demucs/official-hf/htdemucs/955717e8.safetensors"
    base_identity = checked_identity(base_path, OFFICIAL_BASE)
    base_contract = safetensors_contract(base_path)
    ft_root = (
        repo_root
        / "models/demucs/candidates/htdemucs-ft/478be8a68f85418addd6f7baefd4be76522a4034"
    )
    members: dict[str, Any] = {}
    for member_id, expected in FT_MEMBERS.items():
        weight_path = ft_root / f"{member_id}.safetensors"
        sidecar_path = ft_root / f"{member_id}.json"
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        target = expected["target"]
        source_index = ["drums", "bass", "other", "vocals"].index(target)
        training_weights = sidecar["training_args"]["weights"]
        if training_weights[source_index] != 1 or sum(training_weights) != 1:
            raise ValueError(f"Unexpected specialist weights for {member_id}")
        final_metrics = sidecar["metrics"][-1]
        with safe_open(str(weight_path), framework="np") as handle:
            metadata = handle.metadata()
        members[member_id] = {
            "targetStem": target,
            "identity": checked_identity(weight_path, expected),
            "architecture": compare_contracts(base_contract, safetensors_contract(weight_path)),
            "weightDeltaFromBase": compare_safetensors(base_path, weight_path),
            "safetensorsMetadata": {
                "klass": metadata.get("klass"),
                "args": metadata.get("args"),
                "kwargsSha256": hashlib.sha256(
                    metadata.get("kwargs", "").encode("utf-8")
                ).hexdigest(),
            },
            "training": {
                "continuedFrom": sidecar["training_args"].get("continue_from"),
                "lossWeights": training_weights,
                "epochs": sidecar["training_args"].get("epochs"),
            },
            "publishedSidecarMetrics": {
                "targetSdrDb": final_metrics.get(f"sdr_{target}"),
                "targetMedianSdrDb": final_metrics.get(f"sdr_med_{target}"),
                "targetNsdrDb": final_metrics.get(f"nsdr_{target}"),
                "targetMedianNsdrDb": final_metrics.get(f"nsdr_med_{target}"),
                "targetSirDb": final_metrics.get(f"sir_{target}"),
                "targetMedianSirDb": final_metrics.get(f"sir_med_{target}"),
            },
        }
    return {"identity": base_identity, "contract": base_contract}, members


def value_info(value: Any) -> dict[str, Any]:
    import onnx

    tensor_type = value.type.tensor_type
    shape = []
    for dimension in tensor_type.shape.dim:
        shape.append(int(dimension.dim_value) if dimension.dim_value else dimension.dim_param)
    return {
        "name": value.name,
        "dtype": onnx.TensorProto.DataType.Name(tensor_type.elem_type).lower(),
        "shape": shape,
    }


def onnx_report(path: Path, expected: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    import numpy as np
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    structure_digest = hashlib.sha256()
    for node in model.graph.node:
        structure_digest.update(node.SerializeToString())
    initializer_values = int(
        sum(np.prod(initializer.dims, dtype=np.int64) for initializer in model.graph.initializer)
    )
    return model, {
        "identity": checked_identity(path, expected),
        "irVersion": int(model.ir_version),
        "producer": {"name": model.producer_name, "version": model.producer_version},
        "opsets": [
            {"domain": item.domain, "version": int(item.version)}
            for item in model.opset_import
        ],
        "nodeCount": len(model.graph.node),
        "operatorHistogram": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
        "initializerCount": len(model.graph.initializer),
        "initializerValueCount": initializer_values,
        "initializerDtypes": dict(
            sorted(
                Counter(
                    onnx.TensorProto.DataType.Name(initializer.data_type).lower()
                    for initializer in model.graph.initializer
                ).items()
            )
        ),
        "inputs": [value_info(value) for value in model.graph.input],
        "outputs": [value_info(value) for value in model.graph.output],
        "nodeStructureSha256": structure_digest.hexdigest(),
    }


def compare_onnx_models(reference: Any, candidate: Any) -> dict[str, Any]:
    from onnx import numpy_helper

    reference_initializers = {item.name: item for item in reference.graph.initializer}
    candidate_initializers = {item.name: item for item in candidate.graph.initializer}
    common = set(reference_initializers) & set(candidate_initializers)
    pairs = [
        (
            name,
            numpy_helper.to_array(reference_initializers[name]),
            numpy_helper.to_array(candidate_initializers[name]),
        )
        for name in sorted(common)
    ]
    result = compare_numpy_pairs(pairs)
    result.update(
        {
            "referenceOnlyInitializers": sorted(set(reference_initializers) - set(candidate_initializers)),
            "candidateOnlyInitializers": sorted(set(candidate_initializers) - set(reference_initializers)),
            "nodeStructureBitwiseEqual": (
                len(reference.graph.node) == len(candidate.graph.node)
                and all(
                    left.SerializeToString() == right.SerializeToString()
                    for left, right in zip(reference.graph.node, candidate.graph.node)
                )
            ),
        }
    )
    return result


def identify_onnx_ft_member(model: Any, ft_root: Path) -> dict[str, Any]:
    from onnx import numpy_helper
    from safetensors import safe_open

    initializers = {item.name: item for item in model.graph.initializer}
    comparisons: dict[str, Any] = {}
    for member_id in FT_MEMBERS:
        path = ft_root / f"{member_id}.safetensors"
        with safe_open(str(path), framework="np") as handle:
            common = sorted(set(initializers) & set(handle.keys()))
            pairs = [
                (
                    name,
                    numpy_helper.to_array(initializers[name]),
                    handle.get_tensor(name),
                )
                for name in common
            ]
        comparisons[member_id] = {
            "commonNamedTensorCount": len(common),
            **compare_numpy_pairs(pairs),
        }
    best = min(
        comparisons,
        key=lambda key: (
            comparisons[key]["changedValueCount"],
            comparisons[key]["rootMeanSquareDelta"],
        ),
    )
    return {"bestMatch": best, "comparisons": comparisons}


def pickle_globals(path: Path) -> list[str]:
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Not a PyTorch ZIP checkpoint: {path}")
    with zipfile.ZipFile(path) as archive:
        pickle_info = next(
            item
            for item in archive.infolist()
            if item.filename == "data.pkl" or item.filename.endswith("/data.pkl")
        )
        payload = archive.read(pickle_info)
    return sorted(
        {
            str(argument)
            for opcode, argument, _ in pickletools.genops(payload)
            if opcode.name == "GLOBAL"
        }
    )


def state_dict_from_payload(payload: Any) -> Mapping[str, Any]:
    import torch

    if isinstance(payload, Mapping) and payload and all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in payload.items()
    ):
        return payload
    if isinstance(payload, Mapping):
        state = payload.get("state_dict")
        if isinstance(state, Mapping) and state and all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in state.items()
        ):
            return state
    raise ValueError("No tensor-only state dictionary found")


def state_contract(state: Mapping[str, Any], prefix: str = "") -> dict[str, tuple[tuple[int, ...], str]]:
    result = {}
    for key, value in state.items():
        normalized_key = key.removeprefix(prefix)
        result[normalized_key] = (
            tuple(int(item) for item in value.shape),
            str(value.dtype).removeprefix("torch."),
        )
    return result


def infer_stem_count(contract: Mapping[str, tuple[tuple[int, ...], str]]) -> dict[str, Any]:
    frequency = [
        shape[0]
        for key, (shape, _) in contract.items()
        if key == "decoder.3.conv_tr.bias" and len(shape) == 1
    ]
    waveform = [
        shape[0]
        for key, (shape, _) in contract.items()
        if key == "tdecoder.3.conv_tr.bias" and len(shape) == 1
    ]
    frequency_stems = frequency[0] // 4 if len(frequency) == 1 else None
    waveform_stems = waveform[0] // 2 if len(waveform) == 1 else None
    return {
        "frequencyOutputChannels": frequency[0] if len(frequency) == 1 else None,
        "waveformOutputChannels": waveform[0] if len(waveform) == 1 else None,
        "inferredStemCount": (
            frequency_stems if frequency_stems == waveform_stems else None
        ),
    }


def summarize_state(state: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    contract = state_contract(state, prefix)
    dtype_counts = Counter(dtype for _, dtype in contract.values())
    return {
        "tensorCount": len(state),
        "valueCount": int(sum(value.numel() for value in state.values())),
        "dtypeTensorCounts": dict(sorted(dtype_counts.items())),
        "keyExamples": list(sorted(contract))[:20],
        "outputInference": infer_stem_count(contract),
        "contract": contract,
    }


def load_unknown_checkpoint(path: Path, *, waspaa: bool) -> tuple[Any, Mapping[str, Any]]:
    import torch

    if waspaa:
        from torch._subclasses.fake_tensor import FakeTensorMode

        with torch.serialization.safe_globals(
            [(SpecsDataModule, "sgmsvs.data_module.SpecsDataModule")]
        ):
            with FakeTensorMode():
                payload = torch.load(
                    path,
                    map_location="cpu",
                    weights_only=True,
                    mmap=True,
                )
    else:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    return payload, state_dict_from_payload(payload)


def compare_torch_states(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    if set(left) != set(right):
        raise ValueError("Torch state key mismatch")
    changed_tensors = []
    changed_values = 0
    compared_values = 0
    error_energy = 0.0
    reference_energy = 0.0
    maximum_error = 0.0
    for key in sorted(left):
        left_tensor = left[key].detach().cpu()
        right_tensor = right[key].detach().cpu()
        if left_tensor.shape != right_tensor.shape:
            raise ValueError(f"Torch state shape mismatch for {key}")
        delta = left_tensor.double() - right_tensor.double()
        compared_values += int(delta.numel())
        nonzero = int(torch.count_nonzero(delta).item())
        if nonzero:
            changed_tensors.append(key)
            changed_values += nonzero
            maximum_error = max(maximum_error, float(delta.abs().max().item()))
        error_energy += float(delta.square().sum().item())
        reference_energy += float(left_tensor.double().square().sum().item())
    return {
        "bitwiseEqual": not changed_tensors,
        "changedTensorCount": len(changed_tensors),
        "changedValueCount": changed_values,
        "comparedValueCount": compared_values,
        "maximumAbsoluteError": maximum_error,
        "rootMeanSquareDelta": math.sqrt(error_energy / compared_values),
        "referenceToDeltaSnrDb": finite_snr(reference_energy, error_energy),
        "changedTensorExamples": changed_tensors[:20],
    }


def main() -> int:
    import numpy
    import onnx
    import safetensors
    import torch

    args = parse_args()
    repo_root = args.repo_root.resolve()
    output = args.output.resolve()

    base, ft_members = official_ft_report(repo_root)
    base_contract = base.pop("contract")
    ft_root = (
        repo_root
        / "models/demucs/candidates/htdemucs-ft/478be8a68f85418addd6f7baefd4be76522a4034"
    )

    psy_path = (
        repo_root
        / "models/demucs/candidates/htdemucs-psy-ft/e725e7eb9204188de4731658e9923dcf049273c4/psytrance_ft.onnx"
    )
    kani_path = (
        repo_root
        / "models/demucs/conversion/htdemucs-ft-ort/12cc9a49c3b5f3badc1b0821ccc26f1a32c1779a/htdemucs_ft.onnx"
    )
    kani_model, kani_report = onnx_report(kani_path, OTHER_ARTIFACTS["kaniReference"])
    psy_model, psy_report = onnx_report(psy_path, OTHER_ARTIFACTS["psytrance"])
    onnx_comparison = compare_onnx_models(kani_model, psy_model)
    kani_identification = identify_onnx_ft_member(kani_model, ft_root)

    raddhuha_root = (
        repo_root
        / "models/demucs/candidates/htdemucs-raddhuha-ft/ebb50b725c6b2a3b408b75e5430ff9a056ebeb2c"
    )
    raddhuha_a_path = raddhuha_root / "htdemucs_finetuned.pt"
    raddhuha_b_path = raddhuha_root / "htdemucs_finetuned (1).pt"
    raddhuha_a_payload, raddhuha_a_state = load_unknown_checkpoint(
        raddhuha_a_path, waspaa=False
    )
    raddhuha_b_payload, raddhuha_b_state = load_unknown_checkpoint(
        raddhuha_b_path, waspaa=False
    )
    del raddhuha_a_payload, raddhuha_b_payload
    raddhuha_a_summary = summarize_state(raddhuha_a_state)
    raddhuha_b_summary = summarize_state(raddhuha_b_state)

    waspaa_path = (
        repo_root
        / "models/demucs/candidates/htdemucs-waspaa2025/6a6d4df0e334c263ff9d820005db808239e70974/htdemucs_epoch=570-sdr=6.38.ckpt"
    )
    waspaa_payload, waspaa_state = load_unknown_checkpoint(waspaa_path, waspaa=True)
    waspaa_summary = summarize_state(waspaa_state, prefix="dnn.")
    hyper_parameters = waspaa_payload.get("hyper_parameters", {})

    report = {
        "schemaVersion": 1,
        "scope": "download and static host audit for four-stem HTDemucs candidates",
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "safetensors": safetensors.__version__,
            "onnx": onnx.__version__,
            "numpy": numpy.__version__,
        },
        "tool": {
            "path": str(Path(__file__).resolve()),
            "byteSize": Path(__file__).stat().st_size,
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "officialBase": base,
        "officialFineTunedBag": {
            "repository": "adefossez/HTDemucs-ft",
            "revision": "478be8a68f85418addd6f7baefd4be76522a4034",
            "license": "MIT",
            "stemOrder": ["drums", "bass", "other", "vocals"],
            "members": ft_members,
        },
        "psytranceOnnx": {
            "repository": "Kani95/htdemucs-psy-ft",
            "revision": "e725e7eb9204188de4731658e9923dcf049273c4",
            "publisherLicenseDeclaration": "Apache-2.0",
            "model": psy_report,
            "conversionReference": kani_report,
            "referenceToOfficialFtIdentification": kani_identification,
            "candidateVsReference": onnx_comparison,
            "contractAssessment": {
                "fourStemOutput": True,
                "neuralCoreBoundary": True,
                "manifestFixedWindowSamples": 343_980,
                "graphWaveformAxisDynamic": True,
                "rawTorchWeightsPublished": False,
                "trainingEvidencePublished": False,
                "qualityMetricsPublished": False,
            },
        },
        "raddhuha": {
            "repository": "raddhuha/HT-Demucs",
            "revision": "ebb50b725c6b2a3b408b75e5430ff9a056ebeb2c",
            "licenseFileBytes": 0,
            "readmeHasNoModelDetails": True,
            "artifacts": {
                "a": {
                    "identity": checked_identity(raddhuha_a_path, OTHER_ARTIFACTS["raddhuhaA"]),
                    "pickleGlobals": pickle_globals(raddhuha_a_path),
                    "state": {key: value for key, value in raddhuha_a_summary.items() if key != "contract"},
                    "vsOfficialFourStemContract": compare_contracts(
                        base_contract, raddhuha_a_summary["contract"]
                    ),
                },
                "b": {
                    "identity": checked_identity(raddhuha_b_path, OTHER_ARTIFACTS["raddhuhaB"]),
                    "pickleGlobals": pickle_globals(raddhuha_b_path),
                    "state": {key: value for key, value in raddhuha_b_summary.items() if key != "contract"},
                    "vsOfficialFourStemContract": compare_contracts(
                        base_contract, raddhuha_b_summary["contract"]
                    ),
                },
            },
            "artifactDelta": compare_torch_states(raddhuha_a_state, raddhuha_b_state),
            "contractAssessment": {
                "inferredStemCount": raddhuha_a_summary["outputInference"]["inferredStemCount"],
                "fourStemCandidate": False,
            },
        },
        "waspaa2025": {
            "repository": "pablebe/htdemucs",
            "revision": "6a6d4df0e334c263ff9d820005db808239e70974",
            "license": "MIT",
            "identity": checked_identity(waspaa_path, OTHER_ARTIFACTS["waspaa"]),
            "pickleGlobals": pickle_globals(waspaa_path),
            "loader": "torch.load(weights_only=True, mmap=True) under FakeTensorMode with an inert allowlisted metadata stand-in",
            "checkpoint": {
                "epoch": waspaa_payload.get("epoch"),
                "globalStep": waspaa_payload.get("global_step"),
                "pytorchLightningVersion": waspaa_payload.get("pytorch-lightning_version"),
                "topLevelKeys": sorted(waspaa_payload),
            },
            "selectedHyperParameters": {
                key: hyper_parameters.get(key)
                for key in (
                    "backbone",
                    "sources",
                    "target_str",
                    "sr",
                    "channels",
                    "depth",
                    "bottom_channels",
                    "t_hidden_scale",
                )
            },
            "state": {key: value for key, value in waspaa_summary.items() if key != "contract"},
            "vsOfficialFourStemContract": compare_contracts(
                base_contract, waspaa_summary["contract"]
            ),
            "contractAssessment": {
                "inferredStemCount": waspaa_summary["outputInference"]["inferredStemCount"],
                "sampleRate": hyper_parameters.get("sr"),
                "fourStemCandidate": False,
            },
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    partial.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(output)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output),
                "officialFtMembers": list(ft_members),
                "psytranceChangedValues": onnx_comparison["changedValueCount"],
                "raddhuhaStemCount": report["raddhuha"]["contractAssessment"]["inferredStemCount"],
                "waspaaStemCount": report["waspaa2025"]["contractAssessment"]["inferredStemCount"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
