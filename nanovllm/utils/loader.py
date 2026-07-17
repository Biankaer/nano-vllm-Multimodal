import os
from dataclasses import dataclass
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


@dataclass(frozen=True, slots=True)
class WeightLoadReport:
    loaded_parameters: tuple[str, ...]
    skipped_checkpoint_weights: tuple[str, ...]


def load_model(
    model: nn.Module,
    path: str,
    *,
    include_prefixes: tuple[str, ...] | None = None,
    strip_prefix: str = "",
    ignore_missing: tuple[str, ...] = (),
) -> WeightLoadReport:
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    parameters = dict(model.named_parameters())
    selected_checkpoint_weights: set[str] = set()
    loaded_parameters: set[str] = set()
    loaded_packed_shards: dict[str, set[str | int]] = {}
    skipped_checkpoint_weights: list[str] = []

    for file in sorted(glob(os.path.join(path, "*.safetensors"))):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                if include_prefixes is not None and not weight_name.startswith(include_prefixes):
                    skipped_checkpoint_weights.append(weight_name)
                    continue
                if weight_name in selected_checkpoint_weights:
                    raise ValueError(f"checkpoint weight loaded twice: {weight_name}")
                selected_checkpoint_weights.add(weight_name)

                param_name = weight_name
                if strip_prefix:
                    if not param_name.startswith(strip_prefix):
                        raise ValueError(
                            f"selected checkpoint weight does not start with {strip_prefix!r}: {weight_name}"
                        )
                    param_name = param_name[len(strip_prefix):]

                for source_name, (packed_name, shard_id) in packed_modules_mapping.items():
                    if source_name in param_name:
                        target_name = param_name.replace(source_name, packed_name)
                        param = parameters.get(target_name)
                        if param is None:
                            raise KeyError(f"unknown selected checkpoint weight: {weight_name}")
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        loaded_parameters.add(target_name)
                        shards = loaded_packed_shards.setdefault(target_name, set())
                        if shard_id in shards:
                            raise ValueError(
                                f"packed shard {shard_id!r} loaded twice for {target_name}"
                            )
                        shards.add(shard_id)
                        break
                else:
                    param = parameters.get(param_name)
                    if param is None:
                        raise KeyError(f"unknown selected checkpoint weight: {weight_name}")
                    if param_name in loaded_parameters:
                        raise ValueError(f"model parameter loaded twice: {param_name}")
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
                    loaded_parameters.add(param_name)

    expected_shards_by_packed_name: dict[str, set[str | int]] = {}
    for packed_name, shard_id in packed_modules_mapping.values():
        expected_shards_by_packed_name.setdefault(packed_name, set()).add(shard_id)
    for param_name, loaded_shards in loaded_packed_shards.items():
        packed_name = next(
            name for name in expected_shards_by_packed_name if name in param_name
        )
        expected_shards = expected_shards_by_packed_name[packed_name]
        if loaded_shards != expected_shards:
            missing_shards = sorted(expected_shards - loaded_shards, key=str)
            raise ValueError(f"packed parameter {param_name} is missing shards: {missing_shards}")

    missing = sorted(set(parameters) - loaded_parameters - set(ignore_missing))
    if missing:
        raise ValueError(f"model parameters were not loaded: {missing}")

    return WeightLoadReport(
        loaded_parameters=tuple(sorted(loaded_parameters)),
        skipped_checkpoint_weights=tuple(sorted(skipped_checkpoint_weights)),
    )
