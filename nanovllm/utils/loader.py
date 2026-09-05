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
    checkpoint_parameter_aliases = getattr(
        model,
        "checkpoint_parameter_aliases",
        {},
    )
    if not isinstance(checkpoint_parameter_aliases, dict):
        raise TypeError("checkpoint_parameter_aliases must be a dictionary")
    parameters = dict(model.named_parameters())
    packed_source_names = tuple(packed_modules_mapping)
    packed_target_names = tuple(
        packed_name for packed_name, _ in packed_modules_mapping.values()
    )
    for alias_name, target_name in checkpoint_parameter_aliases.items():
        if not isinstance(alias_name, str) or not isinstance(target_name, str):
            raise TypeError("checkpoint parameter aliases must map strings to strings")
        if alias_name == target_name:
            raise ValueError(f"checkpoint parameter alias cannot target itself: {alias_name}")
        if target_name not in parameters:
            raise ValueError(
                f"checkpoint parameter alias target does not exist: {target_name}"
            )
        if alias_name in parameters:
            raise ValueError(
                f"checkpoint parameter alias shadows a model parameter: {alias_name}"
            )
        if any(source_name in alias_name for source_name in packed_source_names):
            raise ValueError(
                f"checkpoint parameter alias cannot reference a packed source: {alias_name}"
            )
        if any(target in target_name for target in packed_target_names):
            raise ValueError(
                f"checkpoint parameter alias cannot target a packed parameter: {target_name}"
            )
    alias_target_names = set(checkpoint_parameter_aliases.values())
    selected_checkpoint_weights: set[str] = set()
    loaded_parameters: set[str] = set()
    loaded_packed_shards: dict[str, set[str | int]] = {}
    skipped_checkpoint_weights: list[str] = []
    loaded_alias_values: dict[str, tuple[str, torch.Tensor]] = {}

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
                    target_name = checkpoint_parameter_aliases.get(
                        param_name,
                        param_name,
                    )
                    param = parameters.get(target_name)
                    if param is None:
                        raise KeyError(f"unknown selected checkpoint weight: {weight_name}")
                    loaded_weight = f.get_tensor(weight_name)
                    previous = loaded_alias_values.get(target_name)
                    if previous is not None:
                        previous_name, previous_weight = previous
                        if (
                            previous_weight.shape != loaded_weight.shape
                            or previous_weight.dtype != loaded_weight.dtype
                            or not torch.equal(previous_weight, loaded_weight)
                        ):
                            raise ValueError(
                                "checkpoint parameter alias values differ: "
                                f"{previous_name} and {weight_name}"
                            )
                        loaded_parameters.add(param_name)
                        continue
                    if target_name in loaded_parameters:
                        raise ValueError(f"model parameter loaded twice: {target_name}")
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_parameters.add(target_name)
                    if param_name != target_name:
                        loaded_parameters.add(param_name)
                    if target_name in alias_target_names:
                        loaded_alias_values[target_name] = (weight_name, loaded_weight)

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
