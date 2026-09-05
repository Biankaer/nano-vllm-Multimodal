from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TypeVar

import torch
import torch.distributed as dist
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nanovllm import LLM, SamplingParams
from nanovllm.engine.model_runner import (
    merge_planned_visual_embeddings,
    plan_visual_replacements,
)
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen25_vl import Qwen25VLForCausalLM
from nanovllm.multimodal.qwen25_vl_processor import Qwen25VLPromptProcessor
from nanovllm.multimodal.runtime import VisionFeatureBundle
from nanovllm.multimodal.vision_provider import VisionFeatureProvider
from nanovllm.utils.context import reset_context, set_context
from nanovllm.utils.loader import load_model


_T = TypeVar("_T")


def run_verification_stage(operation: Callable[[], _T]) -> _T:
    try:
        return operation()
    finally:
        # The operation frame must be gone before empty_cache so model/tensor
        # references have actually been released.  This matters when --mode
        # all starts a second engine in the same Python process.
        gc.collect()
        torch.cuda.empty_cache()


def qwen25_final_embedding(bundle: VisionFeatureBundle) -> torch.Tensor:
    if not isinstance(bundle, VisionFeatureBundle):
        raise TypeError("Qwen2.5-VL vision provider must return a feature bundle")
    if bundle.deepstack_embeddings:
        raise ValueError("Qwen2.5-VL feature bundle must not contain DeepStack embeddings")
    return bundle.final_embedding


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the native Qwen2.5-VL runtime on one GPU")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", default="assets/logo.png")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("alignment", "cache", "all"), default="all")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    return parser.parse_args(argv)


def make_messages(*, system: bool = False, long_prompt: bool = False) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    if system:
        messages.append({"role": "system", "content": "Be concise."})
    text = "Describe this image briefly."
    if long_prompt:
        text = " ".join(["Describe the image carefully."] * 70)
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        }
    )
    return messages


def verify_alignment(model_path: str, image: Image.Image, device: str) -> dict[str, object]:
    dtype = torch.bfloat16
    rank = torch.device(device).index or 0
    handle, rendezvous_path = tempfile.mkstemp(prefix="nanovllm-dist-")
    os.close(handle)
    os.unlink(rendezvous_path)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous_path}",
        world_size=1,
        rank=0,
    )
    torch.cuda.set_device(rank)

    processor = AutoProcessor.from_pretrained(model_path)
    messages = make_messages()
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    hf_inputs = processor(
        text=[prompt],
        images=[image],
        videos=None,
        padding=False,
        return_tensors="pt",
    ).to(device)
    hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device).eval()
    with torch.inference_mode():
        hf_output = hf_model(**hf_inputs, use_cache=False)
        hf_logits = hf_output.logits[0, -1].float()
        hf_features = hf_model.visual(
            hf_inputs.pixel_values.to(dtype),
            grid_thw=hf_inputs.image_grid_thw,
        )

    prompt_processor = Qwen25VLPromptProcessor.from_pretrained(model_path)
    materialized = prompt_processor.materialize(messages, [image], ["auto"])
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    try:
        native_model = Qwen25VLForCausalLM(hf_model.config)
        load_model(native_model, model_path, include_prefixes=("model.",))
        provider = VisionFeatureProvider.from_pretrained(
            model_path,
            byte_budget=64 * 1024**2,
            dtype=dtype,
            device=device,
            attn_implementation="sdpa",
        )
        provider.register_inputs(materialized.vision_inputs)
        sequence = Sequence(
            list(materialized.token_ids),
            multimodal_metadata=materialized.metadata,
        )
        sequence.num_scheduled_tokens = len(sequence)
        input_ids = torch.tensor(materialized.token_ids, dtype=torch.int64, device=device)
        positions = torch.tensor(materialized.metadata.position_ids, dtype=torch.int64, device=device)
        plan = plan_visual_replacements([sequence])
        with torch.inference_mode(), provider.resolve_pinned(plan.required_feature_keys) as features:
            native_features = qwen25_final_embedding(
                features[plan.required_feature_keys[0]]
            )
            inputs_embeds = merge_planned_visual_embeddings(
                native_model.embed(input_ids),
                plan,
                features,
            )
            token_count = len(sequence)
            cu_seqlens = torch.tensor([0, token_count], dtype=torch.int32, device=device)
            set_context(
                True,
                cu_seqlens,
                cu_seqlens,
                token_count,
                token_count,
                torch.empty(0, dtype=torch.int32, device=device),
            )
            native_logits = native_model.compute_logits(
                native_model(None, positions, inputs_embeds=inputs_embeds)
            )[0].float()
            reset_context()

        visual_difference = (native_features.float() - hf_features.float()).abs()
        logits_difference = (native_logits - hf_logits).abs()
        logits_cosine = torch.nn.functional.cosine_similarity(
            native_logits,
            hf_logits,
            dim=0,
        ).item()
        native_top10 = torch.topk(native_logits, 10).indices.tolist()
        hf_top10 = torch.topk(hf_logits, 10).indices.tolist()
        top10_overlap = len(set(native_top10) & set(hf_top10))
        result = {
            "visual": {
                "exact": torch.equal(native_features, hf_features),
                "max_abs": visual_difference.max().item(),
                "mean_abs": visual_difference.mean().item(),
            },
            "logits": {
                "cosine": logits_cosine,
                "max_abs": logits_difference.max().item(),
                "mean_abs": logits_difference.mean().item(),
                "native_argmax": native_logits.argmax().item(),
                "hf_argmax": hf_logits.argmax().item(),
                "top10_overlap": top10_overlap,
            },
        }
        if not result["visual"]["exact"]:
            raise AssertionError(f"vision features are not exact: {result['visual']}")
        if logits_cosine < 0.9999 or native_logits.argmax() != hf_logits.argmax() or top10_overlap < 9:
            raise AssertionError(f"native logits failed alignment: {result['logits']}")
        return result
    finally:
        reset_context()
        for name in ("provider", "native_model", "hf_model"):
            if name in locals():
                del locals()[name]
        torch.set_default_device("cpu")
        torch.set_default_dtype(original_dtype)
        torch.cuda.empty_cache()
        dist.destroy_process_group()
        Path(rendezvous_path).unlink(missing_ok=True)


def verify_cache(
    model_path: str,
    image: Image.Image,
    *,
    gpu_memory_utilization: float = 0.4,
) -> dict[str, object]:
    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_batched_tokens=1536,
        max_num_seqs=4,
        max_model_len=1536,
        gpu_memory_utilization=gpu_memory_utilization,
        vision_feature_cache_bytes=256 * 1024**2,
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=1)
    snapshots = []
    try:
        for label, messages in (
            ("first", make_messages(long_prompt=True)),
            ("feature_reuse", make_messages(system=True, long_prompt=True)),
            ("prefix_reuse", make_messages(long_prompt=True)),
        ):
            llm.generate_multimodal(
                [messages],
                [[image]],
                [["auto"]],
                sampling_params,
                use_tqdm=False,
            )
            snapshots.append({"label": label, **asdict(llm.vision_stats())})
    finally:
        llm.exit()

    first, feature_reuse, prefix_reuse = snapshots
    if first["encoded_images"] != 1 or first["cache_misses"] != 1:
        raise AssertionError(f"first request did not encode exactly one image: {first}")
    if feature_reuse["cache_hits"] < 1 or feature_reuse["encoded_images"] != 1:
        raise AssertionError(f"feature cache was not reused: {feature_reuse}")
    if prefix_reuse["prefix_skipped_features"] < 1 or prefix_reuse["encoded_images"] != 1:
        raise AssertionError(f"paged KV prefix did not skip vision encoding: {prefix_reuse}")
    return {"snapshots": snapshots}


def main() -> None:
    args = parse_args()
    image = Image.open(args.image).convert("RGB")
    results = {}
    if args.mode in {"alignment", "all"}:
        results["alignment"] = run_verification_stage(
            lambda: verify_alignment(args.model, image, args.device)
        )
    if args.mode in {"cache", "all"}:
        results["cache"] = run_verification_stage(
            lambda: verify_cache(
                args.model,
                image,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
