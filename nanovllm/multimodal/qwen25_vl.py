from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Qwen25VLConfig:
    model_path: str
    device_map: str = "auto"
    torch_dtype: str = "auto"
    attn_implementation: str | None = None


class Qwen25VLBackend:
    """HuggingFace Qwen2.5-VL backend for image-text chat.

    This backend is intentionally separate from the nano-vLLM text runtime. It
    gives the project a real open-source multimodal model path first, while later
    phases can progressively move vision encoding, image feature caching, and
    multimodal batching into the NanoInfer engine.
    """

    def __init__(self, config: Qwen25VLConfig):
        self.config = config
        self._model = None
        self._processor = None
        self._process_vision_info = None

    def generate_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        ignore_eos: bool = False,
    ) -> str:
        return self.generate_chat_batch(
            [messages],
            image_inputs_batch=None,
            image_details_batch=None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            ignore_eos=ignore_eos,
        )[0]

    def generate_chat_batch(
        self,
        messages_batch: list[list[dict[str, Any]]],
        *,
        image_inputs_batch: list[list[object]] | None,
        image_details_batch: list[list[str]] | None,
        max_new_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        ignore_eos: bool = False,
    ) -> list[str]:
        self._ensure_loaded()
        prompts = [
            self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in messages_batch
        ]
        if image_inputs_batch is None:
            image_inputs, video_inputs = self._process_vision_info(messages_batch)
        else:
            image_inputs = [image for request_images in image_inputs_batch for image in request_images]
            video_inputs = None
        inputs = self._processor(
            text=prompts,
            images=image_inputs or None,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(next(self._model.parameters()).device)
        do_sample = temperature > 1e-5
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if ignore_eos:
            generate_kwargs["eos_token_id"] = []
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p
        generated_ids = self._model.generate(**inputs, **generate_kwargs)
        generated_ids = [
            output_ids[input_ids.shape[-1]:]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return outputs

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        try:
            import torch
            import transformers
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Qwen2.5-VL backend requires multimodal dependencies. "
                "Install with: pip install -e '.[multimodal]'"
            ) from exc

        model_cls = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None)
        if model_cls is None:
            model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
        if model_cls is None:
            raise RuntimeError("Installed transformers version does not expose a Qwen2.5-VL compatible model class.")

        kwargs: dict[str, Any] = {
            "torch_dtype": self._resolve_dtype(torch),
            "device_map": self.config.device_map,
        }
        if self.config.attn_implementation:
            kwargs["attn_implementation"] = self.config.attn_implementation

        self._model = model_cls.from_pretrained(self.config.model_path, **kwargs)
        self._processor = AutoProcessor.from_pretrained(self.config.model_path)
        self._process_vision_info = process_vision_info

    def _resolve_dtype(self, torch_module):
        if self.config.torch_dtype == "auto":
            return "auto"
        dtype = getattr(torch_module, self.config.torch_dtype, None)
        if dtype is None:
            raise ValueError(f"Unsupported torch dtype: {self.config.torch_dtype}")
        return dtype
