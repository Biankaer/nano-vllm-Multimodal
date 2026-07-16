from __future__ import annotations

import os

from nanovllm.serving.server import create_app


MODEL_PATH = os.environ.get("NANOINFER_MODEL", os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
ENFORCE_EAGER = os.environ.get("NANOINFER_ENFORCE_EAGER", "1") not in {"0", "false", "False"}
TENSOR_PARALLEL_SIZE = int(os.environ.get("NANOINFER_TP_SIZE", "1"))
MULTIMODAL_MODEL_PATH = os.environ.get("NANOINFER_MM_MODEL")
MULTIMODAL_DEVICE_MAP = os.environ.get("NANOINFER_MM_DEVICE_MAP", "auto")
MULTIMODAL_TORCH_DTYPE = os.environ.get("NANOINFER_MM_DTYPE", "auto")
MULTIMODAL_ATTN_IMPL = os.environ.get("NANOINFER_MM_ATTN_IMPL") or None

app = create_app(
    MODEL_PATH,
    enforce_eager=ENFORCE_EAGER,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    multimodal_model_path=MULTIMODAL_MODEL_PATH,
    multimodal_device_map=MULTIMODAL_DEVICE_MAP,
    multimodal_torch_dtype=MULTIMODAL_TORCH_DTYPE,
    multimodal_attn_implementation=MULTIMODAL_ATTN_IMPL,
)
