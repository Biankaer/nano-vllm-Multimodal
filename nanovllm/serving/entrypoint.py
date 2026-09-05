from __future__ import annotations

import os

from nanovllm.serving.server import create_app


MODEL_PATH = os.environ.get("NANOINFER_MODEL", os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
ENFORCE_EAGER = os.environ.get("NANOINFER_ENFORCE_EAGER", "0") not in {"0", "false", "False"}
GPU_MEMORY_UTILIZATION = float(os.environ.get("NANOINFER_GPU_MEMORY_UTILIZATION", "0.5"))
TENSOR_PARALLEL_SIZE = int(os.environ.get("NANOINFER_TP_SIZE", "1"))
MULTIMODAL_MODEL_PATH = os.environ.get("NANOINFER_MM_MODEL")
MULTIMODAL_BACKEND = os.environ.get("NANOINFER_MM_BACKEND", "native")
MULTIMODAL_DEVICE_MAP = os.environ.get("NANOINFER_MM_DEVICE_MAP", "auto")
MULTIMODAL_TORCH_DTYPE = os.environ.get("NANOINFER_MM_DTYPE", "auto")
MULTIMODAL_ATTN_IMPL = os.environ.get("NANOINFER_MM_ATTN_IMPL") or None
MULTIMODAL_MAX_BATCH_SIZE = int(os.environ.get("NANOINFER_MM_MAX_BATCH_SIZE", "8"))
MULTIMODAL_MAX_IMAGES_PER_BATCH = int(os.environ.get("NANOINFER_MM_MAX_IMAGES_PER_BATCH", "8"))
_LEGACY_BATCH_WAIT_MS = os.environ.get("NANOINFER_MM_BATCH_WAIT_MS")
MULTIMODAL_BATCH_WAIT_MS = float(_LEGACY_BATCH_WAIT_MS) if _LEGACY_BATCH_WAIT_MS is not None else None
MULTIMODAL_BATCH_QUIET_MS = float(os.environ.get("NANOINFER_MM_BATCH_QUIET_MS", "3"))
MULTIMODAL_BATCH_MAX_WAIT_MS = float(os.environ.get("NANOINFER_MM_BATCH_MAX_WAIT_MS", "20"))
MULTIMODAL_CACHE_CAPACITY = int(os.environ.get("NANOINFER_MM_CACHE_CAPACITY", "256"))
MULTIMODAL_FEATURE_CACHE_BYTES = int(
    os.environ.get("NANOINFER_MM_FEATURE_CACHE_BYTES", str(2 * 1024**3))
)
MULTIMODAL_ALLOW_LOCAL_FILES = os.environ.get("NANOINFER_MM_ALLOW_LOCAL_FILES", "0") in {"1", "true", "True"}

app = create_app(
    MODEL_PATH,
    enforce_eager=ENFORCE_EAGER,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    multimodal_model_path=MULTIMODAL_MODEL_PATH,
    multimodal_backend=MULTIMODAL_BACKEND,
    multimodal_device_map=MULTIMODAL_DEVICE_MAP,
    multimodal_torch_dtype=MULTIMODAL_TORCH_DTYPE,
    multimodal_attn_implementation=MULTIMODAL_ATTN_IMPL,
    multimodal_max_batch_size=MULTIMODAL_MAX_BATCH_SIZE,
    multimodal_max_images_per_batch=MULTIMODAL_MAX_IMAGES_PER_BATCH,
    multimodal_batch_wait_ms=MULTIMODAL_BATCH_WAIT_MS,
    multimodal_batch_quiet_ms=MULTIMODAL_BATCH_QUIET_MS,
    multimodal_batch_max_wait_ms=MULTIMODAL_BATCH_MAX_WAIT_MS,
    multimodal_cache_capacity=MULTIMODAL_CACHE_CAPACITY,
    multimodal_feature_cache_bytes=MULTIMODAL_FEATURE_CACHE_BYTES,
    multimodal_allow_local_files=MULTIMODAL_ALLOW_LOCAL_FILES,
)
