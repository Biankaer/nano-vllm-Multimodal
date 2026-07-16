from __future__ import annotations

import os

from nanovllm.serving.server import create_app


MODEL_PATH = os.environ.get("NANOINFER_MODEL", os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
ENFORCE_EAGER = os.environ.get("NANOINFER_ENFORCE_EAGER", "1") not in {"0", "false", "False"}
TENSOR_PARALLEL_SIZE = int(os.environ.get("NANOINFER_TP_SIZE", "1"))

app = create_app(
    MODEL_PATH,
    enforce_eager=ENFORCE_EAGER,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
)
