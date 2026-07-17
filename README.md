# NanoInfer

NanoInfer is a job-oriented fork of `nano-vllm`. The project goal is to evolve a
compact offline inference engine into a lightweight vLLM-style LLM/VLM serving
system with clear architecture, scheduling optimizations, multimodal support, and
reproducible benchmarks.

The original offline `nanovllm.LLM` API is preserved. New work is added around it
in serving, multimodal, benchmark, and engine-refactor modules.

## Current Status

Implemented milestones:

- Added OpenAI-style serving schema definitions.
- Added FastAPI app skeleton.
- Added `/v1/completions` and `/v1/chat/completions` non-streaming endpoints.
- Added multimodal request, image loading, image feature-cache hook, and processor skeletons.
- Added the first EngineCore / Executor / GPUWorker split while preserving the original offline API.
- Added metrics foundation for scheduler state, KV cache usage, per-step throughput, TTFT, TPOT, and request latency.
- Added a Qwen2.5-VL HuggingFace compatibility backend for explicit fallback.
- Added an OpenAI-to-Qwen2.5-VL request adapter for URL, base64, file URL, and trusted local-path images.
- Added engine-owned multimodal request batching, image/text token budgets, and decoded-image LRU caching.
- Added a concurrent multimodal benchmark with latency percentiles and server-side cache/batch metrics.
- Added the first native-runtime foundation: Qwen2.5-VL MRoPE prompt positions,
  strict visual embedding replacement, and image-aware paged-KV prefix identities.
- Added a native Qwen2.5-VL decoder using the same Scheduler, BlockManager,
  paged KV cache, chunked prefill, sampler, and CUDA execution path as Qwen3.
- Added a worker-owned, byte-budgeted GPU vision feature LRU with miss
  micro-batching, pinning, request reference counts, and prefix-cache-aware lazy
  vision resolution.

## Roadmap

Planned milestones:

1. Serving skeleton and API compatibility.
2. EngineCore / Executor / GPUWorker architecture split.
3. Continuous batching and token-budget based chunked prefill.
4. N-gram speculative decoding.
5. Multimodal VLM serving with image preprocessing, image feature cache, image-text batching, and multimodal benchmarks.
6. Benchmark reports for TTFT, TPOT, throughput, latency percentiles, KV cache usage, and multimodal overhead.

## Installation

Base install:

```bash
pip install -e .
```

Serving extras:

```bash
pip install -e ".[serving]"
```

Multimodal extras:

```bash
pip install -e ".[all]"
```

Qwen2.5-VL requires `transformers>=4.51.0`. Check the server environment before
loading the multimodal model:

```bash
python -c "import transformers; print(transformers.__version__)"
```

## Model Download

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Offline Usage

```python
from nanovllm import LLM, SamplingParams

llm = LLM("~/huggingface/Qwen3-0.6B/", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=128)
outputs = llm.generate(["Hello, NanoInfer."], sampling_params)
print(outputs[0]["text"])
```

Native static-image generation uses the same `LLM` instance:

```python
from PIL import Image
from nanovllm import LLM, SamplingParams

llm = LLM("~/huggingface/Qwen2.5-VL-3B-Instruct/", enforce_eager=True)
messages = [{
    "role": "user",
    "content": [
        {"type": "image"},
        {"type": "text", "text": "Describe this image."},
    ],
}]
outputs = llm.generate_multimodal(
    [messages],
    [[Image.open("assets/logo.png").convert("RGB")]],
    [["auto"]],
    SamplingParams(temperature=0.2, max_tokens=64),
)
```

## Serving Usage

Start the server:

```bash
NANOINFER_MODEL=~/huggingface/Qwen3-0.6B/ \
uvicorn nanovllm.serving.entrypoint:app --host 0.0.0.0 --port 8000
```

Start the unified native Qwen2.5-VL runtime:

```bash
NANOINFER_MODEL=~/huggingface/Qwen2.5-VL-3B-Instruct/ \
NANOINFER_MM_BACKEND=native \
NANOINFER_MM_FEATURE_CACHE_BYTES=2147483648 \
NANOINFER_MM_MAX_BATCH_SIZE=8 \
NANOINFER_MM_BATCH_WAIT_MS=5 \
NANOINFER_MM_CACHE_CAPACITY=256 \
uvicorn nanovllm.serving.entrypoint:app --host 0.0.0.0 --port 8000
```

Set `NANOINFER_MM_BACKEND=hf` and `NANOINFER_MM_MODEL` to retain the Transformers
`generate()` compatibility path. Native mode currently supports static images
on one GPU (`tensor_parallel_size=1`); video and top-p sampling are rejected
explicitly.

The `image_url.url` field accepts HTTP(S) URLs and base64 data URLs. Multiple
image and text parts may be interleaved in one message. Trusted deployments can
also enable `file://` URLs and server-local paths with
`NANOINFER_MM_ALLOW_LOCAL_FILES=1`; local file access is disabled by default.

## Multimodal Benchmark

Run concurrent requests against a local image:

```bash
python bench_multimodal.py \
  --server-url http://127.0.0.1:8000 \
  --model qwen2.5-vl \
  --image ./assets/demo.jpg \
  --prompt "Describe the image and list the visible objects." \
  --num-requests 32 \
  --concurrency 8 \
  --max-tokens 128 \
  --output-json benchmark-results/qwen25-vl.json
```

Repeat `--image` to benchmark multi-image requests. The report includes request
throughput, average/P50/P90/P99 latency, failures, actual average server batch
size, and image cache counters from `/metrics/multimodal`. TTFT and TPOT remain
empty until streaming SSE is implemented.

Completion request:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "prompt": "Explain KV cache in one paragraph.",
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

Chat request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [
      {"role": "user", "content": "What is continuous batching?"}
    ],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

Image-text chat request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-vl",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
          {"type": "text", "text": "Describe this image."}
        ]
      }
    ],
    "max_tokens": 128,
    "temperature": 0.2
  }'
```

## Native Multimodal Runtime

Native mode owns the full decode path:

```text
OpenAI request -> decoded-image LRU -> Qwen processor
  -> token IDs + 3-axis MRoPE + CPU image patches
  -> Sequence / MultimodalBatcher / Scheduler
  -> lazy VisionFeatureProvider resolution
      -> byte-budgeted GPU feature LRU
  -> native Qwen2.5 decoder prefill
  -> image-aware BlockManager + paged KV cache
  -> native decode + sampler
```

When paged-KV prefix reuse covers an entire visual span, `ModelRunner` does not
resolve the image feature at all. If the text prefix differs but the image is
the same, the GPU feature cache avoids rerunning the vision tower.

Run the CPU-only foundation tests with:

```bash
python -m unittest discover -s tests -v
```

Run the A100 correctness and cache gates with:

```bash
python verify_qwen25_vl_native.py \
  --model ~/huggingface/Qwen2.5-VL-3B-Instruct \
  --image assets/logo.png \
  --mode all
```

Optimization goals:

- Async image preprocessing.
- Vision feature reuse.
- Image-text batching.
- Visual token compression.
- Multimodal KV cache accounting.
- Multimodal benchmark reports.

## Credits

This project is based on the original `nano-vllm` implementation by Xingkai Yu.
The fork is used as a learning and job-oriented AI infrastructure project.
