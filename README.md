# NanoInfer

NanoInfer is a job-oriented fork of `nano-vllm`. The project goal is to evolve a
compact offline inference engine into a lightweight vLLM-style LLM/VLM serving
system with clear architecture, scheduling optimizations, multimodal support, and
reproducible benchmarks.

The original offline `nanovllm.LLM` API is preserved. New work is added around it
in serving, multimodal, benchmark, and engine-refactor modules.

## Current Status

The current implementation spans the serving, engine refactor, metrics, and
multimodal batching milestones:

- Added project modification plan: `docs/MODIFICATION_PLAN.md`.
- Added OpenAI-style serving schema definitions.
- Added FastAPI app skeleton.
- Added `/v1/completions` and `/v1/chat/completions` non-streaming endpoints.
- Added multimodal request, image loading, image feature-cache hook, and processor skeletons.
- Added the first EngineCore / Executor / GPUWorker split while preserving the original offline API.
- Added metrics foundation for scheduler state, KV cache usage, per-step throughput, TTFT, TPOT, and request latency.
- Added a Qwen2.5-VL HuggingFace backend for image-text chat requests.
- Added an OpenAI-to-Qwen2.5-VL request adapter for URL, base64, file URL, and trusted local-path images.
- Added engine-owned multimodal request batching, image/text token budgets, and decoded-image LRU caching.
- Added a concurrent multimodal benchmark with latency percentiles and server-side cache/batch metrics.

Architecture notes:

- [docs/COMPLETE_LEARNING_GUIDE.md](docs/COMPLETE_LEARNING_GUIDE.md)
- [docs/LEARNING_GUIDE.md](docs/LEARNING_GUIDE.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/IMPLEMENTATION_RETROSPECTIVE.md](docs/IMPLEMENTATION_RETROSPECTIVE.md)

## Roadmap

Planned milestones:

1. Serving skeleton and API compatibility.
2. EngineCore / Executor / GPUWorker architecture split.
3. Continuous batching and token-budget based chunked prefill.
4. N-gram speculative decoding.
5. Multimodal VLM serving with image preprocessing, image feature cache, image-text batching, and multimodal benchmarks.
6. Benchmark reports for TTFT, TPOT, throughput, latency percentiles, KV cache usage, and multimodal overhead.

See [docs/MODIFICATION_PLAN.md](docs/MODIFICATION_PLAN.md) for the full plan.

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

## Serving Usage

Start the server:

```bash
NANOINFER_MODEL=~/huggingface/Qwen3-0.6B/ \
uvicorn nanovllm.serving.entrypoint:app --host 0.0.0.0 --port 8000
```

Start with a Qwen2.5-VL multimodal backend:

```bash
NANOINFER_MODEL=~/huggingface/Qwen3-0.6B/ \
NANOINFER_MM_MODEL=~/huggingface/Qwen2.5-VL-3B-Instruct/ \
NANOINFER_MM_DTYPE=bfloat16 \
NANOINFER_MM_MAX_BATCH_SIZE=8 \
NANOINFER_MM_BATCH_WAIT_MS=5 \
NANOINFER_MM_CACHE_CAPACITY=256 \
uvicorn nanovllm.serving.entrypoint:app --host 0.0.0.0 --port 8000
```

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

## Multimodal Direction

The first multimodal milestone introduces request schemas and model-agnostic image
feature caching. The first real VLM backend is Qwen2.5-VL through HuggingFace
Transformers:

```text
image + text
  -> image loader / preprocessing
  -> Qwen2.5-VL processor
  -> Qwen2.5-VL model generate
  -> OpenAI-compatible response
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
