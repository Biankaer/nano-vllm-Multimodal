# NanoInfer

NanoInfer is a job-oriented fork of `nano-vllm`. The project goal is to evolve a
compact offline inference engine into a lightweight vLLM-style LLM/VLM serving
system with clear architecture, scheduling optimizations, multimodal support, and
reproducible benchmarks.

The original offline `nanovllm.LLM` API is preserved. New work is added around it
in serving, multimodal, benchmark, and engine-refactor modules.

## Current Status

Phase 1 has started:

- Added project modification plan: `docs/MODIFICATION_PLAN.md`.
- Added OpenAI-style serving schema definitions.
- Added FastAPI app skeleton.
- Added `/v1/completions` and `/v1/chat/completions` non-streaming endpoints.
- Added multimodal request, image loading, image feature cache, and processor skeletons.
- Added the first EngineCore / Executor / GPUWorker split while preserving the original offline API.

Architecture notes:

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

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

## Multimodal Direction

The first multimodal milestone introduces request schemas and model-agnostic image
feature caching. Later phases will connect these abstractions to a real VLM path:

```text
image + text
  -> image loader / preprocessing
  -> vision encoder
  -> image feature cache
  -> multimodal prompt assembly
  -> scheduler
  -> LLM decode
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
