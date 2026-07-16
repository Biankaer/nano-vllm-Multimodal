# NanoInfer Architecture

## Current Layering

Phase 2 introduces the first architecture split:

```text
LLMEngine
  -> EngineCore
      -> Scheduler
      -> Executor
          -> GPUWorker
              -> ModelRunner
                  -> Qwen3ForCausalLM
```

## Responsibilities

### LLMEngine

User-facing offline API.

Responsibilities:

- Build `Config`.
- Load tokenizer.
- Convert prompts into token ids.
- Create `Sequence` objects.
- Decode completion token ids into text.
- Preserve the original `LLM.generate()` interface.

### EngineCore

Request lifecycle and scheduling orchestration.

Responsibilities:

- Own `Scheduler`.
- Own `Executor`.
- Accept internal `Sequence` objects.
- Run one engine step:
  - schedule sequences
  - execute model
  - postprocess generated token ids
  - return newly finished outputs

Future responsibilities:

- Async request admission.
- Per-request state tracking.
- Streaming token handoff.
- TTFT and TPOT measurement.

Current metrics exposed by `EngineCore`:

- `scheduler_stats()`: waiting/running sequence count, KV block usage, prefix cache entries.
- `latest_step_stats`: last prefill/decode step latency and throughput.
- `request_stats(seq_id)`: completed request queue time, TTFT, TPOT, and total latency.

### Executor

Model execution abstraction.

Responsibilities:

- Start tensor-parallel child processes.
- Own the driver `GPUWorker`.
- Forward execution requests from `EngineCore` to GPU workers.
- Shutdown workers and join child processes.

Future responsibilities:

- Switch between local GPU, remote worker, or mock executor.
- Expose worker health and execution metrics.
- Support multimodal workers.

### GPUWorker

Single GPU worker wrapper.

Responsibilities:

- Wrap the existing `ModelRunner`.
- Keep CUDA/model/KV-cache details out of `EngineCore`.
- Provide stable `run()` and `shutdown()` methods.

### ModelRunner

Low-level model runtime from the original nano-vLLM.

Responsibilities:

- Load model weights.
- Allocate KV cache.
- Prepare prefill/decode tensors.
- Execute model forward.
- Run sampler.
- Handle tensor-parallel shared-memory protocol.

## Why This Split Matters

The original nano-vLLM code is compact and readable, but the API layer, scheduler,
worker startup, and model execution path are tightly coupled. This split creates
clear extension points for:

- OpenAI-compatible serving.
- Continuous batching.
- Chunked prefill scheduling.
- Speculative decoding.
- Multimodal image-text preprocessing and worker execution.
- Benchmark and metrics instrumentation.

## Current Execution Path

Offline generation still starts from:

```python
outputs = llm.generate(prompts, sampling_params)
```

Internally it now flows through:

```text
LLM.generate()
  -> LLM.add_request()
      -> EngineCore.add_sequence()
  -> while not LLM.is_finished()
      -> LLM.step()
          -> EngineCore.step()
              -> Scheduler.schedule()
              -> Executor.run()
                  -> GPUWorker.run()
                      -> ModelRunner.call("run", ...)
              -> Scheduler.postprocess()
  -> tokenizer.decode()
```

This preserves behavior while making the engine easier to evolve.

## Metrics Flow

Phase 3 adds a lightweight internal metrics layer:

```text
Sequence
  -> records arrival_time / running_time / first_token_time / finish_time
  -> exposes RequestTimingStats

Scheduler
  -> exposes SchedulerStats
  -> reports waiting/running queue sizes
  -> reports KV cache free/used blocks

EngineCore
  -> records EngineStepStats after every step
  -> stores RequestTimingStats for finished sequences

LLMEngine
  -> exposes scheduler_stats(), step_stats(), request_stats(seq_id)
```

These stats are intentionally dependency-free. Serving and benchmark modules can
later export them to Prometheus, JSON benchmark reports, or logs.

## Multimodal Path

Phase 5 starts with a pragmatic two-backend design:

```text
Text-only request
  -> OpenAI API
  -> EngineClient.generate()
  -> NanoInfer text engine

Image-text request
  -> OpenAI API
  -> EngineClient.generate_multimodal_chat()
  -> Qwen25VLBackend
  -> HuggingFace Qwen2.5-VL
```

This keeps the original nano-vLLM text runtime intact while adding a real
open-source VLM backend. The current Qwen2.5-VL backend is a functional bridge,
not the final optimized serving path. Later phases should move these pieces into
the engine:

- image preprocessing queue
- image feature cache
- vision encoder batching
- image-token budget accounting
- multimodal KV cache statistics
- mixed text/image scheduling
