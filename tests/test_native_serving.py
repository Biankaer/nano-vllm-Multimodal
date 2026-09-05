import unittest
import json
import inspect
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from pydantic import ValidationError

from nanovllm.engine.metrics import RequestTimingStats, SchedulerStats
from nanovllm.engine.streaming import GenerationStreamEvent, TokenEvent
from nanovllm.multimodal.manager import MultimodalManagerStats
from nanovllm.multimodal.schemas import ImageInput, MultimodalRequest
from nanovllm.multimodal.vision_provider import VisionProviderStats
from nanovllm.serving.engine_client import (
    EngineClient,
    EngineClientConfig,
    NativeQwen25VLBackend,
    NativeQwenVLBackend,
)
from nanovllm.serving.schemas import ChatCompletionRequest, CompletionRequest
from nanovllm.serving.server import create_app


class NativeServingTests(unittest.TestCase):
    def test_legacy_native_backend_is_compatibility_alias(self):
        self.assertIs(NativeQwen25VLBackend, NativeQwenVLBackend)

    def test_hf_backend_rejects_qwen3_before_constructing_hf_backend(self):
        with TemporaryDirectory() as model_dir:
            Path(model_dir, "config.json").write_text(
                json.dumps({"model_type": "qwen3_vl"}), encoding="utf-8"
            )
            client = EngineClient(EngineClientConfig(
                model_path="text", multimodal_model_path=model_dir,
                multimodal_backend="hf"
            ))
            with patch("nanovllm.serving.engine_client.Qwen25VLBackend") as backend:
                with self.assertRaisesRegex(ValueError, "Qwen3-VL.*native"):
                    _ = client.multimodal_backend
        backend.assert_not_called()

    def test_qwen25_hf_backend_still_uses_existing_backend(self):
        with TemporaryDirectory() as model_dir:
            Path(model_dir, "config.json").write_text(
                json.dumps({"model_type": "qwen2_5_vl"}), encoding="utf-8"
            )
            client = EngineClient(EngineClientConfig(
                model_path="text", multimodal_model_path=model_dir,
                multimodal_backend="hf",
            ))
            with patch(
                "nanovllm.serving.engine_client.Qwen25VLBackend",
                return_value="hf-backend",
            ) as backend:
                self.assertEqual(client.multimodal_backend, "hf-backend")
        backend.assert_called_once()
    def test_completion_schema_allows_greedy_temperature(self):
        request = CompletionRequest(prompt="hello", temperature=0)

        self.assertEqual(request.temperature, 0)

    def test_chat_schema_allows_greedy_temperature(self):
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}],
            temperature=0,
        )

        self.assertEqual(request.temperature, 0)

    def test_completion_and_chat_schemas_reject_negative_temperature(self):
        with self.assertRaises(ValidationError):
            CompletionRequest(prompt="hello", temperature=-0.1)
        with self.assertRaises(ValidationError):
            ChatCompletionRequest(
                messages=[{"role": "user", "content": "hello"}],
                temperature=-0.1,
            )

    def test_native_is_the_default_multimodal_backend(self):
        config = EngineClientConfig(model_path="text", multimodal_model_path="vision")

        self.assertEqual(config.multimodal_backend, "native")

    def test_native_runtime_uses_auto_vision_attention_by_default(self):
        config = EngineClientConfig(
            model_path="text-model",
            multimodal_model_path="vision-model",
            multimodal_backend="native",
            multimodal_feature_cache_bytes=1234,
        )
        client = EngineClient(config)

        with patch("nanovllm.serving.engine_client.LLM", return_value=Mock()) as llm_cls:
            _ = client.llm

        llm_cls.assert_called_once_with(
            "vision-model",
            enforce_eager=True,
            tensor_parallel_size=1,
            vision_feature_cache_bytes=1234,
            vision_attn_implementation="auto",
        )

    def test_native_runtime_preserves_explicit_vision_attention_backend(self):
        config = EngineClientConfig(
            model_path="text-model",
            multimodal_model_path="vision-model",
            multimodal_backend="native",
            multimodal_attn_implementation="sdpa",
        )
        client = EngineClient(config)

        with patch("nanovllm.serving.engine_client.LLM", return_value=Mock()) as llm_cls:
            _ = client.llm

        self.assertEqual(
            llm_cls.call_args.kwargs["vision_attn_implementation"],
            "sdpa",
        )

    def test_native_image_request_calls_unified_llm(self):
        request = MultimodalRequest(
            text="ignored",
            images=[ImageInput("image.png", source_type="path", detail="high")],
            messages=[{"role": "user", "content": [{"type": "image"}]}],
        )
        client = EngineClient(
            EngineClientConfig(
                model_path="vision-model",
                multimodal_backend="native",
            )
        )
        client._multimodal_core = Mock()
        client._multimodal_core.generate.return_value = "native answer"

        answer = client.generate_multimodal_chat(
            request,
            temperature=0.2,
            max_tokens=8,
        )

        self.assertEqual(answer, "native answer")
        client._multimodal_core.generate.assert_called_once()

    def test_native_batch_backend_submits_one_unified_llm_batch(self):
        llm = Mock()
        llm.generate_multimodal.return_value = [
            {"text": "first"},
            {"text": "second"},
        ]
        backend = NativeQwen25VLBackend(lambda: llm, nullcontext())
        messages = [
            [{"role": "user", "content": "one"}],
            [{"role": "user", "content": "two"}],
        ]
        images = [[object()], [object()]]
        details = [["high"], ["low"]]

        outputs = backend.generate_chat_batch(
            messages,
            image_inputs_batch=images,
            image_details_batch=details,
            max_new_tokens=8,
            temperature=0.2,
            top_p=1.0,
        )

        self.assertEqual(outputs, ["first", "second"])
        args = llm.generate_multimodal.call_args.args
        self.assertEqual(args[:3], (messages, images, details))

    def test_native_batch_backend_streams_one_unified_llm_batch(self):
        llm = Mock()
        expected = GenerationStreamEvent(0, 1, 2, "ok", True, "stop", 4, 1)
        llm.generate_multimodal_stream.return_value = iter([expected])
        backend = NativeQwenVLBackend(lambda: llm, nullcontext())

        events = list(backend.generate_chat_batch_stream(
            [[{"role": "user", "content": "one"}]],
            image_inputs_batch=[[]],
            image_details_batch=[[]],
            max_new_tokens=8,
            temperature=0,
            top_p=1.0,
        ))

        self.assertEqual(events, [expected])
        llm.generate_multimodal_stream.assert_called_once()

    def test_native_backend_dynamic_admission_steps_and_decodes_by_sequence(self):
        llm = Mock()
        llm.admit_multimodal_requests.return_value = [11]
        llm.tokenizer.decode.side_effect = lambda token_ids, **kwargs: "".join(
            chr(token_id) for token_id in token_ids
        )
        llm.core.drain_token_events.return_value = [
            TokenEvent(11, ord("A"), True, "length")
        ]
        llm.request_stats.return_value = RequestTimingStats(
            seq_id=11,
            prompt_tokens=4,
            completion_tokens=1,
            queue_time_sec=0.1,
            ttft_sec=0.2,
            total_latency_sec=0.3,
            tpot_sec=None,
        )
        backend = NativeQwenVLBackend(lambda: llm, nullcontext())

        with backend.dynamic_session():
            sequence_ids = backend.admit_chat_batch(
                [[{"role": "user", "content": "one"}]],
                image_inputs_batch=[[]],
                image_details_batch=[[]],
                max_new_tokens=1,
                temperature=0,
                top_p=1.0,
            )
            events = backend.step_dynamic()

        self.assertEqual(sequence_ids, [11])
        self.assertEqual(events, [
            GenerationStreamEvent(0, 11, ord("A"), "A", True, "length", 4, 1)
        ])
        llm.step.assert_called_once_with()

    def test_native_stream_backend_forwards_ignore_eos_to_sampling_params(self):
        llm = Mock()
        llm.generate_multimodal_stream.return_value = iter([])
        backend = NativeQwenVLBackend(lambda: llm, nullcontext())

        list(backend.generate_chat_batch_stream(
            [[{"role": "user", "content": "one"}]],
            image_inputs_batch=[[]],
            image_details_batch=[[]],
            max_new_tokens=8,
            temperature=0,
            top_p=1.0,
            ignore_eos=True,
        ))

        sampling_params = llm.generate_multimodal_stream.call_args.args[3]
        self.assertTrue(sampling_params.ignore_eos)

    def test_hf_backend_remains_explicitly_available(self):
        client = EngineClient(
            EngineClientConfig(
                model_path="text-model",
                multimodal_model_path="vision-model",
                multimodal_backend="hf",
            )
        )
        client._multimodal_core = Mock()
        client._multimodal_core.generate.return_value = "hf answer"
        request = MultimodalRequest(text="hello")

        answer = client.generate_multimodal_chat(
            request,
            temperature=0.2,
            max_tokens=8,
        )

        self.assertEqual(answer, "hf answer")
        client._multimodal_core.generate.assert_called_once()

    def test_native_metrics_expose_real_gpu_feature_cache(self):
        client = EngineClient(
            EngineClientConfig(model_path="vision-model", multimodal_backend="native")
        )
        client._llm = Mock()
        client._llm.vision_stats.return_value = VisionProviderStats(
            cache_entries=1,
            cache_resident_bytes=2048,
            cache_byte_budget=4096,
            cache_hits=3,
            cache_misses=1,
            cache_evictions=0,
            encoder_batches=1,
            encoded_images=1,
            registered_inputs=0,
            prefix_skipped_features=2,
        )
        client._llm.scheduler_stats.return_value = SchedulerStats(0, 0, 10, 0, 10, 4)
        client._multimodal_core = Mock()
        client._multimodal_core.manager.stats.return_value = MultimodalManagerStats(
            2, 2, 1, 1, 1, 0, 0, 0, 0, 0
        )
        client._multimodal_core.stats.return_value = Mock(
            waiting_requests=0,
            submitted_requests=4,
            completed_requests=4,
            failed_requests=0,
            executed_batches=2,
            average_batch_size=2.0,
            batch_size_histogram={2: 2},
            average_queue_wait_ms=4.5,
            average_coalescing_wait_ms=2.5,
            average_execution_ms=30.0,
        )

        metrics = client.multimodal_stats()

        self.assertEqual(metrics["backend"], "native")
        self.assertEqual(metrics["vision_feature_cache"]["cache_entries"], 1)
        self.assertEqual(metrics["vision_feature_cache"]["cache_hits"], 3)
        self.assertEqual(metrics["vision_feature_cache"]["prefix_skipped_features"], 2)
        self.assertEqual(metrics["batching"]["average_batch_size"], 2.0)
        self.assertEqual(metrics["batching"]["batch_size_histogram"], {2: 2})
        self.assertEqual(metrics["batching"]["average_queue_wait_ms"], 4.5)
        self.assertEqual(metrics["batching"]["average_coalescing_wait_ms"], 2.5)
        self.assertEqual(metrics["batching"]["average_execution_ms"], 30.0)

    def test_native_serving_defaults_to_adaptive_batching(self):
        config = EngineClientConfig(model_path="vision-model")

        self.assertTrue(config.enforce_eager)
        self.assertEqual(config.multimodal_batch_quiet_ms, 3.0)
        self.assertEqual(config.multimodal_batch_max_wait_ms, 20.0)
        self.assertIsNone(config.multimodal_batch_wait_ms)

    def test_http_server_defaults_to_half_gpu_memory_budget(self):
        parameter = inspect.signature(create_app).parameters["gpu_memory_utilization"]

        self.assertEqual(parameter.default, 0.5)

    def test_gpu_memory_budget_is_forwarded_to_native_llm(self):
        client = EngineClient(EngineClientConfig(
            model_path="vision-model",
            gpu_memory_utilization=0.5,
        ))

        with patch("nanovllm.serving.engine_client.LLM", return_value=Mock()) as llm_cls:
            _ = client.llm

        self.assertEqual(llm_cls.call_args.kwargs["gpu_memory_utilization"], 0.5)


if __name__ == "__main__":
    unittest.main()
