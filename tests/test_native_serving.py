import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch

from nanovllm.engine.metrics import SchedulerStats
from nanovllm.multimodal.manager import MultimodalManagerStats
from nanovllm.multimodal.schemas import ImageInput, MultimodalRequest
from nanovllm.multimodal.vision_provider import VisionProviderStats
from nanovllm.serving.engine_client import (
    EngineClient,
    EngineClientConfig,
    NativeQwen25VLBackend,
)


class NativeServingTests(unittest.TestCase):
    def test_native_is_the_default_multimodal_backend(self):
        config = EngineClientConfig(model_path="text", multimodal_model_path="vision")

        self.assertEqual(config.multimodal_backend, "native")

    def test_native_runtime_uses_multimodal_model_for_all_requests(self):
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
            vision_attn_implementation="sdpa",
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
        )

        metrics = client.multimodal_stats()

        self.assertEqual(metrics["backend"], "native")
        self.assertEqual(metrics["vision_feature_cache"]["cache_entries"], 1)
        self.assertEqual(metrics["vision_feature_cache"]["cache_hits"], 3)
        self.assertEqual(metrics["vision_feature_cache"]["prefix_skipped_features"], 2)
        self.assertEqual(metrics["batching"]["average_batch_size"], 2.0)


if __name__ == "__main__":
    unittest.main()
