import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor

from nanovllm.multimodal.qwen3_vl_processor import (
    Qwen3VLPromptProcessor,
    require_transformers_version,
)
from nanovllm.multimodal.runtime import MaterializedMultimodalPrompt, VisionInput


class FakeQwen3Processor:
    def __init__(self):
        self.tokenizer = SimpleNamespace(decode=lambda tokens: "decoded")
        self.input_ids = (10, 99, 99, 99, 99, 20, 99, 99, 30)
        self.grids = ((1, 4, 4), (1, 2, 4))
        self.patch_count = 24

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.messages = messages
        return "rendered prompt"

    def __call__(self, *, text, images, videos, padding, return_tensors):
        result = {"input_ids": torch.tensor([self.input_ids])}
        if images:
            result.update(
                pixel_values=torch.arange(
                    self.patch_count * 4,
                    dtype=torch.float32,
                ).reshape(self.patch_count, 4),
                image_grid_thw=torch.tensor(self.grids),
            )
        return SimpleNamespace(**result)


class Qwen3VLPromptProcessorTests(unittest.TestCase):
    def setUp(self):
        self.raw_processor = FakeQwen3Processor()
        self.processor = Qwen3VLPromptProcessor(
            self.raw_processor,
            model_fingerprint="model-a",
            processor_fingerprint="processor-a",
            image_token_id=99,
            video_token_id=100,
            spatial_merge_size=2,
        )
        self.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": "describe"},
                ],
            }
        ]
        self.image_a = Image.new("RGB", (2, 2), color=(1, 2, 3))
        self.image_b = Image.new("RGB", (2, 4), color=(4, 5, 6))

    def _installed_transformers_version(self, distribution_name):
        self.assertEqual(distribution_name, "transformers")
        return "4.57.6"

    def test_materializes_multiple_images_into_common_runtime_types(self):
        result = self.processor.materialize(
            self.messages,
            [self.image_a, self.image_b],
            ["auto", "high"],
        )

        self.assertIsInstance(result, MaterializedMultimodalPrompt)
        self.assertTrue(all(isinstance(item, VisionInput) for item in result.vision_inputs))
        self.assertEqual(
            [item.pixel_values.shape[0] for item in result.vision_inputs],
            [16, 8],
        )
        self.assertEqual(
            [item.grid_thw for item in result.vision_inputs],
            [(1, 4, 4), (1, 2, 4)],
        )
        self.assertEqual(result.token_ids, self.raw_processor.input_ids)
        result.metadata.validate(len(result.token_ids))

    def test_materializes_text_only_prompt(self):
        self.raw_processor.input_ids = (4, 5, 6)
        self.raw_processor.grids = ()

        result = self.processor.materialize(
            [{"role": "user", "content": "hello"}],
            [],
            [],
        )

        self.assertEqual(result.token_ids, (4, 5, 6))
        self.assertEqual(result.vision_inputs, ())
        self.assertEqual(result.metadata.position_ids, ((0, 1, 2),) * 3)

    def test_repeated_images_have_distinct_occurrence_keys_in_one_stable_group(self):
        self.raw_processor.input_ids = (10, 99, 99, 99, 99, 20, 99, 99, 99, 99, 30)
        self.raw_processor.grids = ((1, 4, 4), (1, 4, 4))
        self.raw_processor.patch_count = 32

        result = self.processor.materialize(
            self.messages,
            [self.image_a, self.image_a.copy()],
            ["auto", "auto"],
        )

        keys = tuple(item.feature_key for item in result.vision_inputs)
        self.assertNotEqual(keys[0], keys[1])
        self.assertEqual(keys[0].rsplit(":", 1)[0], keys[1].rsplit(":", 1)[0])

        repeated = self.processor.materialize(
            self.messages,
            [self.image_a.copy(), self.image_a.copy()],
            ["auto", "auto"],
        )
        self.assertEqual(keys, tuple(item.feature_key for item in repeated.vision_inputs))

    def test_single_image_key_does_not_alias_the_same_image_in_a_multi_image_group(self):
        self.raw_processor.input_ids = (10, 99, 99, 99, 99, 30)
        self.raw_processor.grids = ((1, 4, 4),)
        self.raw_processor.patch_count = 16
        single = self.processor.materialize(
            [{"role": "user", "content": [{"type": "image"}]}],
            [self.image_a],
            ["auto"],
        )

        self.raw_processor.input_ids = (10, 99, 99, 99, 99, 20, 99, 99, 99, 99, 30)
        self.raw_processor.grids = ((1, 4, 4), (1, 4, 4))
        self.raw_processor.patch_count = 32
        repeated = self.processor.materialize(
            self.messages,
            [self.image_a, self.image_a.copy()],
            ["auto", "auto"],
        )

        self.assertNotEqual(
            single.vision_inputs[0].feature_key,
            repeated.vision_inputs[0].feature_key,
        )

    def test_rejects_patch_count_that_does_not_match_grids(self):
        self.raw_processor.patch_count = 23

        with self.assertRaisesRegex(ValueError, "patch"):
            self.processor.materialize(
                self.messages,
                [self.image_a, self.image_b],
                ["auto", "high"],
            )

    def test_rejects_placeholder_count_that_does_not_match_grids(self):
        self.raw_processor.input_ids = (10, 99, 99, 20, 99, 99, 30)

        with self.assertRaisesRegex(ValueError, "placeholder"):
            self.processor.materialize(
                self.messages,
                [self.image_a, self.image_b],
                ["auto", "high"],
            )

    def test_explicitly_rejects_video_tokens(self):
        self.raw_processor.input_ids = (10, 100, 20)
        self.raw_processor.grids = ()

        with self.assertRaisesRegex(ValueError, "Qwen3-VL.*video"):
            self.processor.materialize(
                [{"role": "user", "content": [{"type": "video"}]}],
                [],
                [],
            )

    def test_requires_transformers_4576_before_loading_checkpoint(self):
        with self.assertRaisesRegex(ImportError, "4\\.57\\.6.*4\\.57\\.5"):
            require_transformers_version("4.57.5")

    def test_version_lookup_only_queries_transformers_distribution(self):
        queried_distributions = []

        def version(distribution_name):
            queried_distributions.append(distribution_name)
            return "4.57.6"

        with patch(
            "nanovllm.multimodal.qwen3_vl_processor.importlib_metadata.version",
            side_effect=version,
        ):
            require_transformers_version()

        self.assertEqual(queried_distributions, ["transformers"])

    def test_from_pretrained_uses_auto_processor_for_qwen3_vl(self):
        raw_processor = SimpleNamespace(
            image_processor=SimpleNamespace(to_dict=lambda: {"merge_size": 2}),
        )
        config = SimpleNamespace(
            model_type="qwen3_vl",
            image_token_id=99,
            video_token_id=100,
            vision_config=SimpleNamespace(spatial_merge_size=2),
        )
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "model.safetensors").write_bytes(b"weights")
            with (
                patch(
                    "nanovllm.multimodal.qwen3_vl_processor.importlib_metadata.version",
                    side_effect=self._installed_transformers_version,
                ),
                patch.object(AutoConfig, "from_pretrained", return_value=config),
                patch.object(
                    AutoProcessor,
                    "from_pretrained",
                    return_value=raw_processor,
                ) as load,
            ):
                result = Qwen3VLPromptProcessor.from_pretrained(directory)

        self.assertIs(result.processor, raw_processor)
        load.assert_called_once_with(directory)

    def test_from_pretrained_rejects_other_model_types(self):
        config = SimpleNamespace(model_type="qwen2_5_vl")
        with (
            patch(
                "nanovllm.multimodal.qwen3_vl_processor.importlib_metadata.version",
                side_effect=self._installed_transformers_version,
            ),
            patch.object(AutoConfig, "from_pretrained", return_value=config),
        ):
            with self.assertRaisesRegex(ValueError, "Qwen3-VL"):
                Qwen3VLPromptProcessor.from_pretrained("unused")


if __name__ == "__main__":
    unittest.main()
