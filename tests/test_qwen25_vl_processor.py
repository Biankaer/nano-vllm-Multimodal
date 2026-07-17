import unittest
from types import SimpleNamespace

import torch
from PIL import Image

from nanovllm.multimodal.qwen25_vl_processor import Qwen25VLPromptProcessor


class FakeQwenProcessor:
    def __init__(self):
        self.tokenizer = SimpleNamespace(decode=lambda tokens: "decoded")
        self.patch_count = 12

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.messages = messages
        return "rendered prompt"

    def __call__(self, *, text, images, videos, padding, return_tensors):
        return SimpleNamespace(
            input_ids=torch.tensor([[10, 99, 20, 99, 99, 30]]),
            pixel_values=torch.arange(
                self.patch_count * 4,
                dtype=torch.float32,
            ).reshape(self.patch_count, 4),
            image_grid_thw=torch.tensor(((1, 2, 2), (1, 2, 4))),
        )


class Qwen25VLPromptProcessorTests(unittest.TestCase):
    def setUp(self):
        self.raw_processor = FakeQwenProcessor()
        self.processor = Qwen25VLPromptProcessor(
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

    def test_splits_combined_patches_by_grid(self):
        result = self.processor.materialize(
            self.messages,
            [self.image_a, self.image_b],
            ["auto", "high"],
        )

        self.assertEqual(
            [item.pixel_values.shape[0] for item in result.vision_inputs],
            [4, 8],
        )
        self.assertEqual(
            [item.grid_thw for item in result.vision_inputs],
            [(1, 2, 2), (1, 2, 4)],
        )
        self.assertEqual(result.token_ids, (10, 99, 20, 99, 99, 30))
        result.metadata.validate(len(result.token_ids))

    def test_same_pixels_share_feature_key(self):
        first = self.processor.materialize(
            self.messages,
            [self.image_a, self.image_b],
            ["auto", "high"],
        )
        second = self.processor.materialize(
            self.messages,
            [self.image_a.copy(), self.image_b.copy()],
            ["auto", "high"],
        )

        self.assertEqual(
            [item.feature_key for item in first.vision_inputs],
            [item.feature_key for item in second.vision_inputs],
        )

    def test_detail_changes_feature_key(self):
        first = self.processor.materialize(
            self.messages,
            [self.image_a, self.image_b],
            ["auto", "high"],
        )
        second = self.processor.materialize(
            self.messages,
            [self.image_a, self.image_b],
            ["low", "high"],
        )

        self.assertNotEqual(
            first.vision_inputs[0].feature_key,
            second.vision_inputs[0].feature_key,
        )

    def test_rejects_patch_count_mismatch(self):
        self.raw_processor.patch_count = 11

        with self.assertRaisesRegex(ValueError, "patch"):
            self.processor.materialize(
                self.messages,
                [self.image_a, self.image_b],
                ["auto", "high"],
            )


if __name__ == "__main__":
    unittest.main()
