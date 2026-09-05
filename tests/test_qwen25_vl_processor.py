import unittest
from types import SimpleNamespace

import torch
from PIL import Image

from nanovllm.multimodal.qwen25_vl_processor import (
    MaterializedQwen25VLPrompt,
    Qwen25VLPromptProcessor,
    VisionInput as LegacyVisionInput,
)
from nanovllm.multimodal.runtime import (
    MaterializedMultimodalPrompt,
    VisionFeatureBundle,
    VisionInput,
)


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

    def test_legacy_types_reexport_common_multimodal_types(self):
        self.assertIs(LegacyVisionInput, VisionInput)
        self.assertIs(MaterializedQwen25VLPrompt, MaterializedMultimodalPrompt)

        result = self.processor.materialize(
            self.messages,
            [self.image_a, self.image_b],
            ["auto", "high"],
        )

        self.assertIsInstance(result, MaterializedMultimodalPrompt)

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


class VisionFeatureBundleTests(unittest.TestCase):
    def test_validates_layers_and_counts_total_bytes(self):
        final = torch.zeros(2, 3, dtype=torch.float32)
        deep = (
            torch.ones(2, 3, dtype=torch.float16),
            torch.full((2, 3), 2.0, dtype=torch.float32),
        )

        bundle = VisionFeatureBundle(final, deep)

        self.assertEqual(bundle.token_count, 2)
        self.assertEqual(bundle.hidden_size, 3)
        self.assertEqual(bundle.nbytes, 24 + 12 + 24)

    def test_rejects_non_matrix_and_inconsistent_deepstack_shapes(self):
        with self.assertRaisesRegex(ValueError, "rank-2"):
            VisionFeatureBundle(torch.zeros(2, 3, 4))
        with self.assertRaisesRegex(ValueError, "token and hidden"):
            VisionFeatureBundle(
                torch.zeros(2, 3),
                (torch.zeros(3, 3),),
            )

    def test_detach_contiguous_and_to_apply_to_every_layer(self):
        final_source = torch.arange(12.0, requires_grad=True).reshape(3, 4).T
        deep_source = (final_source + 1,)
        bundle = VisionFeatureBundle(final_source, deep_source)

        stored = bundle.detach().contiguous().to(dtype=torch.float64)

        for tensor in (stored.final_embedding, *stored.deepstack_embeddings):
            self.assertFalse(tensor.requires_grad)
            self.assertTrue(tensor.is_contiguous())
            self.assertEqual(tensor.dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()
