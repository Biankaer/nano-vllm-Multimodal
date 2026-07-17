import subprocess
import sys
import unittest
from pathlib import Path

import torch

from nanovllm.multimodal.qwen25_vl_runtime import (
    build_vision_feature_key,
    build_qwen25_vl_prompt_metadata,
    merge_visual_embeddings,
)
from nanovllm.multimodal.runtime import MultimodalPromptMetadata, VisualTokenSpan


class MultimodalPromptMetadataTests(unittest.TestCase):

    def test_runtime_helpers_import_without_pillow(self):
        project_root = Path(__file__).resolve().parents[1]
        script = """
import sys

class BlockPillowImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'PIL' or fullname.startswith('PIL.'):
            raise ModuleNotFoundError('Pillow intentionally unavailable')
        return None

sys.meta_path.insert(0, BlockPillowImports())
from nanovllm.multimodal.qwen25_vl_runtime import build_qwen25_vl_prompt_metadata
metadata = build_qwen25_vl_prompt_metadata(
    token_ids=[1, 2],
    image_grid_thw=[],
    image_feature_keys=[],
    image_token_id=11,
    spatial_merge_size=2,
)
assert metadata.rope_delta == 0
"""

        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_visual_identity_contains_feature_key_and_feature_index(self):
        metadata = MultimodalPromptMetadata(
            position_ids=((0, 1, 1), (0, 1, 1), (0, 1, 2)),
            rope_delta=-1,
            visual_spans=(VisualTokenSpan(start=1, length=2, feature_key="image:a"),),
        )

        metadata.validate(prompt_length=3)

        self.assertIsNone(metadata.visual_identity_at(0))
        self.assertEqual(metadata.visual_identity_at(1), "image:a:0")
        self.assertEqual(metadata.visual_identity_at(2), "image:a:1")

    def test_overlapping_visual_spans_are_rejected(self):
        metadata = MultimodalPromptMetadata(
            position_ids=((0, 1, 2),) * 3,
            rope_delta=0,
            visual_spans=(
                VisualTokenSpan(start=0, length=2, feature_key="a"),
                VisualTokenSpan(start=1, length=2, feature_key="b"),
            ),
        )

        with self.assertRaisesRegex(ValueError, "overlap"):
            metadata.validate(prompt_length=3)

    def test_position_axis_length_must_match_prompt(self):
        metadata = MultimodalPromptMetadata(
            position_ids=((0, 1), (0, 1), (0, 1)),
            rope_delta=0,
        )

        with self.assertRaisesRegex(ValueError, "prompt length"):
            metadata.validate(prompt_length=3)


class Qwen25VLPositionTests(unittest.TestCase):

    def test_vision_feature_key_is_deterministic(self):
        arguments = dict(
            model_fingerprint="qwen25-vl:model-revision",
            processor_fingerprint="processor:min_pixels=1:max_pixels=2",
            image_fingerprint="sha256:image-a",
            grid_thw=(1, 4, 4),
        )

        self.assertEqual(build_vision_feature_key(**arguments), build_vision_feature_key(**arguments))

    def test_vision_feature_key_changes_with_each_feature_input(self):
        baseline = build_vision_feature_key(
            model_fingerprint="model-a",
            processor_fingerprint="processor-a",
            image_fingerprint="image-a",
            grid_thw=(1, 4, 4),
        )

        variants = (
            build_vision_feature_key("model-b", "processor-a", "image-a", (1, 4, 4)),
            build_vision_feature_key("model-a", "processor-b", "image-a", (1, 4, 4)),
            build_vision_feature_key("model-a", "processor-a", "image-b", (1, 4, 4)),
            build_vision_feature_key("model-a", "processor-a", "image-a", (1, 8, 4)),
        )

        self.assertTrue(all(key != baseline for key in variants))

    def test_builds_three_axis_positions_for_image_and_text(self):
        metadata = build_qwen25_vl_prompt_metadata(
            token_ids=[10, 11, 11, 11, 11, 20, 21],
            image_grid_thw=[(1, 4, 4)],
            image_feature_keys=["model:processor:image-a:grid-1x4x4"],
            image_token_id=11,
            spatial_merge_size=2,
        )

        self.assertEqual(
            metadata.position_ids,
            (
                (0, 1, 1, 1, 1, 3, 4),
                (0, 1, 1, 2, 2, 3, 4),
                (0, 1, 2, 1, 2, 3, 4),
            ),
        )
        self.assertEqual(metadata.rope_delta, -2)
        self.assertEqual(
            metadata.visual_spans,
            (
                VisualTokenSpan(
                    start=1,
                    length=4,
                    feature_key="model:processor:image-a:grid-1x4x4",
                ),
            ),
        )

    def test_text_only_positions_are_standard_one_dimensional_positions(self):
        metadata = build_qwen25_vl_prompt_metadata(
            token_ids=[4, 5, 6],
            image_grid_thw=[],
            image_feature_keys=[],
            image_token_id=11,
            spatial_merge_size=2,
        )

        self.assertEqual(metadata.position_ids, ((0, 1, 2),) * 3)
        self.assertEqual(metadata.rope_delta, 0)
        self.assertEqual(metadata.visual_spans, ())

    def test_static_image_temporal_positions_do_not_advance(self):
        metadata = build_qwen25_vl_prompt_metadata(
            token_ids=[11, 11],
            image_grid_thw=[(2, 2, 2)],
            image_feature_keys=["image-a"],
            image_token_id=11,
            spatial_merge_size=2,
        )

        self.assertEqual(metadata.position_ids, ((0, 0), (0, 0), (0, 0)))
        self.assertEqual(metadata.rope_delta, -1)

    def test_rejects_placeholder_count_that_does_not_match_grid(self):
        with self.assertRaisesRegex(ValueError, "placeholder"):
            build_qwen25_vl_prompt_metadata(
                token_ids=[10, 11, 11, 20],
                image_grid_thw=[(1, 4, 4)],
                image_feature_keys=["image-a"],
                image_token_id=11,
                spatial_merge_size=2,
            )

    def test_rejects_grid_not_divisible_by_spatial_merge_size(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            build_qwen25_vl_prompt_metadata(
                token_ids=[10, 11, 20],
                image_grid_thw=[(1, 3, 4)],
                image_feature_keys=["image-a"],
                image_token_id=11,
                spatial_merge_size=2,
            )


class VisualEmbeddingMergeTests(unittest.TestCase):

    def test_replaces_only_visual_placeholder_rows(self):
        input_ids = torch.tensor([4, 11, 11, 5])
        token_embeddings = torch.tensor(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
        )
        visual_embeddings = torch.tensor([[20.0, 21.0], [30.0, 31.0]])

        merged = merge_visual_embeddings(
            input_ids,
            token_embeddings,
            visual_embeddings,
            image_token_id=11,
        )

        torch.testing.assert_close(
            merged,
            torch.tensor([[1.0, 1.0], [20.0, 21.0], [30.0, 31.0], [4.0, 4.0]]),
        )
        torch.testing.assert_close(
            token_embeddings,
            torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]),
        )

    def test_rejects_visual_feature_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "tokens.*features"):
            merge_visual_embeddings(
                torch.tensor([11, 11]),
                torch.zeros(2, 4),
                torch.zeros(1, 4),
                image_token_id=11,
            )

    def test_rejects_hidden_size_mismatch(self):
        with self.assertRaisesRegex(ValueError, "hidden size"):
            merge_visual_embeddings(
                torch.tensor([11]),
                torch.zeros(1, 4),
                torch.zeros(1, 3),
                image_token_id=11,
            )


if __name__ == "__main__":
    unittest.main()
