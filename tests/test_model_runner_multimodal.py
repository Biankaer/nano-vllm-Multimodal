import unittest

import torch

from nanovllm.engine.model_runner import (
    build_decode_position_rows,
    build_prefill_position_rows,
    merge_planned_visual_embeddings,
    plan_visual_replacements,
)
from nanovllm.engine.sequence import Sequence
from nanovllm.multimodal.runtime import MultimodalPromptMetadata, VisualTokenSpan


def make_sequence(
    *,
    prompt_length=8,
    span_start=1,
    span_length=3,
    feature_key="vision:a",
):
    positions = tuple(range(prompt_length))
    metadata = MultimodalPromptMetadata(
        position_ids=(positions, positions, positions),
        rope_delta=0,
        visual_spans=(
            VisualTokenSpan(
                start=span_start,
                length=span_length,
                feature_key=feature_key,
            ),
        ),
    )
    return Sequence(list(range(prompt_length)), multimodal_metadata=metadata)


class VisualReplacementPlanTests(unittest.TestCase):
    def test_selects_only_visual_rows_in_scheduled_range(self):
        sequence = make_sequence()
        sequence.num_cached_tokens = 0
        sequence.num_scheduled_tokens = 4

        plan = plan_visual_replacements([sequence])

        self.assertEqual(plan.required_feature_keys, ("vision:a",))
        self.assertEqual(
            [replacement.output_row for replacement in plan.replacements],
            [1, 2, 3],
        )
        self.assertEqual(
            [replacement.feature_row for replacement in plan.replacements],
            [0, 1, 2],
        )

    def test_prefix_covered_visual_span_requires_no_feature(self):
        sequence = make_sequence()
        sequence.num_cached_tokens = 4
        sequence.num_scheduled_tokens = 2

        plan = plan_visual_replacements([sequence])

        self.assertEqual(plan.required_feature_keys, ())
        self.assertEqual(plan.replacements, ())
        self.assertEqual(plan.prefix_skipped_features, 1)

    def test_packed_rows_include_preceding_sequence_tokens(self):
        first = make_sequence(prompt_length=5, span_start=1, span_length=1)
        first.num_scheduled_tokens = 5
        second = make_sequence(
            prompt_length=6,
            span_start=2,
            span_length=2,
            feature_key="vision:b",
        )
        second.num_scheduled_tokens = 4

        plan = plan_visual_replacements([first, second])

        self.assertEqual(plan.required_feature_keys, ("vision:a", "vision:b"))
        self.assertEqual(
            [replacement.output_row for replacement in plan.replacements],
            [1, 7, 8],
        )
        self.assertEqual(
            [replacement.feature_row for replacement in plan.replacements],
            [0, 0, 1],
        )

    def test_chunked_prefill_uses_feature_slice(self):
        sequence = make_sequence(span_start=1, span_length=5)
        sequence.num_cached_tokens = 3
        sequence.num_scheduled_tokens = 2

        plan = plan_visual_replacements([sequence])

        self.assertEqual(
            [replacement.output_row for replacement in plan.replacements],
            [0, 1],
        )
        self.assertEqual(
            [replacement.feature_row for replacement in plan.replacements],
            [2, 3],
        )


class VisualEmbeddingMergeTests(unittest.TestCase):
    def test_replaces_only_planned_embedding_rows(self):
        token_embeddings = torch.zeros(5, 3)
        plan = plan_visual_replacements([
            self._scheduled_sequence(span_start=1, span_length=2),
        ])
        features = {
            "vision:a": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        }

        merged = merge_planned_visual_embeddings(token_embeddings, plan, features)

        self.assertTrue(torch.equal(merged[0], token_embeddings[0]))
        self.assertTrue(torch.equal(merged[1:3], features["vision:a"]))
        self.assertTrue(torch.equal(merged[3:], token_embeddings[3:]))
        self.assertTrue(torch.equal(token_embeddings, torch.zeros_like(token_embeddings)))

    def test_rejects_feature_hidden_size_mismatch(self):
        token_embeddings = torch.zeros(3, 4)
        plan = plan_visual_replacements([
            self._scheduled_sequence(prompt_length=3, span_start=0, span_length=1),
        ])

        with self.assertRaisesRegex(ValueError, "hidden size"):
            merge_planned_visual_embeddings(
                token_embeddings,
                plan,
                {"vision:a": torch.zeros(1, 3)},
            )

    def test_rejects_missing_feature(self):
        token_embeddings = torch.zeros(3, 4)
        plan = plan_visual_replacements([
            self._scheduled_sequence(prompt_length=3, span_start=0, span_length=1),
        ])

        with self.assertRaisesRegex(KeyError, "vision:a"):
            merge_planned_visual_embeddings(token_embeddings, plan, {})

    @staticmethod
    def _scheduled_sequence(**kwargs):
        sequence = make_sequence(**kwargs)
        sequence.num_scheduled_tokens = len(sequence)
        return sequence


class PackedPositionTests(unittest.TestCase):
    def test_prefill_slices_three_axis_metadata(self):
        positions = (
            (0, 1, 1, 3, 4),
            (0, 1, 2, 3, 4),
            (0, 2, 1, 3, 4),
        )
        metadata = MultimodalPromptMetadata(
            position_ids=positions,
            rope_delta=0,
        )
        sequence = Sequence(
            [10, 11, 12, 13, 14],
            multimodal_metadata=metadata,
        )
        sequence.num_cached_tokens = 1
        sequence.num_scheduled_tokens = 3

        self.assertEqual(
            build_prefill_position_rows([sequence], position_axes=3),
            ((1, 1, 3), (1, 2, 3), (2, 1, 3)),
        )

    def test_text_prefill_repeats_positions_for_three_axes(self):
        sequence = Sequence([10, 11, 12])
        sequence.num_scheduled_tokens = 3

        self.assertEqual(
            build_prefill_position_rows([sequence], position_axes=3),
            ((0, 1, 2), (0, 1, 2), (0, 1, 2)),
        )

    def test_decode_applies_mrope_delta_on_all_axes(self):
        sequence = make_sequence(prompt_length=5)
        sequence.mrope_position_delta = -2
        sequence.append_token(42)

        self.assertEqual(
            build_decode_position_rows([sequence], position_axes=3),
            ((3,), (3,), (3,)),
        )

    def test_qwen3_positions_remain_one_dimensional(self):
        sequence = Sequence([10, 11, 12])
        sequence.num_scheduled_tokens = 3

        self.assertEqual(
            build_prefill_position_rows([sequence], position_axes=1),
            ((0, 1, 2),),
        )

    def test_reprefill_after_preemption_extends_positions_past_prompt(self):
        positions = (
            (0, 1, 1, 3, 4),
            (0, 1, 2, 3, 4),
            (0, 2, 1, 3, 4),
        )
        metadata = MultimodalPromptMetadata(
            position_ids=positions,
            rope_delta=-2,
        )
        sequence = Sequence(
            [10, 11, 12, 13, 14],
            multimodal_metadata=metadata,
        )
        sequence.append_token(20)
        sequence.append_token(21)
        sequence.num_cached_tokens = 4
        sequence.num_scheduled_tokens = 3

        self.assertEqual(
            build_prefill_position_rows([sequence], position_axes=3),
            ((4, 3, 4), (4, 3, 4), (4, 3, 4)),
        )


if __name__ == "__main__":
    unittest.main()
