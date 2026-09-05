import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from nanovllm.engine.model_runner import (
    ModelRunner,
    VisualReplacement,
    VisualReplacementPlan,
    build_planned_visual_embeddings,
    build_decode_position_rows,
    build_prefill_position_rows,
    merge_planned_visual_embeddings,
    plan_visual_replacements,
)
from nanovllm.engine.sequence import Sequence
from nanovllm.multimodal.runtime import MultimodalPromptMetadata, VisualTokenSpan
from nanovllm.multimodal.runtime import VisionFeatureBundle


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
    def test_rejects_overlapping_visual_spans_that_duplicate_output_rows(self):
        positions = tuple(range(5))
        metadata = MultimodalPromptMetadata(
            position_ids=(positions, positions, positions), rope_delta=0,
            visual_spans=(
                VisualTokenSpan(1, 2, "a"), VisualTokenSpan(2, 2, "b"),
            ),
        )
        sequence = SimpleNamespace(
            multimodal_metadata=metadata,
            num_cached_tokens=0,
            num_scheduled_tokens=5,
        )
        with self.assertRaisesRegex(ValueError, "strictly increasing and unique"):
            plan_visual_replacements([sequence])
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
    def test_large_interleaved_plan_batches_transfers_by_feature_key(self):
        replacements = tuple(
            VisualReplacement(
                output_row=1 + index * 2,
                feature_key=f"vision:{'a' if index % 2 == 0 else 'b'}",
                feature_row=(index // 2) % 8,
            )
            for index in range(64)
        )
        plan = VisualReplacementPlan(
            replacements=replacements,
            required_feature_keys=("vision:a", "vision:b"),
            prefix_skipped_features=0,
        )
        token_embeddings = torch.full((130, 4), -1.0)
        final_a = torch.arange(32.0).reshape(8, 4)
        final_b = torch.arange(100.0, 132.0).reshape(8, 4)
        bundles = {
            "vision:a": VisionFeatureBundle(
                final_a,
                (final_a + 1_000, final_a + 2_000),
            ),
            "vision:b": VisionFeatureBundle(
                final_b,
                (final_b + 1_000, final_b + 2_000),
            ),
        }
        original_to = torch.Tensor.to
        to_calls = 0

        def counting_to(tensor, *args, **kwargs):
            nonlocal to_calls
            to_calls += 1
            return original_to(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, "to", counting_to):
            packed = build_planned_visual_embeddings(
                token_embeddings,
                plan,
                bundles,
            )

        expected_final_rows = torch.stack([
            bundles[replacement.feature_key].final_embedding[replacement.feature_row]
            for replacement in replacements
        ])
        expected_deepstack_rows = tuple(
            torch.stack([
                bundles[replacement.feature_key].deepstack_embeddings[layer_index][
                    replacement.feature_row
                ]
                for replacement in replacements
            ])
            for layer_index in range(2)
        )
        output_rows = torch.tensor(
            [replacement.output_row for replacement in replacements]
        )

        self.assertTrue(
            torch.equal(packed.final_embedding.index_select(0, output_rows), expected_final_rows)
        )
        self.assertEqual(packed.visual_rows.tolist(), output_rows.tolist())
        for actual, expected in zip(
            packed.deepstack_embeddings,
            expected_deepstack_rows,
        ):
            self.assertTrue(torch.equal(actual, expected))
        self.assertLessEqual(to_calls, len(bundles) * 3)

    def test_builds_final_and_deepstack_rows_from_the_same_plan(self):
        token_embeddings = torch.zeros(5, 3)
        plan = plan_visual_replacements([
            self._scheduled_sequence(span_start=1, span_length=2),
        ])
        bundle = VisionFeatureBundle(
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            (
                torch.tensor([[11.0, 12.0, 13.0], [14.0, 15.0, 16.0]]),
                torch.tensor([[21.0, 22.0, 23.0], [24.0, 25.0, 26.0]]),
            ),
        )

        packed = build_planned_visual_embeddings(
            token_embeddings,
            plan,
            {"vision:a": bundle},
        )

        self.assertTrue(torch.equal(packed.final_embedding[1:3], bundle.final_embedding))
        self.assertEqual(packed.visual_rows.tolist(), [1, 2])
        self.assertEqual(len(packed.deepstack_embeddings), 2)
        self.assertTrue(
            torch.equal(packed.deepstack_embeddings[0], bundle.deepstack_embeddings[0])
        )
        self.assertTrue(
            torch.equal(packed.deepstack_embeddings[1], bundle.deepstack_embeddings[1])
        )

    def test_deepstack_packing_honors_chunked_feature_slices(self):
        token_embeddings = torch.zeros(2, 2)
        sequence = self._scheduled_sequence(span_start=1, span_length=5)
        sequence.num_cached_tokens = 3
        sequence.num_scheduled_tokens = 2
        plan = plan_visual_replacements([sequence])
        rows = torch.arange(10.0).reshape(5, 2)

        packed = build_planned_visual_embeddings(
            token_embeddings,
            plan,
            {"vision:a": VisionFeatureBundle(rows, (rows + 100,))},
        )

        self.assertTrue(torch.equal(packed.final_embedding, rows[2:4]))
        self.assertTrue(torch.equal(packed.deepstack_embeddings[0], rows[2:4] + 100))
        self.assertEqual(packed.visual_rows.tolist(), [0, 1])

    def test_rejects_mixed_deepstack_layer_counts(self):
        first = self._scheduled_sequence(span_start=0, span_length=1)
        second = self._scheduled_sequence(
            span_start=0,
            span_length=1,
            feature_key="vision:b",
        )
        plan = plan_visual_replacements([first, second])

        with self.assertRaisesRegex(ValueError, "deepstack layer count"):
            build_planned_visual_embeddings(
                torch.zeros(16, 3),
                plan,
                {
                    "vision:a": VisionFeatureBundle(
                        torch.zeros(1, 3),
                        (torch.zeros(1, 3),),
                    ),
                    "vision:b": VisionFeatureBundle(torch.zeros(1, 3)),
                },
            )

    def test_final_only_bundle_has_no_deepstack_payload(self):
        token_embeddings = torch.zeros(3, 2)
        plan = plan_visual_replacements([
            self._scheduled_sequence(prompt_length=3, span_start=0, span_length=1),
        ])

        packed = build_planned_visual_embeddings(
            token_embeddings,
            plan,
            {"vision:a": VisionFeatureBundle(torch.ones(1, 2))},
        )

        self.assertEqual(packed.deepstack_embeddings, ())
        self.assertEqual(packed.visual_rows.tolist(), [0])

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


class ModelRunnerVisualForwardTests(unittest.TestCase):
    def test_run_model_forwards_prepared_deepstack_payload(self):
        class RecordingModel:
            def __init__(self):
                self.kwargs = None

            def __call__(self, input_ids, positions, **kwargs):
                self.kwargs = kwargs
                return torch.ones(2, 3)

            def compute_logits(self, hidden_states):
                return hidden_states

        runner = ModelRunner.__new__(ModelRunner)
        runner.enforce_eager = True
        runner.model = RecordingModel()
        inputs_embeds = torch.zeros(2, 3)
        visual_rows = torch.tensor([1])
        deepstack = (torch.ones(1, 3),)

        runner.run_model(
            torch.tensor([10, 11]),
            torch.tensor([0, 1]),
            True,
            inputs_embeds,
            visual_rows,
            deepstack,
        )

        self.assertIs(runner.model.kwargs["inputs_embeds"], inputs_embeds)
        self.assertIs(runner.model.kwargs["visual_token_rows"], visual_rows)
        self.assertIs(runner.model.kwargs["deepstack_visual_embeds"], deepstack)

    def test_run_carries_prefill_visual_payload_to_run_model(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.rank = 1
        runner.prepare_prefill = Mock(
            return_value=(
                torch.tensor([10]),
                torch.tensor([0]),
                torch.zeros(1, 3),
                torch.tensor([0]),
                (torch.ones(1, 3),),
            )
        )
        runner.run_model = Mock(return_value=torch.zeros(1, 3))

        result = runner.run([SimpleNamespace()], is_prefill=True)

        self.assertIsNone(result)
        args = runner.run_model.call_args.args
        self.assertEqual(len(args), 6)
        self.assertEqual(args[4].tolist(), [0])
        self.assertEqual(len(args[5]), 1)

    def test_decode_passes_no_visual_payload(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.rank = 1
        runner.prepare_decode = Mock(
            return_value=(torch.tensor([10]), torch.tensor([0]))
        )
        runner.run_model = Mock(return_value=torch.zeros(1, 3))

        runner.run([SimpleNamespace()], is_prefill=False)

        args = runner.run_model.call_args.args
        self.assertIsNone(args[3])
        self.assertIsNone(args[4])
        self.assertEqual(args[5], ())

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
