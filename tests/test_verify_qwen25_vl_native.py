import unittest
import weakref
from unittest.mock import patch

import torch
import verify_qwen25_vl_native as verification_module

from nanovllm.multimodal.runtime import VisionFeatureBundle
from verify_qwen25_vl_native import qwen25_final_embedding


class Qwen25VerificationCompatibilityTests(unittest.TestCase):
    def test_cache_gpu_memory_utilization_is_configurable(self):
        args = verification_module.parse_args([
            "--model", "/models/qwen25-vl",
            "--gpu-memory-utilization", "0.7",
        ])

        self.assertEqual(args.gpu_memory_utilization, 0.7)

    def test_stage_releases_cyclic_objects_before_emptying_cuda_cache(self):
        references = []

        class Probe:
            pass

        def operation():
            probe = Probe()
            probe.cycle = probe
            references.append(weakref.ref(probe))
            return "finished"

        def assert_released():
            self.assertIsNone(references[0]())

        with patch.object(torch.cuda, "empty_cache", side_effect=assert_released):
            result = verification_module.run_verification_stage(operation)

        self.assertEqual(result, "finished")

    def test_extracts_final_embedding_from_bundle(self):
        final = torch.randn(3, 4)

        actual = qwen25_final_embedding(VisionFeatureBundle(final))

        self.assertIs(actual, final)

    def test_rejects_unexpected_deepstack_features(self):
        final = torch.randn(3, 4)
        bundle = VisionFeatureBundle(final, (torch.randn(3, 4),))

        with self.assertRaisesRegex(ValueError, "DeepStack"):
            qwen25_final_embedding(bundle)


if __name__ == "__main__":
    unittest.main()
