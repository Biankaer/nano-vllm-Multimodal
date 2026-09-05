import unittest
from unittest.mock import patch

import torch

from nanovllm.layers.sampler import Sampler
from nanovllm.sampling_params import SamplingParams


class SamplingParamsTests(unittest.TestCase):
    def test_negative_temperature_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "temperature cannot be negative"):
            SamplingParams(temperature=-0.1)

    def test_zero_temperature_is_allowed(self):
        params = SamplingParams(temperature=0)

        self.assertEqual(params.temperature, 0)


class SamplerTests(unittest.TestCase):
    def test_zero_temperature_uses_argmax(self):
        logits = torch.tensor([[1.0, 5.0, 3.0], [7.0, -1.0, 2.0]])

        tokens = Sampler()(
            logits,
            torch.tensor([0.0, 0.0]),
            all_greedy=True,
        )

        torch.testing.assert_close(tokens, torch.tensor([1, 0]))

    def test_all_greedy_sampling_does_not_advance_rng_state(self):
        logits = torch.tensor([[1.0, 5.0, 3.0], [7.0, -1.0, 2.0]])
        temperatures = torch.zeros(2)
        torch.manual_seed(1234)
        rng_state = torch.random.get_rng_state()

        with patch("nanovllm.layers.sampler.torch.all") as tensor_all:
            Sampler()(logits, temperatures, all_greedy=True)

        tensor_all.assert_not_called()
        torch.testing.assert_close(torch.random.get_rng_state(), rng_state)

    def test_mixed_batch_keeps_greedy_rows_exact_and_samples_positive_rows(self):
        batch_size = 128
        logits = torch.zeros(batch_size, 3)
        logits[0] = torch.tensor([1.0, 5.0, 3.0])
        temperatures = torch.ones(batch_size)
        temperatures[0] = 0

        torch.manual_seed(7)
        tokens = Sampler()(logits, temperatures, all_greedy=False)

        self.assertEqual(tokens[0].item(), 1)
        self.assertGreater(torch.unique(tokens[1:]).numel(), 1)


if __name__ == "__main__":
    unittest.main()
