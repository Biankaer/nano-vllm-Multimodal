import torch
from torch import nn


class Sampler(nn.Module):

    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        all_greedy: bool = False,
    ):
        if all_greedy:
            return logits.argmax(dim=-1)
        return self._sample(logits, temperatures)

    @torch.compile
    def _sample(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float()
        greedy_mask = temperatures == 0
        safe_temperatures = torch.where(
            greedy_mask,
            torch.ones_like(temperatures),
            temperatures,
        )
        probs = torch.softmax(logits / safe_temperatures.unsqueeze(dim=1), dim=-1)
        sampled_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        greedy_tokens = logits.argmax(dim=-1)
        return torch.where(greedy_mask, greedy_tokens, sampled_tokens)
