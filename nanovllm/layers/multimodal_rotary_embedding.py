from __future__ import annotations

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_multimodal_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    sections: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if query.ndim != 3 or key.ndim != 3:
        raise ValueError("query and key must have shape [num_tokens, num_heads, head_dim]")
    if cos.ndim != 3 or sin.ndim != 3 or cos.shape != sin.shape:
        raise ValueError("cos and sin must have equal shape [3, num_tokens, head_dim]")
    if cos.shape[0] != 3 or cos.shape[1] != query.shape[0]:
        raise ValueError("rotary positions must match the query token count")
    if query.shape[-1] != key.shape[-1] or query.shape[-1] != cos.shape[-1]:
        raise ValueError("query, key, cos, and sin must use the same head dimension")

    repeated_sections = sections + sections
    if sum(repeated_sections) != query.shape[-1]:
        raise ValueError("multimodal rotary sections must cover the head dimension")

    selected_cos = torch.cat(
        [part[index % 3] for index, part in enumerate(cos.split(repeated_sections, dim=-1))],
        dim=-1,
    ).unsqueeze(1)
    selected_sin = torch.cat(
        [part[index % 3] for index, part in enumerate(sin.split(repeated_sections, dim=-1))],
        dim=-1,
    ).unsqueeze(1)

    query_dtype = query.dtype
    key_dtype = key.dtype
    query = query.float()
    key = key.float()
    selected_cos = selected_cos.float()
    selected_sin = selected_sin.float()
    query = query * selected_cos + rotate_half(query) * selected_sin
    key = key * selected_cos + rotate_half(key) * selected_sin
    return query.to(query_dtype), key.to(key_dtype)


class MultimodalRotaryEmbedding(nn.Module):
    def __init__(
        self,
        *,
        head_dim: int,
        base: float,
        sections: tuple[int, int, int],
    ) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2:
            raise ValueError("head_dim must be a positive even number")
        if len(sections) != 3 or any(section <= 0 for section in sections):
            raise ValueError("sections must contain three positive dimensions")
        if sum(sections) != head_dim // 2:
            raise ValueError("multimodal rotary sections must sum to half the head dimension")

        self.head_dim = head_dim
        self.sections = sections
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def cos_sin(
        self,
        positions: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.ndim != 2 or positions.shape[0] != 3:
            raise ValueError("positions must have shape [3, num_tokens]")

        frequencies = positions.to(device=self.inv_freq.device, dtype=torch.float32).unsqueeze(-1)
        frequencies = frequencies * self.inv_freq.view(1, 1, -1)
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        return embeddings.cos().to(dtype), embeddings.sin().to(dtype)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.cos_sin(positions, query.dtype)
        return apply_multimodal_rotary_emb(
            query,
            key,
            cos,
            sin,
            self.sections,
        )
