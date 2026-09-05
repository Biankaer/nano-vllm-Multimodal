from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention, SDPAAttention
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from nanovllm.layers.multimodal_rotary_embedding import MultimodalRotaryEmbedding
from nanovllm.utils.context import get_context


def hf_aligned_linear(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run per-request 3D GEMMs using HF's full-prefill row shape.

    Linear projections are row-independent, so cached prefix rows can be zero
    padded without changing current-token values. Matching HF's [1, seq, dim]
    GEMM shape avoids BF16 kernel-selection drift during chunked prefill.
    """
    context = get_context()
    if not context.is_prefill or context.cu_seqlens_q is None:
        return nn.functional.linear(hidden_states.unsqueeze(0), weight, bias).squeeze(0)

    outputs = []
    request_count = context.cu_seqlens_q.numel() - 1
    for request_index in range(request_count):
        q_start = int(context.cu_seqlens_q[request_index])
        q_end = int(context.cu_seqlens_q[request_index + 1])
        query_length = q_end - q_start
        key_length = int(context.cu_seqlens_k[request_index + 1] - context.cu_seqlens_k[request_index])
        full_length = (
            int(context.full_context_lens[request_index])
            if context.full_context_lens is not None
            else key_length
        )
        query_start = (
            int(context.query_start_lens[request_index])
            if context.query_start_lens is not None
            else key_length - query_length
        )
        current = hidden_states[q_start:q_end]
        suffix_length = full_length - query_start - query_length
        if query_start < 0 or suffix_length < 0:
            raise ValueError("HF-aligned linear received invalid full-context metadata")
        current = torch.cat(
            (
                current.new_zeros(query_start, current.shape[-1]),
                current,
                current.new_zeros(suffix_length, current.shape[-1]),
            ),
            dim=0,
        )
        projected = nn.functional.linear(current.unsqueeze(0), weight, bias)
        outputs.append(projected[0, query_start:query_start + query_length])
    return torch.cat(outputs, dim=0)


class Qwen3VLTextRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.to(torch.float32)
        variance = normalized.pow(2).mean(-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(input_dtype)


class Qwen3VLAttention(nn.Module):
    def __init__(self, config, *, attention_backend: str = "flash_attention_2") -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        total_num_heads = config.num_attention_heads
        total_num_kv_heads = config.num_key_value_heads
        if total_num_heads % tp_size or total_num_kv_heads % tp_size:
            raise ValueError("attention heads must be divisible by tensor parallel size")

        self.num_heads = total_num_heads // tp_size
        self.num_kv_heads = total_num_kv_heads // tp_size
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // total_num_heads,
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        attention_bias = getattr(config, "attention_bias", False)
        if attention_backend not in {"sdpa", "flash_attention_2"}:
            raise ValueError(f"unsupported Qwen3-VL language attention backend: {attention_backend}")
        self.attention_backend = attention_backend
        self.projection_mode = "separate" if attention_backend == "sdpa" else "fused"
        self.output_projection_mode = (
            "hf_aligned" if attention_backend == "sdpa" else "fused"
        )
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            total_num_heads,
            total_num_kv_heads,
            bias=attention_bias,
        )
        self.o_proj = RowParallelLinear(
            total_num_heads * self.head_dim,
            config.hidden_size,
            bias=attention_bias,
        )
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        rope_scaling = getattr(config, "rope_scaling", None) or {}
        sections = tuple(rope_scaling.get("mrope_section", ()))
        if len(sections) != 3:
            raise ValueError("Qwen3-VL requires three multimodal RoPE sections")
        if rope_scaling.get("mrope_interleaved", True) is not True:
            raise ValueError("Qwen3-VL requires interleaved multimodal RoPE")
        self.rotary_emb = MultimodalRotaryEmbedding(
            head_dim=self.head_dim,
            base=getattr(config, "rope_theta", 5_000_000.0),
            sections=sections,
            strategy="interleaved",
        )
        attention_class = SDPAAttention if attention_backend == "sdpa" else Attention
        self.attn = attention_class(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.projection_mode == "separate":
            q_weight, k_weight, v_weight = self.qkv_proj.weight.split(
                (self.q_size, self.kv_size, self.kv_size), dim=0
            )
            qkv = torch.cat(
                (
                    hf_aligned_linear(hidden_states, q_weight),
                    hf_aligned_linear(hidden_states, k_weight),
                    hf_aligned_linear(hidden_states, v_weight),
                ),
                dim=-1,
            )
        else:
            qkv = self.qkv_proj(hidden_states)
        query, key, value = qkv.split(
            (self.q_size, self.kv_size, self.kv_size),
            dim=-1,
        )
        query = query.view(-1, self.num_heads, self.head_dim)
        key = key.view(-1, self.num_kv_heads, self.head_dim)
        value = value.view(-1, self.num_kv_heads, self.head_dim)
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = self.rotary_emb(positions, query, key)
        output = self.attn(query, key, value)
        output = output.flatten(1, -1)
        if self.output_projection_mode == "hf_aligned":
            return hf_aligned_linear(output, self.o_proj.weight, self.o_proj.bias)
        return self.o_proj(output)


class Qwen3VLMLP(nn.Module):
    def __init__(self, config, *, attention_backend: str = "flash_attention_2") -> None:
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError(f"unsupported activation: {config.hidden_act}")
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )
        self.act_fn = SiluAndMul()
        self.projection_mode = (
            "separate_3d" if attention_backend == "sdpa" else "fused"
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.projection_mode == "separate_3d":
            gate_weight, up_weight = self.gate_up_proj.weight.chunk(2, dim=0)
            gate = hf_aligned_linear(hidden_states, gate_weight)
            up = hf_aligned_linear(hidden_states, up_weight)
            return hf_aligned_linear(
                nn.functional.silu(gate) * up,
                self.down_proj.weight,
                self.down_proj.bias,
            )
        return self.down_proj(self.act_fn(self.gate_up_proj(hidden_states)))


class Qwen3VLDecoderLayer(nn.Module):
    def __init__(self, config, *, attention_backend: str = "flash_attention_2") -> None:
        super().__init__()
        self.self_attn = Qwen3VLAttention(config, attention_backend=attention_backend)
        self.mlp = Qwen3VLMLP(config, attention_backend=attention_backend)
        self.input_layernorm = Qwen3VLTextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            hidden_states = hidden_states + residual
        attention_residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = attention_residual + self.self_attn(positions, hidden_states)
        mlp_residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = mlp_residual + self.mlp(hidden_states)
        return hidden_states, None


class Qwen3VLLanguageModel(nn.Module):
    deepstack_layer_count = 3

    def __init__(self, config, *, attention_backend: str = "flash_attention_2") -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            Qwen3VLDecoderLayer(config, attention_backend=attention_backend)
            for _ in range(config.num_hidden_layers)
        )
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _validate_inputs(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
    ) -> None:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        token_count = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        if input_ids is not None:
            if input_ids.ndim != 1:
                raise ValueError("input_ids must have shape [num_tokens]")
            if input_ids.dtype != torch.long:
                raise TypeError("input_ids must use torch.long dtype")
        else:
            if inputs_embeds.ndim != 2 or inputs_embeds.shape[1] != self.hidden_size:
                raise ValueError(
                    "inputs_embeds must have shape [num_tokens, hidden_size]"
                )
            if not inputs_embeds.is_floating_point():
                raise TypeError("inputs_embeds must use a floating-point dtype")
        if positions.ndim != 2 or positions.shape != (3, token_count):
            raise ValueError("positions must have shape [3, num_tokens]")
        if positions.dtype != torch.long:
            raise TypeError("positions must use torch.long dtype")

    def _validate_deepstack(
        self,
        hidden_states: torch.Tensor,
        visual_token_rows: torch.Tensor | None,
        deepstack_visual_embeds: tuple[torch.Tensor, ...],
    ) -> None:
        has_rows = visual_token_rows is not None
        has_features = bool(deepstack_visual_embeds)
        if has_rows != has_features:
            raise ValueError(
                "visual_token_rows and DeepStack embeddings must be provided together"
            )
        if not has_features:
            return
        if len(deepstack_visual_embeds) != self.deepstack_layer_count:
            raise ValueError("Qwen3-VL requires exactly three DeepStack embeddings")
        if len(deepstack_visual_embeds) > len(self.layers):
            raise ValueError("DeepStack embedding count exceeds the language layer count")
        if visual_token_rows.ndim != 1:
            raise ValueError("visual_token_rows must be rank-1")
        if visual_token_rows.dtype != torch.long:
            raise TypeError("visual_token_rows must use torch.long dtype")
        if visual_token_rows.device != hidden_states.device:
            raise ValueError("visual_token_rows must be on the hidden-state device")
        if visual_token_rows.numel():
            if visual_token_rows.is_cuda:
                # Trusted runtime contract: the replacement-plan builder supplies
                # sorted, unique, in-range rows. Reading CUDA values here would
                # synchronize every visual prefill and interfere with graph use.
                pass
            else:
                if torch.any(visual_token_rows < 0) or torch.any(
                    visual_token_rows >= hidden_states.shape[0]
                ):
                    raise IndexError("visual_token_rows contains an out-of-range row")
                if torch.unique(visual_token_rows).numel() != visual_token_rows.numel():
                    raise ValueError("visual_token_rows must not contain duplicate rows")

        expected_shape = (visual_token_rows.numel(), self.hidden_size)
        for layer_index, embedding in enumerate(deepstack_visual_embeds):
            if embedding.ndim != 2 or tuple(embedding.shape) != expected_shape:
                raise ValueError(
                    "DeepStack embedding shape mismatch at layer "
                    f"{layer_index}: expected {expected_shape}, got {tuple(embedding.shape)}"
                )
            if embedding.device != hidden_states.device:
                raise ValueError("DeepStack embeddings must be on the hidden-state device")
            if embedding.dtype != hidden_states.dtype:
                raise TypeError("DeepStack embeddings must match hidden-state dtype")

    @staticmethod
    def _add_deepstack_rows(
        hidden_states: torch.Tensor,
        visual_token_rows: torch.Tensor,
        visual_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if visual_token_rows.numel() == 0:
            return hidden_states
        selected = hidden_states.index_select(0, visual_token_rows)
        updated = selected + visual_embedding
        output = hidden_states.clone()
        output.index_copy_(0, visual_token_rows, updated)
        return output

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        *,
        inputs_embeds: torch.Tensor | None = None,
        visual_token_rows: torch.Tensor | None = None,
        deepstack_visual_embeds: tuple[torch.Tensor, ...] = (),
    ) -> torch.Tensor:
        self._validate_inputs(input_ids, positions, inputs_embeds)
        hidden_states = (
            self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        )
        self._validate_deepstack(
            hidden_states,
            visual_token_rows,
            deepstack_visual_embeds,
        )

        residual = None
        for layer_index, layer in enumerate(self.layers):
            hidden_states, residual = layer(positions, hidden_states, residual)
            if residual is not None:
                hidden_states = hidden_states + residual
                residual = None
            if layer_index < len(deepstack_visual_embeds):
                hidden_states = self._add_deepstack_rows(
                    hidden_states,
                    visual_token_rows,
                    deepstack_visual_embeds[layer_index],
                )

        return self.norm(hidden_states)


class Qwen3VLModel(nn.Module):
    def __init__(self, config, *, attention_backend: str = "flash_attention_2") -> None:
        super().__init__()
        self.language_model = Qwen3VLLanguageModel(
            config, attention_backend=attention_backend
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        *,
        inputs_embeds: torch.Tensor | None = None,
        visual_token_rows: torch.Tensor | None = None,
        deepstack_visual_embeds: tuple[torch.Tensor, ...] = (),
    ) -> torch.Tensor:
        return self.language_model(
            input_ids,
            positions,
            inputs_embeds=inputs_embeds,
            visual_token_rows=visual_token_rows,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )


class Qwen3VLForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    checkpoint_parameter_aliases = {
        "lm_head.weight": "model.language_model.embed_tokens.weight",
    }
    position_axes = 3

    def __init__(self, config, *, attention_backend: str = "flash_attention_2") -> None:
        super().__init__()
        if not config.tie_word_embeddings:
            raise ValueError("Qwen3-VL requires tied token embedding and LM head weights")
        self.model = Qwen3VLModel(config, attention_backend=attention_backend)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.language_model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        *,
        inputs_embeds: torch.Tensor | None = None,
        visual_token_rows: torch.Tensor | None = None,
        deepstack_visual_embeds: tuple[torch.Tensor, ...] = (),
    ) -> torch.Tensor:
        return self.model(
            input_ids,
            positions,
            inputs_embeds=inputs_embeds,
            visual_token_rows=visual_token_rows,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


__all__ = [
    "Qwen3VLAttention",
    "Qwen3VLDecoderLayer",
    "Qwen3VLForCausalLM",
    "Qwen3VLLanguageModel",
    "Qwen3VLMLP",
    "Qwen3VLModel",
    "Qwen3VLTextRMSNorm",
    "hf_aligned_linear",
]
