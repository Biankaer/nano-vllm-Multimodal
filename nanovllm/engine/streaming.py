from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class TokenEvent:
    seq_id: int
    token_id: int
    finished: bool
    finish_reason: str | None


@dataclass(frozen=True, slots=True)
class GenerationStreamEvent:
    request_index: int
    seq_id: int
    token_id: int
    text_delta: str
    finished: bool
    finish_reason: str | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class TokenizerLike(Protocol):
    def decode(self, token_ids: list[int], **kwargs) -> str:
        ...


class CumulativeTextDecoder:
    """Decode cumulative token ids and emit only the stable new suffix."""

    def __init__(self, tokenizer: TokenizerLike) -> None:
        self._tokenizer = tokenizer
        self._token_ids: list[int] = []
        self._decoded = ""

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(self._token_ids)

    @property
    def text(self) -> str:
        return self._decoded

    def append(self, token_id: int) -> str:
        self._token_ids.append(token_id)
        decoded = self._tokenizer.decode(
            self._token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not decoded.startswith(self._decoded):
            raise RuntimeError(
                "cumulative tokenizer decode changed an already emitted prefix"
            )
        delta = decoded[len(self._decoded) :]
        self._decoded = decoded
        return delta
