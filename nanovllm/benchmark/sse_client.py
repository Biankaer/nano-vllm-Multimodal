from __future__ import annotations

import codecs
import json
from dataclasses import dataclass
from typing import Any


class SSEProtocolError(RuntimeError):
    """The HTTP body is not a complete, valid OpenAI SSE stream."""


@dataclass(frozen=True, slots=True)
class SSEEvent:
    data: str
    event: str | None = None
    event_id: str | None = None


class SSEDecoder:
    """Incrementally frame SSE events independently of HTTP/TCP chunking."""

    def __init__(self) -> None:
        self._utf8 = codecs.getincrementaldecoder("utf-8")(errors="strict")
        self._buffer = ""
        self._data_lines: list[str] = []
        self._event: str | None = None
        self._event_id: str | None = None
        self._closed = False

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        if self._closed:
            raise SSEProtocolError("cannot feed a closed SSE decoder")
        try:
            self._buffer += self._utf8.decode(chunk, final=False)
        except UnicodeDecodeError as exc:
            raise SSEProtocolError("SSE stream contains invalid UTF-8") from exc
        return self._drain_complete_lines(final=False)

    def close(self) -> list[SSEEvent]:
        if self._closed:
            return []
        self._closed = True
        try:
            self._buffer += self._utf8.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            raise SSEProtocolError("SSE stream ended within a UTF-8 character") from exc
        events = self._drain_complete_lines(final=True)
        if self._data_lines:
            emitted = self._dispatch()
            if emitted is not None:
                events.append(emitted)
        return events

    def _drain_complete_lines(self, *, final: bool) -> list[SSEEvent]:
        events: list[SSEEvent] = []
        while True:
            boundary = self._next_line_boundary(final=final)
            if boundary is None:
                break
            index, width = boundary
            line = self._buffer[:index]
            self._buffer = self._buffer[index + width :]
            emitted = self._consume_line(line)
            if emitted is not None:
                events.append(emitted)
        if final and self._buffer:
            emitted = self._consume_line(self._buffer)
            self._buffer = ""
            if emitted is not None:
                events.append(emitted)
        return events

    def _next_line_boundary(self, *, final: bool) -> tuple[int, int] | None:
        for index, character in enumerate(self._buffer):
            if character == "\n":
                return index, 1
            if character == "\r":
                if index + 1 == len(self._buffer) and not final:
                    return None
                width = 2 if self._buffer[index + 1 : index + 2] == "\n" else 1
                return index, width
        return None

    def _consume_line(self, line: str) -> SSEEvent | None:
        if not line:
            return self._dispatch()
        if line.startswith(":"):
            return None
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "data":
            self._data_lines.append(value)
        elif field == "event":
            self._event = value
        elif field == "id" and "\x00" not in value:
            self._event_id = value
        return None

    def _dispatch(self) -> SSEEvent | None:
        if not self._data_lines:
            self._event = None
            return None
        emitted = SSEEvent(
            data="\n".join(self._data_lines),
            event=self._event,
            event_id=self._event_id,
        )
        self._data_lines = []
        self._event = None
        return emitted


@dataclass(frozen=True, slots=True)
class StreamUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class StreamObservation:
    output_text: str
    finish_reason: str | None
    request_start: float
    first_content_at: float
    stream_done_at: float
    content_chunk_times: tuple[float, ...]
    usage: StreamUsage | None
    prompt_tokens: int | None
    output_tokens: int | None
    token_count_source: str


class OpenAIStreamAccumulator:
    def __init__(self, *, request_start: float) -> None:
        self.request_start = request_start
        self.first_content_at: float | None = None
        self._content: list[str] = []
        self._content_times: list[float] = []
        self._finish_reason: str | None = None
        self._usage: StreamUsage | None = None
        self._done_at: float | None = None

    def consume(self, event: SSEEvent, *, received_at: float) -> None:
        if self._done_at is not None:
            raise SSEProtocolError("received an SSE event after [DONE]")
        if event.data == "[DONE]":
            self._done_at = received_at
            return
        try:
            payload: Any = json.loads(event.data)
        except (json.JSONDecodeError, TypeError) as exc:
            raise SSEProtocolError("SSE data is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise SSEProtocolError("OpenAI SSE JSON payload must be an object")
        if "error" in payload:
            error = payload["error"]
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise SSEProtocolError(f"OpenAI SSE error: {message or 'unknown error'}")
        self._consume_usage(payload.get("usage"))
        choices = payload.get("choices", [])
        if not isinstance(choices, list):
            raise SSEProtocolError("OpenAI SSE choices must be a list")
        for choice in choices:
            if not isinstance(choice, dict):
                raise SSEProtocolError("OpenAI SSE choice must be an object")
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                self._finish_reason = str(finish_reason)
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                raise SSEProtocolError("OpenAI SSE delta must be an object")
            content = delta.get("content")
            if content:
                if not isinstance(content, str):
                    raise SSEProtocolError("OpenAI SSE delta.content must be text")
                if self.first_content_at is None:
                    self.first_content_at = received_at
                self._content.append(content)
                self._content_times.append(received_at)

    def _consume_usage(self, usage: Any) -> None:
        if usage is None:
            return
        if not isinstance(usage, dict):
            raise SSEProtocolError("OpenAI SSE usage must be an object")
        try:
            parsed = StreamUsage(
                prompt_tokens=int(usage["prompt_tokens"]),
                completion_tokens=int(usage["completion_tokens"]),
                total_tokens=int(usage["total_tokens"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SSEProtocolError("OpenAI SSE usage has invalid token counts") from exc
        if min(parsed.prompt_tokens, parsed.completion_tokens, parsed.total_tokens) < 0:
            raise SSEProtocolError("OpenAI SSE usage token counts must be non-negative")
        self._usage = parsed

    def finish(self) -> StreamObservation:
        if self._done_at is None:
            raise SSEProtocolError("SSE stream ended without [DONE]")
        if self.first_content_at is None:
            raise SSEProtocolError("SSE stream ended without non-empty content")
        return StreamObservation(
            output_text="".join(self._content),
            finish_reason=self._finish_reason,
            request_start=self.request_start,
            first_content_at=self.first_content_at,
            stream_done_at=self._done_at,
            content_chunk_times=tuple(self._content_times),
            usage=self._usage,
            prompt_tokens=None if self._usage is None else self._usage.prompt_tokens,
            output_tokens=None if self._usage is None else self._usage.completion_tokens,
            token_count_source="usage" if self._usage is not None else "missing",
        )
