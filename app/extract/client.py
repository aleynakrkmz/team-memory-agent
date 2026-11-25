"""LLM client for the one extraction call. Tests pass a fake; Ollama is optional."""

import os
import re
from typing import Any, Protocol

from dotenv import load_dotenv

from app.agent.config import (
    DEFAULT_NUM_PREDICT,
    DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    load_agent_config,
    resolve_agent_model,
)

# Design §3.5: every LLM setup uses a 32,768-token context.
NUM_CTX = 32768

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


class OllamaTimeout(TimeoutError):
    """One Ollama call exceeded its per-call limit."""

    def __init__(self, timeout: float):
        self.timeout = float(timeout)
        super().__init__(f"Ollama call timed out after {self.timeout:g} seconds")


class LLMClient(Protocol):
    """One JSON completion. `system` is trusted. `user` contains the transcript as data."""

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any],
        timeout: float | None = None,
    ) -> str:
        """Return the model text, which should be one JSON object.

        `timeout`, when set, is the most seconds this call may take.
        """
        ...


def strip_thinking(text: str) -> str:
    """Remove qwen-style reasoning so a `<think>` block cannot break JSON parsing."""
    cleaned = _THINK_BLOCK.sub("", text)
    opened = _THINK_OPEN.search(cleaned)
    if opened:
        cleaned = cleaned[: opened.start()]
    return cleaned.strip()


class OllamaClient:
    """Ollama chat with a JSON schema. The model name comes from the caller, not from code.

    Thinking is off unless `think` is set. Each call uses an HTTP timeout, and
    `num_predict` caps how many tokens a JSON reply can run to.
    """

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        *,
        chat: Any = None,
        think: bool = False,
        timeout: float = DEFAULT_OLLAMA_TIMEOUT_SECONDS,
        num_predict: int = DEFAULT_NUM_PREDICT,
    ):
        if not model.strip():
            raise ValueError("model name is empty")
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0")
        if num_predict < 1:
            raise ValueError("num_predict must be at least 1")
        self.model = model.strip()
        self.host = host
        self._chat = chat
        self.think = bool(think)
        self.timeout_seconds = float(timeout)
        self.num_predict = int(num_predict)

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any],
        timeout: float | None = None,
    ) -> str:
        limit = self._effective_timeout(timeout)
        try:
            response = self._send(limit, system, user, json_schema)
        except OllamaTimeout:
            raise
        except TimeoutError as exc:
            raise OllamaTimeout(limit) from exc
        except Exception as exc:
            if _is_http_timeout(exc):
                raise OllamaTimeout(limit) from exc
            raise
        message = response["message"] if isinstance(response, dict) else response.message
        content = message["content"] if isinstance(message, dict) else message.content
        if not isinstance(content, str):
            raise RuntimeError("Ollama returned no text")
        return strip_thinking(content)

    def _effective_timeout(self, requested: float | None) -> float:
        limit = self.timeout_seconds
        if requested is None:
            return limit
        if requested <= 0:
            raise OllamaTimeout(requested)
        return min(limit, float(requested))

    def _send(self, limit: float, system: str, user: str, json_schema: dict[str, Any]):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": json_schema,
            "options": {"temperature": 0, "num_ctx": NUM_CTX, "num_predict": self.num_predict},
            "think": self.think,
        }
        # `timeout` is the HTTP limit. The ollama chat method has no such argument;
        # a fake chat records it, and the real client gets it on the HTTP transport.
        if self._chat is not None:
            return self._chat(**payload, timeout=limit)
        from ollama import Client

        return Client(host=self.host, timeout=limit).chat(**payload)


def ollama_client_from_env() -> OllamaClient:
    """Build a client from AGENT_MODEL and OLLAMA_HOST. Never reads a key from source.

    A blank AGENT_MODEL is an error. If the variable is unset, the model in
    config/agent.json is used (qwen3:8b). OLLAMA_THINK, OLLAMA_TIMEOUT_S, and
    OLLAMA_NUM_PREDICT override the same file. Thinking defaults to off.
    """
    load_dotenv()
    model = resolve_agent_model()
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").strip() or "http://localhost:11434"
    config = load_agent_config()
    think = _env_bool("OLLAMA_THINK")
    timeout = _env_float("OLLAMA_TIMEOUT_S")
    num_predict = _env_int("OLLAMA_NUM_PREDICT")
    return OllamaClient(
        model=model,
        host=host,
        think=config.think if think is None else think,
        timeout=config.ollama_timeout_seconds if timeout is None else timeout,
        num_predict=config.num_predict if num_predict is None else num_predict,
    )


def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip().lower()
    if not value:
        return None
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise RuntimeError(f"{name} must be true or false, got {raw!r}")


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw.strip())
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number of seconds, got {raw!r}") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than 0, got {raw!r}")
    return value


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be at least 1, got {raw!r}")
    return value


def _is_http_timeout(exc: BaseException) -> bool:
    try:
        import httpx
    except ImportError:
        return False
    return isinstance(exc, httpx.TimeoutException)
