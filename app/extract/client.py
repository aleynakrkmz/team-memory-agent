"""LLM client for the one extraction call. Tests pass a fake; Ollama is optional."""

import os
from typing import Any, Protocol

from dotenv import load_dotenv

# Design §3.5: every LLM setup uses a 32,768-token context.
NUM_CTX = 32768


class LLMClient(Protocol):
    """One JSON completion. `system` is trusted. `user` contains the transcript as data."""

    def complete_json(self, *, system: str, user: str, json_schema: dict[str, Any]) -> str:
        """Return the model text, which should be one JSON object."""
        ...


class OllamaClient:
    """Ollama chat with a JSON schema. The model name comes from the caller, not from code."""

    def __init__(self, model: str, host: str = "http://localhost:11434", *, chat: Any = None):
        if not model.strip():
            raise ValueError("model name is empty")
        self.model = model.strip()
        self.host = host
        self._chat = chat

    def complete_json(self, *, system: str, user: str, json_schema: dict[str, Any]) -> str:
        chat = self._chat or self._default_chat()
        response = chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format=json_schema,
            options={"temperature": 0, "num_ctx": NUM_CTX},
        )
        message = response["message"] if isinstance(response, dict) else response.message
        content = message["content"] if isinstance(message, dict) else message.content
        if not isinstance(content, str):
            raise RuntimeError("Ollama returned no text")
        return content

    def _default_chat(self):
        from ollama import Client

        return Client(host=self.host).chat


def ollama_client_from_env() -> OllamaClient:
    """Build a client from AGENT_MODEL and OLLAMA_HOST. Never reads a key from source."""
    load_dotenv()
    model = os.environ.get("AGENT_MODEL", "").strip()
    if not model:
        raise RuntimeError(
            "AGENT_MODEL is empty. Set it in .env (see .env.example) to the Ollama model name."
        )
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").strip() or "http://localhost:11434"
    return OllamaClient(model=model, host=host)
