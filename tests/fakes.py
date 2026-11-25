"""Offline stand-in for the extraction LLM."""

from typing import Any


class FakeLLM:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any],
        timeout: float | None = None,
    ) -> str:
        self.calls.append(
            {"system": system, "user": user, "json_schema": json_schema, "timeout": timeout}
        )
        if not self._responses:
            raise AssertionError("FakeLLM has no responses left")
        return self._responses.pop(0)
