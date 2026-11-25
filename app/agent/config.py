"""Limits for the agent loop. The model name is not a secret."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "agent.json"

# One JSON call. Long enough for a meeting's candidates, short enough to stop a runaway reply.
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 120.0
DEFAULT_NUM_PREDICT = 4096


class AgentConfig(BaseModel):
    """Starting limits from design §3.5. Tune on the development set, then freeze."""

    model_config = ConfigDict(extra="ignore")

    max_steps: int = Field(default=6, ge=1)
    time_budget_seconds: float = Field(default=300, gt=0)
    max_invalid_outputs: int = Field(default=2, ge=1)
    model: str = "qwen3:8b"
    search_max_results: int = Field(default=5, ge=1, le=20)
    think: bool = False
    ollama_timeout_seconds: float = Field(default=DEFAULT_OLLAMA_TIMEOUT_SECONDS, gt=0)
    num_predict: int = Field(default=DEFAULT_NUM_PREDICT, ge=1)


def load_agent_config(path: Path | None = None) -> AgentConfig:
    config_path = path or CONFIG_PATH
    data = json.loads(config_path.read_text(encoding="utf-8"))
    config = AgentConfig.model_validate(data)
    if not config.model.strip():
        raise ValueError("agent model name is empty")
    return config


def resolve_agent_model() -> str:
    """AGENT_MODEL when it is set to a name. Unset uses config/agent.json.

    A blank value is an error, so an empty .env entry is not silently replaced.
    """
    load_dotenv()
    raw = os.environ.get("AGENT_MODEL")
    if raw is None:
        return load_agent_config().model.strip()
    model = raw.strip()
    if not model:
        raise RuntimeError(
            "AGENT_MODEL is empty. Set it in .env (see .env.example) to the Ollama model name. "
            "The team default is qwen3:8b."
        )
    return model
