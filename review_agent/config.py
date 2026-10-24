from __future__ import annotations

import os
from dataclasses import dataclass


MAX_SOURCE_LINES = 2000
MAX_SOURCE_BYTES = 100 * 1024
MAX_MODEL_CALLS = 8
MAX_PLAN_ROUNDS = 5
MAX_TOOL_CALLS_PER_ROUND = 5
MAX_TOOL_OUTPUT_CHARS = 6000
MAX_EVIDENCE_CHARS = 8000


@dataclass(frozen=True)
class ModelConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-flash"
    timeout: float = 40.0
    max_tokens: int = 4096
    max_retries: int = 2

    @classmethod
    def from_env(cls) -> "ModelConfig":
        return cls(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        )

    def public_dict(self) -> dict[str, object]:
        return {"base_url": self.base_url, "model": self.model, "timeout": self.timeout,
                "max_tokens": self.max_tokens}
