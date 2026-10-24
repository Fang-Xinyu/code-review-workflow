from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_openai import ChatOpenAI

from .config import MAX_MODEL_CALLS, ModelConfig
from .models import ModelAttemptStats, ModelCallStats


@dataclass
class ModelResult:
    content: str
    tool_calls: list[dict[str, Any]]
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None
    reasoning_tokens: int | None = None


class ModelLike(Protocol):
    def invoke(self, system: str, user: str, tools: list[dict[str, Any]] | None = None) -> ModelResult: ...


class ModelRequestError(RuntimeError):
    def __init__(self, message: str, attempts: int = 1):
        super().__init__(message)
        self.attempts = attempts


class DeepSeekModel:
    """Small adapter that never puts the API key in state, reports, or errors."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = ChatOpenAI(api_key=config.api_key, base_url=config.base_url, model=config.model,
                                 timeout=config.timeout, max_tokens=config.max_tokens,
                                 max_retries=0, temperature=0,
                                 extra_body={"thinking": {"type": "disabled"}})
        self.attempts = 0
        self.request_stats: list[ModelAttemptStats] = []

    def invoke(self, system: str, user: str, tools: list[dict[str, Any]] | None = None) -> ModelResult:
        client: Any = self.client
        if tools:
            openai_tools = [{"type": "function", "function": t} for t in tools]
            client = client.bind_tools(openai_tools)
        last: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            self.attempts += 1
            started = time.perf_counter()
            result = None
            try:
                response = client.invoke([("system", system), ("human", user)])
                usage = getattr(response, "usage_metadata", None) or {}
                calls = []
                for call in getattr(response, "tool_calls", []) or []:
                    calls.append({"id": call.get("id", ""), "name": call.get("name", ""), "args": call.get("args", {})})
                # LangChain separates malformed JSON arguments from parsed tool calls.
                # Preserve them so the executor can return a bounded, structured error.
                for call in getattr(response, "invalid_tool_calls", []) or []:
                    calls.append({"id": call.get("id", ""), "name": call.get("name") or "", "args": call.get("args", "")})
                metadata = getattr(response, "response_metadata", None) or {}
                token_usage = metadata.get("token_usage") or {}
                details = usage.get("output_token_details") or token_usage.get("completion_tokens_details") or {}
                result = ModelResult(content=str(response.content or ""), tool_calls=calls,
                    input_tokens=usage.get("input_tokens", token_usage.get("prompt_tokens")),
                    output_tokens=usage.get("output_tokens", token_usage.get("completion_tokens")),
                    finish_reason=metadata.get("finish_reason"),
                    reasoning_tokens=details.get("reasoning", details.get("reasoning_tokens")))
                return result
            except Exception as exc:  # library exception classes differ by version
                last = exc
                status = _status_code(exc)
                retryable = status in {408, 429, 500, 502, 503, 504} or status is None
                if not retryable or attempt >= self.config.max_retries:
                    kind = "配置或请求错误" if status in {400, 401, 402, 403, 404} else "模型服务不可用"
                    raise ModelRequestError(f"{kind}（HTTP {status or '未知'}）", attempt + 1) from None
            finally:
                self.request_stats.append(_attempt_stats(started, result))
            time.sleep(0.2 * (2 ** attempt))
        raise ModelRequestError("模型服务不可用", self.config.max_retries + 1) from last


def _status_code(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", None)
    if isinstance(value, int):
        return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def parse_json_object(text: str) -> dict[str, Any]:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("模型 JSON 顶层必须是对象")
    return data


def test_connection(model: ModelLike) -> str:
    result = model.invoke("你是连接测试器。只返回 JSON。", 'Return exactly {"ok":true}.')
    data = parse_json_object(result.content)
    if data.get("ok") is not True:
        raise ModelRequestError("连接成功但返回格式不符")
    return "连接成功"


def _attempt_stats(started: float, result: ModelResult | None) -> ModelAttemptStats:
    return ModelAttemptStats(elapsed_ms=int((time.perf_counter() - started) * 1000),
        input_tokens=result.input_tokens if result else None,
        output_tokens=result.output_tokens if result else None,
        answer_chars=len(result.content) if result else 0,
        finish_reason=result.finish_reason if result else None,
        reasoning_tokens=result.reasoning_tokens if result else None, failed=result is None)


class ModelBudgetExceeded(RuntimeError):
    """The operation cannot issue another logical model call."""


class RecordingModel:
    """Per-run metadata only: no prompts, keys, source or private reasoning retained."""

    def __init__(self, model: ModelLike, max_calls: int = MAX_MODEL_CALLS):
        self.model = model
        self.max_calls = max_calls
        self.calls: list[ModelCallStats] = []
        self.stage = "review"

    @property
    def remaining(self) -> int:
        return max(0, self.max_calls - len(self.calls))

    @property
    def attempts(self) -> int:
        return sum(x.http_attempts for x in self.calls)

    def invoke(self, system: str, user: str, tools=None) -> ModelResult:
        if not self.remaining:
            raise ModelBudgetExceeded(f"本次操作已达到 {self.max_calls} 次模型调用上限")
        started = time.perf_counter()
        before = getattr(self.model, "attempts", 0)
        offset = len(getattr(self.model, "request_stats", []))
        result = None
        try:
            result = self.model.invoke(system, user, tools)
            return result
        finally:
            attempts = list(getattr(self.model, "request_stats", [])[offset:])
            count = getattr(self.model, "attempts", before + 1) - before
            stats = _attempt_stats(started, result).model_dump()
            if attempts:
                for field in ("input_tokens", "output_tokens", "reasoning_tokens"):
                    values = [getattr(x, field) for x in attempts if getattr(x, field) is not None]
                    stats[field] = sum(values) if values else None
            self.calls.append(ModelCallStats(**stats, stage=self.stage,
                http_attempts=max(1, count), attempts=attempts))

    def totals(self) -> dict[str, Any]:
        totals = {"model_calls": len(self.calls), "http_attempts": self.attempts}
        for field in ("input_tokens", "output_tokens"):
            values = [getattr(x, field) for x in self.calls if getattr(x, field) is not None]
            totals[field] = sum(values) if values else None
        return totals
