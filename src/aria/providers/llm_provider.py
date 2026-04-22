"""ARIA Engine - LLM Provider Abstraction Layer

LiteLLM 기반 멀티 프로바이더 LLM 호출
- Provider Agnostic: Claude/GPT/Gemini/DeepSeek 한 줄 교체
- Cost Tracking: 요청별 비용 추적
- Fallback: 장애 시 자동 대체 모델 전환
- KillSwitch: 비용 상한 초과 시 자동 차단
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import litellm
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from aria.core.config import AriaConfig, get_config

logger = structlog.get_logger()

# LiteLLM 전역 설정
litellm.drop_params = True  # 지원 안 하는 파라미터 자동 무시
litellm.set_verbose = False


@dataclass
class UsageRecord:
    """단일 API 호출 사용량 기록"""

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class CostTracker:
    """일별/월별 비용 추적기 (KillSwitch 연동)"""

    daily_cost: float = 0.0
    monthly_cost: float = 0.0
    current_date: str = ""
    current_month: str = ""
    records: list[UsageRecord] = field(default_factory=list)

    def add(self, record: UsageRecord) -> None:
        today = date.today()
        today_str = today.isoformat()
        month_str = today.strftime("%Y-%m")

        # 날짜 변경 시 리셋
        if self.current_date != today_str:
            self.daily_cost = 0.0
            self.current_date = today_str

        if self.current_month != month_str:
            self.monthly_cost = 0.0
            self.current_month = month_str

        self.daily_cost += record.cost_usd
        self.monthly_cost += record.cost_usd
        self.records.append(record)

    def check_limits(self, config: AriaConfig) -> tuple[bool, str]:
        """비용 상한 체크 → (허용여부, 사유)"""
        if self.daily_cost >= config.cost_control.daily_cost_limit_usd:
            return False, f"일일 비용 상한 도달: ${self.daily_cost:.2f} / ${config.cost_control.daily_cost_limit_usd}"
        if self.monthly_cost >= config.cost_control.monthly_cost_limit_usd:
            return False, f"월간 비용 상한 도달: ${self.monthly_cost:.2f} / ${config.cost_control.monthly_cost_limit_usd}"
        return True, "OK"


class LLMProvider:
    """LiteLLM 기반 멀티 프로바이더 LLM 클라이언트

    사용법:
        provider = LLMProvider()
        response = await provider.complete("오늘 날씨 어때?")
        response = await provider.complete("복잡한 분석", model_tier="heavy")
        response = await provider.complete("간단한 분류", model_tier="cheap")
    """

    def __init__(self, config: AriaConfig | None = None) -> None:
        self.config = config or get_config()
        self.cost_tracker = CostTracker()

        # 모델 티어 매핑
        self._model_tiers: dict[str, str] = {
            "default": self.config.llm.default_model,
            "heavy": self.config.llm.default_model,  # 복잡한 추론
            "cheap": self.config.llm.cheap_model,     # 분류/태깅/짧은 응답
            "fallback": self.config.llm.fallback_model,
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def complete(
        self,
        prompt: str,
        *,
        model_tier: str = "default",
        model: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """LLM 호출 (자동 비용 추적 + KillSwitch)

        Args:
            prompt: 사용자 프롬프트
            model_tier: "default" | "heavy" | "cheap" | "fallback"
            model: 직접 모델명 지정 (tier보다 우선)
            system_prompt: 시스템 프롬프트
            max_tokens: 최대 출력 토큰
            temperature: 생성 온도
            response_format: 응답 포맷 (JSON mode 등)

        Returns:
            {"content": str, "model": str, "usage": UsageRecord}

        Raises:
            RuntimeError: KillSwitch 발동 시
        """
        # KillSwitch 체크
        allowed, reason = self.cost_tracker.check_limits(self.config)
        if not allowed:
            logger.error("killswitch_triggered", reason=reason)
            raise RuntimeError(f"🛑 ARIA KillSwitch: {reason}")

        # 모델 결정
        target_model = model or self._model_tiers.get(model_tier, self.config.llm.default_model)

        # 메시지 구성
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # LiteLLM 호출
        start_time = time.time()

        try:
            kwargs: dict[str, Any] = {
                "model": target_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens or self.config.llm.max_tokens_per_request,
            }
            if response_format:
                kwargs["response_format"] = response_format

            response = await litellm.acompletion(**kwargs)

        except Exception as e:
            logger.warning("llm_call_failed", model=target_model, error=str(e))

            # Fallback 시도
            if target_model != self.config.llm.fallback_model:
                logger.info("fallback_triggered", from_model=target_model, to_model=self.config.llm.fallback_model)
                kwargs["model"] = self.config.llm.fallback_model
                response = await litellm.acompletion(**kwargs)
            else:
                raise

        latency_ms = (time.time() - start_time) * 1000

        # 사용량 기록
        usage = response.usage
        cost = litellm.completion_cost(completion_response=response)

        record = UsageRecord(
            model=response.model or target_model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cost_usd=cost or 0.0,
            latency_ms=latency_ms,
        )
        self.cost_tracker.add(record)

        content = response.choices[0].message.content or ""

        logger.info(
            "llm_call_success",
            model=record.model,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            cost_usd=f"${record.cost_usd:.4f}",
            latency_ms=f"{record.latency_ms:.0f}ms",
        )

        return {
            "content": content,
            "model": record.model,
            "usage": record,
        }

    async def complete_with_messages(
        self,
        messages: list[dict[str, str]],
        *,
        model_tier: str = "default",
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """멀티턴 대화용 LLM 호출 (messages 직접 전달)"""
        allowed, reason = self.cost_tracker.check_limits(self.config)
        if not allowed:
            raise RuntimeError(f"🛑 ARIA KillSwitch: {reason}")

        target_model = model or self._model_tiers.get(model_tier, self.config.llm.default_model)

        start_time = time.time()
        response = await litellm.acompletion(
            model=target_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens or self.config.llm.max_tokens_per_request,
        )
        latency_ms = (time.time() - start_time) * 1000

        usage = response.usage
        cost = litellm.completion_cost(completion_response=response)

        record = UsageRecord(
            model=response.model or target_model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cost_usd=cost or 0.0,
            latency_ms=latency_ms,
        )
        self.cost_tracker.add(record)

        return {
            "content": response.choices[0].message.content or "",
            "model": record.model,
            "usage": record,
        }

    def get_cost_summary(self) -> dict[str, Any]:
        """현재 비용 요약"""
        return {
            "daily_cost_usd": round(self.cost_tracker.daily_cost, 4),
            "monthly_cost_usd": round(self.cost_tracker.monthly_cost, 4),
            "daily_limit_usd": self.config.cost_control.daily_cost_limit_usd,
            "monthly_limit_usd": self.config.cost_control.monthly_cost_limit_usd,
            "total_requests": len(self.cost_tracker.records),
        }
