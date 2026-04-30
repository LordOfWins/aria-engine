"""ARIA Engine - LLM Provider Abstraction Layer

LiteLLM 기반 멀티 프로바이더 LLM 호출
- Provider Agnostic: Claude/GPT/Gemini/DeepSeek 한 줄 교체
- Cost Tracking: 요청별 비용 추적
- Fallback Chain: 장애 시 자동 대체 모델 전환 (모든 메서드 공통)
- KillSwitch: 비용 상한 초과 시 자동 차단
- Prompt Caching: 반복 시스템 프롬프트 캐싱으로 토큰 절감
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import litellm
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from aria.core.config import AriaConfig, get_config
from aria.core.exceptions import (
    KillSwitchError,
    LLMAllProvidersExhaustedError,
    LLMProviderError,
)

logger = structlog.get_logger()

# LiteLLM 전역 설정
litellm.drop_params = True  # 지원 안 하는 파라미터 자동 무시
litellm.set_verbose = False

# Retry 대상 예외 (rate limit / 일시적 서버 에러)
_RETRYABLE_EXCEPTIONS = (
    litellm.RateLimitError,
    litellm.ServiceUnavailableError,
    litellm.Timeout,
    litellm.InternalServerError,
)


@dataclass
class UsageRecord:
    """단일 API 호출 사용량 기록"""

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    cached_tokens: int = 0
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

        # Fallback 체인: 요청 모델 → fallback 모델
        # 각 tier별 fallback 순서를 결정
        self._fallback_chains: dict[str, list[str]] = {
            "default": [self.config.llm.default_model, self.config.llm.fallback_model],
            "heavy": [self.config.llm.default_model, self.config.llm.fallback_model],
            "cheap": [self.config.llm.cheap_model, self.config.llm.fallback_model],
            "fallback": [self.config.llm.fallback_model],
        }

    def _check_killswitch(self) -> None:
        """KillSwitch 체크 → 발동 시 KillSwitchError raise"""
        allowed, reason = self.cost_tracker.check_limits(self.config)
        if not allowed:
            logger.error("killswitch_triggered", reason=reason)
            raise KillSwitchError(
                reason,
                daily_cost=self.cost_tracker.daily_cost,
                monthly_cost=self.cost_tracker.monthly_cost,
            )

    def _record_usage(self, response: Any, target_model: str, latency_ms: float) -> UsageRecord:
        """API 응답에서 사용량 기록 추출 + CostTracker에 등록"""
        usage = response.usage
        cost = litellm.completion_cost(completion_response=response)

        # 캐시 토큰 추출 (Anthropic prompt caching)
        cached_tokens = 0
        if usage and hasattr(usage, "cache_read_input_tokens"):
            cached_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0

        record = UsageRecord(
            model=response.model or target_model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cost_usd=cost or 0.0,
            latency_ms=latency_ms,
            cached_tokens=cached_tokens,
        )
        self.cost_tracker.add(record)

        logger.info(
            "llm_call_success",
            model=record.model,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            cached_tokens=record.cached_tokens,
            cost_usd=f"${record.cost_usd:.4f}",
            latency_ms=f"{record.latency_ms:.0f}ms",
        )
        return record

    async def _call_llm_with_fallback(
        self,
        kwargs_base: dict[str, Any],
        model_tier: str,
        explicit_model: str | None = None,
    ) -> Any:
        """Fallback 체인을 따라 LLM 호출 시도

        Args:
            kwargs_base: litellm.acompletion에 전달할 파라미터 (model 제외)
            model_tier: 모델 티어
            explicit_model: 직접 지정 모델 (지정 시 해당 모델 → fallback 순서)

        Returns:
            litellm response object

        Raises:
            LLMAllProvidersExhaustedError: 모든 모델 실패
        """
        if explicit_model:
            chain = [explicit_model]
            if explicit_model != self.config.llm.fallback_model:
                chain.append(self.config.llm.fallback_model)
        else:
            chain = self._fallback_chains.get(model_tier, [self.config.llm.default_model])

        # 중복 제거 (동일 모델이 체인에 여러 번 있을 수 있음)
        seen = set()
        unique_chain = []
        for m in chain:
            if m not in seen:
                seen.add(m)
                unique_chain.append(m)

        attempts: list[dict] = []

        for i, model in enumerate(unique_chain):
            try:
                kwargs = {**kwargs_base, "model": model}
                response = await litellm.acompletion(**kwargs)
                return response

            except _RETRYABLE_EXCEPTIONS as e:
                # Rate limit / 서버 에러: 같은 모델 1회 재시도 후 다음 모델로
                logger.warning(
                    "llm_retryable_error",
                    model=model,
                    error_type=type(e).__name__,
                    error=str(e)[:200],
                    attempt=i + 1,
                )
                attempts.append({
                    "model": model,
                    "error_type": type(e).__name__,
                    "error": str(e)[:200],
                    "retryable": True,
                })

                # 같은 모델 1회 재시도 (exponential backoff)
                import asyncio
                await asyncio.sleep(min(2 ** i, 8))
                try:
                    response = await litellm.acompletion(**{**kwargs_base, "model": model})
                    return response
                except Exception:
                    if i < len(unique_chain) - 1:
                        logger.info("fallback_triggered", from_model=model, to_model=unique_chain[i + 1])
                    continue

            except litellm.AuthenticationError as e:
                # API 키 에러: 해당 프로바이더 스킵 → 다음 모델로
                logger.error("llm_auth_error", model=model, error=str(e)[:200])
                attempts.append({
                    "model": model,
                    "error_type": "AuthenticationError",
                    "error": str(e)[:200],
                    "retryable": False,
                })
                if i < len(unique_chain) - 1:
                    logger.info("fallback_triggered", from_model=model, to_model=unique_chain[i + 1])
                continue

            except litellm.BadRequestError as e:
                # 잘못된 요청 (컨텍스트 초과 등): 재시도 무의미
                logger.error("llm_bad_request", model=model, error=str(e)[:200])
                attempts.append({
                    "model": model,
                    "error_type": "BadRequestError",
                    "error": str(e)[:200],
                    "retryable": False,
                })
                # 컨텍스트 길이 초과는 다른 모델에서도 발생할 수 있지만 시도는 함
                if i < len(unique_chain) - 1:
                    logger.info("fallback_triggered", from_model=model, to_model=unique_chain[i + 1])
                continue

            except Exception as e:
                # 기타 예상 못한 에러
                logger.error("llm_unexpected_error", model=model, error_type=type(e).__name__, error=str(e)[:200])
                attempts.append({
                    "model": model,
                    "error_type": type(e).__name__,
                    "error": str(e)[:200],
                    "retryable": False,
                })
                if i < len(unique_chain) - 1:
                    logger.info("fallback_triggered", from_model=model, to_model=unique_chain[i + 1])
                continue

        # 모든 모델 실패
        raise LLMAllProvidersExhaustedError(attempts)

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
        cache_system_prompt: bool = False,
    ) -> dict[str, Any]:
        """LLM 호출 (자동 비용 추적 + KillSwitch + Fallback Chain)

        Args:
            prompt: 사용자 프롬프트
            model_tier: "default" | "heavy" | "cheap" | "fallback"
            model: 직접 모델명 지정 (tier보다 우선)
            system_prompt: 시스템 프롬프트
            max_tokens: 최대 출력 토큰
            temperature: 생성 온도
            response_format: 응답 포맷 (JSON mode 등)
            cache_system_prompt: True면 시스템 프롬프트에 캐시 마커 추가 (Anthropic 전용)

        Returns:
            {"content": str, "model": str, "usage": UsageRecord}

        Raises:
            KillSwitchError: 비용 상한 초과
            LLMAllProvidersExhaustedError: 모든 모델 실패
        """
        self._check_killswitch()

        # 메시지 구성
        messages: list[dict[str, Any]] = []
        if system_prompt:
            sys_msg: dict[str, Any] = {"role": "system", "content": system_prompt}
            if cache_system_prompt:
                # Anthropic prompt caching: cache_control 마커 추가
                sys_msg["content"] = [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            messages.append(sys_msg)
        messages.append({"role": "user", "content": prompt})

        # LLM 호출 파라미터
        kwargs_base: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.config.llm.max_tokens_per_request,
        }
        if response_format:
            kwargs_base["response_format"] = response_format

        start_time = time.time()
        response = await self._call_llm_with_fallback(kwargs_base, model_tier, model)
        latency_ms = (time.time() - start_time) * 1000

        record = self._record_usage(response, model or self._model_tiers.get(model_tier, ""), latency_ms)
        content = response.choices[0].message.content or ""

        return {
            "content": content,
            "model": record.model,
            "usage": record,
        }

    async def complete_with_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        model_tier: str = "default",
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """멀티턴 대화용 LLM 호출 (messages 직접 전달 + Fallback Chain)

        Raises:
            KillSwitchError: 비용 상한 초과
            LLMAllProvidersExhaustedError: 모든 모델 실패
        """
        self._check_killswitch()

        kwargs_base: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.config.llm.max_tokens_per_request,
        }

        start_time = time.time()
        response = await self._call_llm_with_fallback(kwargs_base, model_tier, model)
        latency_ms = (time.time() - start_time) * 1000

        record = self._record_usage(response, model or self._model_tiers.get(model_tier, ""), latency_ms)

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
            "total_cached_tokens": sum(r.cached_tokens for r in self.cost_tracker.records),
        }
