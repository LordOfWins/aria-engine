"""ARIA Engine - LLM Provider Abstraction Layer

LiteLLM 기반 멀티 프로바이더 LLM 호출
- Provider Agnostic: Claude/GPT/Gemini/DeepSeek 한 줄 교체
- Cost Tracking: 요청별 비용 추적
- Fallback Chain: 장애 시 자동 대체 모델 전환 (모든 메서드 공통)
- KillSwitch: 비용 상한 초과 시 자동 차단
- Prompt Caching: 반복 시스템 프롬프트 캐싱으로 토큰 절감
- Dynamic Boundary: 고정/동적 시스템 프롬프트 분리 (SYSTEM_PROMPT_DYNAMIC_BOUNDARY)
    → 고정 영역만 캐시하여 동적 컨텍스트(메모리 등) 변경 시에도 캐시 유지
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
    NoAPIKeyError,
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

        # 키 없는 모델 경고 로그
        missing = self.config.get_missing_key_models()
        if missing:
            for model, env_var in missing:
                logger.warning(
                    "api_key_missing",
                    model=model,
                    env_var=env_var,
                    hint=f".env에 {env_var}를 설정하세요",
                )

        available = self.config.get_available_models()
        if not available:
            logger.error(
                "no_api_keys_configured",
                message="사용 가능한 API 키가 없습니다. 모든 LLM 호출이 실패합니다.",
                configured_models=[
                    self.config.llm.default_model,
                    self.config.llm.fallback_model,
                    self.config.llm.cheap_model,
                ],
            )
        else:
            logger.info("llm_provider_ready", available_models=available)

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
            NoAPIKeyError: 체인 내 모든 모델의 API 키 미설정
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

        # 키 없는 모델 사전 감지: 체인 내 모든 모델이 키 미설정이면 즉시 실패
        models_without_keys = [
            m for m in unique_chain
            if not self.config.has_api_key_for_model(m)
        ]
        if len(models_without_keys) == len(unique_chain):
            # 모든 모델이 키 미설정 → API 호출 시도 없이 명확한 에러
            missing_info = self.config.get_missing_key_models()
            if missing_info:
                model, env_var = missing_info[0]
                raise NoAPIKeyError(model=model, env_var=env_var)
            # fallback: get_missing_key_models가 빈 경우 (매핑에 없는 모델)
            raise NoAPIKeyError(model=unique_chain[0], env_var="UNKNOWN_KEY")

        # 키 있는 모델을 우선 시도하도록 정렬 (키 없는 모델은 뒤로)
        unique_chain.sort(key=lambda m: 0 if self.config.has_api_key_for_model(m) else 1)

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
        system_prompt_dynamic: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        response_format: dict[str, Any] | None = None,
        cache_system_prompt: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """LLM 호출 (자동 비용 추적 + KillSwitch + Fallback Chain)

        Args:
            prompt: 사용자 프롬프트
            model_tier: "default" | "heavy" | "cheap" | "fallback"
            model: 직접 모델명 지정 (tier보다 우선)
            system_prompt: 시스템 프롬프트 (고정 영역 — 캐시 대상)
            system_prompt_dynamic: 시스템 프롬프트 동적 영역 (캐시 경계 이후 — 매 턴 변경 가능)
                cache_system_prompt=True일 때만 의미 있음.
                고정 영역에 cache_control을 걸고 동적 영역은 캐시 밖에 배치하여
                고정 부분만 90% 비용 절감 (SYSTEM_PROMPT_DYNAMIC_BOUNDARY 패턴)
            max_tokens: 최대 출력 토큰
            temperature: 생성 온도
            response_format: 응답 포맷 (JSON mode 등)
            cache_system_prompt: True면 시스템 프롬프트에 캐시 마커 추가 (Anthropic 전용)
            tools: LLM function calling 도구 목록
                [{"type": "function", "function": {"name": ..., "parameters": ...}}, ...]
                None이면 도구 미사용 (기존 동작 유지)

        Returns:
            {"content": str, "model": str, "usage": UsageRecord,
             "tool_calls": list[dict] | None}

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
                # SYSTEM_PROMPT_DYNAMIC_BOUNDARY 패턴:
                # [static block + cache_control] → 캐시됨 (90% 비용 절감)
                # [dynamic block] → 캐시 경계 바깥 (매 턴 변경 가능)
                content_blocks: list[dict[str, Any]] = [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
                if system_prompt_dynamic:
                    content_blocks.append({
                        "type": "text",
                        "text": system_prompt_dynamic,
                    })
                sys_msg["content"] = content_blocks
            elif system_prompt_dynamic:
                # 캐싱 비활성 상태에서도 동적 영역은 시스템 프롬프트에 합산
                sys_msg["content"] = f"{system_prompt}\n\n{system_prompt_dynamic}"
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
        if tools:
            kwargs_base["tools"] = tools

        start_time = time.time()
        response = await self._call_llm_with_fallback(kwargs_base, model_tier, model)
        latency_ms = (time.time() - start_time) * 1000

        record = self._record_usage(response, model or self._model_tiers.get(model_tier, ""), latency_ms)
        content = response.choices[0].message.content or ""
        tool_calls = self._extract_tool_calls(response)

        return {
            "content": content,
            "model": record.model,
            "usage": record,
            "tool_calls": tool_calls,
        }

    async def complete_with_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        model_tier: str = "default",
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """멀티턴 대화용 LLM 호출 (messages 직접 전달 + Fallback Chain)

        Args:
            messages: 대화 메시지 목록 (role/content/tool_calls/tool_call_id)
            tools: LLM function calling 도구 목록 (None이면 도구 미사용)

        Returns:
            {"content": str, "model": str, "usage": UsageRecord,
             "tool_calls": list[dict] | None}

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
        if tools:
            kwargs_base["tools"] = tools

        start_time = time.time()
        response = await self._call_llm_with_fallback(kwargs_base, model_tier, model)
        latency_ms = (time.time() - start_time) * 1000

        record = self._record_usage(response, model or self._model_tiers.get(model_tier, ""), latency_ms)
        tool_calls = self._extract_tool_calls(response)

        return {
            "content": response.choices[0].message.content or "",
            "model": record.model,
            "usage": record,
            "tool_calls": tool_calls,
        }

    @staticmethod
    def _extract_tool_calls(response: Any) -> list[dict[str, Any]] | None:
        """LLM 응답에서 tool_calls 추출

        LiteLLM은 provider별 tool_call 포맷을 통일된 형식으로 반환.
        None이면 도구 호출 없음 (일반 텍스트 응답).
        """
        message = response.choices[0].message
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            return None

        return [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in message.tool_calls
        ]

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
