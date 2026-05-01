"""ARIA Engine - Tool Registry

도구 등록/발견/실행을 관리하는 중앙 레지스트리
- register / unregister: 도구 등록/제거
- get / list_tools: 도구 조회
- to_llm_tools: LiteLLM function calling 포맷 변환
- execute: Critic 평가 → 실행 (통합 파이프라인)

Critic 연동 플로우:
    execute() → Critic 활성화?
        → evaluate_and_enforce(ToolAction)
            → SAFE → executor.execute()
            → NEEDS_CONFIRMATION → ToolResult(pending=True) 반환
            → UNSAFE → ToolExecutionBlockedError raise
        → Critic 비활성화? → 바로 executor.execute()
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import structlog

from aria.core.exceptions import AriaError, ToolExecutionBlockedError
from aria.tools.critic import CriticEvaluator
from aria.tools.critic_types import CriticConfig, SafetyLevel, ToolAction
from aria.tools.pending_store import PendingAction, PendingStore
from aria.tools.tool_types import (
    ToolCategory,
    ToolDefinition,
    ToolExecutor,
    ToolResult,
)

logger = structlog.get_logger()


class ToolNotFoundError(AriaError):
    """등록되지 않은 도구 호출 시 발생"""

    def __init__(self, tool_name: str) -> None:
        super().__init__(
            f"등록되지 않은 도구: '{tool_name}'",
            code="TOOL_NOT_FOUND",
            details={"tool_name": tool_name},
        )


class ToolAlreadyRegisteredError(AriaError):
    """이미 등록된 도구명으로 재등록 시도 시 발생"""

    def __init__(self, tool_name: str) -> None:
        super().__init__(
            f"이미 등록된 도구: '{tool_name}'",
            code="TOOL_ALREADY_REGISTERED",
            details={"tool_name": tool_name},
        )


class ToolParameterError(AriaError):
    """도구 파라미터 검증 실패"""

    def __init__(self, tool_name: str, message: str) -> None:
        super().__init__(
            f"도구 '{tool_name}' 파라미터 오류: {message}",
            code="TOOL_PARAMETER_ERROR",
            details={"tool_name": tool_name, "validation_error": message},
        )


class ToolRegistry:
    """도구 중앙 레지스트리

    도구 등록/조회/실행을 관리하며 Critic 평가를 실행 전에 자동 수행

    Args:
        critic: CriticEvaluator 인스턴스 (None이면 Critic 비활성 → 모든 도구 즉시 실행)

    사용법:
        registry = ToolRegistry(critic=critic_evaluator)
        registry.register_executor(my_tool_executor)
        result = await registry.execute("my_tool", {"param": "value"}, context="대화 요약")
    """

    def __init__(
        self,
        critic: CriticEvaluator | None = None,
        pending_store: PendingStore | None = None,
    ) -> None:
        self._tools: dict[str, tuple[ToolDefinition, ToolExecutor]] = {}
        self._critic = critic
        self._pending_store = pending_store or PendingStore()

    # === 등록/제거 ===

    def register(self, definition: ToolDefinition, executor: ToolExecutor) -> None:
        """도구 등록

        Args:
            definition: 도구 스펙
            executor: 도구 실행기

        Raises:
            ToolAlreadyRegisteredError: 동일 이름 도구가 이미 등록됨
        """
        if definition.name in self._tools:
            raise ToolAlreadyRegisteredError(definition.name)

        self._tools[definition.name] = (definition, executor)
        logger.info(
            "tool_registered",
            name=definition.name,
            category=definition.category.value,
            safety_hint=definition.safety_hint.value,
        )

    def register_executor(self, executor: ToolExecutor) -> None:
        """executor에서 definition을 자동 추출하여 등록

        ToolExecutor.get_definition()을 호출하여 ToolDefinition을 얻음
        → register(definition, executor) 호출

        Args:
            executor: get_definition()을 구현한 ToolExecutor
        """
        definition = executor.get_definition()
        self.register(definition, executor)

    def unregister(self, name: str) -> None:
        """도구 제거

        Args:
            name: 도구 이름

        Raises:
            ToolNotFoundError: 등록되지 않은 도구
        """
        if name not in self._tools:
            raise ToolNotFoundError(name)

        del self._tools[name]
        logger.info("tool_unregistered", name=name)

    # === 조회 ===

    def get(self, name: str) -> tuple[ToolDefinition, ToolExecutor]:
        """이름으로 도구 조회

        Returns:
            (ToolDefinition, ToolExecutor) 튜플

        Raises:
            ToolNotFoundError: 등록되지 않은 도구
        """
        if name not in self._tools:
            raise ToolNotFoundError(name)
        return self._tools[name]

    def list_tools(
        self,
        *,
        category: ToolCategory | None = None,
        enabled_only: bool = True,
    ) -> list[ToolDefinition]:
        """등록된 도구 목록 반환

        Args:
            category: 특정 카테고리만 필터 (None이면 전체)
            enabled_only: True면 활성화된 도구만 (기본)
        """
        definitions: list[ToolDefinition] = []
        for defn, _ in self._tools.values():
            if enabled_only and not defn.enabled:
                continue
            if category is not None and defn.category != category:
                continue
            definitions.append(defn)
        return definitions

    def has_tool(self, name: str) -> bool:
        """도구 등록 여부 확인"""
        return name in self._tools

    @property
    def tool_count(self) -> int:
        """등록된 도구 수"""
        return len(self._tools)

    # === LLM Function Calling 포맷 변환 ===

    def to_llm_tools(self, *, enabled_only: bool = True) -> list[dict[str, Any]]:
        """등록된 도구를 LiteLLM function calling 포맷으로 변환

        ReAct 에이전트가 LLM에게 사용 가능한 도구 목록을 전달할 때 사용

        Returns:
            [{"type": "function", "function": {"name": ..., ...}}, ...]
        """
        return [
            defn.to_llm_tool()
            for defn in self.list_tools(enabled_only=enabled_only)
        ]

    # === 실행 ===

    async def execute(
        self,
        name: str,
        parameters: dict[str, Any],
        *,
        context: str = "",
        skip_critic: bool = False,
    ) -> ToolResult:
        """도구 실행 (Critic 평가 → 실행)

        Args:
            name: 도구 이름
            parameters: 실행 파라미터
            context: 실행 컨텍스트 (대화 이력 요약 — Critic에 전달)
            skip_critic: True면 Critic 건너뛰기 (테스트/내부 호출용)

        Returns:
            ToolResult

        Raises:
            ToolNotFoundError: 등록되지 않은 도구
            ToolParameterError: 파라미터 검증 실패
            ToolExecutionBlockedError: Critic이 UNSAFE로 판단하여 차단
        """
        # 1. 도구 조회
        definition, executor = self.get(name)

        # 2. 활성화 체크
        if not definition.enabled:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"도구 '{name}'이(가) 비활성화 상태입니다",
            )

        # 3. 필수 파라미터 검증
        self._validate_parameters(definition, parameters)

        # 4. Critic 평가
        if not skip_critic and self._critic is not None:
            critic_result = await self._run_critic(definition, parameters, context)
            if critic_result is not None:
                # NEEDS_CONFIRMATION → pending result 반환
                return critic_result
            # SAFE → 계속 진행 / UNSAFE → ToolExecutionBlockedError가 이미 raise됨

        # 5. 실행
        start_time = time.time()
        try:
            result = await executor.execute(parameters)
            result.latency_ms = (time.time() - start_time) * 1000
            logger.info(
                "tool_executed",
                name=name,
                success=result.success,
                latency_ms=f"{result.latency_ms:.0f}ms",
            )
            return result
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(
                "tool_execution_failed",
                name=name,
                error=str(e)[:200],
                latency_ms=f"{latency_ms:.0f}ms",
            )
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"도구 실행 중 오류: {str(e)[:500]}",
                latency_ms=latency_ms,
            )

    async def execute_pending(self, confirmation_id: str) -> ToolResult:
        """승인된 대기 액션 실행

        PendingStore에서 확인 ID로 액션을 조회하여 실행
        Critic 재평가 없이 직접 실행 (이미 사용자가 승인함)

        Args:
            confirmation_id: 확인 ID

        Returns:
            ToolResult: 실행 결과

        Raises:
            ToolNotFoundError: 대기 액션 없음 또는 만료
        """
        action = self._pending_store.get(confirmation_id)
        if action is None:
            raise ToolNotFoundError(f"pending:{confirmation_id}")

        # 도구 존재 여부 확인
        if action.tool_name not in self._tools:
            self._pending_store.remove(confirmation_id)
            raise ToolNotFoundError(action.tool_name)

        definition, executor = self._tools[action.tool_name]

        # 대기 액션 삭제 (1회 실행 후 폐기)
        self._pending_store.remove(confirmation_id)

        # Critic 스킵 — 사용자가 이미 승인
        start = time.monotonic()
        try:
            result = await executor.execute(action.parameters)
            result.latency_ms = (time.monotonic() - start) * 1000
            logger.info(
                "pending_tool_executed",
                confirmation_id=confirmation_id,
                tool_name=action.tool_name,
                success=result.success,
            )
            return result
        except Exception as e:
            latency_ms = (time.monotonic() - start) * 1000
            logger.error(
                "pending_tool_execution_failed",
                confirmation_id=confirmation_id,
                tool_name=action.tool_name,
                error=str(e)[:200],
            )
            return ToolResult(
                tool_name=action.tool_name,
                success=False,
                error=f"도구 실행 중 오류: {str(e)[:500]}",
                latency_ms=latency_ms,
            )

    def deny_pending(self, confirmation_id: str) -> bool:
        """대기 액션 거부 (삭제)

        Returns:
            True if action existed and was removed
        """
        removed = self._pending_store.remove(confirmation_id)
        if removed:
            logger.info("pending_tool_denied", confirmation_id=confirmation_id)
        return removed

    @property
    def pending_store(self) -> PendingStore:
        """PendingStore 인스턴스 접근"""
        return self._pending_store

    # === Private Methods ===

    def _validate_parameters(
        self,
        definition: ToolDefinition,
        parameters: dict[str, Any],
    ) -> None:
        """필수 파라미터 존재 여부 검증

        Raises:
            ToolParameterError: 필수 파라미터 누락
        """
        required_names = {p.name for p in definition.parameters if p.required}
        provided_names = set(parameters.keys())
        missing = required_names - provided_names

        if missing:
            raise ToolParameterError(
                definition.name,
                f"필수 파라미터 누락: {', '.join(sorted(missing))}",
            )

    async def _run_critic(
        self,
        definition: ToolDefinition,
        parameters: dict[str, Any],
        context: str,
    ) -> ToolResult | None:
        """Critic 평가 실행

        Returns:
            None: SAFE (계속 실행)
            ToolResult: NEEDS_CONFIRMATION (pending 상태)

        Raises:
            ToolExecutionBlockedError: UNSAFE (차단)
        """
        action = ToolAction(
            tool_name=definition.name,
            action=definition.description,
            parameters=parameters,
            context=context,
        )

        # evaluate_and_enforce: UNSAFE → ToolExecutionBlockedError raise
        judgment = await self._critic.evaluate_and_enforce(action)

        if judgment.safety_level == SafetyLevel.NEEDS_CONFIRMATION:
            confirmation_id = f"confirm-{uuid.uuid4().hex[:12]}"
            logger.info(
                "tool_needs_confirmation",
                name=definition.name,
                confirmation_id=confirmation_id,
                reason=judgment.reason[:100],
            )

            # PendingStore에 저장 → 사용자 승인 시 실행 가능
            self._pending_store.add(PendingAction(
                confirmation_id=confirmation_id,
                tool_name=definition.name,
                parameters=parameters,
                context=context,
            ))

            return ToolResult(
                tool_name=definition.name,
                success=False,
                output=None,
                error="",
                pending_confirmation=True,
                confirmation_id=confirmation_id,
            )

        # SAFE → None 반환 (실행 계속)
        return None
