"""Critic 패턴 단위 테스트

도구 실행 전 안전성 판단 레이어 검증:
- CriticTypes: ToolAction / CriticJudgment / SafetyLevel / CriticConfig
- CriticEvaluator: cheap 모델 기반 평가 + bypass + disable + fallback
- evaluate_and_enforce: UNSAFE 차단 + NEEDS_CONFIRMATION 반환
- ToolExecutionBlockedError: 예외 구조
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aria.core.exceptions import ToolExecutionBlockedError
from aria.tools.critic_types import (
    CriticConfig,
    CriticJudgment,
    SafetyLevel,
    ToolAction,
)
from aria.tools.critic import (
    CriticEvaluator,
    CRITIC_SYSTEM,
    CRITIC_USER,
)


# === ToolAction / CriticJudgment 타입 테스트 ===

class TestCriticTypes:
    def test_tool_action_creation(self):
        action = ToolAction(
            tool_name="file_write",
            action="설정 파일 수정",
            parameters={"path": "/config.json", "content": "{}"},
            context="사용자가 설정 변경 요청",
        )
        assert action.tool_name == "file_write"
        assert action.parameters["path"] == "/config.json"

    def test_tool_action_minimal(self):
        action = ToolAction(tool_name="health_check", action="서버 상태 확인")
        assert action.parameters == {}
        assert action.context == ""

    def test_safety_level_values(self):
        assert SafetyLevel.SAFE == "safe"
        assert SafetyLevel.NEEDS_CONFIRMATION == "needs_confirmation"
        assert SafetyLevel.UNSAFE == "unsafe"

    def test_critic_judgment_creation(self):
        judgment = CriticJudgment(
            safety_level=SafetyLevel.SAFE,
            reason="읽기 전용 작업",
        )
        assert judgment.safety_level == SafetyLevel.SAFE
        assert judgment.risk_factors == []
        assert judgment.suggested_mitigation == ""
        assert judgment.evaluated_at is not None

    def test_critic_config_defaults(self):
        config = CriticConfig()
        assert config.enabled is True
        assert config.bypass_tools == []
        assert config.block_on_unsafe is True
        assert config.require_confirmation_on_needs_confirm is True

    def test_critic_config_custom(self):
        config = CriticConfig(
            enabled=True,
            bypass_tools=["health_check", "read_file"],
            block_on_unsafe=False,
        )
        assert len(config.bypass_tools) == 2
        assert config.block_on_unsafe is False


# === CriticEvaluator 테스트 ===

class TestCriticEvaluator:
    @pytest.fixture
    def mock_llm(self):
        return MagicMock()

    @pytest.fixture
    def evaluator(self, mock_llm):
        return CriticEvaluator(mock_llm)

    @pytest.fixture
    def read_action(self):
        return ToolAction(
            tool_name="search_knowledge",
            action="벡터DB 검색",
            parameters={"query": "테스트", "top_k": 5},
        )

    @pytest.fixture
    def write_action(self):
        return ToolAction(
            tool_name="file_write",
            action="프로젝트 설정 파일 수정",
            parameters={"path": "/etc/passwd", "content": "malicious"},
            context="사용자가 시스템 파일 접근 시도",
        )

    # --- 비활성화 테스트 ---

    @pytest.mark.asyncio
    async def test_disabled_returns_safe(self, mock_llm, read_action):
        """Critic 비활성화 → 무조건 SAFE"""
        evaluator = CriticEvaluator(mock_llm, CriticConfig(enabled=False))
        judgment = await evaluator.evaluate(read_action)

        assert judgment.safety_level == SafetyLevel.SAFE
        assert "비활성화" in judgment.reason
        mock_llm.complete.assert_not_called()

    # --- bypass 테스트 ---

    @pytest.mark.asyncio
    async def test_bypass_tool_returns_safe(self, mock_llm, read_action):
        """bypass_tools에 포함된 도구 → 무조건 SAFE (LLM 호출 없음)"""
        config = CriticConfig(bypass_tools=["search_knowledge"])
        evaluator = CriticEvaluator(mock_llm, config)
        judgment = await evaluator.evaluate(read_action)

        assert judgment.safety_level == SafetyLevel.SAFE
        assert "bypass" in judgment.reason
        mock_llm.complete.assert_not_called()

    # --- SAFE 판단 테스트 ---

    @pytest.mark.asyncio
    async def test_evaluate_safe(self, mock_llm, read_action):
        """LLM이 safe 판단 → CriticJudgment.safety_level == SAFE"""
        mock_llm.complete = AsyncMock(return_value={
            "content": '{"safety_level":"safe","reason":"읽기 전용 검색","risk_factors":[],"suggested_mitigation":""}',
            "model": "claude-haiku",
        })

        evaluator = CriticEvaluator(mock_llm)
        judgment = await evaluator.evaluate(read_action)

        assert judgment.safety_level == SafetyLevel.SAFE
        assert judgment.model_used == "claude-haiku"
        assert judgment.latency_ms > 0

    # --- NEEDS_CONFIRMATION 판단 테스트 ---

    @pytest.mark.asyncio
    async def test_evaluate_needs_confirmation(self, mock_llm):
        """데이터 수정 작업 → NEEDS_CONFIRMATION"""
        mock_llm.complete = AsyncMock(return_value={
            "content": '{"safety_level":"needs_confirmation","reason":"데이터 수정 작업","risk_factors":["data_modification"],"suggested_mitigation":"사용자 확인 요청"}',
            "model": "claude-haiku",
        })

        action = ToolAction(
            tool_name="db_update",
            action="사용자 프로필 삭제",
            parameters={"user_id": "123"},
        )
        evaluator = CriticEvaluator(mock_llm)
        judgment = await evaluator.evaluate(action)

        assert judgment.safety_level == SafetyLevel.NEEDS_CONFIRMATION
        assert "data_modification" in judgment.risk_factors

    # --- UNSAFE 판단 테스트 ---

    @pytest.mark.asyncio
    async def test_evaluate_unsafe(self, mock_llm, write_action):
        """시스템 파일 접근 → UNSAFE"""
        mock_llm.complete = AsyncMock(return_value={
            "content": '{"safety_level":"unsafe","reason":"시스템 파일 접근 시도","risk_factors":["system_file_access","privilege_escalation"],"suggested_mitigation":"해당 작업은 허용되지 않습니다"}',
            "model": "claude-haiku",
        })

        evaluator = CriticEvaluator(mock_llm)
        judgment = await evaluator.evaluate(write_action)

        assert judgment.safety_level == SafetyLevel.UNSAFE
        assert len(judgment.risk_factors) == 2

    # --- LLM 호출 파라미터 검증 ---

    @pytest.mark.asyncio
    async def test_uses_cheap_model(self, mock_llm, read_action):
        """Critic은 cheap 모델(Haiku) 사용"""
        mock_llm.complete = AsyncMock(return_value={
            "content": '{"safety_level":"safe","reason":"ok","risk_factors":[],"suggested_mitigation":""}',
            "model": "claude-haiku",
        })

        evaluator = CriticEvaluator(mock_llm)
        await evaluator.evaluate(read_action)

        call_kwargs = mock_llm.complete.call_args
        assert call_kwargs.kwargs["model_tier"] == "cheap"
        assert call_kwargs.kwargs["temperature"] == 0.1
        assert call_kwargs.kwargs["cache_system_prompt"] is True

    # --- JSON 파싱 실패 fallback ---

    @pytest.mark.asyncio
    async def test_json_parse_failure_fallback(self, mock_llm, read_action):
        """LLM 응답 JSON 파싱 실패 → NEEDS_CONFIRMATION fallback"""
        mock_llm.complete = AsyncMock(return_value={
            "content": "이건 JSON이 아닙니다",
            "model": "claude-haiku",
        })

        evaluator = CriticEvaluator(mock_llm)
        judgment = await evaluator.evaluate(read_action)

        assert judgment.safety_level == SafetyLevel.NEEDS_CONFIRMATION
        assert "파싱 실패" in judgment.reason

    # --- LLM 호출 자체 실패 fallback ---

    @pytest.mark.asyncio
    async def test_llm_call_failure_fallback(self, mock_llm, read_action):
        """LLM 호출 예외 → NEEDS_CONFIRMATION fallback"""
        mock_llm.complete = AsyncMock(side_effect=RuntimeError("LLM 연결 실패"))

        evaluator = CriticEvaluator(mock_llm)
        judgment = await evaluator.evaluate(read_action)

        assert judgment.safety_level == SafetyLevel.NEEDS_CONFIRMATION
        assert "critic_evaluation_error" in judgment.risk_factors

    # --- 알 수 없는 safety_level fallback ---

    @pytest.mark.asyncio
    async def test_unknown_safety_level_fallback(self, mock_llm, read_action):
        """알 수 없는 safety_level → NEEDS_CONFIRMATION"""
        mock_llm.complete = AsyncMock(return_value={
            "content": '{"safety_level":"unknown_level","reason":"test","risk_factors":[],"suggested_mitigation":""}',
            "model": "test",
        })

        evaluator = CriticEvaluator(mock_llm)
        judgment = await evaluator.evaluate(read_action)

        assert judgment.safety_level == SafetyLevel.NEEDS_CONFIRMATION


# === evaluate_and_enforce 테스트 ===

class TestCriticEnforcement:
    @pytest.fixture
    def mock_llm(self):
        return MagicMock()

    @pytest.mark.asyncio
    async def test_enforce_unsafe_raises(self, mock_llm):
        """UNSAFE + block_on_unsafe=True → ToolExecutionBlockedError"""
        mock_llm.complete = AsyncMock(return_value={
            "content": '{"safety_level":"unsafe","reason":"위험 행위","risk_factors":["danger"],"suggested_mitigation":""}',
            "model": "test",
        })

        evaluator = CriticEvaluator(mock_llm, CriticConfig(block_on_unsafe=True))
        action = ToolAction(tool_name="rm_rf", action="전체 삭제")

        with pytest.raises(ToolExecutionBlockedError) as exc_info:
            await evaluator.evaluate_and_enforce(action)

        assert exc_info.value.code == "TOOL_EXECUTION_BLOCKED"
        assert "rm_rf" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_enforce_unsafe_no_block(self, mock_llm):
        """UNSAFE + block_on_unsafe=False → 경고만 (에러 없음)"""
        mock_llm.complete = AsyncMock(return_value={
            "content": '{"safety_level":"unsafe","reason":"위험","risk_factors":[],"suggested_mitigation":""}',
            "model": "test",
        })

        evaluator = CriticEvaluator(mock_llm, CriticConfig(block_on_unsafe=False))
        action = ToolAction(tool_name="test", action="테스트")

        judgment = await evaluator.evaluate_and_enforce(action)
        assert judgment.safety_level == SafetyLevel.UNSAFE  # 차단 안 됨

    @pytest.mark.asyncio
    async def test_enforce_safe_returns_normally(self, mock_llm):
        """SAFE → 정상 반환"""
        mock_llm.complete = AsyncMock(return_value={
            "content": '{"safety_level":"safe","reason":"안전","risk_factors":[],"suggested_mitigation":""}',
            "model": "test",
        })

        evaluator = CriticEvaluator(mock_llm)
        action = ToolAction(tool_name="search", action="검색")

        judgment = await evaluator.evaluate_and_enforce(action)
        assert judgment.safety_level == SafetyLevel.SAFE

    @pytest.mark.asyncio
    async def test_enforce_needs_confirmation_returns(self, mock_llm):
        """NEEDS_CONFIRMATION → 판단 결과 반환 (호출자가 처리)"""
        mock_llm.complete = AsyncMock(return_value={
            "content": '{"safety_level":"needs_confirmation","reason":"확인 필요","risk_factors":["data_change"],"suggested_mitigation":"확인 요청"}',
            "model": "test",
        })

        evaluator = CriticEvaluator(mock_llm)
        action = ToolAction(tool_name="update", action="업데이트")

        judgment = await evaluator.evaluate_and_enforce(action)
        assert judgment.safety_level == SafetyLevel.NEEDS_CONFIRMATION


# === ToolExecutionBlockedError 테스트 ===

class TestToolExecutionBlockedError:
    def test_error_attributes(self):
        err = ToolExecutionBlockedError(
            tool_name="dangerous_tool",
            reason="시스템 파일 접근",
            risk_factors=["system_access", "privilege_escalation"],
        )
        assert err.code == "TOOL_EXECUTION_BLOCKED"
        assert err.details["tool_name"] == "dangerous_tool"
        assert len(err.details["risk_factors"]) == 2

    def test_error_inherits_aria_error(self):
        from aria.core.exceptions import AriaError
        err = ToolExecutionBlockedError(
            tool_name="test",
            reason="test",
        )
        assert isinstance(err, AriaError)

    def test_error_without_risk_factors(self):
        err = ToolExecutionBlockedError(tool_name="t", reason="r")
        assert err.details["risk_factors"] == []


# === Critic 프롬프트 안정성 테스트 ===

class TestCriticPrompts:
    def test_critic_system_is_cacheable(self):
        """CRITIC_SYSTEM에 포맷 변수 없어야 캐시 안정"""
        import re
        cleaned = CRITIC_SYSTEM.replace("{{", "").replace("}}", "")
        assert not re.search(r"[{}]", cleaned), \
            "CRITIC_SYSTEM에 이스케이프 안 된 {variable}이 있으면 캐시 불안정"

    def test_critic_user_has_format_vars(self):
        """CRITIC_USER는 포맷 변수를 포함해야 함"""
        assert "{tool_name}" in CRITIC_USER
        assert "{action}" in CRITIC_USER
        assert "{parameters}" in CRITIC_USER
        assert "{context}" in CRITIC_USER

    def test_critic_user_format_works(self):
        """CRITIC_USER 포맷 정상 동작 확인"""
        result = CRITIC_USER.format(
            tool_name="test_tool",
            action="테스트 동작",
            parameters="{'key': 'value'}",
            context="테스트 컨텍스트",
        )
        assert "test_tool" in result
        assert "테스트 동작" in result
