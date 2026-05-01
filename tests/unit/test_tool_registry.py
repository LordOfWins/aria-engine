"""ARIA Engine - Tool Registry + Tool Types 단위 테스트

테스트 범위:
- ToolDefinition: 생성 / 검증 / LLM 포맷 변환
- ToolParameter: 타입 / 필수 여부
- ToolResult: 성공/실패/대기 상태 / to_observation()
- ToolExecutor: ABC 인터페이스 구현
- ToolRegistry: 등록 / 조회 / 제거 / 실행 / Critic 연동
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aria.tools.tool_types import (
    SafetyLevelHint,
    ToolCategory,
    ToolDefinition,
    ToolExecutor,
    ToolParameter,
    ToolResult,
)
from aria.tools.tool_registry import (
    ToolAlreadyRegisteredError,
    ToolNotFoundError,
    ToolParameterError,
    ToolRegistry,
)
from aria.tools.critic_types import (
    CriticConfig,
    CriticJudgment,
    SafetyLevel,
)
from aria.core.exceptions import ToolExecutionBlockedError


# === Test Helpers ===


class DummyExecutor(ToolExecutor):
    """테스트용 더미 도구 실행기"""

    def __init__(
        self,
        name: str = "dummy_tool",
        output: Any = "dummy output",
        should_fail: bool = False,
        error_msg: str = "dummy error",
    ) -> None:
        self._name = name
        self._output = output
        self._should_fail = should_fail
        self._error_msg = error_msg

    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        if self._should_fail:
            return ToolResult(
                tool_name=self._name,
                success=False,
                error=self._error_msg,
            )
        return ToolResult(
            tool_name=self._name,
            success=True,
            output=self._output,
        )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=f"{self._name} 테스트 도구",
            parameters=[
                ToolParameter(name="input", type="string", required=True),
            ],
            category=ToolCategory.BUILTIN,
            safety_hint=SafetyLevelHint.READ_ONLY,
        )


class CrashingExecutor(ToolExecutor):
    """실행 시 예외를 던지는 도구"""

    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        raise RuntimeError("executor crashed")

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="crashing_tool",
            description="충돌 테스트 도구",
        )


def _make_definition(
    name: str = "test_tool",
    *,
    params: list[ToolParameter] | None = None,
    category: ToolCategory = ToolCategory.BUILTIN,
    safety_hint: SafetyLevelHint = SafetyLevelHint.READ_ONLY,
    enabled: bool = True,
) -> ToolDefinition:
    """테스트용 ToolDefinition 팩토리"""
    return ToolDefinition(
        name=name,
        description=f"{name} 테스트 도구",
        parameters=params or [],
        category=category,
        safety_hint=safety_hint,
        enabled=enabled,
    )


# === ToolParameter Tests ===


class TestToolParameter:
    def test_default_values(self) -> None:
        param = ToolParameter(name="query")
        assert param.type == "string"
        assert param.required is False
        assert param.enum is None
        assert param.default is None

    def test_required_param(self) -> None:
        param = ToolParameter(name="query", type="string", required=True)
        assert param.required is True

    def test_enum_param(self) -> None:
        param = ToolParameter(name="mode", type="string", enum=["fast", "slow"])
        assert param.enum == ["fast", "slow"]


# === ToolDefinition Tests ===


class TestToolDefinition:
    def test_valid_creation(self) -> None:
        defn = _make_definition("my_tool")
        assert defn.name == "my_tool"
        assert defn.category == ToolCategory.BUILTIN
        assert defn.enabled is True

    def test_snake_case_name_validation(self) -> None:
        """도구 이름은 snake_case만 허용"""
        _make_definition("valid_name_123")  # OK
        _make_definition("a")  # OK — 1글자

        with pytest.raises(Exception):  # pydantic ValidationError
            _make_definition("InvalidName")  # 대문자 불가

        with pytest.raises(Exception):
            _make_definition("123_start")  # 숫자 시작 불가

        with pytest.raises(Exception):
            _make_definition("has-dash")  # 하이픈 불가

    def test_reserved_name_rejected(self) -> None:
        """예약어는 도구 이름으로 사용 불가"""
        with pytest.raises(Exception):
            _make_definition("self")

        with pytest.raises(Exception):
            _make_definition("none")

    def test_to_llm_tool_no_params(self) -> None:
        """파라미터 없는 도구의 LLM 포맷 변환"""
        defn = _make_definition("simple_tool")
        llm_tool = defn.to_llm_tool()

        assert llm_tool["type"] == "function"
        assert llm_tool["function"]["name"] == "simple_tool"
        assert llm_tool["function"]["description"] == "simple_tool 테스트 도구"
        assert llm_tool["function"]["parameters"]["type"] == "object"
        assert llm_tool["function"]["parameters"]["properties"] == {}
        assert "required" not in llm_tool["function"]["parameters"]

    def test_to_llm_tool_with_params(self) -> None:
        """파라미터 있는 도구의 LLM 포맷 변환"""
        defn = _make_definition(
            "search_tool",
            params=[
                ToolParameter(name="query", type="string", required=True, description="검색어"),
                ToolParameter(name="limit", type="integer", required=False, description="결과 수"),
                ToolParameter(name="mode", type="string", enum=["fast", "accurate"]),
            ],
        )
        llm_tool = defn.to_llm_tool()
        func = llm_tool["function"]
        props = func["parameters"]["properties"]

        assert "query" in props
        assert props["query"]["type"] == "string"
        assert props["query"]["description"] == "검색어"
        assert "limit" in props
        assert props["limit"]["type"] == "integer"
        assert "mode" in props
        assert props["mode"]["enum"] == ["fast", "accurate"]
        assert func["parameters"]["required"] == ["query"]

    def test_disabled_tool(self) -> None:
        defn = _make_definition("disabled_tool", enabled=False)
        assert defn.enabled is False


# === ToolResult Tests ===


class TestToolResult:
    def test_success_result(self) -> None:
        result = ToolResult(tool_name="test", success=True, output="hello")
        assert result.success is True
        assert result.output == "hello"
        assert result.pending_confirmation is False

    def test_failure_result(self) -> None:
        result = ToolResult(tool_name="test", success=False, error="failed")
        assert result.success is False
        assert result.error == "failed"

    def test_pending_result(self) -> None:
        result = ToolResult(
            tool_name="test",
            success=False,
            pending_confirmation=True,
            confirmation_id="confirm-abc123",
        )
        assert result.pending_confirmation is True
        assert result.confirmation_id == "confirm-abc123"

    def test_to_observation_success(self) -> None:
        result = ToolResult(tool_name="search", success=True, output="결과 데이터")
        obs = result.to_observation()
        assert "[도구: search] 성공" in obs
        assert "결과 데이터" in obs

    def test_to_observation_failure(self) -> None:
        result = ToolResult(tool_name="search", success=False, error="연결 실패")
        obs = result.to_observation()
        assert "[도구: search] 실패" in obs
        assert "연결 실패" in obs

    def test_to_observation_pending(self) -> None:
        result = ToolResult(
            tool_name="write",
            success=False,
            pending_confirmation=True,
            confirmation_id="confirm-xyz",
        )
        obs = result.to_observation()
        assert "사용자 확인 대기 중" in obs
        assert "confirm-xyz" in obs

    def test_to_observation_truncates_long_output(self) -> None:
        """3000자 초과 출력은 잘림"""
        long_output = "x" * 5000
        result = ToolResult(tool_name="test", success=True, output=long_output)
        obs = result.to_observation()
        assert "결과 잘림" in obs
        assert len(obs) < 5000

    def test_to_observation_none_output(self) -> None:
        result = ToolResult(tool_name="test", success=True, output=None)
        obs = result.to_observation()
        assert "(빈 결과)" in obs


# === ToolExecutor Tests ===


class TestToolExecutor:
    @pytest.mark.asyncio
    async def test_dummy_executor_success(self) -> None:
        executor = DummyExecutor(output="test result")
        result = await executor.execute({"input": "hello"})
        assert result.success is True
        assert result.output == "test result"

    @pytest.mark.asyncio
    async def test_dummy_executor_failure(self) -> None:
        executor = DummyExecutor(should_fail=True, error_msg="bad input")
        result = await executor.execute({"input": "bad"})
        assert result.success is False
        assert result.error == "bad input"

    def test_get_definition(self) -> None:
        executor = DummyExecutor(name="my_tool")
        defn = executor.get_definition()
        assert defn.name == "my_tool"
        assert len(defn.parameters) == 1


# === ToolRegistry Tests ===


class TestToolRegistryRegistration:
    def test_register_and_get(self) -> None:
        registry = ToolRegistry()
        defn = _make_definition("tool_a")
        executor = DummyExecutor(name="tool_a")
        registry.register(defn, executor)

        got_defn, got_exec = registry.get("tool_a")
        assert got_defn.name == "tool_a"
        assert got_exec is executor

    def test_register_executor_shortcut(self) -> None:
        """register_executor로 definition 자동 추출"""
        registry = ToolRegistry()
        executor = DummyExecutor(name="auto_tool")
        registry.register_executor(executor)

        assert registry.has_tool("auto_tool")
        defn, _ = registry.get("auto_tool")
        assert defn.description == "auto_tool 테스트 도구"

    def test_duplicate_registration_raises(self) -> None:
        registry = ToolRegistry()
        defn = _make_definition("tool_a")
        registry.register(defn, DummyExecutor(name="tool_a"))

        with pytest.raises(ToolAlreadyRegisteredError):
            registry.register(defn, DummyExecutor(name="tool_a"))

    def test_unregister(self) -> None:
        registry = ToolRegistry()
        defn = _make_definition("tool_a")
        registry.register(defn, DummyExecutor(name="tool_a"))
        assert registry.has_tool("tool_a")

        registry.unregister("tool_a")
        assert not registry.has_tool("tool_a")

    def test_unregister_nonexistent_raises(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            registry.unregister("nonexistent")

    def test_get_nonexistent_raises(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            registry.get("nonexistent")

    def test_tool_count(self) -> None:
        registry = ToolRegistry()
        assert registry.tool_count == 0
        registry.register(_make_definition("a"), DummyExecutor(name="a"))
        registry.register(_make_definition("b"), DummyExecutor(name="b"))
        assert registry.tool_count == 2


class TestToolRegistryListing:
    def test_list_all(self) -> None:
        registry = ToolRegistry()
        registry.register(_make_definition("a"), DummyExecutor(name="a"))
        registry.register(_make_definition("b"), DummyExecutor(name="b"))
        tools = registry.list_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"a", "b"}

    def test_list_by_category(self) -> None:
        registry = ToolRegistry()
        registry.register(
            _make_definition("builtin_tool", category=ToolCategory.BUILTIN),
            DummyExecutor(name="builtin_tool"),
        )
        registry.register(
            _make_definition("mcp_tool", category=ToolCategory.MCP),
            DummyExecutor(name="mcp_tool"),
        )

        builtin = registry.list_tools(category=ToolCategory.BUILTIN)
        assert len(builtin) == 1
        assert builtin[0].name == "builtin_tool"

        mcp = registry.list_tools(category=ToolCategory.MCP)
        assert len(mcp) == 1
        assert mcp[0].name == "mcp_tool"

    def test_list_excludes_disabled(self) -> None:
        registry = ToolRegistry()
        registry.register(_make_definition("enabled_tool"), DummyExecutor(name="enabled_tool"))
        registry.register(
            _make_definition("disabled_tool", enabled=False),
            DummyExecutor(name="disabled_tool"),
        )

        enabled = registry.list_tools(enabled_only=True)
        assert len(enabled) == 1
        assert enabled[0].name == "enabled_tool"

        all_tools = registry.list_tools(enabled_only=False)
        assert len(all_tools) == 2

    def test_to_llm_tools(self) -> None:
        registry = ToolRegistry()
        registry.register(
            _make_definition("search", params=[
                ToolParameter(name="query", type="string", required=True),
            ]),
            DummyExecutor(name="search"),
        )
        llm_tools = registry.to_llm_tools()
        assert len(llm_tools) == 1
        assert llm_tools[0]["type"] == "function"
        assert llm_tools[0]["function"]["name"] == "search"


class TestToolRegistryExecution:
    @pytest.mark.asyncio
    async def test_execute_success_no_critic(self) -> None:
        """Critic 없이 도구 실행 성공"""
        registry = ToolRegistry(critic=None)
        registry.register(
            _make_definition("tool_a", params=[
                ToolParameter(name="input", type="string", required=True),
            ]),
            DummyExecutor(name="tool_a", output="result"),
        )

        result = await registry.execute("tool_a", {"input": "hello"})
        assert result.success is True
        assert result.output == "result"
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool_raises(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            await registry.execute("nonexistent", {})

    @pytest.mark.asyncio
    async def test_execute_missing_required_param_raises(self) -> None:
        registry = ToolRegistry()
        registry.register(
            _make_definition("tool_a", params=[
                ToolParameter(name="required_param", type="string", required=True),
            ]),
            DummyExecutor(name="tool_a"),
        )

        with pytest.raises(ToolParameterError) as exc_info:
            await registry.execute("tool_a", {})
        assert "required_param" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_disabled_tool_returns_failure(self) -> None:
        registry = ToolRegistry()
        registry.register(
            _make_definition("disabled_tool", enabled=False),
            DummyExecutor(name="disabled_tool"),
        )

        result = await registry.execute("disabled_tool", {})
        assert result.success is False
        assert "비활성화" in result.error

    @pytest.mark.asyncio
    async def test_execute_executor_crash_returns_failure(self) -> None:
        """executor가 예외를 던지면 실패 ToolResult 반환 (전파 안 됨)"""
        registry = ToolRegistry()
        registry.register(
            _make_definition("crashing_tool"),
            CrashingExecutor(),
        )

        result = await registry.execute("crashing_tool", {})
        assert result.success is False
        assert "executor crashed" in result.error
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_execute_skip_critic(self) -> None:
        """skip_critic=True면 Critic 건너뜀"""
        mock_critic = MagicMock()
        mock_critic.evaluate_and_enforce = AsyncMock()

        registry = ToolRegistry(critic=mock_critic)
        registry.register(
            _make_definition("tool_a"),
            DummyExecutor(name="tool_a", output="direct"),
        )

        result = await registry.execute("tool_a", {}, skip_critic=True)
        assert result.success is True
        assert result.output == "direct"
        mock_critic.evaluate_and_enforce.assert_not_called()


class TestToolRegistryCriticIntegration:
    @pytest.mark.asyncio
    async def test_critic_safe_allows_execution(self) -> None:
        """Critic이 SAFE 판단 → 도구 실행"""
        mock_critic = MagicMock()
        mock_critic.evaluate_and_enforce = AsyncMock(return_value=CriticJudgment(
            safety_level=SafetyLevel.SAFE,
            reason="읽기 전용 작업",
            model_used="test",
        ))

        registry = ToolRegistry(critic=mock_critic)
        registry.register(
            _make_definition("safe_tool"),
            DummyExecutor(name="safe_tool", output="safe result"),
        )

        result = await registry.execute("safe_tool", {}, context="테스트 컨텍스트")
        assert result.success is True
        assert result.output == "safe result"
        mock_critic.evaluate_and_enforce.assert_called_once()

    @pytest.mark.asyncio
    async def test_critic_needs_confirmation_returns_pending(self) -> None:
        """Critic이 NEEDS_CONFIRMATION → pending result 반환"""
        mock_critic = MagicMock()
        mock_critic.evaluate_and_enforce = AsyncMock(return_value=CriticJudgment(
            safety_level=SafetyLevel.NEEDS_CONFIRMATION,
            reason="데이터 수정 작업",
            model_used="test",
        ))

        registry = ToolRegistry(critic=mock_critic)
        registry.register(
            _make_definition("write_tool", safety_hint=SafetyLevelHint.WRITE),
            DummyExecutor(name="write_tool"),
        )

        result = await registry.execute("write_tool", {})
        assert result.success is False
        assert result.pending_confirmation is True
        assert result.confirmation_id.startswith("confirm-")

    @pytest.mark.asyncio
    async def test_critic_unsafe_raises_blocked(self) -> None:
        """Critic이 UNSAFE → ToolExecutionBlockedError raise"""
        mock_critic = MagicMock()
        mock_critic.evaluate_and_enforce = AsyncMock(
            side_effect=ToolExecutionBlockedError(
                tool_name="dangerous_tool",
                reason="시스템 파일 접근 시도",
                risk_factors=["system_access"],
            )
        )

        registry = ToolRegistry(critic=mock_critic)
        registry.register(
            _make_definition("dangerous_tool", safety_hint=SafetyLevelHint.DESTRUCTIVE),
            DummyExecutor(name="dangerous_tool"),
        )

        with pytest.raises(ToolExecutionBlockedError) as exc_info:
            await registry.execute("dangerous_tool", {})
        assert "시스템 파일 접근" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_critic_receives_correct_tool_action(self) -> None:
        """Critic에 전달되는 ToolAction 검증"""
        mock_critic = MagicMock()
        mock_critic.evaluate_and_enforce = AsyncMock(return_value=CriticJudgment(
            safety_level=SafetyLevel.SAFE,
            reason="safe",
            model_used="test",
        ))

        registry = ToolRegistry(critic=mock_critic)
        registry.register(
            _make_definition("tool_x"),
            DummyExecutor(name="tool_x"),
        )

        await registry.execute(
            "tool_x",
            {"key": "value"},
            context="사용자가 데이터 조회 요청",
        )

        call_args = mock_critic.evaluate_and_enforce.call_args
        action = call_args[0][0]
        assert action.tool_name == "tool_x"
        assert action.parameters == {"key": "value"}
        assert action.context == "사용자가 데이터 조회 요청"


# === Category/Enum Tests ===


class TestEnums:
    def test_tool_category_values(self) -> None:
        assert ToolCategory.BUILTIN.value == "builtin"
        assert ToolCategory.MCP.value == "mcp"
        assert ToolCategory.CUSTOM.value == "custom"

    def test_safety_level_hint_values(self) -> None:
        assert SafetyLevelHint.READ_ONLY.value == "read_only"
        assert SafetyLevelHint.WRITE.value == "write"
        assert SafetyLevelHint.DESTRUCTIVE.value == "destructive"
        assert SafetyLevelHint.EXTERNAL.value == "external"


# === Exception Tests ===


class TestExceptions:
    def test_tool_not_found_error(self) -> None:
        err = ToolNotFoundError("missing_tool")
        assert err.code == "TOOL_NOT_FOUND"
        assert "missing_tool" in err.message
        assert err.details["tool_name"] == "missing_tool"

    def test_tool_already_registered_error(self) -> None:
        err = ToolAlreadyRegisteredError("dup_tool")
        assert err.code == "TOOL_ALREADY_REGISTERED"
        assert "dup_tool" in err.message

    def test_tool_parameter_error(self) -> None:
        err = ToolParameterError("my_tool", "필수 파라미터 누락: query")
        assert err.code == "TOOL_PARAMETER_ERROR"
        assert "query" in err.message
        assert err.details["tool_name"] == "my_tool"
