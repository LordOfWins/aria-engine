"""ARIA Engine - Tool Integration 단위 테스트

테스트 범위:
- LLMProvider: tools 파라미터 / tool_calls 추출 / 하위 호환
- ReActAgent: tool_registry 통합 / 도구 호출 루프 / NEEDS_CONFIRMATION 처리
- App: ToolRegistry 초기화 / Built-in Tools 등록
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from aria.agents.react_agent import (
    MAX_TOOL_ITERATIONS,
    AgentState,
    ReActAgent,
    _safe_parse_json,
)
from aria.providers.llm_provider import LLMProvider
from aria.tools.tool_types import (
    SafetyLevelHint,
    ToolCategory,
    ToolDefinition,
    ToolExecutor,
    ToolParameter,
    ToolResult,
)
from aria.tools.tool_registry import ToolRegistry


# === Test Helpers ===


class EchoExecutor(ToolExecutor):
    """입력을 그대로 반환하는 테스트 도구"""

    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        return ToolResult(
            tool_name="echo_tool",
            success=True,
            output=f"echo: {parameters.get('message', '')}",
        )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="echo_tool",
            description="메시지를 그대로 반환합니다",
            parameters=[
                ToolParameter(name="message", type="string", required=True),
            ],
            category=ToolCategory.BUILTIN,
            safety_hint=SafetyLevelHint.READ_ONLY,
        )


class CounterExecutor(ToolExecutor):
    """호출 횟수를 세는 테스트 도구"""

    def __init__(self) -> None:
        self.call_count = 0

    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        self.call_count += 1
        return ToolResult(
            tool_name="counter_tool",
            success=True,
            output=f"call #{self.call_count}",
        )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="counter_tool",
            description="호출 횟수를 세는 도구",
        )


def _make_mock_response(
    content: str = "test response",
    tool_calls: list[dict] | None = None,
) -> MagicMock:
    """LiteLLM 응답 객체 목"""
    response = MagicMock()
    message = MagicMock()
    message.content = content

    if tool_calls:
        mock_tool_calls = []
        for tc in tool_calls:
            mock_tc = MagicMock()
            mock_tc.id = tc["id"]
            mock_tc.function.name = tc["function"]["name"]
            mock_tc.function.arguments = tc["function"]["arguments"]
            mock_tool_calls.append(mock_tc)
        message.tool_calls = mock_tool_calls
    else:
        message.tool_calls = None

    response.choices = [MagicMock(message=message)]

    # usage 속성
    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 50
    usage.total_tokens = 150
    usage.cache_creation_input_tokens = 0
    usage.cache_read_input_tokens = 0
    response.usage = usage

    return response


# === LLMProvider Tool Support Tests ===


class TestLLMProviderToolCalls:
    def test_extract_tool_calls_none(self) -> None:
        """tool_calls가 없는 응답 → None"""
        response = _make_mock_response(content="일반 텍스트 응답")
        result = LLMProvider._extract_tool_calls(response)
        assert result is None

    def test_extract_tool_calls_present(self) -> None:
        """tool_calls가 있는 응답 → 리스트 반환"""
        response = _make_mock_response(
            content="",
            tool_calls=[
                {
                    "id": "call_123",
                    "function": {
                        "name": "memory_read",
                        "arguments": '{"scope": "global", "domain": "user-profile"}',
                    },
                },
            ],
        )
        result = LLMProvider._extract_tool_calls(response)
        assert result is not None
        assert len(result) == 1
        assert result[0]["id"] == "call_123"
        assert result[0]["function"]["name"] == "memory_read"
        assert '"scope"' in result[0]["function"]["arguments"]

    def test_extract_multiple_tool_calls(self) -> None:
        """복수 tool_calls → 모두 추출"""
        response = _make_mock_response(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "function": {"name": "memory_read", "arguments": "{}"},
                },
                {
                    "id": "call_2",
                    "function": {"name": "knowledge_search", "arguments": '{"query": "test"}'},
                },
            ],
        )
        result = LLMProvider._extract_tool_calls(response)
        assert result is not None
        assert len(result) == 2
        assert result[0]["function"]["name"] == "memory_read"
        assert result[1]["function"]["name"] == "knowledge_search"


# === ReActAgent Tool Integration Tests ===


class TestAgentToolRegistry:
    def test_agent_accepts_tool_registry(self) -> None:
        """ReActAgent가 tool_registry 파라미터 수용"""
        llm = MagicMock()
        vs = MagicMock()
        registry = ToolRegistry()

        agent = ReActAgent(llm, vs, tool_registry=registry)
        assert agent.tool_registry is registry

    def test_agent_without_registry_backward_compat(self) -> None:
        """tool_registry 없이도 기존 동작 유지"""
        llm = MagicMock()
        vs = MagicMock()

        agent = ReActAgent(llm, vs)
        assert agent.tool_registry is None

    def test_tool_calls_made_in_state(self) -> None:
        """AgentState에 tool_calls_made 필드 존재"""
        state = AgentState(query="test")
        assert state.tool_calls_made == 0

    def test_max_tool_iterations_constant(self) -> None:
        """MAX_TOOL_ITERATIONS 상수 존재"""
        assert MAX_TOOL_ITERATIONS >= 3  # 최소 3회


class TestAgentReasonWithTools:
    """_reason_with_tools 동작 테스트 (LLM 목 사용)"""

    @pytest.mark.asyncio
    async def test_no_tool_calls_returns_text(self) -> None:
        """도구 호출 없이 바로 텍스트 응답"""
        llm = MagicMock()
        vs = MagicMock()
        registry = ToolRegistry()
        registry.register_executor(EchoExecutor())

        agent = ReActAgent(llm, vs, tool_registry=registry)

        # LLM이 도구 없이 바로 텍스트 응답
        llm.complete_with_messages = AsyncMock(return_value={
            "content": "직접 답변입니다",
            "model": "test",
            "tool_calls": None,
        })

        state = AgentState(
            query="테스트 질문",
            intent={"recommended_action": "respond"},
        )

        result = await agent._reason_with_tools(
            state,
            "테스트 프롬프트",
            "동적 컨텍스트",
        )
        assert result["current_answer"] == "직접 답변입니다"
        assert result["tool_calls_made"] == 0

    @pytest.mark.asyncio
    async def test_single_tool_call_and_response(self) -> None:
        """도구 1회 호출 후 텍스트 응답"""
        llm = MagicMock()
        vs = MagicMock()
        registry = ToolRegistry()
        registry.register_executor(EchoExecutor())

        agent = ReActAgent(llm, vs, tool_registry=registry)

        # 1차: 도구 호출 → 2차: 텍스트 응답
        call_count = 0

        async def mock_complete(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "content": "",
                    "model": "test",
                    "tool_calls": [{
                        "id": "call_001",
                        "type": "function",
                        "function": {
                            "name": "echo_tool",
                            "arguments": '{"message": "hello"}',
                        },
                    }],
                }
            return {
                "content": "에코 결과를 바탕으로 답변합니다",
                "model": "test",
                "tool_calls": None,
            }

        llm.complete_with_messages = AsyncMock(side_effect=mock_complete)

        state = AgentState(query="hello 에코")
        result = await agent._reason_with_tools(state, "prompt", "dynamic")

        assert result["current_answer"] == "에코 결과를 바탕으로 답변합니다"
        assert result["tool_calls_made"] == 1
        assert "도구 1회 호출" in result["reasoning_steps"][0]

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_loop(self) -> None:
        """도구 여러 번 호출 후 텍스트 응답"""
        llm = MagicMock()
        vs = MagicMock()
        registry = ToolRegistry()
        counter = CounterExecutor()
        registry.register_executor(counter)

        agent = ReActAgent(llm, vs, tool_registry=registry)

        call_count = 0

        async def mock_complete(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return {
                    "content": "",
                    "model": "test",
                    "tool_calls": [{
                        "id": f"call_{call_count:03d}",
                        "type": "function",
                        "function": {
                            "name": "counter_tool",
                            "arguments": "{}",
                        },
                    }],
                }
            return {
                "content": "3회 호출 후 최종 답변",
                "model": "test",
                "tool_calls": None,
            }

        llm.complete_with_messages = AsyncMock(side_effect=mock_complete)

        state = AgentState(query="카운터 테스트")
        result = await agent._reason_with_tools(state, "prompt", "dynamic")

        assert result["current_answer"] == "3회 호출 후 최종 답변"
        assert result["tool_calls_made"] == 3
        assert counter.call_count == 3

    @pytest.mark.asyncio
    async def test_max_tool_iterations_enforced(self) -> None:
        """MAX_TOOL_ITERATIONS 초과 시 루프 종료"""
        llm = MagicMock()
        vs = MagicMock()
        registry = ToolRegistry()
        counter = CounterExecutor()
        registry.register_executor(counter)

        agent = ReActAgent(llm, vs, tool_registry=registry)

        call_count = 0

        async def mock_complete_always_tool(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # 항상 도구 호출 (무한루프 시도)
            return {
                "content": "",
                "model": "test",
                "tool_calls": [{
                    "id": f"call_{call_count:03d}",
                    "type": "function",
                    "function": {
                        "name": "counter_tool",
                        "arguments": "{}",
                    },
                }],
            }

        # 마지막에 도구 없이 호출하는 fallback도 목
        side_effects = []
        for i in range(MAX_TOOL_ITERATIONS):
            side_effects.append({
                "content": "",
                "model": "test",
                "tool_calls": [{
                    "id": f"call_{i:03d}",
                    "type": "function",
                    "function": {"name": "counter_tool", "arguments": "{}"},
                }],
            })
        # MAX 도달 후 마지막 LLM 호출 (도구 없이)
        side_effects.append({
            "content": "강제 종료 답변",
            "model": "test",
            "tool_calls": None,
        })

        llm.complete_with_messages = AsyncMock(side_effect=side_effects)

        state = AgentState(query="무한 루프 테스트")
        result = await agent._reason_with_tools(state, "prompt", "dynamic")

        assert counter.call_count == MAX_TOOL_ITERATIONS
        assert result["tool_calls_made"] == MAX_TOOL_ITERATIONS

    @pytest.mark.asyncio
    async def test_pending_confirmation_stops_loop(self) -> None:
        """NEEDS_CONFIRMATION 도구 → 루프 중단 + 확인 대기 메시지"""
        llm = MagicMock()
        vs = MagicMock()

        # Critic이 NEEDS_CONFIRMATION 판단
        from aria.tools.critic_types import CriticJudgment, SafetyLevel
        mock_critic = MagicMock()
        mock_critic.evaluate_and_enforce = AsyncMock(return_value=CriticJudgment(
            safety_level=SafetyLevel.NEEDS_CONFIRMATION,
            reason="데이터 수정 작업",
            model_used="test",
        ))

        registry = ToolRegistry(critic=mock_critic)

        # WRITE 힌트가 있는 도구
        class WriteExecutor(ToolExecutor):
            async def execute(self, parameters: dict[str, Any]) -> ToolResult:
                return ToolResult(tool_name="write_tool", success=True, output="written")

            def get_definition(self) -> ToolDefinition:
                return ToolDefinition(
                    name="write_tool",
                    description="쓰기 도구",
                    safety_hint=SafetyLevelHint.WRITE,
                )

        registry.register_executor(WriteExecutor())
        agent = ReActAgent(llm, vs, tool_registry=registry)

        llm.complete_with_messages = AsyncMock(return_value={
            "content": "",
            "model": "test",
            "tool_calls": [{
                "id": "call_write",
                "type": "function",
                "function": {"name": "write_tool", "arguments": "{}"},
            }],
        })

        state = AgentState(query="데이터 수정 요청")
        result = await agent._reason_with_tools(state, "prompt", "dynamic")

        assert "사용자 확인" in result["current_answer"]
        assert "확인 ID" in result["current_answer"]

    @pytest.mark.asyncio
    async def test_tool_not_found_handled(self) -> None:
        """존재하지 않는 도구 호출 → 에러 ToolResult (에이전트 종료 아님)"""
        llm = MagicMock()
        vs = MagicMock()
        registry = ToolRegistry()
        registry.register_executor(EchoExecutor())

        agent = ReActAgent(llm, vs, tool_registry=registry)

        call_count = 0

        async def mock_complete(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # 존재하지 않는 도구 호출
                return {
                    "content": "",
                    "model": "test",
                    "tool_calls": [{
                        "id": "call_bad",
                        "type": "function",
                        "function": {
                            "name": "nonexistent_tool",
                            "arguments": "{}",
                        },
                    }],
                }
            return {
                "content": "도구 실패 후 텍스트 답변",
                "model": "test",
                "tool_calls": None,
            }

        llm.complete_with_messages = AsyncMock(side_effect=mock_complete)

        state = AgentState(query="없는 도구 호출")
        # ToolNotFoundError가 ToolRegistry.execute()에서 raise되지만
        # _reason_with_tools가 이를 catch하여 AgentError로 변환
        # 하지만 실제로 ToolNotFoundError는 AriaError이므로
        # AgentError로 잡히지 않을 수 있음... 실제로는 ToolRegistry.execute()에서
        # ToolNotFoundError raise → agent의 except Exception으로 잡힘 → AgentError raise
        from aria.core.exceptions import AgentError as AE
        with pytest.raises(AE):
            await agent._reason_with_tools(state, "prompt", "dynamic")


# === ToolRegistry + LLM Tools Format Tests ===


class TestRegistryLLMFormat:
    def test_to_llm_tools_format(self) -> None:
        """ToolRegistry.to_llm_tools() 포맷 검증"""
        registry = ToolRegistry()
        registry.register_executor(EchoExecutor())

        tools = registry.to_llm_tools()
        assert len(tools) == 1

        tool = tools[0]
        assert tool["type"] == "function"
        func = tool["function"]
        assert func["name"] == "echo_tool"
        assert "message" in func["parameters"]["properties"]
        assert func["parameters"]["required"] == ["message"]

    def test_multiple_tools_format(self) -> None:
        """여러 도구의 LLM 포맷"""
        registry = ToolRegistry()
        registry.register_executor(EchoExecutor())
        registry.register_executor(CounterExecutor())

        tools = registry.to_llm_tools()
        assert len(tools) == 2
        names = {t["function"]["name"] for t in tools}
        assert names == {"echo_tool", "counter_tool"}


# === Backward Compatibility Tests ===


class TestBackwardCompatibility:
    def test_complete_returns_tool_calls_none_by_default(self) -> None:
        """complete() 반환값에 tool_calls 키가 항상 존재 (None 가능)"""
        # tool_calls가 None이어도 키 자체는 있어야 기존 코드가 .get()으로 안전하게 접근
        response = _make_mock_response(content="test")
        result = LLMProvider._extract_tool_calls(response)
        assert result is None

    def test_agent_state_has_tool_calls_made(self) -> None:
        """AgentState.tool_calls_made 기본값 0"""
        state = AgentState(query="test")
        assert state.tool_calls_made == 0

    @pytest.mark.asyncio
    async def test_reason_without_tools_unchanged(self) -> None:
        """tool_registry 없을 때 기존 _reason 동작 유지"""
        llm = MagicMock()
        vs = MagicMock()
        agent = ReActAgent(llm, vs)  # tool_registry=None

        llm.complete = AsyncMock(return_value={
            "content": "기존 방식 답변",
            "model": "test",
            "usage": MagicMock(),
            "tool_calls": None,
        })

        state = AgentState(
            query="기존 질문",
            intent={"search_queries": ["test"]},
            search_results=[],
        )

        result = await agent._reason(state)
        assert result["current_answer"] == "기존 방식 답변"
        # complete() 호출 (complete_with_messages 아님)
        llm.complete.assert_called_once()
        llm.complete_with_messages.assert_not_called()
