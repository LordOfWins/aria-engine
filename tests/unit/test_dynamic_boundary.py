"""SYSTEM_PROMPT_DYNAMIC_BOUNDARY 패턴 단위 테스트

프롬프트 캐시 경계 패턴 검증:
- 고정 시스템 프롬프트: cache_control 마커 포함 → 캐시됨
- 동적 시스템 프롬프트: cache_control 없음 → 캐시 경계 바깥
- 하위 호환: system_prompt_dynamic 미지정 시 기존 동작 유지
"""

from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aria.core.config import AriaConfig
from aria.providers.llm_provider import LLMProvider
from aria.agents.react_agent import (
    INTENT_ANALYSIS_SYSTEM,
    REASONING_SYSTEM,
    REASONING_SYSTEM_DYNAMIC,
    SELF_REFLECTION_SYSTEM,
    ReActAgent,
    AgentState,
)


# === Helper: LiteLLM 응답 mock ===

def _make_litellm_response(content: str = "test response", model: str = "test-model"):
    """litellm.acompletion 응답 mock 생성"""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    mock_response.model = model
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    mock_response.usage.cache_read_input_tokens = 80
    return mock_response


def _has_unescaped_braces(text: str) -> bool:
    """포맷 가능한 {variable} 패턴이 있는지 확인 ({{ }} 이스케이프 제외)"""
    # {{ 와 }} 를 제거한 후 남는 { } 가 있으면 True
    cleaned = text.replace("{{", "").replace("}}", "")
    return bool(re.search(r"[{}]", cleaned))


# === LLMProvider.complete() 메시지 구성 테스트 ===

class TestDynamicBoundaryMessageConstruction:
    """시스템 프롬프트 동적 경계 메시지 구성 검증"""

    @pytest.fixture
    def provider(self):
        return LLMProvider(AriaConfig())

    @pytest.mark.asyncio
    async def test_cache_with_dynamic_creates_two_content_blocks(self, provider):
        """cache_system_prompt=True + system_prompt_dynamic → 2개 content block"""
        captured_kwargs = {}

        async def mock_fallback(kwargs_base, model_tier, explicit_model=None):
            captured_kwargs.update(kwargs_base)
            return _make_litellm_response()

        with patch.object(provider, "_call_llm_with_fallback", side_effect=mock_fallback), \
             patch("litellm.completion_cost", return_value=0.001):
            await provider.complete(
                "user prompt",
                system_prompt="고정 지시",
                system_prompt_dynamic="동적 메모리 컨텍스트",
                cache_system_prompt=True,
            )

        messages = captured_kwargs["messages"]
        sys_msg = messages[0]

        assert sys_msg["role"] == "system"
        assert isinstance(sys_msg["content"], list)
        assert len(sys_msg["content"]) == 2

        # 첫 번째 블록: 고정 영역 + cache_control
        static_block = sys_msg["content"][0]
        assert static_block["type"] == "text"
        assert static_block["text"] == "고정 지시"
        assert static_block["cache_control"] == {"type": "ephemeral"}

        # 두 번째 블록: 동적 영역 + cache_control 없음
        dynamic_block = sys_msg["content"][1]
        assert dynamic_block["type"] == "text"
        assert dynamic_block["text"] == "동적 메모리 컨텍스트"
        assert "cache_control" not in dynamic_block

    @pytest.mark.asyncio
    async def test_cache_without_dynamic_creates_one_block(self, provider):
        """cache_system_prompt=True + system_prompt_dynamic=None → 1개 block (하위 호환)"""
        captured_kwargs = {}

        async def mock_fallback(kwargs_base, model_tier, explicit_model=None):
            captured_kwargs.update(kwargs_base)
            return _make_litellm_response()

        with patch.object(provider, "_call_llm_with_fallback", side_effect=mock_fallback), \
             patch("litellm.completion_cost", return_value=0.001):
            await provider.complete(
                "user prompt",
                system_prompt="고정 지시만",
                cache_system_prompt=True,
            )

        messages = captured_kwargs["messages"]
        sys_msg = messages[0]

        assert isinstance(sys_msg["content"], list)
        assert len(sys_msg["content"]) == 1
        assert sys_msg["content"][0]["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_no_cache_with_dynamic_concatenates(self, provider):
        """cache_system_prompt=False + system_prompt_dynamic → 텍스트 합산"""
        captured_kwargs = {}

        async def mock_fallback(kwargs_base, model_tier, explicit_model=None):
            captured_kwargs.update(kwargs_base)
            return _make_litellm_response()

        with patch.object(provider, "_call_llm_with_fallback", side_effect=mock_fallback), \
             patch("litellm.completion_cost", return_value=0.001):
            await provider.complete(
                "user prompt",
                system_prompt="고정 지시",
                system_prompt_dynamic="동적 영역",
                cache_system_prompt=False,
            )

        messages = captured_kwargs["messages"]
        sys_msg = messages[0]

        assert isinstance(sys_msg["content"], str)
        assert "고정 지시" in sys_msg["content"]
        assert "동적 영역" in sys_msg["content"]

    @pytest.mark.asyncio
    async def test_no_cache_no_dynamic_plain_text(self, provider):
        """cache_system_prompt=False + system_prompt_dynamic=None → 기존 동작"""
        captured_kwargs = {}

        async def mock_fallback(kwargs_base, model_tier, explicit_model=None):
            captured_kwargs.update(kwargs_base)
            return _make_litellm_response()

        with patch.object(provider, "_call_llm_with_fallback", side_effect=mock_fallback), \
             patch("litellm.completion_cost", return_value=0.001):
            await provider.complete(
                "user prompt",
                system_prompt="기존 프롬프트",
                cache_system_prompt=False,
            )

        messages = captured_kwargs["messages"]
        sys_msg = messages[0]

        assert isinstance(sys_msg["content"], str)
        assert sys_msg["content"] == "기존 프롬프트"

    @pytest.mark.asyncio
    async def test_user_message_always_last(self, provider):
        """사용자 메시지는 항상 마지막"""
        captured_kwargs = {}

        async def mock_fallback(kwargs_base, model_tier, explicit_model=None):
            captured_kwargs.update(kwargs_base)
            return _make_litellm_response()

        with patch.object(provider, "_call_llm_with_fallback", side_effect=mock_fallback), \
             patch("litellm.completion_cost", return_value=0.001):
            await provider.complete(
                "사용자 질문",
                system_prompt="시스템",
                system_prompt_dynamic="동적",
                cache_system_prompt=True,
            )

        messages = captured_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "사용자 질문"


# === REASONING_SYSTEM_DYNAMIC 템플릿 테스트 ===

class TestReasoningDynamicTemplate:
    """추론 동적 경계 템플릿 검증"""

    def test_template_format_with_memory(self):
        """메모리 있을 때 동적 영역 포맷"""
        result = REASONING_SYSTEM_DYNAMIC.format(memory_context="사용자 프로필 데이터")
        assert "사용자 프로필 데이터" in result
        assert "메모리 컨텍스트" in result

    def test_template_format_without_memory(self):
        """메모리 없을 때 동적 영역"""
        result = REASONING_SYSTEM_DYNAMIC.format(memory_context="메모리 없음")
        assert "메모리 없음" in result

    def test_static_prompts_no_format_variables(self):
        """고정 시스템 프롬프트에 포맷 가능한 변수가 없어야 캐시 안정성 보장

        {{}} 이스케이프된 중괄호(JSON 예시)는 허용 — 실제 format() 호출 불가
        """
        assert not _has_unescaped_braces(REASONING_SYSTEM), \
            "REASONING_SYSTEM에 이스케이프 안 된 {variable}이 있으면 캐시 불안정"

    def test_intent_system_has_only_escaped_braces(self):
        """INTENT_ANALYSIS_SYSTEM의 {{ }}는 JSON 예시용 이스케이프"""
        # {{ }} 는 str.format()에서 리터럴 { } 로 변환 → 캐시에 영향 없음
        assert not _has_unescaped_braces(INTENT_ANALYSIS_SYSTEM)

    def test_reflection_system_has_only_escaped_braces(self):
        """SELF_REFLECTION_SYSTEM의 {{ }}는 JSON 예시용 이스케이프"""
        assert not _has_unescaped_braces(SELF_REFLECTION_SYSTEM)


# === ReActAgent 동적 경계 통합 테스트 ===

class TestAgentDynamicBoundary:
    """에이전트의 동적 경계 패턴 통합 동작 검증"""

    @pytest.fixture
    def mock_llm(self):
        provider = MagicMock(spec=LLMProvider)
        provider.get_cost_summary.return_value = {
            "daily_cost_usd": 0.0,
            "monthly_cost_usd": 0.0,
            "total_requests": 0,
            "total_cached_tokens": 0,
        }
        return provider

    @pytest.fixture
    def mock_vector_store(self):
        return MagicMock()

    @pytest.mark.asyncio
    async def test_reason_passes_dynamic_boundary(self, mock_llm, mock_vector_store):
        """_reason이 system_prompt_dynamic을 전달하는지 확인"""
        # LLM complete 호출 기록
        mock_llm.complete = AsyncMock(return_value={
            "content": "추론 결과",
            "model": "test",
            "usage": MagicMock(cached_tokens=80),
        })

        agent = ReActAgent(mock_llm, mock_vector_store)
        state = AgentState(
            query="테스트 질문",
            intent={"surface_intent": "test"},
            memory_context="## 사용자 프로필\n이름: 승재",
            search_results=[],
        )

        result = await agent._reason(state)

        # llm.complete 호출 확인
        call_kwargs = mock_llm.complete.call_args
        assert call_kwargs.kwargs["system_prompt"] == REASONING_SYSTEM
        assert "system_prompt_dynamic" in call_kwargs.kwargs
        assert "승재" in call_kwargs.kwargs["system_prompt_dynamic"]
        assert call_kwargs.kwargs["cache_system_prompt"] is True

    @pytest.mark.asyncio
    async def test_reason_without_memory_still_passes_dynamic(self, mock_llm, mock_vector_store):
        """메모리 없어도 dynamic 영역 전달 (메모리 없음 텍스트)"""
        mock_llm.complete = AsyncMock(return_value={
            "content": "추론 결과",
            "model": "test",
            "usage": MagicMock(cached_tokens=0),
        })

        agent = ReActAgent(mock_llm, mock_vector_store)
        state = AgentState(
            query="테스트",
            intent={"surface_intent": "test"},
            memory_context="",
            search_results=[],
        )

        await agent._reason(state)

        call_kwargs = mock_llm.complete.call_args
        assert "메모리 없음" in call_kwargs.kwargs["system_prompt_dynamic"]

    @pytest.mark.asyncio
    async def test_intent_analysis_no_dynamic(self, mock_llm, mock_vector_store):
        """의도 분석은 동적 경계 불필요 — system_prompt_dynamic 미전달"""
        mock_llm.complete = AsyncMock(return_value={
            "content": '{"surface_intent":"t","deeper_intent":"","required_knowledge":[],'
                       '"search_queries":["t"],"complexity":"simple","recommended_action":"respond"}',
            "model": "test",
            "usage": MagicMock(cached_tokens=0),
        })

        agent = ReActAgent(mock_llm, mock_vector_store)
        state = AgentState(query="테스트")

        await agent._analyze_intent(state)

        call_kwargs = mock_llm.complete.call_args
        # 의도 분석은 system_prompt_dynamic을 전달하지 않아야 함
        assert call_kwargs.kwargs.get("system_prompt_dynamic") is None

    @pytest.mark.asyncio
    async def test_self_reflect_no_dynamic(self, mock_llm, mock_vector_store):
        """자기 성찰도 동적 경계 불필요"""
        mock_llm.complete = AsyncMock(return_value={
            "content": '{"quality_score":0.8,"issues":[],"should_retry":false,"improvement_suggestion":"good"}',
            "model": "test",
            "usage": MagicMock(cached_tokens=0),
        })

        agent = ReActAgent(mock_llm, mock_vector_store)
        state = AgentState(
            query="테스트",
            intent={"surface_intent": "test"},
            current_answer="답변",
            iteration=1,
        )

        await agent._self_reflect(state)

        call_kwargs = mock_llm.complete.call_args
        # 성찰은 system_prompt_dynamic을 전달하지 않아야 함
        assert call_kwargs.kwargs.get("system_prompt_dynamic") is None
