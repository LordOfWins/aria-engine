"""프롬프트 튜닝 + Fast Path 단위 테스트

테스트 범위:
- _extract_answer: <answer> 태그 추출 / 태그 없는 경우 하위 호환
- REASONING_SYSTEM: 추론+답변 분리 프롬프트 구조 검증
- fast_respond: simple 의도 → LLM 1회 호출 즉답
- _route_after_intent: fast path 라우팅 결정
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aria.providers.llm_provider import LLMProvider
from aria.agents.react_agent import (
    FAST_RESPOND_SYSTEM,
    FAST_RESPOND_USER,
    REASONING_SYSTEM,
    ReActAgent,
    AgentState,
    _extract_answer,
)


# === _extract_answer 테스트 ===


class TestExtractAnswer:
    """<answer> 태그 기반 답변 추출"""

    def test_extracts_answer_from_tags(self):
        """정상 <reasoning>+<answer> 구조에서 answer만 추출"""
        content = (
            "<reasoning>\n"
            "사용자가 인사를 했다. 간단한 인사 응답이 필요하다.\n"
            "확신도: 0.95\n"
            "</reasoning>\n\n"
            "<answer>\n"
            "안녕하세요! 무엇을 도와드릴까요?\n"
            "</answer>"
        )
        assert _extract_answer(content) == "안녕하세요! 무엇을 도와드릴까요?"

    def test_extracts_answer_with_whitespace(self):
        """앞뒤 공백/개행 정리"""
        content = "<answer>\n\n  답변입니다  \n\n</answer>"
        assert _extract_answer(content) == "답변입니다"

    def test_extracts_multiline_answer(self):
        """여러 줄 답변"""
        content = (
            "<reasoning>사고 과정</reasoning>\n"
            "<answer>\n"
            "첫째줄\n"
            "둘째줄\n"
            "셋째줄\n"
            "</answer>"
        )
        result = _extract_answer(content)
        assert "첫째줄" in result
        assert "둘째줄" in result
        assert "셋째줄" in result

    def test_no_tags_returns_original(self):
        """태그 없으면 원본 반환 (하위 호환)"""
        content = "이것은 태그 없는 일반 텍스트 답변입니다."
        assert _extract_answer(content) == content

    def test_empty_answer_tag(self):
        """빈 <answer> 태그"""
        content = "<reasoning>사고</reasoning>\n<answer></answer>"
        assert _extract_answer(content) == ""

    def test_answer_without_reasoning(self):
        """<reasoning> 없이 <answer>만 있는 경우"""
        content = "<answer>직접 답변</answer>"
        assert _extract_answer(content) == "직접 답변"

    def test_reasoning_not_leaked_in_result(self):
        """추론 과정이 결과에 포함되지 않아야 함"""
        content = (
            "<reasoning>\n"
            "1. 알려진 정보: Python은 프로그래밍 언어\n"
            "2. 부족한 정보: 없음\n"
            "3. 추론: 기본 설명 제공\n"
            "4. 확신도: 0.9\n"
            "</reasoning>\n"
            "<answer>\n"
            "Python은 범용 프로그래밍 언어입니다.\n"
            "</answer>"
        )
        result = _extract_answer(content)
        assert "확신도" not in result
        assert "알려진 정보" not in result
        assert "추론" not in result
        assert "Python은 범용 프로그래밍 언어입니다." == result

    def test_answer_with_markdown(self):
        """마크다운 포함 답변"""
        content = (
            "<reasoning>분석 완료</reasoning>\n"
            "<answer>\n"
            "## 요약\n"
            "- 항목 1\n"
            "- 항목 2\n"
            "</answer>"
        )
        result = _extract_answer(content)
        assert "## 요약" in result
        assert "- 항목 1" in result


# === REASONING_SYSTEM 프롬프트 구조 검증 ===


class TestReasoningSystemPrompt:
    """REASONING_SYSTEM 프롬프트 검증"""

    def test_contains_answer_tag_instruction(self):
        """<answer> 태그 사용 지시가 포함되어야 함"""
        assert "<answer>" in REASONING_SYSTEM
        assert "</answer>" in REASONING_SYSTEM

    def test_contains_reasoning_tag_instruction(self):
        """<reasoning> 태그 사용 지시가 포함되어야 함"""
        assert "<reasoning>" in REASONING_SYSTEM
        assert "</reasoning>" in REASONING_SYSTEM

    def test_instructs_answer_only_visible(self):
        """사용자에게 answer만 보인다는 지시가 포함"""
        assert "최종 답변만" in REASONING_SYSTEM or "최종 답변" in REASONING_SYSTEM

    def test_no_old_numbered_format(self):
        """이전 번호 형식(1. 현재까지/5. 확신도 (0~1))이 없어야 함
        참고: <reasoning> 내부의 사고 단계 번호(1. 질문 이해 등)는 허용 (사용자에게 비공개)
        """
        assert "1. 현재까지" not in REASONING_SYSTEM
        assert "5. 확신도 (0" not in REASONING_SYSTEM  # 이전 형식: "5. 확신도 (0~1)"


# === FAST_RESPOND_SYSTEM 프롬프트 검증 ===


class TestFastRespondPrompt:
    """Fast path 프롬프트 검증"""

    def test_exists_and_not_empty(self):
        assert FAST_RESPOND_SYSTEM
        assert len(FAST_RESPOND_SYSTEM) > 10

    def test_is_concise(self):
        """fast path 프롬프트는 캐시 효율을 위해 간결해야 함"""
        assert len(FAST_RESPOND_SYSTEM) < 500

    def test_fast_respond_user_has_query_placeholder(self):
        assert "{query}" in FAST_RESPOND_USER


# === 라우팅 테스트 ===


class TestRouteAfterIntent:
    """의도 분석 후 라우팅 결정 테스트"""

    @pytest.fixture
    def agent(self):
        llm = MagicMock(spec=LLMProvider)
        vector_store = MagicMock()
        return ReActAgent(llm, vector_store)

    def test_simple_respond_goes_to_fast_respond(self, agent):
        """simple + respond → fast_respond"""
        state = AgentState(
            intent={"recommended_action": "respond", "complexity": "simple"},
        )
        assert agent._route_after_intent(state) == "fast_respond"

    def test_simple_clarify_goes_to_fast_respond(self, agent):
        """simple + clarify → fast_respond"""
        state = AgentState(
            intent={"recommended_action": "clarify", "complexity": "simple"},
        )
        assert agent._route_after_intent(state) == "fast_respond"

    def test_moderate_respond_goes_to_reason(self, agent):
        """moderate + respond → reason (fast path 아님)"""
        state = AgentState(
            intent={"recommended_action": "respond", "complexity": "moderate"},
        )
        assert agent._route_after_intent(state) == "reason"

    def test_complex_respond_goes_to_reason(self, agent):
        """complex + respond → reason"""
        state = AgentState(
            intent={"recommended_action": "respond", "complexity": "complex"},
        )
        assert agent._route_after_intent(state) == "reason"

    def test_simple_search_goes_to_search(self, agent):
        """simple + search_knowledge → search_knowledge (검색 필요하면 fast path 아님)"""
        state = AgentState(
            intent={"recommended_action": "search_knowledge", "complexity": "simple"},
        )
        assert agent._route_after_intent(state) == "search_knowledge"

    def test_default_complexity_goes_to_search(self, agent):
        """기본값(moderate + search_knowledge) → search_knowledge"""
        state = AgentState(
            intent={"recommended_action": "search_knowledge"},
        )
        assert agent._route_after_intent(state) == "search_knowledge"

    def test_missing_intent_keys_safe(self, agent):
        """intent에 키가 없어도 안전하게 동작"""
        state = AgentState(intent={})
        # 기본값: moderate + search_knowledge → search_knowledge
        assert agent._route_after_intent(state) == "search_knowledge"


# === fast_respond 노드 테스트 ===


class TestFastRespondNode:
    """_fast_respond 노드 동작 테스트"""

    @pytest.fixture
    def mock_llm(self):
        provider = MagicMock(spec=LLMProvider)
        provider.get_cost_summary.return_value = {
            "daily_cost_usd": 0.0, "monthly_cost_usd": 0.0,
            "total_requests": 0, "total_cached_tokens": 0,
        }
        return provider

    @pytest.mark.asyncio
    async def test_fast_respond_returns_answer(self, mock_llm):
        """fast_respond가 LLM 응답을 current_answer로 반환"""
        mock_llm.complete = AsyncMock(return_value={
            "content": "안녕하세요! 도와드릴까요?",
            "model": "test", "usage": MagicMock(cached_tokens=0),
        })

        agent = ReActAgent(mock_llm, MagicMock())
        state = AgentState(query="안녕")
        result = await agent._fast_respond(state)

        assert result["current_answer"] == "안녕하세요! 도와드릴까요?"
        assert result["confidence"] == 0.9
        assert result["iteration"] == 1

    @pytest.mark.asyncio
    async def test_fast_respond_uses_cheap_model(self, mock_llm):
        """fast_respond는 cheap 모델을 사용"""
        mock_llm.complete = AsyncMock(return_value={
            "content": "hi",
            "model": "test", "usage": MagicMock(cached_tokens=0),
        })

        agent = ReActAgent(mock_llm, MagicMock())
        state = AgentState(query="hi")
        await agent._fast_respond(state)

        call_kwargs = mock_llm.complete.call_args
        assert call_kwargs.kwargs["model_tier"] == "cheap"

    @pytest.mark.asyncio
    async def test_fast_respond_graceful_on_error(self, mock_llm):
        """fast_respond LLM 실패 시 빈 답변 반환"""
        mock_llm.complete = AsyncMock(side_effect=RuntimeError("LLM down"))

        agent = ReActAgent(mock_llm, MagicMock())
        state = AgentState(query="hi")
        result = await agent._fast_respond(state)

        assert result["current_answer"] == ""
        assert result["confidence"] == 0.5

    @pytest.mark.asyncio
    async def test_fast_respond_includes_messages(self, mock_llm):
        """fast_respond가 assistant 메시지를 포함"""
        mock_llm.complete = AsyncMock(return_value={
            "content": "반가워요!",
            "model": "test", "usage": MagicMock(cached_tokens=0),
        })

        agent = ReActAgent(mock_llm, MagicMock())
        state = AgentState(query="반가워")
        result = await agent._fast_respond(state)

        assert result["messages"] == [{"role": "assistant", "content": "반가워요!"}]

    @pytest.mark.asyncio
    async def test_fast_respond_uses_fast_system_prompt(self, mock_llm):
        """fast_respond는 FAST_RESPOND_SYSTEM 프롬프트를 사용"""
        mock_llm.complete = AsyncMock(return_value={
            "content": "답변",
            "model": "test", "usage": MagicMock(cached_tokens=0),
        })

        agent = ReActAgent(mock_llm, MagicMock())
        state = AgentState(query="hi")
        await agent._fast_respond(state)

        call_kwargs = mock_llm.complete.call_args
        assert call_kwargs.kwargs["system_prompt"] == FAST_RESPOND_SYSTEM

    @pytest.mark.asyncio
    async def test_fast_respond_propagates_killswitch(self, mock_llm):
        """KillSwitch 에러는 fast_respond에서 catch 안 됨"""
        from aria.core.exceptions import KillSwitchError
        mock_llm.complete = AsyncMock(side_effect=KillSwitchError(
            "비용 상한 초과", daily_cost=100, monthly_cost=200
        ))

        agent = ReActAgent(mock_llm, MagicMock())
        state = AgentState(query="hi")
        with pytest.raises(KillSwitchError):
            await agent._fast_respond(state)


# === _reason에서 answer 추출 테스트 ===


class TestReasonAnswerExtraction:
    """_reason 노드에서 <answer> 태그 추출이 적용되는지 검증"""

    @pytest.fixture
    def mock_llm(self):
        provider = MagicMock(spec=LLMProvider)
        provider.get_cost_summary.return_value = {
            "daily_cost_usd": 0.0, "monthly_cost_usd": 0.0,
            "total_requests": 0, "total_cached_tokens": 0,
        }
        return provider

    @pytest.mark.asyncio
    async def test_reason_extracts_answer_tag(self, mock_llm):
        """_reason이 <answer> 태그에서 답변만 추출"""
        llm_response = (
            "<reasoning>\n내부 사고 과정\n</reasoning>\n"
            "<answer>\n깨끗한 답변만 여기에\n</answer>"
        )
        mock_llm.complete = AsyncMock(return_value={
            "content": llm_response,
            "model": "test", "usage": MagicMock(cached_tokens=0),
        })

        agent = ReActAgent(mock_llm, MagicMock())
        state = AgentState(
            query="테스트",
            intent={"recommended_action": "respond", "search_queries": []},
        )
        result = await agent._reason(state)

        assert result["current_answer"] == "깨끗한 답변만 여기에"
        assert "내부 사고 과정" not in result["current_answer"]

    @pytest.mark.asyncio
    async def test_reason_backward_compatible_no_tags(self, mock_llm):
        """태그 없는 LLM 응답도 정상 동작 (하위 호환)"""
        mock_llm.complete = AsyncMock(return_value={
            "content": "태그 없는 일반 응답입니다.",
            "model": "test", "usage": MagicMock(cached_tokens=0),
        })

        agent = ReActAgent(mock_llm, MagicMock())
        state = AgentState(
            query="테스트",
            intent={"recommended_action": "respond", "search_queries": []},
        )
        result = await agent._reason(state)

        assert result["current_answer"] == "태그 없는 일반 응답입니다."

    @pytest.mark.asyncio
    async def test_reasoning_steps_keep_full_content(self, mock_llm):
        """reasoning_steps에는 전체 내용(태그 포함)이 기록됨"""
        llm_response = (
            "<reasoning>추론 과정</reasoning>\n"
            "<answer>깨끗한 답변</answer>"
        )
        mock_llm.complete = AsyncMock(return_value={
            "content": llm_response,
            "model": "test", "usage": MagicMock(cached_tokens=0),
        })

        agent = ReActAgent(mock_llm, MagicMock())
        state = AgentState(
            query="테스트",
            intent={"recommended_action": "respond", "search_queries": []},
        )
        result = await agent._reason(state)

        # reasoning_steps에는 전체 원본이 기록됨
        steps_text = " ".join(result["reasoning_steps"])
        assert "<reasoning>" in steps_text or "추론 과정" in steps_text
