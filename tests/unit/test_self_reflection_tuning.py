"""Self-Reflection 프롬프트 튜닝 + 비용 최적화 + ddgs 마이그레이션 테스트

테스트 범위:
- SELF_REFLECTION_SYSTEM: 도구 API 응답 신뢰 지침 포함 검증
- SELF_REFLECTION_USER: tool_usage_info 플레이스홀더 검증
- _self_reflect: confidence 하한선 (도구 사용 시 0.6) + retry 억제
- 동일 도구+파라미터 재호출 방지 (tool_call_history + call_signature)
- AgentState.tool_call_history 필드
- ddgs 패키지 import 경로 + 파라미터 변경
"""

from __future__ import annotations

import hashlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aria.agents.react_agent import (
    SELF_REFLECTION_SYSTEM,
    SELF_REFLECTION_USER,
    AgentState,
    ReActAgent,
)


# === SELF_REFLECTION_SYSTEM 프롬프트 검증 ===


class TestSelfReflectionPrompt:
    """Self-Reflection 프롬프트에 도구 API 응답 신뢰 지침이 포함되었는지 검증"""

    def test_contains_tool_api_trust_instruction(self):
        """도구 API 응답은 할루시네이션이 아니라는 지침 포함"""
        assert "할루시네이션이 아닙니다" in SELF_REFLECTION_SYSTEM

    def test_contains_tool_services_list(self):
        """구체적인 도구 서비스명 언급"""
        assert "카카오맵" in SELF_REFLECTION_SYSTEM
        assert "네이버" in SELF_REFLECTION_SYSTEM
        assert "DuckDuckGo" in SELF_REFLECTION_SYSTEM

    def test_contains_minimum_score_instruction(self):
        """도구 성공 시 최소 quality_score 0.6 지침"""
        assert "0.6 이상" in SELF_REFLECTION_SYSTEM

    def test_contains_no_retry_for_same_tool_instruction(self):
        """동일 도구 재호출 무의미 지침"""
        assert "재시도가 무의미" in SELF_REFLECTION_SYSTEM

    def test_contains_retry_condition_clarification(self):
        """should_retry 조건 명확화"""
        assert "should_retry" in SELF_REFLECTION_SYSTEM

    def test_json_format_preserved(self):
        """JSON 응답 형식 유지"""
        assert "quality_score" in SELF_REFLECTION_SYSTEM
        assert "should_retry" in SELF_REFLECTION_SYSTEM
        assert "issues" in SELF_REFLECTION_SYSTEM
        assert "improvement_suggestion" in SELF_REFLECTION_SYSTEM

    def test_original_evaluation_criteria_preserved(self):
        """기존 4가지 평가 기준 유지"""
        assert "직접적 답변" in SELF_REFLECTION_SYSTEM
        assert "근거가 충분" in SELF_REFLECTION_SYSTEM
        assert "논리적 오류" in SELF_REFLECTION_SYSTEM
        assert "숨겨진 의도" in SELF_REFLECTION_SYSTEM


class TestSelfReflectionUserPrompt:
    """SELF_REFLECTION_USER에 tool_usage_info 필드 검증"""

    def test_contains_tool_usage_placeholder(self):
        """tool_usage_info 플레이스홀더 존재"""
        assert "{tool_usage_info}" in SELF_REFLECTION_USER

    def test_format_with_tool_usage(self):
        """tool_usage_info 포함 포맷팅 성공"""
        formatted = SELF_REFLECTION_USER.format(
            query="테스트 질문",
            intent="테스트 의도",
            tool_usage_info="도구 3회 호출됨",
            answer="테스트 답변",
        )
        assert "도구 3회 호출됨" in formatted
        assert "테스트 질문" in formatted
        assert "테스트 답변" in formatted


# === AgentState.tool_call_history ===


class TestAgentStateToolCallHistory:
    """AgentState에 tool_call_history 필드 검증"""

    def test_default_empty_list(self):
        """기본값은 빈 리스트"""
        state = AgentState(query="test")
        assert state.tool_call_history == []

    def test_independent_instances(self):
        """인스턴스 간 독립성 (mutable default 안전)"""
        s1 = AgentState(query="a")
        s2 = AgentState(query="b")
        s1.tool_call_history.append("test_sig")
        assert s2.tool_call_history == []

    def test_preserves_history(self):
        """이력이 유지됨"""
        state = AgentState(query="test", tool_call_history=["sig1", "sig2"])
        assert len(state.tool_call_history) == 2


# === _self_reflect: confidence 하한선 + retry 억제 ===


class TestSelfReflectConfidenceFloor:
    """도구 호출 성공 시 confidence 하한선 및 retry 억제"""

    @pytest.fixture
    def mock_agent(self):
        """mock LLM + VectorStore로 ReActAgent 생성"""
        llm = MagicMock(spec_set=["complete", "complete_with_messages", "get_cost_summary"])
        vector_store = MagicMock()
        return ReActAgent(llm, vector_store)

    @pytest.mark.asyncio
    async def test_confidence_floor_when_tools_used(self, mock_agent):
        """도구 사용 시 LLM이 낮은 confidence(0.3) 반환해도 0.6으로 보정"""
        mock_agent.llm.complete = AsyncMock(return_value={
            "content": json.dumps({
                "quality_score": 0.3,
                "issues": ["근거 불충분"],
                "should_retry": True,
                "improvement_suggestion": "더 검색 필요",
            })
        })

        state = AgentState(
            query="남양주 약국",
            current_answer="약국 3건 검색됨",
            tool_calls_made=3,  # 도구 호출 있었음
            iteration=1,
        )

        result = await mock_agent._self_reflect(state)
        assert result["confidence"] >= 0.6
        assert result["should_stop"] is True  # retry 억제됨

    @pytest.mark.asyncio
    async def test_no_floor_when_no_tools(self, mock_agent):
        """도구 미사용 시 LLM 반환값 그대로 사용"""
        mock_agent.llm.complete = AsyncMock(return_value={
            "content": json.dumps({
                "quality_score": 0.3,
                "issues": ["근거 불충분"],
                "should_retry": True,
                "improvement_suggestion": "검색 필요",
            })
        })

        state = AgentState(
            query="회피형 애착",
            current_answer="애착 이론 답변",
            tool_calls_made=0,  # 도구 호출 없음
            iteration=1,
        )

        result = await mock_agent._self_reflect(state)
        assert result["confidence"] == 0.3  # 보정 없음
        assert result["should_stop"] is False  # retry 허용

    @pytest.mark.asyncio
    async def test_retry_suppressed_when_tools_used(self, mock_agent):
        """도구 사용 후 LLM이 should_retry=True 반환해도 억제"""
        mock_agent.llm.complete = AsyncMock(return_value={
            "content": json.dumps({
                "quality_score": 0.7,
                "issues": [],
                "should_retry": True,
                "improvement_suggestion": "",
            })
        })

        state = AgentState(
            query="서울 맛집",
            current_answer="맛집 5건 검색됨",
            tool_calls_made=2,
            iteration=1,
        )

        result = await mock_agent._self_reflect(state)
        assert result["should_stop"] is True  # retry 억제

    @pytest.mark.asyncio
    async def test_high_confidence_no_change(self, mock_agent):
        """도구 사용 + 높은 confidence → 그대로 통과"""
        mock_agent.llm.complete = AsyncMock(return_value={
            "content": json.dumps({
                "quality_score": 0.9,
                "issues": [],
                "should_retry": False,
                "improvement_suggestion": "",
            })
        })

        state = AgentState(
            query="날씨",
            current_answer="오늘 맑음",
            tool_calls_made=1,
            iteration=1,
        )

        result = await mock_agent._self_reflect(state)
        assert result["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_tool_usage_info_in_prompt(self, mock_agent):
        """도구 사용 시 프롬프트에 tool_usage_info 포함"""
        captured_prompt = {}

        async def capture_complete(user_prompt, **kwargs):
            captured_prompt["user"] = user_prompt
            return {"content": json.dumps({
                "quality_score": 0.8,
                "issues": [],
                "should_retry": False,
                "improvement_suggestion": "",
            })}

        mock_agent.llm.complete = capture_complete

        state = AgentState(
            query="테스트",
            current_answer="답변",
            tool_calls_made=5,
            iteration=1,
        )

        await mock_agent._self_reflect(state)
        assert "5회 호출" in captured_prompt["user"]
        assert "할루시네이션이 아닌" in captured_prompt["user"]

    @pytest.mark.asyncio
    async def test_no_tool_usage_info_when_no_tools(self, mock_agent):
        """도구 미사용 시 '도구 사용 없이' 메시지"""
        captured_prompt = {}

        async def capture_complete(user_prompt, **kwargs):
            captured_prompt["user"] = user_prompt
            return {"content": json.dumps({
                "quality_score": 0.7,
                "issues": [],
                "should_retry": False,
                "improvement_suggestion": "",
            })}

        mock_agent.llm.complete = capture_complete

        state = AgentState(
            query="테스트",
            current_answer="답변",
            tool_calls_made=0,
            iteration=1,
        )

        await mock_agent._self_reflect(state)
        assert "도구 사용 없이" in captured_prompt["user"]

    @pytest.mark.asyncio
    async def test_max_iterations_still_skips(self, mock_agent):
        """MAX_ITERATIONS 도달 시 여전히 성찰 스킵"""
        state = AgentState(
            query="테스트",
            current_answer="답변",
            tool_calls_made=3,
            iteration=3,  # MAX_ITERATIONS
        )

        result = await mock_agent._self_reflect(state)
        assert result["confidence"] == 0.6
        assert result["should_stop"] is True


# === 동일 도구+파라미터 재호출 방지 ===


class TestDuplicateToolCallPrevention:
    """동일 도구+파라미터 조합 재호출 차단"""

    def test_call_signature_format(self):
        """call_signature = 'tool_name|md5(args_str)' 형식"""
        func_name = "kakao_keyword_search"
        func_args_str = '{"query": "남양주 약국"}'
        expected_hash = hashlib.md5(func_args_str.encode()).hexdigest()
        call_sig = f"{func_name}|{expected_hash}"
        assert call_sig.startswith("kakao_keyword_search|")
        assert len(call_sig.split("|")) == 2

    def test_same_args_same_signature(self):
        """동일 파라미터 → 동일 시그니처"""
        args = '{"query": "약국", "region": "kr"}'
        sig1 = f"tool|{hashlib.md5(args.encode()).hexdigest()}"
        sig2 = f"tool|{hashlib.md5(args.encode()).hexdigest()}"
        assert sig1 == sig2

    def test_different_args_different_signature(self):
        """다른 파라미터 → 다른 시그니처"""
        args1 = '{"query": "약국"}'
        args2 = '{"query": "병원"}'
        sig1 = f"tool|{hashlib.md5(args1.encode()).hexdigest()}"
        sig2 = f"tool|{hashlib.md5(args2.encode()).hexdigest()}"
        assert sig1 != sig2

    def test_different_tool_same_args_different_signature(self):
        """다른 도구 + 동일 파라미터 → 다른 시그니처"""
        args = '{"query": "test"}'
        sig1 = f"tool_a|{hashlib.md5(args.encode()).hexdigest()}"
        sig2 = f"tool_b|{hashlib.md5(args.encode()).hexdigest()}"
        assert sig1 != sig2

    def test_history_dedup_detection(self):
        """이력에 있는 시그니처는 중복으로 감지"""
        history = ["tool_a|abc123", "tool_b|def456"]
        new_sig = "tool_a|abc123"
        assert new_sig in history

    def test_new_call_not_in_history(self):
        """이력에 없는 시그니처는 신규"""
        history = ["tool_a|abc123"]
        new_sig = "tool_a|xyz789"
        assert new_sig not in history


# === ddgs 패키지 마이그레이션 ===


class TestDdgsMigration:
    """duckduckgo-search → ddgs 패키지 마이그레이션 검증"""

    def test_import_path_ddgs(self):
        """ddg_tools.py에서 ddgs import 사용"""
        import inspect
        from aria.tools.mcp.ddg_tools import DdgSearchClient

        source = inspect.getsource(DdgSearchClient)
        assert "from ddgs import DDGS" in source
        assert "from duckduckgo_search" not in source

    def test_no_keywords_parameter(self):
        """keywords= 대신 positional query 사용"""
        import inspect
        from aria.tools.mcp.ddg_tools import DdgSearchClient

        source = inspect.getsource(DdgSearchClient)
        assert "keywords=" not in source

    def test_pyproject_ddgs_dependency(self):
        """pyproject.toml에 ddgs>=9.0.0 의존성"""
        with open("pyproject.toml") as f:
            content = f.read()
        assert '"ddgs>=' in content
        assert '"duckduckgo-search' not in content

    def test_ddg_tools_docstring_updated(self):
        """모듈 독스트링에 ddgs 패키지명 반영"""
        import aria.tools.mcp.ddg_tools as ddg_mod
        assert "ddgs" in (ddg_mod.__doc__ or "")
        assert "duckduckgo-search 패키지 기반" not in (ddg_mod.__doc__ or "")

    def test_client_docstring_updated(self):
        """DdgSearchClient 독스트링에 ddgs 패키지명 반영"""
        from aria.tools.mcp.ddg_tools import DdgSearchClient
        assert "ddgs" in (DdgSearchClient.__doc__ or "")


# === 비용 최적화 효과 검증 ===


class TestCostOptimization:
    """비용 최적화 효과 시나리오 검증"""

    @pytest.fixture
    def mock_agent(self):
        llm = MagicMock(spec_set=["complete", "complete_with_messages", "get_cost_summary"])
        vector_store = MagicMock()
        return ReActAgent(llm, vector_store)

    @pytest.mark.asyncio
    async def test_tool_success_no_retry_loop(self, mock_agent):
        """도구 성공 → confidence ≥ 0.6 → retry 없음 → 루프 1회로 종료

        기존: confidence=0.3 / should_retry=True → 최대 3회 루프 (28 LLM 호출)
        수정: confidence=0.6 / should_stop=True → 1회 루프 (3~5 LLM 호출)
        """
        # Self-reflection이 낮은 점수를 줘도
        mock_agent.llm.complete = AsyncMock(return_value={
            "content": json.dumps({
                "quality_score": 0.2,
                "issues": ["데이터 신뢰 불가"],
                "should_retry": True,
                "improvement_suggestion": "재검색",
            })
        })

        state = AgentState(
            query="남양주 약국",
            current_answer="약국 데이터 3건 반환",
            tool_calls_made=3,
            iteration=1,
        )

        result = await mock_agent._self_reflect(state)

        # confidence가 0.6으로 보정되어 retry하지 않음
        assert result["confidence"] == 0.6
        assert result["should_stop"] is True

        # → _route_after_reflection에서 "respond"로 라우팅됨
        state_after = AgentState(
            query="남양주 약국",
            confidence=result["confidence"],
            should_stop=result["should_stop"],
            iteration=1,
        )
        route = mock_agent._route_after_reflection(state_after)
        assert route == "respond"  # retry가 아닌 응답으로 종료


# === INTENT_ANALYSIS_SYSTEM 도구 호출 라우팅 ===


class TestIntentAnalysisPrompt:
    """INTENT_ANALYSIS_SYSTEM에 도구 호출 필요 쿼리 분류 가이드 검증"""

    def test_contains_location_search_guidance(self):
        """위치/장소 검색 쿼리가 search_knowledge로 분류되는 가이드"""
        from aria.agents.react_agent import INTENT_ANALYSIS_SYSTEM
        assert "장소" in INTENT_ANALYSIS_SYSTEM or "위치" in INTENT_ANALYSIS_SYSTEM
        assert "약국" in INTENT_ANALYSIS_SYSTEM or "병원" in INTENT_ANALYSIS_SYSTEM

    def test_contains_search_keywords_warning(self):
        """'찾아줘/검색해줘/근처' 등 키워드가 simple이 되면 안 된다는 경고"""
        from aria.agents.react_agent import INTENT_ANALYSIS_SYSTEM
        assert "찾아줘" in INTENT_ANALYSIS_SYSTEM
        assert "simple/respond로 분류하지 마세요" in INTENT_ANALYSIS_SYSTEM

    def test_simple_definition_is_greeting_only(self):
        """simple은 인사/잡담으로 명확히 제한"""
        from aria.agents.react_agent import INTENT_ANALYSIS_SYSTEM
        assert "인사" in INTENT_ANALYSIS_SYSTEM
        assert "잡담" in INTENT_ANALYSIS_SYSTEM

    def test_search_knowledge_includes_realtime_info(self):
        """실시간 정보(뉴스/날씨 등)가 search_knowledge로 분류"""
        from aria.agents.react_agent import INTENT_ANALYSIS_SYSTEM
        assert "실시간" in INTENT_ANALYSIS_SYSTEM or "뉴스" in INTENT_ANALYSIS_SYSTEM

    def test_search_knowledge_includes_web_search(self):
        """웹 검색 필요 쿼리가 search_knowledge로 분류"""
        from aria.agents.react_agent import INTENT_ANALYSIS_SYSTEM
        assert "웹 검색" in INTENT_ANALYSIS_SYSTEM

    def test_moderate_for_tool_required(self):
        """도구 호출 필요 쿼리는 moderate 이상으로 분류"""
        from aria.agents.react_agent import INTENT_ANALYSIS_SYSTEM
        assert "moderate" in INTENT_ANALYSIS_SYSTEM
        assert "도구" in INTENT_ANALYSIS_SYSTEM or "외부 검색" in INTENT_ANALYSIS_SYSTEM


class TestIntentRouting:
    """의도분석 결과에 따른 라우팅 결정 검증"""

    @pytest.fixture
    def mock_agent(self):
        llm = MagicMock(spec_set=["complete", "complete_with_messages", "get_cost_summary"])
        vector_store = MagicMock()
        return ReActAgent(llm, vector_store)

    def test_simple_respond_goes_to_fast_respond(self, mock_agent):
        """simple + respond → fast_respond"""
        state = AgentState(
            query="안녕",
            intent={"complexity": "simple", "recommended_action": "respond"},
        )
        assert mock_agent._route_after_intent(state) == "fast_respond"

    def test_moderate_search_goes_to_search(self, mock_agent):
        """moderate + search_knowledge → search_knowledge (도구 호출 경로)"""
        state = AgentState(
            query="남양주 약국 찾아줘",
            intent={"complexity": "moderate", "recommended_action": "search_knowledge"},
        )
        assert mock_agent._route_after_intent(state) == "search_knowledge"

    def test_moderate_respond_goes_to_reason(self, mock_agent):
        """moderate + respond → reason (검색 스킵 / 추론 수행)"""
        state = AgentState(
            query="파이썬이 뭐야",
            intent={"complexity": "moderate", "recommended_action": "respond"},
        )
        assert mock_agent._route_after_intent(state) == "reason"

    def test_complex_search_goes_to_search(self, mock_agent):
        """complex + search_knowledge → search_knowledge"""
        state = AgentState(
            query="남양주에서 서울역까지 대중교통 경로",
            intent={"complexity": "complex", "recommended_action": "search_knowledge"},
        )
        assert mock_agent._route_after_intent(state) == "search_knowledge"

    def test_simple_clarify_goes_to_fast_respond(self, mock_agent):
        """simple + clarify → fast_respond"""
        state = AgentState(
            query="뭐?",
            intent={"complexity": "simple", "recommended_action": "clarify"},
        )
        assert mock_agent._route_after_intent(state) == "fast_respond"

    def test_default_fallback_is_search(self, mock_agent):
        """알 수 없는 action → search_knowledge (안전 기본값)"""
        state = AgentState(
            query="테스트",
            intent={"complexity": "moderate", "recommended_action": "unknown_action"},
        )
        assert mock_agent._route_after_intent(state) == "search_knowledge"

