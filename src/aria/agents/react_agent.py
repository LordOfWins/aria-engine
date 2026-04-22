"""ARIA Engine - ReAct Agent (LangGraph)

Think → Act → Observe 루프 기반 자율 추론 에이전트
- 질문 분석 → 검색 전략 결정 → 실행 → 결과 검증 → 답변
- Self-Reflection 노드로 답변 품질 자체 평가
- 최대 반복 횟수 제한으로 무한루프 방지
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Annotated

import structlog
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from aria.providers.llm_provider import LLMProvider
from aria.rag.vector_store import VectorStore

logger = structlog.get_logger()

MAX_ITERATIONS = 3  # 최대 ReAct 루프 반복 횟수


class AgentAction(str, Enum):
    """에이전트가 취할 수 있는 행동"""
    SEARCH_KNOWLEDGE = "search_knowledge"      # 벡터DB 검색
    SEARCH_WEB = "search_web"                  # 웹 검색 (추후 구현)
    REASON = "reason"                          # 추가 추론
    RESPOND = "respond"                        # 최종 답변
    CLARIFY = "clarify"                        # 사용자에게 명확화 요청


@dataclass
class AgentState:
    """에이전트 상태 (LangGraph State)"""
    messages: Annotated[list[dict[str, str]], add_messages] = field(default_factory=list)
    query: str = ""
    intent: dict[str, Any] = field(default_factory=dict)
    search_results: list[dict[str, Any]] = field(default_factory=list)
    reasoning_steps: list[str] = field(default_factory=list)
    current_answer: str = ""
    confidence: float = 0.0
    iteration: int = 0
    should_stop: bool = False


INTENT_ANALYSIS_PROMPT = """당신은 사용자 의도 분석 전문가입니다.

사용자 질문을 분석하여 다음 JSON 형태로 응답하세요:
{{
    "surface_intent": "표면적 질문의 핵심",
    "deeper_intent": "질문 뒤에 숨겨진 실제 의도/욕구",
    "required_knowledge": ["필요한 지식 영역 목록"],
    "search_queries": ["벡터DB 검색에 사용할 쿼리 목록 (최대 3개)"],
    "complexity": "simple|moderate|complex",
    "recommended_action": "search_knowledge|reason|respond|clarify"
}}

사용자 질문: {query}"""


REASONING_PROMPT = """당신은 논리적 추론 전문가입니다.

주어진 정보를 바탕으로 단계적으로 사고하여 답변을 도출하세요.

## 사용자 질문
{query}

## 의도 분석
{intent}

## 검색된 관련 정보
{search_results}

## 이전 추론 단계
{previous_reasoning}

다음 형식으로 응답하세요:
1. 현재까지 알고 있는 것을 정리
2. 부족한 정보가 있다면 무엇인지
3. 논리적 추론 과정
4. 결론 및 답변
5. 확신도 (0.0 ~ 1.0)"""


SELF_REFLECTION_PROMPT = """당신은 답변 품질 평가 전문가입니다.

다음 답변의 품질을 평가하세요:

## 원래 질문
{query}

## 사용자의 실제 의도
{intent}

## 현재 답변
{answer}

평가 기준:
1. 질문에 대한 직접적 답변인가?
2. 근거가 충분한가?
3. 논리적 오류가 없는가?
4. 사용자의 숨겨진 의도도 충족하는가?

JSON으로 응답하세요:
{{
    "quality_score": 0.0~1.0,
    "issues": ["발견된 문제점"],
    "should_retry": true/false,
    "improvement_suggestion": "개선 방향"
}}"""


class ReActAgent:
    """LangGraph 기반 ReAct 에이전트

    사용법:
        agent = ReActAgent(llm_provider, vector_store)
        result = await agent.run("회피형 애착 패턴에 대해 설명해줘", collection="psychology_kb")
    """

    def __init__(
        self,
        llm: LLMProvider,
        vector_store: VectorStore,
    ) -> None:
        self.llm = llm
        self.vector_store = vector_store
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """LangGraph 워크플로우 구성"""
        workflow = StateGraph(AgentState)

        # 노드 등록
        workflow.add_node("analyze_intent", self._analyze_intent)
        workflow.add_node("search_knowledge", self._search_knowledge)
        workflow.add_node("reason", self._reason)
        workflow.add_node("self_reflect", self._self_reflect)
        workflow.add_node("respond", self._respond)

        # 엣지 연결
        workflow.set_entry_point("analyze_intent")
        workflow.add_conditional_edges(
            "analyze_intent",
            self._route_after_intent,
            {
                "search_knowledge": "search_knowledge",
                "reason": "reason",
                "respond": "respond",
            },
        )
        workflow.add_edge("search_knowledge", "reason")
        workflow.add_edge("reason", "self_reflect")
        workflow.add_conditional_edges(
            "self_reflect",
            self._route_after_reflection,
            {
                "retry": "search_knowledge",
                "respond": "respond",
            },
        )
        workflow.add_edge("respond", END)

        return workflow.compile()

    async def _analyze_intent(self, state: AgentState) -> dict[str, Any]:
        """Step 1: 사용자 의도 분석"""
        prompt = INTENT_ANALYSIS_PROMPT.format(query=state.query)

        result = await self.llm.complete(
            prompt,
            model_tier="cheap",  # 의도 분석은 저비용 모델로 충분
            temperature=0.3,
        )

        # JSON 파싱 시도
        import json
        try:
            content = result["content"]
            # JSON 블록 추출
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            intent = json.loads(content.strip())
        except (json.JSONDecodeError, IndexError):
            intent = {
                "surface_intent": state.query,
                "deeper_intent": "",
                "required_knowledge": [],
                "search_queries": [state.query],
                "complexity": "moderate",
                "recommended_action": "search_knowledge",
            }

        logger.info("intent_analyzed", intent=intent)
        return {"intent": intent}

    def _route_after_intent(self, state: AgentState) -> str:
        """의도 분석 후 라우팅 결정"""
        action = state.intent.get("recommended_action", "search_knowledge")
        complexity = state.intent.get("complexity", "moderate")

        if action == "respond" and complexity == "simple":
            return "respond"
        if action == "clarify":
            return "respond"  # 명확화 요청도 응답으로 처리
        return "search_knowledge"

    async def _search_knowledge(self, state: AgentState) -> dict[str, Any]:
        """Step 2: 지식 검색 (RAG)"""
        # 이미 검색 결과가 있으면 재검색 안 함
        if state.search_results:
            logger.debug("search_skipped", reason="already_has_results")
            return {"search_results": state.search_results}

        queries = state.intent.get("search_queries", [state.query])
        all_results: list[dict[str, Any]] = []

        for query in queries[:3]:  # 최대 3개 쿼리
            try:
                results = self.vector_store.search(
                    collection_name=self._collection,
                    query=query,
                    top_k=3,
                )
                all_results.extend(results)
            except Exception as e:
                logger.warning("search_failed", query=query, error=str(e))

        # 중복 제거 + 점수순 정렬
        seen_texts: set[str] = set()
        unique_results = []
        for r in sorted(all_results, key=lambda x: x["score"], reverse=True):
            text_hash = r["text"][:100]
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_results.append(r)

        return {"search_results": unique_results[:10]}

    async def _reason(self, state: AgentState) -> dict[str, Any]:
        """Step 3: 논리적 추론"""
        search_context = "\n\n".join(
            f"[관련도: {r['score']:.2f}] {r['text']}"
            for r in state.search_results
        ) if state.search_results else "검색 결과 없음"

        previous = "\n".join(state.reasoning_steps) if state.reasoning_steps else "첫 번째 추론 단계"

        prompt = REASONING_PROMPT.format(
            query=state.query,
            intent=str(state.intent),
            search_results=search_context,
            previous_reasoning=previous,
        )

        result = await self.llm.complete(
            prompt,
            model_tier="default",
            temperature=0.5,
        )

        new_steps = state.reasoning_steps + [f"[Iteration {state.iteration + 1}] {result['content'][:500]}"]

        return {
            "current_answer": result["content"],
            "reasoning_steps": new_steps,
            "iteration": state.iteration + 1,
        }

    async def _self_reflect(self, state: AgentState) -> dict[str, Any]:
        """Step 4: 자기 성찰 - 답변 품질 평가"""
        prompt = SELF_REFLECTION_PROMPT.format(
            query=state.query,
            intent=str(state.intent),
            answer=state.current_answer,
        )

        result = await self.llm.complete(
            prompt,
            model_tier="cheap",
            temperature=0.2,
        )

        import json
        try:
            content = result["content"]
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            reflection = json.loads(content.strip())
        except (json.JSONDecodeError, IndexError):
            reflection = {"quality_score": 0.7, "should_retry": False}

        confidence = reflection.get("quality_score", 0.7)
        should_retry = reflection.get("should_retry", False)

        # 최대 반복 횟수 도달 시 강제 종료
        if state.iteration >= MAX_ITERATIONS:
            should_retry = False

        return {
            "confidence": confidence,
            "should_stop": not should_retry,
        }

    def _route_after_reflection(self, state: AgentState) -> str:
        """자기 성찰 후 라우팅"""
        if state.should_stop or state.iteration >= MAX_ITERATIONS:
            return "respond"
        if state.confidence >= 0.6:
            return "respond"
        return "retry"

    async def _respond(self, state: AgentState) -> dict[str, Any]:
        """Step 5: 최종 답변 생성"""
        return {
            "messages": [{"role": "assistant", "content": state.current_answer}],
        }

    async def run(
        self,
        query: str,
        collection: str = "default",
        context: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """에이전트 실행

        Args:
            query: 사용자 질문
            collection: 검색 대상 벡터DB 컬렉션
            context: 이전 대화 이력

        Returns:
            {"answer": str, "confidence": float, "reasoning_steps": list, "usage_summary": dict}
        """
        self._collection = collection

        initial_state = AgentState(
            query=query,
            messages=context or [],
        )

        # LangGraph 실행
        final_state = await self._graph.ainvoke(initial_state)

        return {
            "answer": final_state.get("current_answer", ""),
            "confidence": final_state.get("confidence", 0.0),
            "reasoning_steps": final_state.get("reasoning_steps", []),
            "iterations": final_state.get("iteration", 0),
            "search_results_count": len(final_state.get("search_results", [])),
            "cost_summary": self.llm.get_cost_summary(),
        }
