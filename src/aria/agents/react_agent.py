"""ARIA Engine - ReAct Agent (LangGraph)

Think → Act → Observe 루프 기반 자율 추론 에이전트
- 질문 분석 → 검색 전략 결정 → 실행 → 결과 검증 → 답변
- Self-Reflection 노드로 답변 품질 자체 평가
- 최대 반복 횟수 제한으로 무한루프 방지
- 구조화된 에러 처리 (AgentError / LLM / VectorStore 예외 분리)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Annotated

import structlog
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from aria.core.exceptions import (
    AgentError,
    CollectionNotFoundError,
    KillSwitchError,
    LLMAllProvidersExhaustedError,
    NoAPIKeyError,
    VectorStoreError,
)
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
    collection: str = "default"  # 컬렉션을 state에 포함 (race condition 방지)
    intent: dict[str, Any] = field(default_factory=dict)
    search_results: list[dict[str, Any]] = field(default_factory=list)
    reasoning_steps: list[str] = field(default_factory=list)
    current_answer: str = ""
    confidence: float = 0.0
    iteration: int = 0
    should_stop: bool = False
    error: str = ""  # 에러 메시지 전달용
    memory_context: str = ""  # 메모리 마크다운 (Layer 1 주입)


INTENT_ANALYSIS_SYSTEM = """당신은 사용자 의도 분석 전문가입니다.

사용자 질문을 분석하여 다음 JSON 형태로 응답하세요:
{{
    "surface_intent": "표면적 질문의 핵심",
    "deeper_intent": "질문 뒤에 숨겨진 실제 의도/욕구",
    "required_knowledge": ["필요한 지식 영역 목록"],
    "search_queries": ["벡터DB 검색에 사용할 쿼리 목록 (최대 3개)"],
    "complexity": "simple|moderate|complex",
    "recommended_action": "search_knowledge|reason|respond|clarify"
}}

반드시 위 JSON 형식으로만 응답하세요. 다른 텍스트를 추가하지 마세요."""

INTENT_ANALYSIS_USER = """사용자 질문: {query}"""


REASONING_SYSTEM = """당신은 논리적 추론 전문가입니다.

주어진 정보를 바탕으로 단계적으로 사고하여 답변을 도출하세요.

다음 형식으로 응답하세요:
1. 현재까지 알고 있는 것을 정리
2. 부족한 정보가 있다면 무엇인지
3. 논리적 추론 과정
4. 결론 및 답변
5. 확신도 (0.0 ~ 1.0)"""

REASONING_USER = """## 사용자 질문
{query}

## 의도 분석
{intent}

## 메모리 컨텍스트
{memory_context}

## 검색된 관련 정보
{search_results}

## 이전 추론 단계
{previous_reasoning}"""


SELF_REFLECTION_SYSTEM = """당신은 답변 품질 평가 전문가입니다.

평가 기준:
1. 질문에 대한 직접적 답변인가?
2. 근거가 충분한가?
3. 논리적 오류가 없는가?
4. 사용자의 숨겨진 의도도 충족하는가?

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트를 추가하지 마세요.
{{
    "quality_score": 0.0~1.0,
    "issues": ["발견된 문제점"],
    "should_retry": true/false,
    "improvement_suggestion": "개선 방향"
}}"""

SELF_REFLECTION_USER = """## 원래 질문
{query}

## 사용자의 실제 의도
{intent}

## 현재 답변
{answer}"""


def _safe_parse_json(content: str) -> dict[str, Any] | None:
    """LLM 응답에서 JSON을 안전하게 추출/파싱

    지원 패턴:
    - 순수 JSON
    - ```json ... ``` 블록
    - ``` ... ``` 블록
    - JSON 앞뒤에 텍스트가 있는 경우
    """
    # 1. ```json 블록 추출
    if "```json" in content:
        try:
            extracted = content.split("```json")[1].split("```")[0]
            return json.loads(extracted.strip())
        except (json.JSONDecodeError, IndexError):
            pass

    # 2. ``` 블록 추출
    if "```" in content:
        try:
            extracted = content.split("```")[1].split("```")[0]
            return json.loads(extracted.strip())
        except (json.JSONDecodeError, IndexError):
            pass

    # 3. 순수 JSON 시도
    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        pass

    # 4. { } 브래킷 범위 추출
    first_brace = content.find("{")
    last_brace = content.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(content[first_brace : last_brace + 1])
        except json.JSONDecodeError:
            pass

    return None


class ReActAgent:
    """LangGraph 기반 ReAct 에이전트

    사용법:
        agent = ReActAgent(llm_provider, vector_store)
        result = await agent.run("회피형 애착 패턴에 대해 설명해줘", collection="psychology_kb")

    Hybrid Retrieval 사용:
        agent = ReActAgent(llm_provider, vector_store, hybrid_retriever=retriever)

    Memory 통합:
        agent = ReActAgent(llm_provider, vector_store, memory_loader=loader)
        result = await agent.run("...", scope="global")
    """

    def __init__(
        self,
        llm: LLMProvider,
        vector_store: VectorStore,
        hybrid_retriever: Any | None = None,
        memory_loader: Any | None = None,
    ) -> None:
        self.llm = llm
        self.vector_store = vector_store
        self.hybrid_retriever = hybrid_retriever
        self.memory_loader = memory_loader
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
        user_prompt = INTENT_ANALYSIS_USER.format(query=state.query)

        try:
            result = await self.llm.complete(
                user_prompt,
                system_prompt=INTENT_ANALYSIS_SYSTEM,
                cache_system_prompt=True,  # Prompt Caching — 반복 시스템 프롬프트 캐싱
                model_tier="cheap",  # 의도 분석은 저비용 모델로 충분
                temperature=0.3,
            )
        except (KillSwitchError, LLMAllProvidersExhaustedError, NoAPIKeyError):
            raise  # 상위로 전파
        except Exception as e:
            logger.error("intent_analysis_failed", error=str(e))
            # 의도 분석 실패해도 기본값으로 계속 진행
            return {
                "intent": {
                    "surface_intent": state.query,
                    "deeper_intent": "",
                    "required_knowledge": [],
                    "search_queries": [state.query],
                    "complexity": "moderate",
                    "recommended_action": "search_knowledge",
                }
            }

        # JSON 파싱 시도
        intent = _safe_parse_json(result["content"])
        if intent is None:
            logger.warning(
                "intent_json_parse_failed",
                content_preview=result["content"][:200],
            )
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
        """Step 2: 지식 검색 (RAG — Hybrid Retrieval 지원)

        hybrid_retriever가 설정되면: 벡터 + BM25 → RRF 병합
        미설정이면: 기존 벡터 검색만 사용 (하위 호환)
        """
        # 이미 검색 결과가 있으면 재검색 안 함 (retry 루프 시)
        if state.search_results and state.iteration > 0:
            logger.debug("search_skipped", reason="already_has_results_in_retry")
            return {"search_results": state.search_results}

        queries = state.intent.get("search_queries", [state.query])
        all_results: list[dict[str, Any]] = []
        collection = state.collection  # state에서 컬렉션 참조 (race condition 방지)
        use_hybrid = self.hybrid_retriever is not None

        for query in queries[:3]:  # 최대 3개 쿼리
            try:
                if use_hybrid:
                    results = self.hybrid_retriever.search(
                        collection_name=collection,
                        query=query,
                        top_k=5,
                    )
                else:
                    results = self.vector_store.search(
                        collection_name=collection,
                        query=query,
                        top_k=3,
                    )
                all_results.extend(results)
            except CollectionNotFoundError:
                logger.warning("collection_not_found", collection=collection, query=query)
                # 컬렉션 없으면 검색 결과 없이 진행 (추론만으로 답변)
                break
            except VectorStoreError as e:
                logger.warning("search_failed", query=query, error=str(e))
                continue

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

        user_prompt = REASONING_USER.format(
            query=state.query,
            intent=str(state.intent),
            memory_context=state.memory_context if state.memory_context else "메모리 없음",
            search_results=search_context,
            previous_reasoning=previous,
        )

        try:
            result = await self.llm.complete(
                user_prompt,
                system_prompt=REASONING_SYSTEM,
                cache_system_prompt=True,  # Prompt Caching — 추론 시스템 프롬프트 캐싱
                model_tier="default",
                temperature=0.5,
            )
        except (KillSwitchError, LLMAllProvidersExhaustedError, NoAPIKeyError):
            raise
        except Exception as e:
            logger.error("reasoning_failed", error=str(e), iteration=state.iteration)
            raise AgentError(
                f"추론 단계 실패: {e}",
                query=state.query,
                iteration=state.iteration,
            ) from e

        new_steps = state.reasoning_steps + [f"[Iteration {state.iteration + 1}] {result['content'][:500]}"]

        return {
            "current_answer": result["content"],
            "reasoning_steps": new_steps,
            "iteration": state.iteration + 1,
        }

    async def _self_reflect(self, state: AgentState) -> dict[str, Any]:
        """Step 4: 자기 성찰 - 답변 품질 평가"""
        # 최대 반복 횟수 도달 시 성찰 스킵 (비용 절감)
        if state.iteration >= MAX_ITERATIONS:
            logger.info("self_reflect_skipped", reason="max_iterations_reached")
            return {"confidence": 0.6, "should_stop": True}

        user_prompt = SELF_REFLECTION_USER.format(
            query=state.query,
            intent=str(state.intent),
            answer=state.current_answer[:2000],  # 너무 긴 답변은 잘라서 평가
        )

        try:
            result = await self.llm.complete(
                user_prompt,
                system_prompt=SELF_REFLECTION_SYSTEM,
                cache_system_prompt=True,  # Prompt Caching — 성찰 시스템 프롬프트 캐싱
                model_tier="cheap",
                temperature=0.2,
            )
        except (KillSwitchError, LLMAllProvidersExhaustedError, NoAPIKeyError):
            raise
        except Exception as e:
            # 성찰 실패해도 현재 답변으로 진행
            logger.warning("self_reflect_failed", error=str(e))
            return {"confidence": 0.6, "should_stop": True}

        reflection = _safe_parse_json(result["content"])
        if reflection is None:
            logger.warning(
                "reflection_json_parse_failed",
                content_preview=result["content"][:200],
            )
            return {"confidence": 0.7, "should_stop": True}

        confidence = reflection.get("quality_score", 0.7)
        should_retry = reflection.get("should_retry", False)

        logger.info(
            "self_reflection_result",
            confidence=confidence,
            should_retry=should_retry,
            issues=reflection.get("issues", []),
        )

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
        answer = state.current_answer
        if not answer and state.error:
            answer = f"요청을 처리하는 중 문제가 발생했습니다: {state.error}"
        elif not answer:
            answer = "죄송합니다. 질문에 대한 답변을 생성하지 못했습니다. 질문을 다시 표현해주세요."

        return {
            "messages": [{"role": "assistant", "content": answer}],
        }

    async def run(
        self,
        query: str,
        collection: str = "default",
        context: list[dict[str, str]] | None = None,
        scope: str = "global",
        memory_domains: list[str] | None = None,
    ) -> dict[str, Any]:
        """에이전트 실행

        Args:
            query: 사용자 질문
            collection: 검색 대상 벡터DB 컬렉션
            context: 이전 대화 이력
            scope: 메모리 스코프 (global/testorum/talksim/autotube)
            memory_domains: 명시적 토픽 지정 (None이면 전체)

        Returns:
            {"answer": str, "confidence": float, "reasoning_steps": list,
             "cost_summary": dict, "memory_loaded": list, ...}

        Raises:
            KillSwitchError: 비용 상한 초과
            LLMAllProvidersExhaustedError: 모든 LLM 프로바이더 실패
            AgentError: 에이전트 내부 에러
        """
        # 메모리 로딩 (memory_loader가 설정된 경우)
        memory_loaded: list[str] = []
        memory_context: str = ""
        if self.memory_loader is not None:
            try:
                load_result = self.memory_loader.load(
                    scope=scope,
                    domains=memory_domains,
                )
                memory_loaded = load_result.loaded_domains
                memory_context = load_result.prompt_markdown
                logger.info(
                    "memory_injected",
                    scope=scope,
                    domains_loaded=len(memory_loaded),
                    total_tokens=load_result.total_tokens,
                )
            except Exception as e:
                # 메모리 로딩 실패해도 에이전트는 계속 동작
                logger.warning("memory_load_failed", scope=scope, error=str(e))

        initial_state = AgentState(
            query=query,
            collection=collection,
            messages=context or [],
            memory_context=memory_context,
        )

        try:
            # LangGraph 실행
            final_state = await self._graph.ainvoke(initial_state)
        except (KillSwitchError, LLMAllProvidersExhaustedError, NoAPIKeyError):
            raise  # 그대로 전파 → API 레이어에서 적절한 HTTP status로 변환
        except Exception as e:
            logger.error("agent_run_failed", query=query[:100], error=str(e))
            raise AgentError(f"에이전트 실행 실패: {e}", query=query) from e

        return {
            "answer": final_state.get("current_answer", ""),
            "confidence": final_state.get("confidence", 0.0),
            "reasoning_steps": final_state.get("reasoning_steps", []),
            "iterations": final_state.get("iteration", 0),
            "search_results_count": len(final_state.get("search_results", [])),
            "cost_summary": self.llm.get_cost_summary(),
            "memory_loaded": memory_loaded,
        }
