"""ARIA Engine - Built-in Tool: Knowledge Search

벡터DB + BM25 하이브리드 검색을 수행하는 내장 도구
에이전트가 지식 베이스에서 관련 문서를 자율적으로 검색할 때 사용

기존 ReAct 에이전트의 _search_knowledge()와 달리
에이전트가 도구 호출로 명시적으로 검색을 트리거할 수 있게 함

사용 시나리오:
- "회피형 애착에 대해 알려줘" → knowledge_search(query="회피형 애착", collection="psychology_kb")
- "Testorum FAQ 검색" → knowledge_search(query="자주 묻는 질문", collection="testorum_docs")
"""

from __future__ import annotations

from typing import Any

import structlog

from aria.core.exceptions import CollectionNotFoundError, VectorStoreError
from aria.rag.hybrid_retriever import HybridRetriever
from aria.tools.tool_types import (
    SafetyLevelHint,
    ToolCategory,
    ToolDefinition,
    ToolExecutor,
    ToolParameter,
    ToolResult,
)

logger = structlog.get_logger()

# 검색 결과 반환 상한 (토큰 절약)
_MAX_RESULTS = 10
_MAX_TEXT_LENGTH = 1000  # 개별 결과 텍스트 잘라서 반환


class KnowledgeSearchTool(ToolExecutor):
    """지식 베이스 하이브리드 검색 도구

    Args:
        hybrid_retriever: HybridRetriever 인스턴스 (벡터 + BM25)

    사용법:
        tool = KnowledgeSearchTool(hybrid_retriever)
        registry.register_executor(tool)
    """

    def __init__(self, hybrid_retriever: HybridRetriever) -> None:
        self._retriever = hybrid_retriever

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="knowledge_search",
            description=(
                "지식 베이스에서 관련 문서를 검색합니다. "
                "벡터 시맨틱 검색과 BM25 키워드 검색을 결합한 하이브리드 검색을 수행합니다. "
                "검색 결과는 관련도 점수순으로 정렬됩니다."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="검색 쿼리 (자연어 질문 또는 키워드)",
                    required=True,
                ),
                ToolParameter(
                    name="collection",
                    type="string",
                    description="검색 대상 컬렉션 이름 (예: psychology_kb / testorum_docs). 기본: default",
                    required=False,
                ),
                ToolParameter(
                    name="top_k",
                    type="integer",
                    description="반환할 최대 결과 수 (기본: 5 / 최대: 10)",
                    required=False,
                ),
            ],
            category=ToolCategory.BUILTIN,
            safety_hint=SafetyLevelHint.READ_ONLY,
            version="1.0.0",
        )

    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        """하이브리드 검색 실행"""
        query = parameters["query"]
        collection = parameters.get("collection", "default")
        top_k = min(parameters.get("top_k", 5), _MAX_RESULTS)

        try:
            results = self._retriever.search(
                collection_name=collection,
                query=query,
                top_k=top_k,
            )

            # 결과 정리 (텍스트 길이 제한 + 핵심 필드만 추출)
            formatted = []
            for r in results:
                text = r["text"]
                if len(text) > _MAX_TEXT_LENGTH:
                    text = text[:_MAX_TEXT_LENGTH] + "..."
                formatted.append({
                    "text": text,
                    "score": round(r["score"], 4),
                    "metadata": r.get("metadata", {}),
                })

            logger.info(
                "knowledge_search_executed",
                query=query[:50],
                collection=collection,
                results_count=len(formatted),
            )

            return ToolResult(
                tool_name="knowledge_search",
                success=True,
                output={
                    "query": query,
                    "collection": collection,
                    "results": formatted,
                    "total_found": len(formatted),
                },
            )

        except CollectionNotFoundError:
            return ToolResult(
                tool_name="knowledge_search",
                success=False,
                error=f"컬렉션을 찾을 수 없습니다: '{collection}'",
            )

        except VectorStoreError as e:
            logger.error(
                "knowledge_search_vector_error",
                collection=collection,
                error=str(e)[:200],
            )
            return ToolResult(
                tool_name="knowledge_search",
                success=False,
                error=f"벡터 검색 오류: {str(e)[:300]}",
            )

        except Exception as e:
            logger.error(
                "knowledge_search_failed",
                query=query[:50],
                collection=collection,
                error=str(e)[:200],
            )
            return ToolResult(
                tool_name="knowledge_search",
                success=False,
                error=f"검색 실패: {str(e)[:300]}",
            )
