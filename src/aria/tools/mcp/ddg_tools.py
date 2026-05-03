"""ARIA Engine - MCP Tool: DuckDuckGo 웹 검색

ddgs 패키지 기반 도구 2종 (구 duckduckgo-search에서 리네임)
- DdgWebSearchTool: 웹 검색 (구글 대안 / 영어+한국어)
- DdgNewsSearchTool: 뉴스 검색 (글로벌 + 한국 뉴스)

인증: 불필요 (API 키 없음 / 무료 / 무제한)
의존성: ddgs (pip install ddgs)

설계 원칙:
- 외부 서비스 의존 제거 → 자체 검색 능력 확보
- Claude API 호출 전에 도구로 실시간 정보 수집 → 비용 절감 + hallucination 방지
- 에러는 ToolResult로 감싸서 반환 (예외 전파 없음)
"""

from __future__ import annotations

from typing import Any

import structlog

from aria.core.config import DuckDuckGoConfig
from aria.tools.tool_types import (
    SafetyLevelHint,
    ToolCategory,
    ToolDefinition,
    ToolExecutor,
    ToolParameter,
    ToolResult,
)

logger = structlog.get_logger()


class DdgSearchClient:
    """DuckDuckGo 검색 클라이언트

    ddgs 패키지의 DDGS 래퍼 (구 duckduckgo-search에서 리네임)
    API 키 불필요 / 무료 / rate limit은 있으나 개인 사용에 충분

    Args:
        config: DuckDuckGoConfig (request_timeout)
    """

    def __init__(self, config: DuckDuckGoConfig) -> None:
        self._config = config

    def web_search(
        self,
        query: str,
        region: str = "wt-wt",
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        """웹 검색 (동기 — ddgs는 동기 API)

        Args:
            query: 검색 쿼리
            region: 지역 코드 (wt-wt: 전세계 / kr-kr: 한국)
            max_results: 최대 결과 수 (1~20)
        """
        from ddgs import DDGS

        with DDGS(timeout=self._config.request_timeout) as ddgs:
            results = list(ddgs.text(
                query,
                region=region,
                max_results=min(max(max_results, 1), 20),
            ))
        return results

    def news_search(
        self,
        query: str,
        region: str = "wt-wt",
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        """뉴스 검색 (동기)

        Args:
            query: 검색 쿼리
            region: 지역 코드 (wt-wt: 전세계 / kr-kr: 한국)
            max_results: 최대 결과 수 (1~20)
        """
        from ddgs import DDGS

        with DDGS(timeout=self._config.request_timeout) as ddgs:
            results = list(ddgs.news(
                query,
                region=region,
                max_results=min(max(max_results, 1), 20),
            ))
        return results


# === 응답 정제 유틸리티 ===


def _simplify_web_result(item: dict[str, Any]) -> dict[str, Any]:
    """웹 검색 결과를 LLM 친화 형태로 정제"""
    return {
        "title": item.get("title", ""),
        "url": item.get("href", ""),
        "snippet": item.get("body", ""),
    }


def _simplify_news_result(item: dict[str, Any]) -> dict[str, Any]:
    """뉴스 검색 결과를 LLM 친화 형태로 정제"""
    return {
        "title": item.get("title", ""),
        "url": item.get("url", ""),
        "snippet": item.get("body", ""),
        "source": item.get("source", ""),
        "date": item.get("date", ""),
    }


# === Tool Executors ===


class DdgWebSearchTool(ToolExecutor):
    """DuckDuckGo 웹 검색 도구

    API 키 없이 글로벌 웹 검색 수행
    영어/한국어 모두 지원 (region 파라미터로 조절)

    Args:
        client: DdgSearchClient 인스턴스
    """

    def __init__(self, client: DdgSearchClient) -> None:
        self._client = client

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="ddg_web_search",
            description=(
                "DuckDuckGo로 웹 검색을 수행합니다. "
                "영어/한국어 등 모든 언어를 지원하며 API 키가 필요 없습니다. "
                "글로벌 기술 문서, 해외 서비스 정보, 영문 자료 검색에 적합합니다. "
                "한국 블로그/뉴스는 네이버 검색이 더 정확합니다."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="검색 쿼리 (예: 'FastAPI best practices 2026')",
                    required=True,
                ),
                ToolParameter(
                    name="region",
                    type="string",
                    description="지역 코드 (wt-wt: 전세계 / kr-kr: 한국 / us-en: 미국)",
                    required=False,
                    default="wt-wt",
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="최대 결과 수 (1~20 / 기본값 5)",
                    required=False,
                ),
            ],
            category=ToolCategory.MCP,
            safety_hint=SafetyLevelHint.READ_ONLY,
            version="1.0.0",
        )

    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        import asyncio

        query = parameters.get("query", "").strip()
        if not query:
            return ToolResult(
                tool_name="ddg_web_search",
                success=False,
                error="query가 비어있습니다",
            )

        region = parameters.get("region", "wt-wt")
        max_results = parameters.get("max_results", 5)

        try:
            # duckduckgo-search는 동기 API → asyncio에서 블로킹 방지
            loop = asyncio.get_event_loop()
            raw_results = await loop.run_in_executor(
                None,
                lambda: self._client.web_search(
                    query=query,
                    region=region,
                    max_results=max_results,
                ),
            )
        except Exception as e:
            error_str = str(e)[:300]
            # rate limit 감지
            if "ratelimit" in error_str.lower() or "429" in error_str:
                error_msg = "DuckDuckGo 검색 속도 제한에 도달했습니다 (잠시 후 다시 시도하세요)"
            else:
                error_msg = f"웹 검색 실패: {error_str}"
            logger.error("ddg_web_search_failed", query=query, error=error_str[:200])
            return ToolResult(
                tool_name="ddg_web_search",
                success=False,
                error=error_msg,
            )

        results = [_simplify_web_result(item) for item in raw_results]

        logger.info(
            "ddg_web_search_success",
            query=query,
            region=region,
            result_count=len(results),
        )

        return ToolResult(
            tool_name="ddg_web_search",
            success=True,
            output={
                "query": query,
                "region": region,
                "results": results,
                "result_count": len(results),
            },
        )


class DdgNewsSearchTool(ToolExecutor):
    """DuckDuckGo 뉴스 검색 도구

    API 키 없이 글로벌 뉴스 검색 수행
    한국 뉴스는 네이버 뉴스가 더 포괄적이지만 글로벌 뉴스에 강점

    Args:
        client: DdgSearchClient 인스턴스
    """

    def __init__(self, client: DdgSearchClient) -> None:
        self._client = client

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="ddg_news_search",
            description=(
                "DuckDuckGo로 뉴스를 검색합니다. "
                "글로벌 뉴스 검색에 강점이 있으며 API 키가 필요 없습니다. "
                "한국 뉴스는 naver_news_search가 더 정확합니다."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="검색 쿼리 (예: 'AI startup funding 2026')",
                    required=True,
                ),
                ToolParameter(
                    name="region",
                    type="string",
                    description="지역 코드 (wt-wt: 전세계 / kr-kr: 한국 / us-en: 미국)",
                    required=False,
                    default="wt-wt",
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="최대 결과 수 (1~20 / 기본값 5)",
                    required=False,
                ),
            ],
            category=ToolCategory.MCP,
            safety_hint=SafetyLevelHint.READ_ONLY,
            version="1.0.0",
        )

    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        import asyncio

        query = parameters.get("query", "").strip()
        if not query:
            return ToolResult(
                tool_name="ddg_news_search",
                success=False,
                error="query가 비어있습니다",
            )

        region = parameters.get("region", "wt-wt")
        max_results = parameters.get("max_results", 5)

        try:
            loop = asyncio.get_event_loop()
            raw_results = await loop.run_in_executor(
                None,
                lambda: self._client.news_search(
                    query=query,
                    region=region,
                    max_results=max_results,
                ),
            )
        except Exception as e:
            error_str = str(e)[:300]
            if "ratelimit" in error_str.lower() or "429" in error_str:
                error_msg = "DuckDuckGo 검색 속도 제한에 도달했습니다 (잠시 후 다시 시도하세요)"
            else:
                error_msg = f"뉴스 검색 실패: {error_str}"
            logger.error("ddg_news_search_failed", query=query, error=error_str[:200])
            return ToolResult(
                tool_name="ddg_news_search",
                success=False,
                error=error_msg,
            )

        results = [_simplify_news_result(item) for item in raw_results]

        logger.info(
            "ddg_news_search_success",
            query=query,
            region=region,
            result_count=len(results),
        )

        return ToolResult(
            tool_name="ddg_news_search",
            success=True,
            output={
                "query": query,
                "region": region,
                "results": results,
                "result_count": len(results),
            },
        )
