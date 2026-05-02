"""ARIA Engine - MCP Tool: 네이버 검색 (Naver Search API)

네이버 검색 API v1 기반 도구 6종
- NaverBlogSearchTool: 블로그 검색
- NaverNewsSearchTool: 뉴스 검색
- NaverCafeSearchTool: 카페 검색
- NaverShopSearchTool: 쇼핑 검색
- NaverKinSearchTool: 지식iN 검색
- NaverLocalSearchTool: 지역(장소) 검색

인증: X-Naver-Client-Id + X-Naver-Client-Secret
API 문서: https://developers.naver.com/docs/serviceapi/search/blog/blog.md

설계 원칙:
- httpx 직접 사용 (추가 의존성 없음)
- 6종 검색은 엔드포인트만 다르고 패턴 동일 → 공통 클라이언트 + 팩토리
- 에러는 ToolResult로 감싸서 반환 (예외 전파 없음)
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog

from aria.core.config import NaverSearchConfig
from aria.tools.tool_types import (
    SafetyLevelHint,
    ToolCategory,
    ToolDefinition,
    ToolExecutor,
    ToolParameter,
    ToolResult,
)

logger = structlog.get_logger()

NAVER_SEARCH_BASE = "https://openapi.naver.com/v1/search"


class NaverSearchClient:
    """네이버 검색 API HTTP 클라이언트

    Client ID / Secret 기반 인증
    httpx.AsyncClient를 재사용하여 커넥션 풀링

    Args:
        config: NaverSearchConfig (client_id / client_secret / request_timeout)
    """

    def __init__(self, config: NaverSearchConfig) -> None:
        self._config = config
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Lazy initialization — 첫 호출 시 클라이언트 생성"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=NAVER_SEARCH_BASE,
                headers={
                    "X-Naver-Client-Id": self._config.client_id,
                    "X-Naver-Client-Secret": self._config.client_secret,
                },
                timeout=httpx.Timeout(self._config.request_timeout),
            )
        return self._client

    async def search(
        self,
        endpoint: str,
        query: str,
        display: int = 5,
        start: int = 1,
        sort: str = "sim",
    ) -> dict[str, Any]:
        """네이버 검색 API 공통 호출

        Args:
            endpoint: 검색 유형 (blog / news / cafearticle / shop / kin / local)
            query: 검색 쿼리
            display: 결과 수 (1~100 / local은 1~5)
            start: 시작 위치 (1~1000)
            sort: 정렬 (sim: 정확도 / date: 날짜순 / shop은 sim/date/asc/dsc)
        """
        # local은 최대 5개
        max_display = 5 if endpoint == "local" else 100
        params: dict[str, Any] = {
            "query": query,
            "display": min(max(display, 1), max_display),
            "start": min(max(start, 1), 1000),
            "sort": sort,
        }

        client = self._get_client()
        response = await client.get(f"/{endpoint}.json", params=params)
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        """클라이언트 종료"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# === 응답 정제 유틸리티 ===


def _strip_html(text: str) -> str:
    """네이버 API가 반환하는 <b> 등 HTML 태그 제거"""
    import re
    return re.sub(r"<[^>]+>", "", text)


def _simplify_blog(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": _strip_html(item.get("title", "")),
        "description": _strip_html(item.get("description", "")),
        "blogger_name": item.get("bloggername", ""),
        "link": item.get("link", ""),
        "post_date": item.get("postdate", ""),
    }


def _simplify_news(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": _strip_html(item.get("title", "")),
        "description": _strip_html(item.get("description", "")),
        "link": item.get("originallink", "") or item.get("link", ""),
        "pub_date": item.get("pubDate", ""),
    }


def _simplify_cafe(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": _strip_html(item.get("title", "")),
        "description": _strip_html(item.get("description", "")),
        "cafe_name": item.get("cafename", ""),
        "link": item.get("link", ""),
    }


def _simplify_shop(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": _strip_html(item.get("title", "")),
        "link": item.get("link", ""),
        "lprice": item.get("lprice", ""),
        "hprice": item.get("hprice", ""),
        "mall_name": item.get("mallName", ""),
        "brand": item.get("brand", ""),
        "category": "/".join(
            filter(None, [
                item.get("category1", ""),
                item.get("category2", ""),
                item.get("category3", ""),
            ])
        ),
    }


def _simplify_kin(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": _strip_html(item.get("title", "")),
        "description": _strip_html(item.get("description", "")),
        "link": item.get("link", ""),
    }


def _simplify_local(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": _strip_html(item.get("title", "")),
        "category": item.get("category", ""),
        "address": item.get("address", ""),
        "road_address": item.get("roadAddress", ""),
        "phone": item.get("telephone", ""),
        "link": item.get("link", ""),
        "mapx": item.get("mapx", ""),
        "mapy": item.get("mapy", ""),
    }


# 엔드포인트별 설정
_SEARCH_CONFIGS: dict[str, dict[str, Any]] = {
    "blog": {
        "endpoint": "blog",
        "tool_name": "naver_blog_search",
        "description": (
            "네이버 블로그를 검색합니다. "
            "리뷰, 여행기, 레시피, 일상 글 등을 찾을 수 있습니다."
        ),
        "simplify": _simplify_blog,
        "sorts": ["sim", "date"],
    },
    "news": {
        "endpoint": "news",
        "tool_name": "naver_news_search",
        "description": (
            "네이버 뉴스를 검색합니다. "
            "최신 뉴스 기사와 보도자료를 찾을 수 있습니다."
        ),
        "simplify": _simplify_news,
        "sorts": ["sim", "date"],
    },
    "cafe": {
        "endpoint": "cafearticle",
        "tool_name": "naver_cafe_search",
        "description": (
            "네이버 카페 게시글을 검색합니다. "
            "커뮤니티 토론, 중고거래, 지역 정보 등을 찾을 수 있습니다."
        ),
        "simplify": _simplify_cafe,
        "sorts": ["sim", "date"],
    },
    "shop": {
        "endpoint": "shop",
        "tool_name": "naver_shop_search",
        "description": (
            "네이버 쇼핑을 검색합니다. "
            "상품 가격 비교와 쇼핑 정보를 찾을 수 있습니다."
        ),
        "simplify": _simplify_shop,
        "sorts": ["sim", "date", "asc", "dsc"],
    },
    "kin": {
        "endpoint": "kin",
        "tool_name": "naver_kin_search",
        "description": (
            "네이버 지식iN을 검색합니다. "
            "Q&A 형태의 질문과 답변을 찾을 수 있습니다."
        ),
        "simplify": _simplify_kin,
        "sorts": ["sim", "date"],
    },
    "local": {
        "endpoint": "local",
        "tool_name": "naver_local_search",
        "description": (
            "네이버 지역 검색으로 업체/장소 정보를 찾습니다. "
            "식당, 병원, 편의시설 등의 상호명, 주소, 전화번호를 제공합니다."
        ),
        "simplify": _simplify_local,
        "sorts": ["random", "comment"],
    },
}


class _NaverSearchToolBase(ToolExecutor):
    """네이버 검색 도구 공통 베이스

    6종 검색 도구가 동일 패턴이므로 base class로 통합
    """

    def __init__(self, client: NaverSearchClient, search_type: str) -> None:
        self._client = client
        self._cfg = _SEARCH_CONFIGS[search_type]

    def get_definition(self) -> ToolDefinition:
        cfg = self._cfg
        sort_param = ToolParameter(
            name="sort",
            type="string",
            description="정렬 기준",
            required=False,
            enum=cfg["sorts"],
            default=cfg["sorts"][0],
        )

        max_display = 5 if cfg["endpoint"] == "local" else 10
        size_desc = f"결과 수 (1~{max_display})"

        return ToolDefinition(
            name=cfg["tool_name"],
            description=cfg["description"],
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="검색 쿼리",
                    required=True,
                ),
                ToolParameter(
                    name="display",
                    type="integer",
                    description=size_desc,
                    required=False,
                ),
                sort_param,
            ],
            category=ToolCategory.MCP,
            safety_hint=SafetyLevelHint.READ_ONLY,
            version="1.0.0",
        )

    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        cfg = self._cfg
        tool_name = cfg["tool_name"]
        query = parameters.get("query", "").strip()

        if not query:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error="query가 비어있습니다",
            )

        try:
            data = await self._client.search(
                endpoint=cfg["endpoint"],
                query=query,
                display=parameters.get("display", 5),
                sort=parameters.get("sort", cfg["sorts"][0]),
            )
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 401:
                error_msg = "네이버 API 인증 실패 (Client ID/Secret을 확인하세요)"
            elif status == 429:
                error_msg = "네이버 API 호출 한도 초과 (잠시 후 다시 시도하세요)"
            else:
                error_msg = f"네이버 API 오류 ({status})"
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=error_msg,
            )
        except Exception as e:
            logger.error(
                f"{tool_name}_failed",
                query=query,
                error=str(e)[:200],
            )
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"검색 실패: {str(e)[:300]}",
            )

        items = data.get("items", [])
        simplify_fn = cfg["simplify"]
        results = [simplify_fn(item) for item in items]

        logger.info(
            f"{tool_name}_success",
            query=query,
            result_count=len(results),
            total=data.get("total", 0),
        )

        return ToolResult(
            tool_name=tool_name,
            success=True,
            output={
                "query": query,
                "results": results,
                "total": data.get("total", 0),
                "display": data.get("display", 0),
            },
        )


# === 개별 클래스 (타입 명확성 + register_executor 호환) ===


class NaverBlogSearchTool(_NaverSearchToolBase):
    def __init__(self, client: NaverSearchClient) -> None:
        super().__init__(client, "blog")


class NaverNewsSearchTool(_NaverSearchToolBase):
    def __init__(self, client: NaverSearchClient) -> None:
        super().__init__(client, "news")


class NaverCafeSearchTool(_NaverSearchToolBase):
    def __init__(self, client: NaverSearchClient) -> None:
        super().__init__(client, "cafe")


class NaverShopSearchTool(_NaverSearchToolBase):
    def __init__(self, client: NaverSearchClient) -> None:
        super().__init__(client, "shop")


class NaverKinSearchTool(_NaverSearchToolBase):
    def __init__(self, client: NaverSearchClient) -> None:
        super().__init__(client, "kin")


class NaverLocalSearchTool(_NaverSearchToolBase):
    def __init__(self, client: NaverSearchClient) -> None:
        super().__init__(client, "local")
