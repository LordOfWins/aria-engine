"""ARIA Engine - MCP Tool: 카카오맵 (Kakao Local API)

카카오 로컬 API v2 기반 도구 3종
- KakaoKeywordSearchTool: 키워드로 장소 검색
- KakaoAddressSearchTool: 주소 → 좌표 변환 (지오코딩)
- KakaoCoord2AddressTool: 좌표 → 주소 변환 (역지오코딩)

인증: REST API 키 (Authorization: KakaoAK {KEY})
API 문서: https://developers.kakao.com/docs/latest/ko/local/dev-guide

설계 원칙:
- httpx 직접 사용 (추가 의존성 없음)
- Notion MCP 도구와 동일 패턴 (ToolExecutor ABC)
- 에러는 ToolResult로 감싸서 반환 (예외 전파 없음)
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog

from aria.core.config import KakaoMapConfig
from aria.tools.tool_types import (
    SafetyLevelHint,
    ToolCategory,
    ToolDefinition,
    ToolExecutor,
    ToolParameter,
    ToolResult,
)

logger = structlog.get_logger()

KAKAO_LOCAL_BASE = "https://dapi.kakao.com/v2/local"


class KakaoMapClient:
    """카카오 로컬 API HTTP 클라이언트

    REST API 키 기반 인증
    httpx.AsyncClient를 재사용하여 커넥션 풀링

    Args:
        config: KakaoMapConfig (rest_api_key / request_timeout)
    """

    def __init__(self, config: KakaoMapConfig) -> None:
        self._config = config
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Lazy initialization — 첫 호출 시 클라이언트 생성"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=KAKAO_LOCAL_BASE,
                headers={
                    "Authorization": f"KakaoAK {self._config.rest_api_key}",
                },
                timeout=httpx.Timeout(self._config.request_timeout),
            )
        return self._client

    async def keyword_search(
        self,
        query: str,
        x: str = "",
        y: str = "",
        radius: int = 0,
        page: int = 1,
        size: int = 5,
        sort: str = "accuracy",
    ) -> dict[str, Any]:
        """키워드로 장소 검색

        Args:
            query: 검색 키워드 (예: "강남역 맛집")
            x: 중심 경도 (longitude)
            y: 중심 위도 (latitude)
            radius: 반경 (미터 / 최대 20000)
            page: 결과 페이지 (1~45)
            size: 한 페이지 결과 수 (1~15)
            sort: 정렬 기준 (accuracy / distance)
        """
        params: dict[str, Any] = {
            "query": query,
            "page": min(max(page, 1), 45),
            "size": min(max(size, 1), 15),
            "sort": sort if sort in ("accuracy", "distance") else "accuracy",
        }
        if x and y:
            params["x"] = x
            params["y"] = y
            if radius > 0:
                params["radius"] = min(radius, 20000)

        client = self._get_client()
        response = await client.get("/search/keyword.json", params=params)
        response.raise_for_status()
        return response.json()

    async def address_search(
        self,
        query: str,
        page: int = 1,
        size: int = 5,
    ) -> dict[str, Any]:
        """주소 → 좌표 변환 (지오코딩)

        Args:
            query: 검색할 주소 (지번/도로명 모두 가능)
            page: 결과 페이지 (1~45)
            size: 한 페이지 결과 수 (1~30)
        """
        params: dict[str, Any] = {
            "query": query,
            "page": min(max(page, 1), 45),
            "size": min(max(size, 1), 30),
        }

        client = self._get_client()
        response = await client.get("/search/address.json", params=params)
        response.raise_for_status()
        return response.json()

    async def coord2address(
        self,
        x: str,
        y: str,
    ) -> dict[str, Any]:
        """좌표 → 주소 변환 (역지오코딩)

        Args:
            x: 경도 (longitude)
            y: 위도 (latitude)
        """
        params: dict[str, Any] = {
            "x": x,
            "y": y,
        }

        client = self._get_client()
        response = await client.get("/geo/coord2address.json", params=params)
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        """클라이언트 종료"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# === 응답 정제 유틸리티 ===


def _simplify_place(doc: dict[str, Any]) -> dict[str, Any]:
    """키워드 검색 결과 문서를 LLM 친화 형태로 정제"""
    return {
        "name": doc.get("place_name", ""),
        "address": doc.get("address_name", ""),
        "road_address": doc.get("road_address_name", ""),
        "phone": doc.get("phone", ""),
        "category": doc.get("category_name", ""),
        "x": doc.get("x", ""),
        "y": doc.get("y", ""),
        "place_url": doc.get("place_url", ""),
        "distance": doc.get("distance", ""),
    }


def _simplify_address_doc(doc: dict[str, Any]) -> dict[str, Any]:
    """주소 검색 결과 문서를 LLM 친화 형태로 정제"""
    result: dict[str, Any] = {
        "address_name": doc.get("address_name", ""),
        "address_type": doc.get("address_type", ""),
        "x": doc.get("x", ""),
        "y": doc.get("y", ""),
    }

    road = doc.get("road_address")
    if road:
        result["road_address"] = road.get("address_name", "")
        result["building_name"] = road.get("building_name", "")

    return result


# === Tool Executors ===


class KakaoKeywordSearchTool(ToolExecutor):
    """카카오맵 키워드 장소 검색 도구

    키워드로 장소를 검색하여 이름/주소/좌표/전화번호/카테고리 등을 반환

    Args:
        client: KakaoMapClient 인스턴스
    """

    def __init__(self, client: KakaoMapClient) -> None:
        self._client = client

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="kakao_keyword_search",
            description=(
                "카카오맵에서 키워드로 장소를 검색합니다. "
                "음식점, 카페, 병원, 주유소, 관공서 등 모든 장소를 검색할 수 있습니다. "
                "좌표(x/y)와 반경(radius)을 함께 지정하면 해당 위치 근처 검색이 가능합니다."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="검색 키워드 (예: '강남역 맛집', '남양주 약국')",
                    required=True,
                ),
                ToolParameter(
                    name="x",
                    type="string",
                    description="중심 경도 (longitude / 예: '127.0276')",
                    required=False,
                ),
                ToolParameter(
                    name="y",
                    type="string",
                    description="중심 위도 (latitude / 예: '37.4979')",
                    required=False,
                ),
                ToolParameter(
                    name="radius",
                    type="integer",
                    description="반경 (미터 / 최대 20000 / x,y 필요)",
                    required=False,
                ),
                ToolParameter(
                    name="size",
                    type="integer",
                    description="결과 수 (1~15 / 기본값 5)",
                    required=False,
                ),
                ToolParameter(
                    name="sort",
                    type="string",
                    description="정렬 기준 (accuracy: 정확도 / distance: 거리순)",
                    required=False,
                    enum=["accuracy", "distance"],
                    default="accuracy",
                ),
            ],
            category=ToolCategory.MCP,
            safety_hint=SafetyLevelHint.READ_ONLY,
            version="1.0.0",
        )

    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        query = parameters.get("query", "").strip()
        if not query:
            return ToolResult(
                tool_name="kakao_keyword_search",
                success=False,
                error="query가 비어있습니다",
            )

        try:
            data = await self._client.keyword_search(
                query=query,
                x=parameters.get("x", ""),
                y=parameters.get("y", ""),
                radius=parameters.get("radius", 0),
                size=parameters.get("size", 5),
                sort=parameters.get("sort", "accuracy"),
            )
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 401:
                error_msg = "카카오 API 인증 실패 (REST API 키를 확인하세요)"
            else:
                error_msg = f"카카오맵 API 오류 ({status})"
            return ToolResult(
                tool_name="kakao_keyword_search",
                success=False,
                error=error_msg,
            )
        except Exception as e:
            logger.error("kakao_keyword_search_failed", query=query, error=str(e)[:200])
            return ToolResult(
                tool_name="kakao_keyword_search",
                success=False,
                error=f"장소 검색 실패: {str(e)[:300]}",
            )

        documents = data.get("documents", [])
        meta = data.get("meta", {})
        results = [_simplify_place(doc) for doc in documents]

        logger.info(
            "kakao_keyword_search_success",
            query=query,
            result_count=len(results),
            total_count=meta.get("total_count", 0),
        )

        return ToolResult(
            tool_name="kakao_keyword_search",
            success=True,
            output={
                "query": query,
                "results": results,
                "total_count": meta.get("total_count", 0),
                "is_end": meta.get("is_end", True),
            },
        )


class KakaoAddressSearchTool(ToolExecutor):
    """카카오맵 주소 → 좌표 변환 도구 (지오코딩)

    지번 주소 또는 도로명 주소를 좌표(경도/위도)로 변환

    Args:
        client: KakaoMapClient 인스턴스
    """

    def __init__(self, client: KakaoMapClient) -> None:
        self._client = client

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="kakao_address_search",
            description=(
                "주소를 좌표(경도/위도)로 변환합니다 (지오코딩). "
                "지번 주소와 도로명 주소 모두 지원합니다. "
                "예: '서울 강남구 삼성동 159' → x: 127.06, y: 37.51"
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="검색할 주소 (예: '서울 강남구 테헤란로 152')",
                    required=True,
                ),
            ],
            category=ToolCategory.MCP,
            safety_hint=SafetyLevelHint.READ_ONLY,
            version="1.0.0",
        )

    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        query = parameters.get("query", "").strip()
        if not query:
            return ToolResult(
                tool_name="kakao_address_search",
                success=False,
                error="query가 비어있습니다",
            )

        try:
            data = await self._client.address_search(query=query)
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 401:
                error_msg = "카카오 API 인증 실패 (REST API 키를 확인하세요)"
            else:
                error_msg = f"카카오맵 API 오류 ({status})"
            return ToolResult(
                tool_name="kakao_address_search",
                success=False,
                error=error_msg,
            )
        except Exception as e:
            logger.error("kakao_address_search_failed", query=query, error=str(e)[:200])
            return ToolResult(
                tool_name="kakao_address_search",
                success=False,
                error=f"주소 검색 실패: {str(e)[:300]}",
            )

        documents = data.get("documents", [])
        meta = data.get("meta", {})
        results = [_simplify_address_doc(doc) for doc in documents]

        logger.info(
            "kakao_address_search_success",
            query=query,
            result_count=len(results),
        )

        return ToolResult(
            tool_name="kakao_address_search",
            success=True,
            output={
                "query": query,
                "results": results,
                "total_count": meta.get("total_count", 0),
            },
        )


class KakaoCoord2AddressTool(ToolExecutor):
    """카카오맵 좌표 → 주소 변환 도구 (역지오코딩)

    경도/위도 좌표를 지번 주소 및 도로명 주소로 변환

    Args:
        client: KakaoMapClient 인스턴스
    """

    def __init__(self, client: KakaoMapClient) -> None:
        self._client = client

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="kakao_coord2address",
            description=(
                "좌표(경도/위도)를 주소로 변환합니다 (역지오코딩). "
                "지번 주소와 도로명 주소를 모두 반환합니다. "
                "x는 경도(longitude), y는 위도(latitude)입니다."
            ),
            parameters=[
                ToolParameter(
                    name="x",
                    type="string",
                    description="경도 (longitude / 예: '127.0276')",
                    required=True,
                ),
                ToolParameter(
                    name="y",
                    type="string",
                    description="위도 (latitude / 예: '37.4979')",
                    required=True,
                ),
            ],
            category=ToolCategory.MCP,
            safety_hint=SafetyLevelHint.READ_ONLY,
            version="1.0.0",
        )

    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        x = parameters.get("x", "").strip()
        y = parameters.get("y", "").strip()
        if not x or not y:
            return ToolResult(
                tool_name="kakao_coord2address",
                success=False,
                error="x(경도)와 y(위도) 모두 필요합니다",
            )

        # 숫자 유효성 검증
        try:
            float(x)
            float(y)
        except ValueError:
            return ToolResult(
                tool_name="kakao_coord2address",
                success=False,
                error="x와 y는 유효한 숫자(좌표)여야 합니다",
            )

        try:
            data = await self._client.coord2address(x=x, y=y)
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 401:
                error_msg = "카카오 API 인증 실패 (REST API 키를 확인하세요)"
            elif status == 400:
                error_msg = "유효하지 않은 좌표입니다"
            else:
                error_msg = f"카카오맵 API 오류 ({status})"
            return ToolResult(
                tool_name="kakao_coord2address",
                success=False,
                error=error_msg,
            )
        except Exception as e:
            logger.error("kakao_coord2address_failed", x=x, y=y, error=str(e)[:200])
            return ToolResult(
                tool_name="kakao_coord2address",
                success=False,
                error=f"좌표→주소 변환 실패: {str(e)[:300]}",
            )

        documents = data.get("documents", [])
        results = []
        for doc in documents:
            result: dict[str, Any] = {"x": x, "y": y}
            addr = doc.get("address")
            if addr:
                result["address"] = addr.get("address_name", "")
                result["region_1depth"] = addr.get("region_1depth_name", "")
                result["region_2depth"] = addr.get("region_2depth_name", "")
                result["region_3depth"] = addr.get("region_3depth_name", "")
            road = doc.get("road_address")
            if road:
                result["road_address"] = road.get("address_name", "")
                result["building_name"] = road.get("building_name", "")
            results.append(result)

        logger.info(
            "kakao_coord2address_success",
            x=x,
            y=y,
            result_count=len(results),
        )

        return ToolResult(
            tool_name="kakao_coord2address",
            success=True,
            output={
                "x": x,
                "y": y,
                "results": results,
            },
        )
