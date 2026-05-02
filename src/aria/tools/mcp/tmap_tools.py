"""ARIA Engine - MCP Tool: TMAP 대중교통 (SK Open API)

TMAP 대중교통 API 기반 도구 1종
- TmapTransitRouteTool: 대중교통 경로 검색

인증: SK Open API App Key (appKey 헤더)
API 문서: https://transit.tmapmobility.com/docs/routes

설계 원칙:
- httpx 직접 사용 (추가 의존성 없음)
- 응답에서 경로 요약 정보만 추출 (전체 좌표 데이터 제외 → 토큰 절약)
- 에러는 ToolResult로 감싸서 반환 (예외 전파 없음)
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog

from aria.core.config import TmapConfig
from aria.tools.tool_types import (
    SafetyLevelHint,
    ToolCategory,
    ToolDefinition,
    ToolExecutor,
    ToolParameter,
    ToolResult,
)

logger = structlog.get_logger()

TMAP_TRANSIT_URL = "https://apis.openapi.sk.com/transit/routes"

# mode 코드 → 한국어 매핑
_MODE_NAMES: dict[str, str] = {
    "WALK": "도보",
    "BUS": "버스",
    "SUBWAY": "지하철",
    "EXPRESSBUS": "고속/시외버스",
    "TRAIN": "기차",
    "AIRPLANE": "항공",
    "FERRY": "해운",
}

# pathType 코드 → 설명
_PATH_TYPE_NAMES: dict[int, str] = {
    1: "지하철",
    2: "버스",
    3: "버스+지하철",
    4: "고속/시외버스",
    5: "기차",
    6: "항공",
    7: "해운",
}


class TmapClient:
    """TMAP 대중교통 API HTTP 클라이언트

    SK Open API App Key 기반 인증
    httpx.AsyncClient를 재사용하여 커넥션 풀링

    Args:
        config: TmapConfig (app_key / request_timeout)
    """

    def __init__(self, config: TmapConfig) -> None:
        self._config = config
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Lazy initialization — 첫 호출 시 클라이언트 생성"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers={
                    "accept": "application/json",
                    "content-type": "application/json",
                    "appKey": self._config.app_key,
                },
                timeout=httpx.Timeout(self._config.request_timeout),
            )
        return self._client

    async def search_transit(
        self,
        start_x: str,
        start_y: str,
        end_x: str,
        end_y: str,
        count: int = 3,
        search_dttm: str = "",
    ) -> dict[str, Any]:
        """대중교통 경로 검색

        Args:
            start_x: 출발지 경도 (WGS84)
            start_y: 출발지 위도 (WGS84)
            end_x: 도착지 경도 (WGS84)
            end_y: 도착지 위도 (WGS84)
            count: 최대 결과 수 (1~10)
            search_dttm: 타임머신 검색 (yyyymmddhhmi / 빈 문자열이면 현재 시각)
        """
        body: dict[str, Any] = {
            "startX": start_x,
            "startY": start_y,
            "endX": end_x,
            "endY": end_y,
            "count": min(max(count, 1), 10),
            "lang": 0,
            "format": "json",
        }
        if search_dttm:
            body["searchDttm"] = search_dttm

        client = self._get_client()
        response = await client.post(TMAP_TRANSIT_URL, json=body)
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        """클라이언트 종료"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# === 응답 정제 유틸리티 ===


def _simplify_leg(leg: dict[str, Any]) -> dict[str, Any]:
    """경로 구간(leg)을 LLM 친화 형태로 정제 (좌표 데이터 제외)"""
    mode = leg.get("mode", "")
    result: dict[str, Any] = {
        "mode": _MODE_NAMES.get(mode, mode),
        "distance_m": leg.get("distance", 0),
        "duration_sec": leg.get("sectionTime", 0),
    }

    start = leg.get("start", {})
    end = leg.get("end", {})
    if start.get("name"):
        result["start_name"] = start["name"]
    if end.get("name"):
        result["end_name"] = end["name"]

    # 대중교통 노선 정보
    if leg.get("route"):
        result["route"] = leg["route"]
    if leg.get("type") is not None:
        result["route_type"] = leg["type"]

    # 정류장 목록 (이름만 추출 → 토큰 절약)
    pass_stop_list = leg.get("passStopList", {})
    stations = pass_stop_list.get("stationList", [])
    if stations:
        result["stops"] = [s.get("stationName", "") for s in stations if s.get("stationName")]
        result["stop_count"] = len(result["stops"])

    return result


def _simplify_itinerary(itinerary: dict[str, Any]) -> dict[str, Any]:
    """경로 후보(itinerary)를 LLM 친화 형태로 정제"""
    fare_info = itinerary.get("fare", {}).get("regular", {})

    result: dict[str, Any] = {
        "total_time_min": round(itinerary.get("totalTime", 0) / 60, 1),
        "total_distance_m": itinerary.get("totalDistance", 0),
        "total_walk_distance_m": itinerary.get("totalWalkDistance", 0),
        "total_walk_time_min": round(itinerary.get("totalWalkTime", 0) / 60, 1),
        "transfer_count": itinerary.get("transferCount", 0),
        "fare": fare_info.get("totalFare", 0),
        "path_type": _PATH_TYPE_NAMES.get(itinerary.get("pathType", 0), ""),
    }

    legs = itinerary.get("legs", [])
    result["legs"] = [_simplify_leg(leg) for leg in legs]

    return result


class TmapTransitRouteTool(ToolExecutor):
    """TMAP 대중교통 경로 검색 도구

    출발지/도착지 좌표 기반으로 대중교통 경로를 검색합니다.
    버스/지하철/고속버스/기차 등 다양한 교통 수단의 최적 경로를 제공합니다.

    Args:
        client: TmapClient 인스턴스
    """

    def __init__(self, client: TmapClient) -> None:
        self._client = client

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="tmap_transit_route",
            description=(
                "대중교통 경로를 검색합니다 (TMAP). "
                "출발지/도착지의 좌표(경도/위도)를 입력하면 "
                "버스, 지하철, 기차 등의 최적 경로와 소요시간, 요금, 환승 정보를 제공합니다. "
                "좌표를 모를 때는 먼저 kakao_address_search로 주소→좌표 변환 후 사용하세요."
            ),
            parameters=[
                ToolParameter(
                    name="start_x",
                    type="string",
                    description="출발지 경도 (longitude / 예: '127.0276')",
                    required=True,
                ),
                ToolParameter(
                    name="start_y",
                    type="string",
                    description="출발지 위도 (latitude / 예: '37.4979')",
                    required=True,
                ),
                ToolParameter(
                    name="end_x",
                    type="string",
                    description="도착지 경도 (longitude)",
                    required=True,
                ),
                ToolParameter(
                    name="end_y",
                    type="string",
                    description="도착지 위도 (latitude)",
                    required=True,
                ),
                ToolParameter(
                    name="count",
                    type="integer",
                    description="최대 결과 수 (1~10 / 기본값 3)",
                    required=False,
                ),
            ],
            category=ToolCategory.MCP,
            safety_hint=SafetyLevelHint.READ_ONLY,
            version="1.0.0",
        )

    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        start_x = parameters.get("start_x", "").strip()
        start_y = parameters.get("start_y", "").strip()
        end_x = parameters.get("end_x", "").strip()
        end_y = parameters.get("end_y", "").strip()

        if not all([start_x, start_y, end_x, end_y]):
            return ToolResult(
                tool_name="tmap_transit_route",
                success=False,
                error="출발지(start_x/y)와 도착지(end_x/y) 좌표 모두 필요합니다",
            )

        # 좌표 유효성 검증
        for label, val in [("start_x", start_x), ("start_y", start_y),
                           ("end_x", end_x), ("end_y", end_y)]:
            try:
                float(val)
            except ValueError:
                return ToolResult(
                    tool_name="tmap_transit_route",
                    success=False,
                    error=f"{label}는 유효한 숫자(좌표)여야 합니다: {val}",
                )

        try:
            data = await self._client.search_transit(
                start_x=start_x,
                start_y=start_y,
                end_x=end_x,
                end_y=end_y,
                count=parameters.get("count", 3),
            )
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 401 or status == 403:
                error_msg = "TMAP API 인증 실패 (SK APP KEY를 확인하세요)"
            elif status == 429:
                error_msg = "TMAP API 호출 한도 초과 (잠시 후 다시 시도하세요)"
            else:
                error_msg = f"TMAP API 오류 ({status})"
            return ToolResult(
                tool_name="tmap_transit_route",
                success=False,
                error=error_msg,
            )
        except Exception as e:
            logger.error(
                "tmap_transit_route_failed",
                start=f"{start_x},{start_y}",
                end=f"{end_x},{end_y}",
                error=str(e)[:200],
            )
            return ToolResult(
                tool_name="tmap_transit_route",
                success=False,
                error=f"대중교통 경로 검색 실패: {str(e)[:300]}",
            )

        # 응답 파싱
        meta = data.get("metaData", {})
        plan = meta.get("plan", {})
        itineraries = plan.get("itineraries", [])

        if not itineraries:
            return ToolResult(
                tool_name="tmap_transit_route",
                success=True,
                output={
                    "start": f"{start_x},{start_y}",
                    "end": f"{end_x},{end_y}",
                    "routes": [],
                    "message": "해당 구간의 대중교통 경로를 찾을 수 없습니다",
                },
            )

        routes = [_simplify_itinerary(it) for it in itineraries]

        logger.info(
            "tmap_transit_route_success",
            start=f"{start_x},{start_y}",
            end=f"{end_x},{end_y}",
            route_count=len(routes),
        )

        return ToolResult(
            tool_name="tmap_transit_route",
            success=True,
            output={
                "start": f"{start_x},{start_y}",
                "end": f"{end_x},{end_y}",
                "routes": routes,
                "route_count": len(routes),
            },
        )
