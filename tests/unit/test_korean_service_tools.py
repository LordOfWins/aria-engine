"""ARIA Engine - 한국 서비스 MCP 도구 단위 테스트

카카오맵 / 네이버 검색 / TMAP 대중교통 도구 테스트
- Config is_configured 테스트
- 클라이언트 HTTP 호출 mock 테스트
- 도구 정의(ToolDefinition) 검증
- 도구 실행(execute) 성공/실패 테스트
- 응답 정제 유틸리티 테스트
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from aria.core.config import KakaoMapConfig, NaverSearchConfig, TmapConfig
from aria.tools.tool_types import ToolCategory, SafetyLevelHint


# ============================================================
# Config Tests
# ============================================================


class TestKakaoMapConfig:
    def test_not_configured_when_empty(self):
        config = KakaoMapConfig(rest_api_key="")
        assert config.is_configured is False

    def test_configured_when_set(self):
        config = KakaoMapConfig(rest_api_key="test-key-123")
        assert config.is_configured is True

    def test_default_timeout(self):
        config = KakaoMapConfig()
        assert config.request_timeout == 10


class TestNaverSearchConfig:
    def test_not_configured_when_empty(self):
        config = NaverSearchConfig(client_id="", client_secret="")
        assert config.is_configured is False

    def test_not_configured_when_partial(self):
        config = NaverSearchConfig(client_id="id", client_secret="")
        assert config.is_configured is False

    def test_configured_when_both_set(self):
        config = NaverSearchConfig(client_id="id", client_secret="secret")
        assert config.is_configured is True

    def test_default_timeout(self):
        config = NaverSearchConfig()
        assert config.request_timeout == 10


class TestTmapConfig:
    def test_not_configured_when_empty(self):
        config = TmapConfig(app_key="")
        assert config.is_configured is False

    def test_configured_when_set(self):
        config = TmapConfig(app_key="test-sk-key")
        assert config.is_configured is True

    def test_default_timeout(self):
        config = TmapConfig()
        assert config.request_timeout == 15


# ============================================================
# Kakao Map Tools Tests
# ============================================================


class TestKakaoMapClient:
    @pytest.fixture
    def config(self):
        return KakaoMapConfig(rest_api_key="test-kakao-key")

    @pytest.fixture
    def client(self, config):
        from aria.tools.mcp.kakao_map_tools import KakaoMapClient
        return KakaoMapClient(config)

    def test_client_lazy_init(self, client):
        """클라이언트는 첫 호출 시 생성되어야 함"""
        assert client._client is None
        http_client = client._get_client()
        assert http_client is not None
        assert "KakaoAK test-kakao-key" in http_client.headers.get("Authorization", "")

    def test_client_reuses_instance(self, client):
        """같은 클라이언트 인스턴스를 재사용해야 함"""
        c1 = client._get_client()
        c2 = client._get_client()
        assert c1 is c2


class TestKakaoKeywordSearchTool:
    @pytest.fixture
    def mock_client(self):
        from aria.tools.mcp.kakao_map_tools import KakaoMapClient
        client = MagicMock(spec=KakaoMapClient)
        return client

    @pytest.fixture
    def tool(self, mock_client):
        from aria.tools.mcp.kakao_map_tools import KakaoKeywordSearchTool
        return KakaoKeywordSearchTool(mock_client)

    def test_definition(self, tool):
        defn = tool.get_definition()
        assert defn.name == "kakao_keyword_search"
        assert defn.category == ToolCategory.MCP
        assert defn.safety_hint == SafetyLevelHint.READ_ONLY
        assert any(p.name == "query" and p.required for p in defn.parameters)

    def test_definition_llm_tool_format(self, tool):
        llm_tool = tool.get_definition().to_llm_tool()
        assert llm_tool["type"] == "function"
        assert llm_tool["function"]["name"] == "kakao_keyword_search"
        assert "query" in llm_tool["function"]["parameters"]["properties"]

    @pytest.mark.asyncio
    async def test_execute_success(self, tool, mock_client):
        mock_client.keyword_search = AsyncMock(return_value={
            "meta": {"total_count": 2, "is_end": True},
            "documents": [
                {
                    "place_name": "강남역 맛집A",
                    "address_name": "서울 강남구 역삼동 123",
                    "road_address_name": "서울 강남구 강남대로 100",
                    "phone": "02-1234-5678",
                    "category_name": "음식점 > 한식",
                    "x": "127.0276",
                    "y": "37.4979",
                    "place_url": "http://place.map.kakao.com/123",
                    "distance": "150",
                },
                {
                    "place_name": "강남역 맛집B",
                    "address_name": "서울 강남구 역삼동 456",
                    "road_address_name": "",
                    "phone": "",
                    "category_name": "음식점 > 일식",
                    "x": "127.0280",
                    "y": "37.4980",
                    "place_url": "http://place.map.kakao.com/456",
                    "distance": "200",
                },
            ],
        })

        result = await tool.execute({"query": "강남역 맛집"})
        assert result.success is True
        assert result.output["total_count"] == 2
        assert len(result.output["results"]) == 2
        assert result.output["results"][0]["name"] == "강남역 맛집A"
        assert result.output["results"][0]["phone"] == "02-1234-5678"

    @pytest.mark.asyncio
    async def test_execute_empty_query(self, tool):
        result = await tool.execute({"query": ""})
        assert result.success is False
        assert "비어있습니다" in result.error

    @pytest.mark.asyncio
    async def test_execute_auth_error(self, tool, mock_client):
        response = MagicMock()
        response.status_code = 401
        mock_client.keyword_search = AsyncMock(
            side_effect=httpx.HTTPStatusError("Unauthorized", request=MagicMock(), response=response)
        )
        result = await tool.execute({"query": "테스트"})
        assert result.success is False
        assert "인증 실패" in result.error

    @pytest.mark.asyncio
    async def test_execute_network_error(self, tool, mock_client):
        mock_client.keyword_search = AsyncMock(side_effect=Exception("Connection timeout"))
        result = await tool.execute({"query": "테스트"})
        assert result.success is False
        assert "검색 실패" in result.error


class TestKakaoAddressSearchTool:
    @pytest.fixture
    def tool(self):
        from aria.tools.mcp.kakao_map_tools import KakaoMapClient, KakaoAddressSearchTool
        client = MagicMock(spec=KakaoMapClient)
        return KakaoAddressSearchTool(client)

    def test_definition(self, tool):
        defn = tool.get_definition()
        assert defn.name == "kakao_address_search"
        assert defn.safety_hint == SafetyLevelHint.READ_ONLY

    @pytest.mark.asyncio
    async def test_execute_success(self, tool):
        tool._client.address_search = AsyncMock(return_value={
            "meta": {"total_count": 1},
            "documents": [{
                "address_name": "서울 강남구 테헤란로 152",
                "address_type": "ROAD_ADDR",
                "x": "127.0340",
                "y": "37.5017",
                "road_address": {
                    "address_name": "서울 강남구 테헤란로 152",
                    "building_name": "강남파이낸스센터",
                },
            }],
        })
        result = await tool.execute({"query": "서울 강남구 테헤란로 152"})
        assert result.success is True
        assert result.output["results"][0]["x"] == "127.0340"

    @pytest.mark.asyncio
    async def test_execute_empty_query(self, tool):
        result = await tool.execute({"query": "   "})
        assert result.success is False


class TestKakaoCoord2AddressTool:
    @pytest.fixture
    def tool(self):
        from aria.tools.mcp.kakao_map_tools import KakaoMapClient, KakaoCoord2AddressTool
        client = MagicMock(spec=KakaoMapClient)
        return KakaoCoord2AddressTool(client)

    def test_definition(self, tool):
        defn = tool.get_definition()
        assert defn.name == "kakao_coord2address"

    @pytest.mark.asyncio
    async def test_execute_success(self, tool):
        tool._client.coord2address = AsyncMock(return_value={
            "meta": {"total_count": 1},
            "documents": [{
                "address": {
                    "address_name": "경기 남양주시 화도읍",
                    "region_1depth_name": "경기",
                    "region_2depth_name": "남양주시",
                    "region_3depth_name": "화도읍",
                },
                "road_address": {
                    "address_name": "경기 남양주시 화도읍 어쩌구",
                    "building_name": "",
                },
            }],
        })
        result = await tool.execute({"x": "127.3", "y": "37.6"})
        assert result.success is True
        assert result.output["results"][0]["address"] == "경기 남양주시 화도읍"

    @pytest.mark.asyncio
    async def test_execute_missing_coords(self, tool):
        result = await tool.execute({"x": "127.0"})
        assert result.success is False
        assert "모두 필요" in result.error

    @pytest.mark.asyncio
    async def test_execute_invalid_coords(self, tool):
        result = await tool.execute({"x": "abc", "y": "37.0"})
        assert result.success is False
        assert "유효한 숫자" in result.error


# ============================================================
# Kakao Map Response Simplifiers
# ============================================================


class TestKakaoSimplifiers:
    def test_simplify_place(self):
        from aria.tools.mcp.kakao_map_tools import _simplify_place
        doc = {
            "place_name": "테스트 장소",
            "address_name": "서울 강남구",
            "road_address_name": "서울 강남구 도로명",
            "phone": "02-111-2222",
            "category_name": "음식점",
            "x": "127.0",
            "y": "37.5",
            "place_url": "http://place.map.kakao.com/1",
            "distance": "100",
        }
        result = _simplify_place(doc)
        assert result["name"] == "테스트 장소"
        assert result["phone"] == "02-111-2222"
        assert result["x"] == "127.0"

    def test_simplify_address_doc(self):
        from aria.tools.mcp.kakao_map_tools import _simplify_address_doc
        doc = {
            "address_name": "서울 강남구 삼성동 159",
            "address_type": "REGION_ADDR",
            "x": "127.06",
            "y": "37.51",
            "road_address": {
                "address_name": "서울 강남구 영동대로 513",
                "building_name": "코엑스",
            },
        }
        result = _simplify_address_doc(doc)
        assert result["address_name"] == "서울 강남구 삼성동 159"
        assert result["road_address"] == "서울 강남구 영동대로 513"
        assert result["building_name"] == "코엑스"


# ============================================================
# Naver Search Tools Tests
# ============================================================


class TestNaverSearchClient:
    @pytest.fixture
    def config(self):
        return NaverSearchConfig(client_id="test-id", client_secret="test-secret")

    @pytest.fixture
    def client(self, config):
        from aria.tools.mcp.naver_search_tools import NaverSearchClient
        return NaverSearchClient(config)

    def test_client_lazy_init(self, client):
        assert client._client is None
        http_client = client._get_client()
        assert http_client is not None
        assert http_client.headers.get("X-Naver-Client-Id") == "test-id"
        assert http_client.headers.get("X-Naver-Client-Secret") == "test-secret"


class TestNaverBlogSearchTool:
    @pytest.fixture
    def mock_client(self):
        from aria.tools.mcp.naver_search_tools import NaverSearchClient
        return MagicMock(spec=NaverSearchClient)

    @pytest.fixture
    def tool(self, mock_client):
        from aria.tools.mcp.naver_search_tools import NaverBlogSearchTool
        return NaverBlogSearchTool(mock_client)

    def test_definition(self, tool):
        defn = tool.get_definition()
        assert defn.name == "naver_blog_search"
        assert defn.category == ToolCategory.MCP
        assert defn.safety_hint == SafetyLevelHint.READ_ONLY

    @pytest.mark.asyncio
    async def test_execute_success(self, tool, mock_client):
        mock_client.search = AsyncMock(return_value={
            "total": 100,
            "display": 5,
            "items": [
                {
                    "title": "<b>남양주</b> 맛집 추천",
                    "description": "오늘은 <b>남양주</b>에서 맛집을 방문...",
                    "bloggername": "맛집탐험가",
                    "link": "https://blog.naver.com/test/123",
                    "postdate": "20260501",
                },
            ],
        })

        result = await tool.execute({"query": "남양주 맛집"})
        assert result.success is True
        assert result.output["total"] == 100
        # HTML 태그 제거 확인
        assert "<b>" not in result.output["results"][0]["title"]
        assert "남양주" in result.output["results"][0]["title"]

    @pytest.mark.asyncio
    async def test_execute_empty_query(self, tool):
        result = await tool.execute({"query": ""})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_rate_limit(self, tool, mock_client):
        response = MagicMock()
        response.status_code = 429
        mock_client.search = AsyncMock(
            side_effect=httpx.HTTPStatusError("Rate limited", request=MagicMock(), response=response)
        )
        result = await tool.execute({"query": "테스트"})
        assert result.success is False
        assert "한도 초과" in result.error


class TestNaverNewsSearchTool:
    @pytest.fixture
    def tool(self):
        from aria.tools.mcp.naver_search_tools import NaverSearchClient, NaverNewsSearchTool
        client = MagicMock(spec=NaverSearchClient)
        return NaverNewsSearchTool(client)

    def test_definition(self, tool):
        defn = tool.get_definition()
        assert defn.name == "naver_news_search"

    @pytest.mark.asyncio
    async def test_execute_success(self, tool):
        tool._client.search = AsyncMock(return_value={
            "total": 50,
            "display": 3,
            "items": [
                {
                    "title": "AI <b>기술</b> 발전",
                    "description": "AI 기술이 빠르게...",
                    "originallink": "https://news.example.com/123",
                    "link": "https://n.news.naver.com/123",
                    "pubDate": "Sat, 03 May 2026 10:00:00 +0900",
                },
            ],
        })
        result = await tool.execute({"query": "AI 기술"})
        assert result.success is True
        assert result.output["results"][0]["link"] == "https://news.example.com/123"


class TestNaverShopSearchTool:
    @pytest.fixture
    def tool(self):
        from aria.tools.mcp.naver_search_tools import NaverSearchClient, NaverShopSearchTool
        client = MagicMock(spec=NaverSearchClient)
        return NaverShopSearchTool(client)

    def test_definition_has_sort_options(self, tool):
        defn = tool.get_definition()
        assert defn.name == "naver_shop_search"
        sort_param = next(p for p in defn.parameters if p.name == "sort")
        assert "asc" in sort_param.enum
        assert "dsc" in sort_param.enum

    @pytest.mark.asyncio
    async def test_simplify_shop_item(self, tool):
        tool._client.search = AsyncMock(return_value={
            "total": 1,
            "display": 1,
            "items": [{
                "title": "<b>맥북</b> 프로",
                "link": "https://shop.example.com",
                "lprice": "2500000",
                "hprice": "3000000",
                "mallName": "Apple Store",
                "brand": "Apple",
                "category1": "컴퓨터",
                "category2": "노트북",
                "category3": "Apple",
            }],
        })
        result = await tool.execute({"query": "맥북"})
        assert result.success is True
        item = result.output["results"][0]
        assert item["lprice"] == "2500000"
        assert item["category"] == "컴퓨터/노트북/Apple"


class TestNaverLocalSearchTool:
    @pytest.fixture
    def tool(self):
        from aria.tools.mcp.naver_search_tools import NaverSearchClient, NaverLocalSearchTool
        client = MagicMock(spec=NaverSearchClient)
        return NaverLocalSearchTool(client)

    def test_definition(self, tool):
        defn = tool.get_definition()
        assert defn.name == "naver_local_search"

    @pytest.mark.asyncio
    async def test_execute_success(self, tool):
        tool._client.search = AsyncMock(return_value={
            "total": 5,
            "display": 1,
            "items": [{
                "title": "<b>화도</b>약국",
                "category": "약국",
                "address": "경기 남양주시 화도읍",
                "roadAddress": "경기 남양주시 화도읍 도로명",
                "telephone": "031-111-2222",
                "link": "https://example.com",
                "mapx": "127300000",
                "mapy": "37600000",
            }],
        })
        result = await tool.execute({"query": "화도 약국"})
        assert result.success is True
        item = result.output["results"][0]
        assert item["title"] == "화도약국"
        assert item["phone"] == "031-111-2222"


class TestNaverAllToolDefinitions:
    """6종 검색 도구의 정의가 모두 유효한지 확인"""

    @pytest.fixture
    def all_tools(self):
        from aria.tools.mcp.naver_search_tools import (
            NaverSearchClient,
            NaverBlogSearchTool, NaverNewsSearchTool,
            NaverCafeSearchTool, NaverShopSearchTool,
            NaverKinSearchTool, NaverLocalSearchTool,
        )
        client = MagicMock(spec=NaverSearchClient)
        return [
            NaverBlogSearchTool(client),
            NaverNewsSearchTool(client),
            NaverCafeSearchTool(client),
            NaverShopSearchTool(client),
            NaverKinSearchTool(client),
            NaverLocalSearchTool(client),
        ]

    def test_all_definitions_valid(self, all_tools):
        names = set()
        for tool in all_tools:
            defn = tool.get_definition()
            assert defn.name.startswith("naver_")
            assert defn.category == ToolCategory.MCP
            assert len(defn.description) > 10
            names.add(defn.name)
        # 6개 고유 이름
        assert len(names) == 6

    def test_all_generate_llm_tool_format(self, all_tools):
        for tool in all_tools:
            llm_tool = tool.get_definition().to_llm_tool()
            assert llm_tool["type"] == "function"
            assert "query" in llm_tool["function"]["parameters"]["properties"]


# ============================================================
# Naver HTML Strip
# ============================================================


class TestNaverStripHtml:
    def test_strip_basic_tags(self):
        from aria.tools.mcp.naver_search_tools import _strip_html
        assert _strip_html("<b>테스트</b>") == "테스트"
        assert _strip_html("a<br>b") == "ab"
        assert _strip_html("plain text") == "plain text"

    def test_strip_nested_tags(self):
        from aria.tools.mcp.naver_search_tools import _strip_html
        assert _strip_html("<p><b>bold</b> text</p>") == "bold text"


# ============================================================
# TMAP Tools Tests
# ============================================================


class TestTmapClient:
    @pytest.fixture
    def config(self):
        return TmapConfig(app_key="test-sk-key")

    @pytest.fixture
    def client(self, config):
        from aria.tools.mcp.tmap_tools import TmapClient
        return TmapClient(config)

    def test_client_lazy_init(self, client):
        assert client._client is None
        http_client = client._get_client()
        assert http_client is not None
        assert http_client.headers.get("appKey") == "test-sk-key"

    def test_client_reuses_instance(self, client):
        c1 = client._get_client()
        c2 = client._get_client()
        assert c1 is c2


class TestTmapTransitRouteTool:
    @pytest.fixture
    def mock_client(self):
        from aria.tools.mcp.tmap_tools import TmapClient
        return MagicMock(spec=TmapClient)

    @pytest.fixture
    def tool(self, mock_client):
        from aria.tools.mcp.tmap_tools import TmapTransitRouteTool
        return TmapTransitRouteTool(mock_client)

    def test_definition(self, tool):
        defn = tool.get_definition()
        assert defn.name == "tmap_transit_route"
        assert defn.category == ToolCategory.MCP
        assert defn.safety_hint == SafetyLevelHint.READ_ONLY
        required_params = [p.name for p in defn.parameters if p.required]
        assert "start_x" in required_params
        assert "start_y" in required_params
        assert "end_x" in required_params
        assert "end_y" in required_params

    @pytest.mark.asyncio
    async def test_execute_success(self, tool, mock_client):
        mock_client.search_transit = AsyncMock(return_value={
            "metaData": {
                "requestParameters": {},
                "plan": {
                    "itineraries": [
                        {
                            "totalTime": 2400,
                            "totalDistance": 5000,
                            "totalWalkDistance": 800,
                            "totalWalkTime": 600,
                            "transferCount": 1,
                            "pathType": 3,
                            "fare": {
                                "regular": {
                                    "totalFare": 1500,
                                    "currency": {"symbol": "￦", "currency": "원", "currencyCode": "KRW"},
                                },
                            },
                            "legs": [
                                {
                                    "mode": "WALK",
                                    "distance": 300,
                                    "sectionTime": 240,
                                    "start": {"name": "", "lat": 37.5, "lon": 127.0},
                                    "end": {"name": "강남역 3번출구", "lat": 37.501, "lon": 127.001},
                                },
                                {
                                    "mode": "SUBWAY",
                                    "distance": 4000,
                                    "sectionTime": 1800,
                                    "route": "2호선",
                                    "type": 1,
                                    "start": {"name": "강남", "lat": 37.501, "lon": 127.001},
                                    "end": {"name": "삼성", "lat": 37.51, "lon": 127.06},
                                    "passStopList": {
                                        "stationList": [
                                            {"index": 0, "stationName": "강남", "stationID": "1"},
                                            {"index": 1, "stationName": "역삼", "stationID": "2"},
                                            {"index": 2, "stationName": "삼성", "stationID": "3"},
                                        ],
                                    },
                                },
                                {
                                    "mode": "WALK",
                                    "distance": 500,
                                    "sectionTime": 360,
                                    "start": {"name": "삼성역 6번출구", "lat": 37.51, "lon": 127.06},
                                    "end": {"name": "", "lat": 37.512, "lon": 127.062},
                                },
                            ],
                        },
                    ],
                },
            },
        })

        result = await tool.execute({
            "start_x": "127.0",
            "start_y": "37.5",
            "end_x": "127.06",
            "end_y": "37.51",
        })
        assert result.success is True
        assert result.output["route_count"] == 1

        route = result.output["routes"][0]
        assert route["total_time_min"] == 40.0
        assert route["transfer_count"] == 1
        assert route["fare"] == 1500
        assert route["path_type"] == "버스+지하철"
        assert len(route["legs"]) == 3
        assert route["legs"][0]["mode"] == "도보"
        assert route["legs"][1]["mode"] == "지하철"
        assert route["legs"][1]["route"] == "2호선"
        assert route["legs"][1]["stops"] == ["강남", "역삼", "삼성"]

    @pytest.mark.asyncio
    async def test_execute_missing_coords(self, tool):
        result = await tool.execute({"start_x": "127.0", "start_y": "37.5"})
        assert result.success is False
        assert "모두 필요" in result.error

    @pytest.mark.asyncio
    async def test_execute_invalid_coords(self, tool):
        result = await tool.execute({
            "start_x": "abc", "start_y": "37.5",
            "end_x": "127.0", "end_y": "37.6",
        })
        assert result.success is False
        assert "유효한 숫자" in result.error

    @pytest.mark.asyncio
    async def test_execute_no_routes(self, tool, mock_client):
        mock_client.search_transit = AsyncMock(return_value={
            "metaData": {
                "requestParameters": {},
                "plan": {"itineraries": []},
            },
        })
        result = await tool.execute({
            "start_x": "127.0", "start_y": "37.5",
            "end_x": "130.0", "end_y": "35.0",
        })
        assert result.success is True
        assert len(result.output["routes"]) == 0
        assert "찾을 수 없습니다" in result.output["message"]

    @pytest.mark.asyncio
    async def test_execute_auth_error(self, tool, mock_client):
        response = MagicMock()
        response.status_code = 401
        mock_client.search_transit = AsyncMock(
            side_effect=httpx.HTTPStatusError("Unauthorized", request=MagicMock(), response=response)
        )
        result = await tool.execute({
            "start_x": "127.0", "start_y": "37.5",
            "end_x": "127.1", "end_y": "37.6",
        })
        assert result.success is False
        assert "인증 실패" in result.error


# ============================================================
# TMAP Response Simplifiers
# ============================================================


class TestTmapSimplifiers:
    def test_simplify_leg_walk(self):
        from aria.tools.mcp.tmap_tools import _simplify_leg
        leg = {
            "mode": "WALK",
            "distance": 300,
            "sectionTime": 240,
            "start": {"name": "", "lat": 37.5, "lon": 127.0},
            "end": {"name": "역삼역", "lat": 37.501, "lon": 127.001},
        }
        result = _simplify_leg(leg)
        assert result["mode"] == "도보"
        assert result["distance_m"] == 300
        assert result["duration_sec"] == 240
        assert result["end_name"] == "역삼역"
        assert "start_name" not in result  # 빈 이름은 포함 안 함

    def test_simplify_leg_subway(self):
        from aria.tools.mcp.tmap_tools import _simplify_leg
        leg = {
            "mode": "SUBWAY",
            "distance": 5000,
            "sectionTime": 1200,
            "route": "2호선",
            "type": 1,
            "start": {"name": "강남", "lat": 37.5, "lon": 127.0},
            "end": {"name": "잠실", "lat": 37.51, "lon": 127.1},
            "passStopList": {
                "stationList": [
                    {"index": 0, "stationName": "강남", "stationID": "1"},
                    {"index": 1, "stationName": "삼성", "stationID": "2"},
                    {"index": 2, "stationName": "잠실", "stationID": "3"},
                ],
            },
        }
        result = _simplify_leg(leg)
        assert result["mode"] == "지하철"
        assert result["route"] == "2호선"
        assert result["stops"] == ["강남", "삼성", "잠실"]
        assert result["stop_count"] == 3

    def test_simplify_itinerary(self):
        from aria.tools.mcp.tmap_tools import _simplify_itinerary
        itinerary = {
            "totalTime": 3600,
            "totalDistance": 10000,
            "totalWalkDistance": 1000,
            "totalWalkTime": 900,
            "transferCount": 2,
            "pathType": 3,
            "fare": {"regular": {"totalFare": 2500}},
            "legs": [],
        }
        result = _simplify_itinerary(itinerary)
        assert result["total_time_min"] == 60.0
        assert result["total_walk_time_min"] == 15.0
        assert result["transfer_count"] == 2
        assert result["fare"] == 2500
        assert result["path_type"] == "버스+지하철"


# ============================================================
# App.py Integration — 조건부 등록 검증
# ============================================================


class TestConditionalRegistration:
    """AriaConfig에 새 서비스 필드가 존재하고 is_configured 프로퍼티가 동작하는지 확인"""

    def test_kakao_config_in_aria_config(self, monkeypatch):
        """AriaConfig에 kakao_map 필드 존재 + 키 없으면 비활성화"""
        monkeypatch.delenv("ARIA_KAKAO_REST_API_KEY", raising=False)
        from aria.core.config import AriaConfig
        config = AriaConfig()
        assert hasattr(config, "kakao_map")
        assert config.kakao_map.is_configured is False

    def test_naver_config_in_aria_config(self, monkeypatch):
        monkeypatch.delenv("ARIA_NAVER_CLIENT_ID", raising=False)
        monkeypatch.delenv("ARIA_NAVER_CLIENT_SECRET", raising=False)
        from aria.core.config import AriaConfig
        config = AriaConfig()
        assert hasattr(config, "naver_search")
        assert config.naver_search.is_configured is False

    def test_tmap_config_in_aria_config(self, monkeypatch):
        monkeypatch.delenv("ARIA_TMAP_APP_KEY", raising=False)
        from aria.core.config import AriaConfig
        config = AriaConfig()
        assert hasattr(config, "tmap")
        assert config.tmap.is_configured is False

    def test_ddg_config_in_aria_config(self):
        from aria.core.config import AriaConfig
        config = AriaConfig()
        assert hasattr(config, "ddg")
        assert config.ddg.is_configured is True  # 기본 활성화


# ============================================================
# DuckDuckGo Tools Tests
# ============================================================


class TestDuckDuckGoConfig:
    def test_default_enabled(self):
        from aria.core.config import DuckDuckGoConfig
        config = DuckDuckGoConfig()
        assert config.enabled is True
        assert config.is_configured is True

    def test_disabled(self):
        from aria.core.config import DuckDuckGoConfig
        config = DuckDuckGoConfig(enabled=False)
        assert config.is_configured is False

    def test_default_timeout(self):
        from aria.core.config import DuckDuckGoConfig
        config = DuckDuckGoConfig()
        assert config.request_timeout == 10


class TestDdgSearchClient:
    @pytest.fixture
    def config(self):
        from aria.core.config import DuckDuckGoConfig
        return DuckDuckGoConfig()

    @pytest.fixture
    def client(self, config):
        from aria.tools.mcp.ddg_tools import DdgSearchClient
        return DdgSearchClient(config)

    def test_client_created(self, client):
        assert client._config.enabled is True


class TestDdgWebSearchTool:
    @pytest.fixture
    def mock_client(self):
        from aria.tools.mcp.ddg_tools import DdgSearchClient
        return MagicMock(spec=DdgSearchClient)

    @pytest.fixture
    def tool(self, mock_client):
        from aria.tools.mcp.ddg_tools import DdgWebSearchTool
        return DdgWebSearchTool(mock_client)

    def test_definition(self, tool):
        defn = tool.get_definition()
        assert defn.name == "ddg_web_search"
        assert defn.category == ToolCategory.MCP
        assert defn.safety_hint == SafetyLevelHint.READ_ONLY
        assert any(p.name == "query" and p.required for p in defn.parameters)

    def test_definition_llm_tool_format(self, tool):
        llm_tool = tool.get_definition().to_llm_tool()
        assert llm_tool["type"] == "function"
        assert "query" in llm_tool["function"]["parameters"]["properties"]

    @pytest.mark.asyncio
    async def test_execute_success(self, tool, mock_client):
        mock_client.web_search = MagicMock(return_value=[
            {
                "title": "FastAPI Best Practices 2026",
                "href": "https://example.com/fastapi",
                "body": "Learn the best practices for FastAPI in 2026...",
            },
            {
                "title": "Python Web Frameworks Comparison",
                "href": "https://example.com/python-web",
                "body": "Comparing Django, FastAPI, and Flask...",
            },
        ])

        result = await tool.execute({"query": "FastAPI best practices"})
        assert result.success is True
        assert result.output["result_count"] == 2
        assert result.output["results"][0]["title"] == "FastAPI Best Practices 2026"
        assert result.output["results"][0]["url"] == "https://example.com/fastapi"

    @pytest.mark.asyncio
    async def test_execute_empty_query(self, tool):
        result = await tool.execute({"query": ""})
        assert result.success is False
        assert "비어있습니다" in result.error

    @pytest.mark.asyncio
    async def test_execute_error(self, tool, mock_client):
        mock_client.web_search = MagicMock(side_effect=Exception("Connection timeout"))
        result = await tool.execute({"query": "test"})
        assert result.success is False
        assert "검색 실패" in result.error

    @pytest.mark.asyncio
    async def test_execute_rate_limit(self, tool, mock_client):
        mock_client.web_search = MagicMock(side_effect=Exception("Ratelimit"))
        result = await tool.execute({"query": "test"})
        assert result.success is False
        assert "속도 제한" in result.error

    @pytest.mark.asyncio
    async def test_execute_with_region(self, tool, mock_client):
        mock_client.web_search = MagicMock(return_value=[])
        result = await tool.execute({"query": "AI 트렌드", "region": "kr-kr"})
        assert result.success is True
        assert result.output["region"] == "kr-kr"
        mock_client.web_search.assert_called_once_with(
            query="AI 트렌드",
            region="kr-kr",
            max_results=5,
        )


class TestDdgNewsSearchTool:
    @pytest.fixture
    def mock_client(self):
        from aria.tools.mcp.ddg_tools import DdgSearchClient
        return MagicMock(spec=DdgSearchClient)

    @pytest.fixture
    def tool(self, mock_client):
        from aria.tools.mcp.ddg_tools import DdgNewsSearchTool
        return DdgNewsSearchTool(mock_client)

    def test_definition(self, tool):
        defn = tool.get_definition()
        assert defn.name == "ddg_news_search"
        assert defn.category == ToolCategory.MCP

    @pytest.mark.asyncio
    async def test_execute_success(self, tool, mock_client):
        mock_client.news_search = MagicMock(return_value=[
            {
                "title": "AI Startup Raises $100M",
                "url": "https://news.example.com/ai-startup",
                "body": "An AI startup raised $100M in funding...",
                "source": "TechCrunch",
                "date": "2026-05-03T10:00:00+00:00",
            },
        ])

        result = await tool.execute({"query": "AI startup funding"})
        assert result.success is True
        assert result.output["results"][0]["source"] == "TechCrunch"
        assert result.output["results"][0]["url"] == "https://news.example.com/ai-startup"

    @pytest.mark.asyncio
    async def test_execute_empty_query(self, tool):
        result = await tool.execute({"query": "   "})
        assert result.success is False


class TestDdgSimplifiers:
    def test_simplify_web_result(self):
        from aria.tools.mcp.ddg_tools import _simplify_web_result
        item = {
            "title": "Test Page",
            "href": "https://example.com",
            "body": "This is a test page...",
        }
        result = _simplify_web_result(item)
        assert result["title"] == "Test Page"
        assert result["url"] == "https://example.com"
        assert result["snippet"] == "This is a test page..."

    def test_simplify_news_result(self):
        from aria.tools.mcp.ddg_tools import _simplify_news_result
        item = {
            "title": "Breaking News",
            "url": "https://news.example.com/123",
            "body": "Breaking news content...",
            "source": "Reuters",
            "date": "2026-05-03",
        }
        result = _simplify_news_result(item)
        assert result["source"] == "Reuters"
        assert result["date"] == "2026-05-03"
