"""ARIA Engine - MCP Tools

외부 서비스 연동 도구 (Model Context Protocol)
- notion: Notion 워크스페이스 검색 / 페이지 읽기 / 페이지 생성
- kakao_map: 카카오맵 장소 검색 / 주소→좌표 / 좌표→주소
- naver_search: 네이버 블로그/뉴스/카페/쇼핑/지식iN/지역 검색
- tmap: TMAP 대중교통 경로 검색
- ddg: DuckDuckGo 웹/뉴스 검색 (API 키 불필요)
"""

from aria.tools.mcp.notion_tools import (
    NotionSearchTool,
    NotionReadPageTool,
    NotionCreatePageTool,
)
from aria.tools.mcp.kakao_map_tools import (
    KakaoKeywordSearchTool,
    KakaoAddressSearchTool,
    KakaoCoord2AddressTool,
)
from aria.tools.mcp.naver_search_tools import (
    NaverBlogSearchTool,
    NaverNewsSearchTool,
    NaverCafeSearchTool,
    NaverShopSearchTool,
    NaverKinSearchTool,
    NaverLocalSearchTool,
)
from aria.tools.mcp.tmap_tools import (
    TmapTransitRouteTool,
)
from aria.tools.mcp.ddg_tools import (
    DdgWebSearchTool,
    DdgNewsSearchTool,
)

__all__ = [
    # Notion
    "NotionSearchTool",
    "NotionReadPageTool",
    "NotionCreatePageTool",
    # Kakao Map
    "KakaoKeywordSearchTool",
    "KakaoAddressSearchTool",
    "KakaoCoord2AddressTool",
    # Naver Search
    "NaverBlogSearchTool",
    "NaverNewsSearchTool",
    "NaverCafeSearchTool",
    "NaverShopSearchTool",
    "NaverKinSearchTool",
    "NaverLocalSearchTool",
    # TMAP
    "TmapTransitRouteTool",
    # DuckDuckGo
    "DdgWebSearchTool",
    "DdgNewsSearchTool",
]
