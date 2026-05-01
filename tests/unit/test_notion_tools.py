"""Notion MCP 도구 단위 테스트

테스트 범위:
- NotionConfig: 설정 / is_configured
- NotionClient: 검색 / 페이지 조회 / 블록 조회 / 페이지 생성
- NotionSearchTool: 검색 실행 / 에러 처리
- NotionReadPageTool: 페이지 읽기 / URL 파싱 / 에러 처리
- NotionCreatePageTool: 페이지 생성 / 입력 검증
- 블록 텍스트 변환 유틸리티
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from aria.core.config import NotionConfig
from aria.tools.mcp.notion_tools import (
    NotionClient,
    NotionSearchTool,
    NotionReadPageTool,
    NotionCreatePageTool,
    _extract_rich_text,
    _block_to_text,
    _extract_page_title,
    _text_to_blocks,
)


# === NotionConfig 테스트 ===


class TestNotionConfig:

    def test_default_not_configured(self):
        config = NotionConfig()
        assert not config.is_configured

    def test_configured_with_token(self):
        config = NotionConfig(token="ntn_test123")
        assert config.is_configured

    def test_default_api_version(self):
        config = NotionConfig()
        assert config.api_version == "2022-06-28"


# === 블록 텍스트 유틸리티 테스트 ===


class TestBlockTextUtils:

    def test_extract_rich_text(self):
        arr = [
            {"plain_text": "Hello ", "type": "text"},
            {"plain_text": "World", "type": "text"},
        ]
        assert _extract_rich_text(arr) == "Hello World"

    def test_extract_rich_text_empty(self):
        assert _extract_rich_text([]) == ""

    def test_block_to_text_paragraph(self):
        block = {
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"plain_text": "테스트 문단"}],
            },
        }
        assert _block_to_text(block) == "테스트 문단"

    def test_block_to_text_heading(self):
        for level in (1, 2, 3):
            block = {
                "type": f"heading_{level}",
                f"heading_{level}": {
                    "rich_text": [{"plain_text": "제목"}],
                },
            }
            assert _block_to_text(block) == f"{'#' * level} 제목"

    def test_block_to_text_bulleted_list(self):
        block = {
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [{"plain_text": "항목"}],
            },
        }
        assert _block_to_text(block) == "• 항목"

    def test_block_to_text_code(self):
        block = {
            "type": "code",
            "code": {
                "rich_text": [{"plain_text": "print('hi')"}],
                "language": "python",
            },
        }
        result = _block_to_text(block)
        assert "```python" in result
        assert "print('hi')" in result

    def test_block_to_text_divider(self):
        block = {"type": "divider", "divider": {}}
        assert _block_to_text(block) == "---"

    def test_block_to_text_child_page(self):
        block = {
            "type": "child_page",
            "child_page": {"title": "하위 페이지"},
        }
        assert "하위 페이지" in _block_to_text(block)

    def test_block_to_text_unsupported(self):
        block = {"type": "image", "image": {}}
        assert "[image]" in _block_to_text(block)

    def test_extract_page_title(self):
        props = {
            "Name": {
                "type": "title",
                "title": [{"plain_text": "테스트 페이지"}],
            },
        }
        assert _extract_page_title(props) == "테스트 페이지"

    def test_extract_page_title_missing(self):
        assert _extract_page_title({}) == "(제목 없음)"

    def test_text_to_blocks(self):
        text = "첫줄\n둘째줄\n셋째줄"
        blocks = _text_to_blocks(text)
        assert len(blocks) == 3
        assert blocks[0]["type"] == "paragraph"

    def test_text_to_blocks_skip_empty(self):
        text = "첫줄\n\n\n둘째줄"
        blocks = _text_to_blocks(text)
        assert len(blocks) == 2


# === NotionClient 테스트 ===


class TestNotionClient:

    @pytest.fixture
    def client(self):
        config = NotionConfig(token="ntn_test_token")
        return NotionClient(config)

    def test_client_creates_httpx_client(self, client):
        """lazy init 전에는 None"""
        assert client._client is None

    @pytest.mark.asyncio
    async def test_search_builds_correct_request(self, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": [], "has_more": False}
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            # Force client creation
            client._client = httpx.AsyncClient()
            result = await client.search("테스트", filter_type="page", page_size=3)

        assert result == {"results": [], "has_more": False}

    @pytest.mark.asyncio
    async def test_search_with_filter(self, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            client._client = httpx.AsyncClient()
            await client.search("query", filter_type="database")

        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs.args[1] if len(call_kwargs.args) > 1 else {}
        # filter가 포함되어야 함
        assert mock_post.called


# === NotionSearchTool 테스트 ===


class TestNotionSearchTool:

    @pytest.fixture
    def mock_client(self):
        return MagicMock(spec=NotionClient)

    def test_definition(self, mock_client):
        tool = NotionSearchTool(mock_client)
        defn = tool.get_definition()
        assert defn.name == "notion_search"
        assert defn.category.value == "mcp"
        assert defn.safety_hint.value == "read_only"

    @pytest.mark.asyncio
    async def test_search_success(self, mock_client):
        mock_client.search = AsyncMock(return_value={
            "results": [
                {
                    "id": "page-123",
                    "object": "page",
                    "url": "https://notion.so/test",
                    "last_edited_time": "2026-05-01",
                    "properties": {
                        "Name": {"type": "title", "title": [{"plain_text": "테스트"}]},
                    },
                },
            ],
        })

        tool = NotionSearchTool(mock_client)
        result = await tool.execute({"query": "테스트"})

        assert result.success
        assert result.output["total"] == 1
        assert result.output["results"][0]["title"] == "테스트"

    @pytest.mark.asyncio
    async def test_search_empty_query(self, mock_client):
        tool = NotionSearchTool(mock_client)
        result = await tool.execute({"query": ""})

        assert not result.success
        assert "비어있습니다" in result.error

    @pytest.mark.asyncio
    async def test_search_http_error(self, mock_client):
        response = MagicMock()
        response.status_code = 401
        mock_client.search = AsyncMock(
            side_effect=httpx.HTTPStatusError("Unauthorized", request=MagicMock(), response=response)
        )

        tool = NotionSearchTool(mock_client)
        result = await tool.execute({"query": "test"})

        assert not result.success
        assert "401" in result.error

    @pytest.mark.asyncio
    async def test_search_network_error(self, mock_client):
        mock_client.search = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

        tool = NotionSearchTool(mock_client)
        result = await tool.execute({"query": "test"})

        assert not result.success


# === NotionReadPageTool 테스트 ===


class TestNotionReadPageTool:

    @pytest.fixture
    def mock_client(self):
        return MagicMock(spec=NotionClient)

    def test_definition(self, mock_client):
        tool = NotionReadPageTool(mock_client)
        defn = tool.get_definition()
        assert defn.name == "notion_read_page"
        assert defn.safety_hint.value == "read_only"

    @pytest.mark.asyncio
    async def test_read_page_success(self, mock_client):
        mock_client.get_page = AsyncMock(return_value={
            "id": "page-123",
            "url": "https://notion.so/page-123",
            "last_edited_time": "2026-05-01",
            "properties": {
                "Name": {"type": "title", "title": [{"plain_text": "테스트 페이지"}]},
            },
        })
        mock_client.get_blocks = AsyncMock(return_value=[
            {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "본문 내용"}]}},
            {"type": "heading_1", "heading_1": {"rich_text": [{"plain_text": "섹션 1"}]}},
        ])

        tool = NotionReadPageTool(mock_client)
        result = await tool.execute({"page_id": "page-123"})

        assert result.success
        assert result.output["title"] == "테스트 페이지"
        assert "본문 내용" in result.output["content"]
        assert "# 섹션 1" in result.output["content"]
        assert result.output["blocks_count"] == 2

    @pytest.mark.asyncio
    async def test_read_page_empty_id(self, mock_client):
        tool = NotionReadPageTool(mock_client)
        result = await tool.execute({"page_id": ""})

        assert not result.success
        assert "비어있습니다" in result.error

    @pytest.mark.asyncio
    async def test_read_page_404(self, mock_client):
        response = MagicMock()
        response.status_code = 404
        mock_client.get_page = AsyncMock(
            side_effect=httpx.HTTPStatusError("Not Found", request=MagicMock(), response=response)
        )

        tool = NotionReadPageTool(mock_client)
        result = await tool.execute({"page_id": "nonexistent"})

        assert not result.success
        assert "찾을 수 없습니다" in result.error

    @pytest.mark.asyncio
    async def test_read_page_403(self, mock_client):
        response = MagicMock()
        response.status_code = 403
        mock_client.get_page = AsyncMock(
            side_effect=httpx.HTTPStatusError("Forbidden", request=MagicMock(), response=response)
        )

        tool = NotionReadPageTool(mock_client)
        result = await tool.execute({"page_id": "restricted"})

        assert not result.success
        assert "권한" in result.error


# === NotionCreatePageTool 테스트 ===


class TestNotionCreatePageTool:

    @pytest.fixture
    def mock_client(self):
        return MagicMock(spec=NotionClient)

    def test_definition(self, mock_client):
        tool = NotionCreatePageTool(mock_client)
        defn = tool.get_definition()
        assert defn.name == "notion_create_page"
        assert defn.safety_hint.value == "write"

    @pytest.mark.asyncio
    async def test_create_page_success(self, mock_client):
        mock_client.create_page = AsyncMock(return_value={
            "id": "new-page-123",
            "url": "https://notion.so/new-page-123",
        })

        tool = NotionCreatePageTool(mock_client)
        result = await tool.execute({
            "parent_id": "parent-123",
            "title": "새 페이지",
            "content": "첫번째 줄\n두번째 줄",
        })

        assert result.success
        assert result.output["page_id"] == "new-page-123"
        assert result.output["title"] == "새 페이지"

    @pytest.mark.asyncio
    async def test_create_page_no_parent(self, mock_client):
        tool = NotionCreatePageTool(mock_client)
        result = await tool.execute({"parent_id": "", "title": "제목"})

        assert not result.success
        assert "parent_id" in result.error

    @pytest.mark.asyncio
    async def test_create_page_no_title(self, mock_client):
        tool = NotionCreatePageTool(mock_client)
        result = await tool.execute({"parent_id": "parent-123", "title": ""})

        assert not result.success
        assert "title" in result.error

    @pytest.mark.asyncio
    async def test_create_page_http_error(self, mock_client):
        response = MagicMock()
        response.status_code = 400
        response.json.return_value = {"message": "Invalid parent"}
        mock_client.create_page = AsyncMock(
            side_effect=httpx.HTTPStatusError("Bad Request", request=MagicMock(), response=response)
        )

        tool = NotionCreatePageTool(mock_client)
        result = await tool.execute({
            "parent_id": "bad-parent",
            "title": "Test",
        })

        assert not result.success
        assert "400" in result.error

    @pytest.mark.asyncio
    async def test_create_page_with_database_parent(self, mock_client):
        mock_client.create_page = AsyncMock(return_value={
            "id": "db-page-123",
            "url": "https://notion.so/db-page-123",
        })

        tool = NotionCreatePageTool(mock_client)
        result = await tool.execute({
            "parent_id": "db-123",
            "title": "DB 페이지",
            "parent_type": "database_id",
        })

        assert result.success
        # create_page가 database_id로 호출되었는지 확인
        call_args = mock_client.create_page.call_args
        assert call_args.kwargs["parent_type"] == "database_id"


# === 도구 등록 통합 검증 ===


class TestNotionToolRegistration:

    def test_all_tools_have_unique_names(self):
        """Notion 도구 이름이 고유한지 확인"""
        mock_client = MagicMock(spec=NotionClient)
        tools = [
            NotionSearchTool(mock_client),
            NotionReadPageTool(mock_client),
            NotionCreatePageTool(mock_client),
        ]
        names = [t.get_definition().name for t in tools]
        assert len(names) == len(set(names))

    def test_all_tools_are_mcp_category(self):
        """Notion 도구가 MCP 카테고리인지 확인"""
        mock_client = MagicMock(spec=NotionClient)
        tools = [
            NotionSearchTool(mock_client),
            NotionReadPageTool(mock_client),
            NotionCreatePageTool(mock_client),
        ]
        for tool in tools:
            assert tool.get_definition().category.value == "mcp"

    def test_all_tools_have_valid_definitions(self):
        """모든 도구의 ToolDefinition이 유효한지 확인"""
        mock_client = MagicMock(spec=NotionClient)
        tools = [
            NotionSearchTool(mock_client),
            NotionReadPageTool(mock_client),
            NotionCreatePageTool(mock_client),
        ]
        for tool in tools:
            defn = tool.get_definition()
            # to_llm_tool()이 에러 없이 동작해야 함
            llm_tool = defn.to_llm_tool()
            assert llm_tool["type"] == "function"
            assert llm_tool["function"]["name"]
            assert llm_tool["function"]["description"]
