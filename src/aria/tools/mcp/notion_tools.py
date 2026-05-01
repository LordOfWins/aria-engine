"""ARIA Engine - MCP Tool: Notion

Notion API v1 기반 도구 3종
- NotionSearchTool: 워크스페이스 검색 (pages / databases)
- NotionReadPageTool: 페이지 속성 + 블록 내용 읽기
- NotionCreatePageTool: 페이지 생성

인증: Internal Integration Token (ARIA_NOTION_TOKEN)
API 문서: https://developers.notion.com/reference

설계 원칙:
- httpx 직접 사용 (추가 의존성 없음)
- 블록 텍스트 추출은 평문 변환 (LLM이 이해하기 쉽게)
- 에러는 ToolResult로 감싸서 반환 (예외 전파 없음)
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog

from aria.core.config import NotionConfig
from aria.tools.tool_types import (
    SafetyLevelHint,
    ToolCategory,
    ToolDefinition,
    ToolExecutor,
    ToolParameter,
    ToolResult,
)

logger = structlog.get_logger()

NOTION_API_BASE = "https://api.notion.com/v1"


class NotionClient:
    """Notion API v1 HTTP 클라이언트

    Internal Integration Token 기반 인증
    httpx.AsyncClient를 재사용하여 커넥션 풀링

    Args:
        config: NotionConfig (token / api_version / timeout)
    """

    def __init__(self, config: NotionConfig) -> None:
        self._config = config
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Lazy initialization — 첫 호출 시 클라이언트 생성"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=NOTION_API_BASE,
                headers={
                    "Authorization": f"Bearer {self._config.token}",
                    "Notion-Version": self._config.api_version,
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(self._config.request_timeout),
            )
        return self._client

    async def search(
        self,
        query: str,
        filter_type: str = "",
        page_size: int = 5,
    ) -> dict[str, Any]:
        """Notion 워크스페이스 검색

        Args:
            query: 검색 쿼리
            filter_type: "page" | "database" | "" (전체)
            page_size: 결과 수 (최대 100)
        """
        body: dict[str, Any] = {
            "query": query,
            "page_size": min(page_size, 100),
        }
        if filter_type in ("page", "database"):
            body["filter"] = {"property": "object", "value": filter_type}

        client = self._get_client()
        response = await client.post("/search", json=body)
        response.raise_for_status()
        return response.json()

    async def get_page(self, page_id: str) -> dict[str, Any]:
        """페이지 속성(properties) 조회"""
        client = self._get_client()
        response = await client.get(f"/pages/{page_id}")
        response.raise_for_status()
        return response.json()

    async def get_blocks(
        self,
        block_id: str,
        page_size: int = 100,
    ) -> list[dict[str, Any]]:
        """블록 하위 자식 목록 조회 (페이지네이션 자동 처리)

        최대 3페이지(300블록)까지 가져옴 — 토큰 절약
        """
        client = self._get_client()
        all_blocks: list[dict[str, Any]] = []
        start_cursor: str | None = None
        max_pages = 3

        for _ in range(max_pages):
            params: dict[str, Any] = {"page_size": page_size}
            if start_cursor:
                params["start_cursor"] = start_cursor

            response = await client.get(
                f"/blocks/{block_id}/children",
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            all_blocks.extend(data.get("results", []))

            if not data.get("has_more"):
                break
            start_cursor = data.get("next_cursor")

        return all_blocks

    async def create_page(
        self,
        parent_id: str,
        parent_type: str,
        title: str,
        content_blocks: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """페이지 생성

        Args:
            parent_id: 부모 ID (page_id 또는 database_id)
            parent_type: "page_id" | "database_id"
            title: 페이지 제목
            content_blocks: Notion 블록 배열 (선택)
        """
        body: dict[str, Any] = {
            "parent": {parent_type: parent_id},
        }

        if parent_type == "database_id":
            # DB 하위 페이지: title 속성은 DB 스키마에 따라 다름
            # 기본적으로 "Name" 또는 "title" 속성 사용
            body["properties"] = {
                "title": {
                    "title": [{"text": {"content": title}}],
                },
            }
        else:
            # 일반 페이지 하위
            body["properties"] = {
                "title": {
                    "title": [{"text": {"content": title}}],
                },
            }

        if content_blocks:
            body["children"] = content_blocks

        client = self._get_client()
        response = await client.post("/pages", json=body)
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        """클라이언트 종료"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# === 블록 텍스트 추출 유틸리티 ===


def _extract_rich_text(rich_text_arr: list[dict[str, Any]]) -> str:
    """Notion rich_text 배열에서 평문 추출"""
    return "".join(item.get("plain_text", "") for item in rich_text_arr)


def _block_to_text(block: dict[str, Any]) -> str:
    """Notion 블록을 평문 텍스트로 변환

    지원 블록 타입: paragraph / heading_1~3 / bulleted_list_item /
    numbered_list_item / to_do / toggle / code / quote / callout / divider
    """
    block_type = block.get("type", "")
    block_data = block.get(block_type, {})

    if block_type in (
        "paragraph", "bulleted_list_item", "numbered_list_item",
        "quote", "callout", "toggle",
    ):
        text = _extract_rich_text(block_data.get("rich_text", []))
        prefix = ""
        if block_type == "bulleted_list_item":
            prefix = "• "
        elif block_type == "numbered_list_item":
            prefix = "- "
        elif block_type == "quote":
            prefix = "> "
        elif block_type == "to_do":
            checked = "✓" if block_data.get("checked") else "☐"
            prefix = f"{checked} "
        return f"{prefix}{text}" if text else ""

    if block_type in ("heading_1", "heading_2", "heading_3"):
        text = _extract_rich_text(block_data.get("rich_text", []))
        level = block_type[-1]
        return f"{'#' * int(level)} {text}" if text else ""

    if block_type == "code":
        text = _extract_rich_text(block_data.get("rich_text", []))
        lang = block_data.get("language", "")
        return f"```{lang}\n{text}\n```" if text else ""

    if block_type == "divider":
        return "---"

    if block_type == "to_do":
        text = _extract_rich_text(block_data.get("rich_text", []))
        checked = "✓" if block_data.get("checked") else "☐"
        return f"{checked} {text}" if text else ""

    if block_type == "child_page":
        return f"📄 [하위 페이지] {block_data.get('title', '')}"

    if block_type == "child_database":
        return f"🗃️ [하위 데이터베이스] {block_data.get('title', '')}"

    # 미지원 블록은 타입명만 표시
    return f"[{block_type}]" if block_type else ""


def _extract_page_title(properties: dict[str, Any]) -> str:
    """페이지 properties에서 title 추출"""
    for prop_data in properties.values():
        if prop_data.get("type") == "title":
            title_arr = prop_data.get("title", [])
            return _extract_rich_text(title_arr)
    return "(제목 없음)"


def _text_to_blocks(text: str) -> list[dict[str, Any]]:
    """평문 텍스트를 Notion 블록 배열로 변환

    간단한 변환: 줄바꿈 → paragraph 블록
    """
    blocks: list[dict[str, Any]] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        blocks.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": line}}],
            },
        })
    return blocks


# === Tool Executors ===


class NotionSearchTool(ToolExecutor):
    """Notion 워크스페이스 검색 도구

    Args:
        client: NotionClient 인스턴스
    """

    def __init__(self, client: NotionClient) -> None:
        self._client = client

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="notion_search",
            description=(
                "Notion 워크스페이스에서 페이지와 데이터베이스를 검색합니다. "
                "제목과 내용을 기반으로 검색하며, 결과에 제목/URL/최근 수정일이 포함됩니다."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="검색 키워드",
                    required=True,
                ),
                ToolParameter(
                    name="filter_type",
                    type="string",
                    description="결과 필터 (page / database / 빈값=전체)",
                    required=False,
                    enum=["page", "database", ""],
                ),
                ToolParameter(
                    name="page_size",
                    type="integer",
                    description="결과 수 (기본 5, 최대 20)",
                    required=False,
                ),
            ],
            category=ToolCategory.MCP,
            safety_hint=SafetyLevelHint.READ_ONLY,
            version="1.0.0",
        )

    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        query = parameters.get("query", "")
        if not query:
            return ToolResult(
                tool_name="notion_search",
                success=False,
                error="검색 쿼리가 비어있습니다",
            )

        filter_type = parameters.get("filter_type", "")
        page_size = min(int(parameters.get("page_size", 5)), 20)

        try:
            data = await self._client.search(
                query=query,
                filter_type=filter_type,
                page_size=page_size,
            )
        except httpx.HTTPStatusError as e:
            logger.error("notion_search_http_error", status=e.response.status_code)
            return ToolResult(
                tool_name="notion_search",
                success=False,
                error=f"Notion API 오류 ({e.response.status_code})",
            )
        except Exception as e:
            logger.error("notion_search_failed", error=str(e)[:200])
            return ToolResult(
                tool_name="notion_search",
                success=False,
                error=f"Notion 검색 실패: {str(e)[:300]}",
            )

        results = []
        for item in data.get("results", []):
            obj_type = item.get("object", "")
            title = ""
            if obj_type == "page":
                title = _extract_page_title(item.get("properties", {}))
            elif obj_type == "database":
                title = _extract_rich_text(item.get("title", []))

            results.append({
                "id": item.get("id", ""),
                "type": obj_type,
                "title": title,
                "url": item.get("url", ""),
                "last_edited": item.get("last_edited_time", ""),
            })

        logger.info("notion_search_success", query=query, result_count=len(results))

        return ToolResult(
            tool_name="notion_search",
            success=True,
            output={
                "query": query,
                "results": results,
                "total": len(results),
            },
        )


class NotionReadPageTool(ToolExecutor):
    """Notion 페이지 내용 읽기 도구

    페이지 속성(properties) + 블록 내용(blocks)을 평문 텍스트로 반환

    Args:
        client: NotionClient 인스턴스
    """

    def __init__(self, client: NotionClient) -> None:
        self._client = client

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="notion_read_page",
            description=(
                "Notion 페이지의 내용을 읽어 텍스트로 반환합니다. "
                "페이지 ID 또는 URL에서 추출한 ID로 조회합니다. "
                "블록 내용은 평문 텍스트로 변환되어 반환됩니다."
            ),
            parameters=[
                ToolParameter(
                    name="page_id",
                    type="string",
                    description="Notion 페이지 ID (UUID 형식 — 대시 포함/미포함 모두 가능)",
                    required=True,
                ),
            ],
            category=ToolCategory.MCP,
            safety_hint=SafetyLevelHint.READ_ONLY,
            version="1.0.0",
        )

    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        page_id = parameters.get("page_id", "").strip()
        if not page_id:
            return ToolResult(
                tool_name="notion_read_page",
                success=False,
                error="page_id가 비어있습니다",
            )

        # URL에서 ID 추출 지원: notion.so/workspace/Page-Title-abc123def456
        if "/" in page_id:
            page_id = page_id.rstrip("/").split("/")[-1].split("-")[-1]
            # 32자 hex면 대시 삽입
            if len(page_id) == 32:
                page_id = (
                    f"{page_id[:8]}-{page_id[8:12]}-{page_id[12:16]}-"
                    f"{page_id[16:20]}-{page_id[20:]}"
                )

        try:
            # 1. 페이지 속성 조회
            page_data = await self._client.get_page(page_id)
            title = _extract_page_title(page_data.get("properties", {}))

            # 2. 블록 내용 조회
            blocks = await self._client.get_blocks(page_id)
            content_lines = []
            for block in blocks:
                text = _block_to_text(block)
                if text:
                    content_lines.append(text)

            content = "\n".join(content_lines)

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 404:
                error_msg = "페이지를 찾을 수 없습니다 (ID를 확인하세요)"
            elif status == 403:
                error_msg = "페이지 접근 권한이 없습니다 (Integration 연결을 확인하세요)"
            else:
                error_msg = f"Notion API 오류 ({status})"
            return ToolResult(
                tool_name="notion_read_page",
                success=False,
                error=error_msg,
            )
        except Exception as e:
            logger.error("notion_read_page_failed", page_id=page_id, error=str(e)[:200])
            return ToolResult(
                tool_name="notion_read_page",
                success=False,
                error=f"페이지 읽기 실패: {str(e)[:300]}",
            )

        logger.info(
            "notion_read_page_success",
            page_id=page_id,
            title=title[:50],
            blocks_count=len(blocks),
        )

        return ToolResult(
            tool_name="notion_read_page",
            success=True,
            output={
                "page_id": page_id,
                "title": title,
                "url": page_data.get("url", ""),
                "last_edited": page_data.get("last_edited_time", ""),
                "content": content[:5000],  # 토큰 절약
                "blocks_count": len(blocks),
                "truncated": len(content) > 5000,
            },
        )


class NotionCreatePageTool(ToolExecutor):
    """Notion 페이지 생성 도구

    부모 페이지 하위에 새 페이지를 생성합니다.
    Critic 패턴에 의해 NEEDS_CONFIRMATION으로 판정될 수 있음

    Args:
        client: NotionClient 인스턴스
    """

    def __init__(self, client: NotionClient) -> None:
        self._client = client

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="notion_create_page",
            description=(
                "Notion에 새 페이지를 생성합니다. "
                "부모 페이지 ID와 제목을 지정하고, 선택적으로 본문 텍스트를 추가할 수 있습니다. "
                "데이터베이스 하위 페이지도 생성 가능합니다."
            ),
            parameters=[
                ToolParameter(
                    name="parent_id",
                    type="string",
                    description="부모 페이지 또는 데이터베이스 ID",
                    required=True,
                ),
                ToolParameter(
                    name="title",
                    type="string",
                    description="페이지 제목",
                    required=True,
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="페이지 본문 텍스트 (줄바꿈으로 구분 → 각 줄이 paragraph 블록)",
                    required=False,
                ),
                ToolParameter(
                    name="parent_type",
                    type="string",
                    description="부모 유형 (page_id / database_id)",
                    required=False,
                    enum=["page_id", "database_id"],
                    default="page_id",
                ),
            ],
            category=ToolCategory.MCP,
            safety_hint=SafetyLevelHint.WRITE,
            version="1.0.0",
        )

    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        parent_id = parameters.get("parent_id", "").strip()
        title = parameters.get("title", "").strip()
        content = parameters.get("content", "")
        parent_type = parameters.get("parent_type", "page_id")

        if not parent_id:
            return ToolResult(
                tool_name="notion_create_page",
                success=False,
                error="parent_id가 비어있습니다",
            )
        if not title:
            return ToolResult(
                tool_name="notion_create_page",
                success=False,
                error="title이 비어있습니다",
            )

        # 본문 텍스트 → 블록 변환
        blocks = _text_to_blocks(content) if content else None

        try:
            result = await self._client.create_page(
                parent_id=parent_id,
                parent_type=parent_type,
                title=title,
                content_blocks=blocks,
            )
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            error_detail = ""
            try:
                error_body = e.response.json()
                error_detail = error_body.get("message", "")
            except Exception:
                pass
            return ToolResult(
                tool_name="notion_create_page",
                success=False,
                error=f"Notion API 오류 ({status}): {error_detail}"[:400],
            )
        except Exception as e:
            logger.error("notion_create_page_failed", error=str(e)[:200])
            return ToolResult(
                tool_name="notion_create_page",
                success=False,
                error=f"페이지 생성 실패: {str(e)[:300]}",
            )

        created_id = result.get("id", "")
        created_url = result.get("url", "")

        logger.info(
            "notion_create_page_success",
            page_id=created_id,
            title=title[:50],
            parent_id=parent_id,
        )

        return ToolResult(
            tool_name="notion_create_page",
            success=True,
            output={
                "page_id": created_id,
                "title": title,
                "url": created_url,
                "parent_id": parent_id,
            },
        )
