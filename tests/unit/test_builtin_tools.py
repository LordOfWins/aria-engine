"""ARIA Engine - Built-in Tools 단위 테스트

테스트 범위:
- MemoryReadTool: 토픽 조회 / 인덱스 조회 / 미존재 에러
- MemoryWriteTool: 신규 생성 / 기존 업데이트 (read-before-write 자동) / 충돌 에러
- KnowledgeSearchTool: 검색 성공 / 컬렉션 미존재 / 에러 핸들링
- 모든 도구: get_definition() / ToolRegistry 등록
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aria.core.exceptions import (
    CollectionNotFoundError,
    MemoryNotFoundError,
    VectorStoreError,
    VersionConflictError,
)
from aria.memory.file_storage import FileStorageAdapter
from aria.memory.index_manager import IndexManager
from aria.memory.types import IndexEntry, MemoryIndex, TopicFile
from aria.tools.builtin.memory_read import MemoryReadTool
from aria.tools.builtin.memory_write import MemoryWriteTool
from aria.tools.builtin.knowledge_search import KnowledgeSearchTool
from aria.tools.tool_types import SafetyLevelHint, ToolCategory
from aria.tools.tool_registry import ToolRegistry


# === Fixtures ===


@pytest.fixture
def temp_memory_dir():
    """임시 메모리 디렉토리"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def index_manager(temp_memory_dir: str) -> IndexManager:
    """실제 파일 기반 IndexManager"""
    storage = FileStorageAdapter(temp_memory_dir)
    return IndexManager(storage)


@pytest.fixture
def memory_read_tool(index_manager: IndexManager) -> MemoryReadTool:
    return MemoryReadTool(index_manager)


@pytest.fixture
def memory_write_tool(index_manager: IndexManager) -> MemoryWriteTool:
    return MemoryWriteTool(index_manager)


@pytest.fixture
def mock_retriever() -> MagicMock:
    """HybridRetriever 목"""
    return MagicMock()


@pytest.fixture
def knowledge_search_tool(mock_retriever: MagicMock) -> KnowledgeSearchTool:
    return KnowledgeSearchTool(mock_retriever)


# === MemoryReadTool Tests ===


class TestMemoryReadToolDefinition:
    def test_get_definition(self, memory_read_tool: MemoryReadTool) -> None:
        defn = memory_read_tool.get_definition()
        assert defn.name == "memory_read"
        assert defn.category == ToolCategory.BUILTIN
        assert defn.safety_hint == SafetyLevelHint.READ_ONLY
        assert len(defn.parameters) == 2
        # scope는 필수 / domain은 선택
        param_map = {p.name: p for p in defn.parameters}
        assert param_map["scope"].required is True
        assert param_map["domain"].required is False

    def test_to_llm_tool_format(self, memory_read_tool: MemoryReadTool) -> None:
        llm_tool = memory_read_tool.get_definition().to_llm_tool()
        assert llm_tool["type"] == "function"
        assert llm_tool["function"]["name"] == "memory_read"


class TestMemoryReadToolExecution:
    @pytest.mark.asyncio
    async def test_read_empty_index(self, memory_read_tool: MemoryReadTool) -> None:
        """빈 인덱스 조회"""
        result = await memory_read_tool.execute({"scope": "global"})
        assert result.success is True
        assert result.output["total_topics"] == 0
        assert result.output["entries"] == []

    @pytest.mark.asyncio
    async def test_read_index_with_topics(
        self,
        memory_read_tool: MemoryReadTool,
        index_manager: IndexManager,
    ) -> None:
        """토픽이 있는 인덱스 조회"""
        index_manager.upsert_topic(
            scope="global",
            domain="test-topic",
            summary="테스트 토픽 요약",
            content="# 테스트\n\n내용입니다",
        )
        result = await memory_read_tool.execute({"scope": "global"})
        assert result.success is True
        assert result.output["total_topics"] == 1
        assert result.output["entries"][0]["domain"] == "test-topic"
        assert result.output["entries"][0]["summary"] == "테스트 토픽 요약"

    @pytest.mark.asyncio
    async def test_read_topic_success(
        self,
        memory_read_tool: MemoryReadTool,
        index_manager: IndexManager,
    ) -> None:
        """토픽 본문 조회 성공"""
        index_manager.upsert_topic(
            scope="global",
            domain="user-profile",
            summary="사용자 프로필",
            content="# 프로필\n\n- 이름: 승재",
        )

        result = await memory_read_tool.execute({
            "scope": "global",
            "domain": "user-profile",
        })
        assert result.success is True
        assert result.output["domain"] == "user-profile"
        assert "승재" in result.output["content"]
        assert result.output["version"] == 1
        assert result.output["summary"] == "사용자 프로필"

    @pytest.mark.asyncio
    async def test_read_nonexistent_topic(self, memory_read_tool: MemoryReadTool) -> None:
        """미존재 토픽 조회 → 실패"""
        result = await memory_read_tool.execute({
            "scope": "global",
            "domain": "nonexistent",
        })
        assert result.success is False
        assert "찾을 수 없습니다" in result.error

    @pytest.mark.asyncio
    async def test_read_empty_domain_returns_index(
        self,
        memory_read_tool: MemoryReadTool,
    ) -> None:
        """domain="" → 인덱스 조회로 동작"""
        result = await memory_read_tool.execute({
            "scope": "global",
            "domain": "",
        })
        assert result.success is True
        assert "entries" in result.output


# === MemoryWriteTool Tests ===


class TestMemoryWriteToolDefinition:
    def test_get_definition(self, memory_write_tool: MemoryWriteTool) -> None:
        defn = memory_write_tool.get_definition()
        assert defn.name == "memory_write"
        assert defn.category == ToolCategory.BUILTIN
        assert defn.safety_hint == SafetyLevelHint.WRITE
        # 4개 파라미터 모두 필수
        assert all(p.required for p in defn.parameters)


class TestMemoryWriteToolExecution:
    @pytest.mark.asyncio
    async def test_create_new_topic(self, memory_write_tool: MemoryWriteTool) -> None:
        """신규 토픽 생성"""
        result = await memory_write_tool.execute({
            "scope": "global",
            "domain": "new-topic",
            "summary": "새 토픽 요약",
            "content": "# 새 토픽\n\n내용입니다",
        })
        assert result.success is True
        assert result.output["action"] == "생성"
        assert result.output["version"] == 1
        assert result.output["domain"] == "new-topic"

    @pytest.mark.asyncio
    async def test_update_existing_topic(
        self,
        memory_write_tool: MemoryWriteTool,
        index_manager: IndexManager,
    ) -> None:
        """기존 토픽 업데이트 (read-before-write 자동)"""
        # 먼저 생성
        index_manager.upsert_topic(
            scope="global",
            domain="existing-topic",
            summary="기존 요약",
            content="# 기존 내용",
        )

        # 도구로 업데이트 → read-before-write 자동 처리
        result = await memory_write_tool.execute({
            "scope": "global",
            "domain": "existing-topic",
            "summary": "업데이트된 요약",
            "content": "# 업데이트된 내용",
        })
        assert result.success is True
        assert result.output["action"] == "업데이트"
        assert result.output["version"] == 2

    @pytest.mark.asyncio
    async def test_multiple_updates(
        self,
        memory_write_tool: MemoryWriteTool,
    ) -> None:
        """연속 업데이트 (매번 read-before-write 자동)"""
        # 생성
        r1 = await memory_write_tool.execute({
            "scope": "global",
            "domain": "counter",
            "summary": "카운터",
            "content": "count: 1",
        })
        assert r1.output["version"] == 1

        # 업데이트 1
        r2 = await memory_write_tool.execute({
            "scope": "global",
            "domain": "counter",
            "summary": "카운터",
            "content": "count: 2",
        })
        assert r2.output["version"] == 2

        # 업데이트 2
        r3 = await memory_write_tool.execute({
            "scope": "global",
            "domain": "counter",
            "summary": "카운터",
            "content": "count: 3",
        })
        assert r3.output["version"] == 3

    @pytest.mark.asyncio
    async def test_write_to_scoped_memory(
        self,
        memory_write_tool: MemoryWriteTool,
    ) -> None:
        """스코프별 메모리 쓰기"""
        result = await memory_write_tool.execute({
            "scope": "testorum",
            "domain": "test-results",
            "summary": "테스트 결과 요약",
            "content": "# Testorum 테스트 결과",
        })
        assert result.success is True
        assert result.output["scope"] == "testorum"


# === KnowledgeSearchTool Tests ===


class TestKnowledgeSearchToolDefinition:
    def test_get_definition(self, knowledge_search_tool: KnowledgeSearchTool) -> None:
        defn = knowledge_search_tool.get_definition()
        assert defn.name == "knowledge_search"
        assert defn.category == ToolCategory.BUILTIN
        assert defn.safety_hint == SafetyLevelHint.READ_ONLY
        param_map = {p.name: p for p in defn.parameters}
        assert param_map["query"].required is True
        assert param_map["collection"].required is False
        assert param_map["top_k"].required is False


class TestKnowledgeSearchToolExecution:
    @pytest.mark.asyncio
    async def test_search_success(
        self,
        knowledge_search_tool: KnowledgeSearchTool,
        mock_retriever: MagicMock,
    ) -> None:
        """검색 성공"""
        mock_retriever.search.return_value = [
            {"text": "회피형 애착은...", "score": 0.85, "metadata": {"source": "psychology"}},
            {"text": "불안형 애착과...", "score": 0.72, "metadata": {"source": "psychology"}},
        ]

        result = await knowledge_search_tool.execute({
            "query": "회피형 애착 패턴",
            "collection": "psychology_kb",
        })
        assert result.success is True
        assert result.output["total_found"] == 2
        assert result.output["results"][0]["score"] == 0.85
        assert "회피형 애착" in result.output["results"][0]["text"]

        mock_retriever.search.assert_called_once_with(
            collection_name="psychology_kb",
            query="회피형 애착 패턴",
            top_k=5,
        )

    @pytest.mark.asyncio
    async def test_search_default_collection(
        self,
        knowledge_search_tool: KnowledgeSearchTool,
        mock_retriever: MagicMock,
    ) -> None:
        """collection 미지정 → default"""
        mock_retriever.search.return_value = []

        await knowledge_search_tool.execute({"query": "test"})

        mock_retriever.search.assert_called_once_with(
            collection_name="default",
            query="test",
            top_k=5,
        )

    @pytest.mark.asyncio
    async def test_search_top_k_capped(
        self,
        knowledge_search_tool: KnowledgeSearchTool,
        mock_retriever: MagicMock,
    ) -> None:
        """top_k 상한 10"""
        mock_retriever.search.return_value = []

        await knowledge_search_tool.execute({
            "query": "test",
            "top_k": 50,  # 50 → 10으로 제한
        })

        mock_retriever.search.assert_called_once_with(
            collection_name="default",
            query="test",
            top_k=10,
        )

    @pytest.mark.asyncio
    async def test_search_long_text_truncated(
        self,
        knowledge_search_tool: KnowledgeSearchTool,
        mock_retriever: MagicMock,
    ) -> None:
        """긴 텍스트는 잘림"""
        long_text = "x" * 5000
        mock_retriever.search.return_value = [
            {"text": long_text, "score": 0.9, "metadata": {}},
        ]

        result = await knowledge_search_tool.execute({"query": "test"})
        assert result.success is True
        assert len(result.output["results"][0]["text"]) < 5000
        assert result.output["results"][0]["text"].endswith("...")

    @pytest.mark.asyncio
    async def test_search_collection_not_found(
        self,
        knowledge_search_tool: KnowledgeSearchTool,
        mock_retriever: MagicMock,
    ) -> None:
        """컬렉션 미존재 에러"""
        mock_retriever.search.side_effect = CollectionNotFoundError("missing_kb")

        result = await knowledge_search_tool.execute({
            "query": "test",
            "collection": "missing_kb",
        })
        assert result.success is False
        assert "missing_kb" in result.error

    @pytest.mark.asyncio
    async def test_search_vector_store_error(
        self,
        knowledge_search_tool: KnowledgeSearchTool,
        mock_retriever: MagicMock,
    ) -> None:
        """벡터 저장소 에러"""
        mock_retriever.search.side_effect = VectorStoreError("connection failed")

        result = await knowledge_search_tool.execute({"query": "test"})
        assert result.success is False
        assert "벡터 검색 오류" in result.error

    @pytest.mark.asyncio
    async def test_search_unexpected_error(
        self,
        knowledge_search_tool: KnowledgeSearchTool,
        mock_retriever: MagicMock,
    ) -> None:
        """예상 못한 에러 → 실패 ToolResult"""
        mock_retriever.search.side_effect = RuntimeError("unexpected")

        result = await knowledge_search_tool.execute({"query": "test"})
        assert result.success is False
        assert "검색 실패" in result.error

    @pytest.mark.asyncio
    async def test_search_empty_results(
        self,
        knowledge_search_tool: KnowledgeSearchTool,
        mock_retriever: MagicMock,
    ) -> None:
        """검색 결과 없음"""
        mock_retriever.search.return_value = []

        result = await knowledge_search_tool.execute({"query": "없는 내용"})
        assert result.success is True
        assert result.output["total_found"] == 0


# === Registry Integration Tests ===


class TestBuiltinToolsRegistration:
    def test_register_all_builtin_tools(self, index_manager: IndexManager) -> None:
        """3개 Built-in Tools 모두 등록 성공"""
        registry = ToolRegistry()
        mock_retriever = MagicMock()

        registry.register_executor(MemoryReadTool(index_manager))
        registry.register_executor(MemoryWriteTool(index_manager))
        registry.register_executor(KnowledgeSearchTool(mock_retriever))

        assert registry.tool_count == 3
        assert registry.has_tool("memory_read")
        assert registry.has_tool("memory_write")
        assert registry.has_tool("knowledge_search")

    def test_to_llm_tools_returns_all(self, index_manager: IndexManager) -> None:
        """LLM 포맷 변환 시 3개 도구 모두 포함"""
        registry = ToolRegistry()
        mock_retriever = MagicMock()

        registry.register_executor(MemoryReadTool(index_manager))
        registry.register_executor(MemoryWriteTool(index_manager))
        registry.register_executor(KnowledgeSearchTool(mock_retriever))

        llm_tools = registry.to_llm_tools()
        assert len(llm_tools) == 3
        names = {t["function"]["name"] for t in llm_tools}
        assert names == {"memory_read", "memory_write", "knowledge_search"}

    @pytest.mark.asyncio
    async def test_execute_through_registry(self, index_manager: IndexManager) -> None:
        """Registry를 통한 도구 실행 (Critic 없이)"""
        registry = ToolRegistry()
        registry.register_executor(MemoryWriteTool(index_manager))
        registry.register_executor(MemoryReadTool(index_manager))

        # 쓰기
        write_result = await registry.execute(
            "memory_write",
            {
                "scope": "global",
                "domain": "registry-test",
                "summary": "레지스트리 통합 테스트",
                "content": "# Registry Test",
            },
        )
        assert write_result.success is True

        # 읽기
        read_result = await registry.execute(
            "memory_read",
            {"scope": "global", "domain": "registry-test"},
        )
        assert read_result.success is True
        assert "Registry Test" in read_result.output["content"]
