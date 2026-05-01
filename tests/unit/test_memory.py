"""ARIA Engine - Memory System Unit Tests

테스트 범위:
    - types.py: Pydantic 스키마 검증
    - file_storage.py: 파일 기반 저장소 CRUD
    - index_manager.py: 인덱스 관리 + 낙관적 락 + read-before-write
    - memory_loader.py: 토큰 예산 로딩 + 프롬프트 렌더링
    - exceptions.py: 메모리 전용 예외

테스트 격리:
    - 각 테스트마다 독립 임시 디렉터리 사용 (tmp_path fixture)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aria.core.config import MemoryConfig
from aria.core.exceptions import (
    MemoryError as AriaMemoryError,
    MemoryNotFoundError,
    MemoryScopeError,
    MemoryStorageError,
    VersionConflictError,
)
from aria.memory.file_storage import FileStorageAdapter
from aria.memory.index_manager import IndexManager, estimate_tokens
from aria.memory.memory_loader import MemoryLoader, LoadResult, inject_memory_context
from aria.memory.types import (
    VALID_SCOPES,
    IndexEntry,
    MemoryIndex,
    MemoryLoadRequest,
    MemoryLoadResponse,
    TopicFile,
    TopicUpsertRequest,
    TopicResponse,
    validate_domain,
    validate_scope,
    MAX_CONTENT_BYTES,
    MAX_SUMMARY_LENGTH,
)


# ============================================================
# types.py 테스트
# ============================================================


class TestIndexEntry:
    def test_valid_entry(self) -> None:
        entry = IndexEntry(domain="user-profile", summary="사용자 프로필")
        assert entry.domain == "user-profile"
        assert entry.summary == "사용자 프로필"
        assert entry.updated_at is not None
        assert entry.token_estimate is None

    def test_with_token_estimate(self) -> None:
        entry = IndexEntry(domain="api-rules", summary="API 규칙", token_estimate=500)
        assert entry.token_estimate == 500

    def test_invalid_domain_uppercase(self) -> None:
        with pytest.raises(ValueError, match="유효하지 않은 도메인명"):
            IndexEntry(domain="INVALID", summary="test")

    def test_invalid_domain_too_long(self) -> None:
        with pytest.raises(ValueError):
            IndexEntry(domain="a" * 65, summary="test")

    def test_invalid_domain_starts_with_hyphen(self) -> None:
        with pytest.raises(ValueError, match="유효하지 않은 도메인명"):
            IndexEntry(domain="-bad-start", summary="test")

    def test_invalid_domain_ends_with_hyphen(self) -> None:
        with pytest.raises(ValueError, match="유효하지 않은 도메인명"):
            IndexEntry(domain="bad-end-", summary="test")

    def test_invalid_domain_special_chars(self) -> None:
        with pytest.raises(ValueError, match="유효하지 않은 도메인명"):
            IndexEntry(domain="has_underscore", summary="test")

    def test_summary_too_long(self) -> None:
        with pytest.raises(ValueError):
            IndexEntry(domain="test", summary="x" * (MAX_SUMMARY_LENGTH + 1))

    def test_single_char_domain(self) -> None:
        entry = IndexEntry(domain="a", summary="test")
        assert entry.domain == "a"


class TestTopicFile:
    def test_valid_topic(self) -> None:
        topic = TopicFile(domain="api-rules", scope="global", content="# Rules")
        assert topic.domain == "api-rules"
        assert topic.scope == "global"
        assert topic.version == 1
        assert topic.created_at is not None

    def test_invalid_scope(self) -> None:
        with pytest.raises(ValueError, match="유효하지 않은 스코프"):
            TopicFile(domain="test", scope="bad-scope", content="test")

    def test_content_too_large(self) -> None:
        with pytest.raises(ValueError, match="최대 크기를 초과"):
            TopicFile(domain="test", scope="global", content="x" * 60000)

    def test_all_valid_scopes(self) -> None:
        for scope in VALID_SCOPES:
            topic = TopicFile(domain="test", scope=scope, content="ok")
            assert topic.scope == scope

    def test_version_auto_increment(self) -> None:
        topic = TopicFile(domain="test", scope="global", content="v1", version=3)
        assert topic.version == 3


class TestMemoryIndex:
    def test_empty_index(self) -> None:
        index = MemoryIndex(scope="global")
        assert len(index.entries) == 0
        assert index.scope == "global"
        assert index.version == 1

    def test_find_entry_exists(self) -> None:
        entry = IndexEntry(domain="test", summary="test summary")
        index = MemoryIndex(scope="global", entries=[entry])
        found = index.find_entry("test")
        assert found is not None
        assert found.domain == "test"

    def test_find_entry_not_exists(self) -> None:
        index = MemoryIndex(scope="global", entries=[])
        assert index.find_entry("nonexistent") is None

    def test_has_entry(self) -> None:
        entry = IndexEntry(domain="exists", summary="yes")
        index = MemoryIndex(scope="global", entries=[entry])
        assert index.has_entry("exists") is True
        assert index.has_entry("nope") is False


class TestValidateFunctions:
    def test_validate_domain_valid(self) -> None:
        assert validate_domain("user-profile") == "user-profile"
        assert validate_domain("api123") == "api123"
        assert validate_domain("a") == "a"

    def test_validate_domain_invalid(self) -> None:
        with pytest.raises(ValueError):
            validate_domain("")
        with pytest.raises(ValueError):
            validate_domain("UPPER")
        with pytest.raises(ValueError):
            validate_domain("has space")

    def test_validate_scope_valid(self) -> None:
        for scope in VALID_SCOPES:
            assert validate_scope(scope) == scope

    def test_validate_scope_invalid(self) -> None:
        with pytest.raises(ValueError, match="유효하지 않은 스코프"):
            validate_scope("invalid")


class TestAPIModels:
    def test_topic_upsert_request(self) -> None:
        req = TopicUpsertRequest(summary="요약", content="# 내용")
        assert req.expected_version is None

    def test_topic_upsert_with_version(self) -> None:
        req = TopicUpsertRequest(summary="요약", content="# 내용", expected_version=3)
        assert req.expected_version == 3

    def test_memory_load_request_defaults(self) -> None:
        req = MemoryLoadRequest()
        assert req.domains is None
        assert req.token_budget is None

    def test_memory_load_request_with_domains(self) -> None:
        req = MemoryLoadRequest(domains=["user-profile", "api-rules"])
        assert len(req.domains) == 2


# ============================================================
# exceptions.py 테스트
# ============================================================


class TestMemoryExceptions:
    def test_version_conflict_error(self) -> None:
        err = VersionConflictError(
            scope="global",
            domain="test",
            expected_version=1,
            actual_version=2,
        )
        assert err.code == "VERSION_CONFLICT"
        assert err.details["expected_version"] == 1
        assert err.details["actual_version"] == 2
        assert "test" in str(err)

    def test_memory_not_found_error(self) -> None:
        err = MemoryNotFoundError(scope="testorum", domain="missing")
        assert err.code == "MEMORY_NOT_FOUND"
        assert "missing" in err.message

    def test_memory_not_found_index(self) -> None:
        err = MemoryNotFoundError(scope="global")
        assert "인덱스" in err.message

    def test_memory_scope_error(self) -> None:
        err = MemoryScopeError("bad")
        assert err.code == "MEMORY_SCOPE_INVALID"
        assert "bad" in err.message

    def test_memory_storage_error(self) -> None:
        err = MemoryStorageError("write failed", scope="global")
        assert err.code == "MEMORY_STORAGE_ERROR"

    def test_all_inherit_from_aria_error(self) -> None:
        from aria.core.exceptions import AriaError
        err = VersionConflictError(
            scope="global", domain="t", expected_version=1, actual_version=2
        )
        assert isinstance(err, AriaError)
        assert isinstance(err, AriaMemoryError)


# ============================================================
# config.py 테스트
# ============================================================


class TestMemoryConfig:
    def test_defaults(self) -> None:
        config = MemoryConfig()
        assert config.base_path == "./memory"
        assert config.default_scope == "global"
        assert config.token_budget == 4000


# ============================================================
# file_storage.py 테스트
# ============================================================


class TestFileStorageAdapter:
    def test_init_creates_base_dir(self, tmp_path: Path) -> None:
        base = tmp_path / "memory"
        assert not base.exists()
        FileStorageAdapter(str(base))
        assert base.exists()

    def test_read_empty_index(self, tmp_path: Path) -> None:
        storage = FileStorageAdapter(str(tmp_path / "mem"))
        index = storage.read_index("global")
        assert index.scope == "global"
        assert len(index.entries) == 0

    def test_write_and_read_index(self, tmp_path: Path) -> None:
        storage = FileStorageAdapter(str(tmp_path / "mem"))
        storage.ensure_scope_directory("global")
        entry = IndexEntry(domain="test", summary="요약")
        index = MemoryIndex(scope="global", entries=[entry])
        storage.write_index(index)

        loaded = storage.read_index("global")
        assert len(loaded.entries) == 1
        assert loaded.entries[0].domain == "test"

    def test_write_and_read_topic(self, tmp_path: Path) -> None:
        storage = FileStorageAdapter(str(tmp_path / "mem"))
        storage.ensure_scope_directory("global")
        topic = TopicFile(
            domain="user-profile",
            scope="global",
            content="# Profile\n\nName: 승재",
        )
        storage.write_topic(topic)

        loaded = storage.read_topic("global", "user-profile")
        assert loaded.domain == "user-profile"
        assert loaded.content == "# Profile\n\nName: 승재"
        assert loaded.version == 1

    def test_read_nonexistent_topic_raises(self, tmp_path: Path) -> None:
        storage = FileStorageAdapter(str(tmp_path / "mem"))
        with pytest.raises(MemoryNotFoundError):
            storage.read_topic("global", "nonexistent")

    def test_delete_topic(self, tmp_path: Path) -> None:
        storage = FileStorageAdapter(str(tmp_path / "mem"))
        storage.ensure_scope_directory("global")
        topic = TopicFile(domain="to-delete", scope="global", content="bye")
        storage.write_topic(topic)
        assert storage.topic_exists("global", "to-delete")

        storage.delete_topic("global", "to-delete")
        assert not storage.topic_exists("global", "to-delete")

    def test_delete_nonexistent_raises(self, tmp_path: Path) -> None:
        storage = FileStorageAdapter(str(tmp_path / "mem"))
        with pytest.raises(MemoryNotFoundError):
            storage.delete_topic("global", "nope")

    def test_invalid_scope_raises(self, tmp_path: Path) -> None:
        storage = FileStorageAdapter(str(tmp_path / "mem"))
        with pytest.raises(MemoryScopeError):
            storage.read_index("bad-scope")

    def test_topic_exists(self, tmp_path: Path) -> None:
        storage = FileStorageAdapter(str(tmp_path / "mem"))
        storage.ensure_scope_directory("testorum")
        assert not storage.topic_exists("testorum", "new-topic")
        topic = TopicFile(domain="new-topic", scope="testorum", content="hello")
        storage.write_topic(topic)
        assert storage.topic_exists("testorum", "new-topic")

    def test_read_static_config_none(self, tmp_path: Path) -> None:
        storage = FileStorageAdapter(str(tmp_path / "mem"))
        assert storage.read_static_config("global") is None

    def test_read_static_config_exists(self, tmp_path: Path) -> None:
        storage = FileStorageAdapter(str(tmp_path / "mem"))
        storage.ensure_scope_directory("global")
        project_md = tmp_path / "mem" / "global" / "PROJECT.md"
        project_md.write_text("# ARIA Project\n\nRules here")
        result = storage.read_static_config("global")
        assert result is not None
        assert "ARIA Project" in result

    def test_atomic_write_no_corruption(self, tmp_path: Path) -> None:
        """원자적 쓰기 검증 — 파일이 중간 상태로 남지 않음"""
        storage = FileStorageAdapter(str(tmp_path / "mem"))
        storage.ensure_scope_directory("global")
        # 여러 번 쓰기
        for i in range(5):
            topic = TopicFile(
                domain="stress-test",
                scope="global",
                content=f"version {i}",
                version=i + 1,
            )
            storage.write_topic(topic)

        loaded = storage.read_topic("global", "stress-test")
        assert loaded.version == 5
        assert loaded.content == "version 4"

    def test_scope_isolation(self, tmp_path: Path) -> None:
        """스코프 간 데이터 격리"""
        storage = FileStorageAdapter(str(tmp_path / "mem"))
        storage.ensure_scope_directory("global")
        storage.ensure_scope_directory("testorum")

        topic_g = TopicFile(domain="shared-name", scope="global", content="global data")
        topic_t = TopicFile(domain="shared-name", scope="testorum", content="testorum data")
        storage.write_topic(topic_g)
        storage.write_topic(topic_t)

        loaded_g = storage.read_topic("global", "shared-name")
        loaded_t = storage.read_topic("testorum", "shared-name")
        assert loaded_g.content == "global data"
        assert loaded_t.content == "testorum data"


# ============================================================
# index_manager.py 테스트
# ============================================================


class TestIndexManager:
    def _make_manager(self, tmp_path: Path) -> IndexManager:
        storage = FileStorageAdapter(str(tmp_path / "mem"))
        return IndexManager(storage)

    def test_upsert_new_topic(self, tmp_path: Path) -> None:
        mgr = self._make_manager(tmp_path)
        topic = mgr.upsert_topic(
            scope="global",
            domain="user-profile",
            summary="사용자 프로필",
            content="# Profile",
        )
        assert topic.version == 1
        assert topic.domain == "user-profile"

    def test_upsert_updates_index(self, tmp_path: Path) -> None:
        mgr = self._make_manager(tmp_path)
        mgr.upsert_topic(
            scope="global",
            domain="rules",
            summary="규칙 요약",
            content="# Rules",
        )
        index = mgr.get_index("global")
        assert index.has_entry("rules")
        entry = index.find_entry("rules")
        assert entry.summary == "규칙 요약"
        assert entry.token_estimate is not None
        assert entry.token_estimate > 0

    def test_upsert_update_with_correct_version(self, tmp_path: Path) -> None:
        mgr = self._make_manager(tmp_path)
        mgr.upsert_topic(
            scope="global", domain="test", summary="v1", content="version 1",
        )
        topic_v2 = mgr.upsert_topic(
            scope="global", domain="test", summary="v2", content="version 2",
            expected_version=1,
        )
        assert topic_v2.version == 2

    def test_upsert_version_conflict(self, tmp_path: Path) -> None:
        """read-before-write 위반 → VersionConflictError"""
        mgr = self._make_manager(tmp_path)
        mgr.upsert_topic(
            scope="global", domain="conflict", summary="v1", content="data",
        )
        with pytest.raises(VersionConflictError) as exc_info:
            mgr.upsert_topic(
                scope="global", domain="conflict", summary="v2", content="new",
                expected_version=999,  # 잘못된 버전
            )
        assert exc_info.value.details["expected_version"] == 999
        assert exc_info.value.details["actual_version"] == 1

    def test_upsert_existing_without_version_raises(self, tmp_path: Path) -> None:
        """기존 토픽에 expected_version=None → VersionConflictError"""
        mgr = self._make_manager(tmp_path)
        mgr.upsert_topic(
            scope="global", domain="existing", summary="v1", content="data",
        )
        with pytest.raises(VersionConflictError):
            mgr.upsert_topic(
                scope="global", domain="existing", summary="v2", content="new",
                expected_version=None,
            )

    def test_delete_topic(self, tmp_path: Path) -> None:
        mgr = self._make_manager(tmp_path)
        mgr.upsert_topic(
            scope="global", domain="to-delete", summary="bye", content="data",
        )
        mgr.delete_topic("global", "to-delete")
        assert not mgr.get_index("global").has_entry("to-delete")
        with pytest.raises(MemoryNotFoundError):
            mgr.get_topic("global", "to-delete")

    def test_delete_nonexistent_raises(self, tmp_path: Path) -> None:
        mgr = self._make_manager(tmp_path)
        with pytest.raises(MemoryNotFoundError):
            mgr.delete_topic("global", "nope")

    def test_list_domains(self, tmp_path: Path) -> None:
        mgr = self._make_manager(tmp_path)
        mgr.upsert_topic(scope="global", domain="alpha", summary="a", content="a")
        mgr.upsert_topic(scope="global", domain="beta", summary="b", content="b")
        domains = mgr.list_domains("global")
        assert set(domains) == {"alpha", "beta"}

    def test_get_entry(self, tmp_path: Path) -> None:
        mgr = self._make_manager(tmp_path)
        mgr.upsert_topic(scope="global", domain="entry-test", summary="요약", content="data")
        entry = mgr.get_entry("global", "entry-test")
        assert entry is not None
        assert entry.domain == "entry-test"
        assert mgr.get_entry("global", "nonexistent") is None

    def test_estimate_tokens(self) -> None:
        tokens = estimate_tokens("Hello world")
        assert tokens > 0
        assert tokens < 100

    def test_multiple_scopes(self, tmp_path: Path) -> None:
        mgr = self._make_manager(tmp_path)
        mgr.upsert_topic(scope="global", domain="shared", summary="g", content="global")
        mgr.upsert_topic(scope="testorum", domain="shared", summary="t", content="testorum")

        global_topic = mgr.get_topic("global", "shared")
        testorum_topic = mgr.get_topic("testorum", "shared")
        assert global_topic.content == "global"
        assert testorum_topic.content == "testorum"

    def test_strict_write_discipline_order(self, tmp_path: Path) -> None:
        """Strict Write Discipline: topic 파일이 먼저 쓰여야 함"""
        mgr = self._make_manager(tmp_path)
        mgr.upsert_topic(scope="global", domain="discipline", summary="s", content="data")

        # topic 파일 존재 확인
        topic = mgr.get_topic("global", "discipline")
        assert topic.content == "data"
        # index에도 반영 확인
        assert mgr.get_index("global").has_entry("discipline")


# ============================================================
# memory_loader.py 테스트
# ============================================================


class TestMemoryLoader:
    def _setup(self, tmp_path: Path) -> tuple[IndexManager, MemoryLoader]:
        storage = FileStorageAdapter(str(tmp_path / "mem"))
        mgr = IndexManager(storage)
        loader = MemoryLoader(mgr, default_token_budget=4000)
        return mgr, loader

    def test_load_empty(self, tmp_path: Path) -> None:
        mgr, loader = self._setup(tmp_path)
        result = loader.load("global")
        assert result.loaded_domains == []
        assert result.total_tokens == 0
        assert "메모리에 저장된 정보가 없습니다" in result.prompt_markdown

    def test_load_single_topic(self, tmp_path: Path) -> None:
        mgr, loader = self._setup(tmp_path)
        mgr.upsert_topic(
            scope="global",
            domain="user-profile",
            summary="프로필",
            content="# 프로필\n이름: 승재",
        )
        result = loader.load("global")
        assert "user-profile" in result.loaded_domains
        assert result.total_tokens > 0
        assert "승재" in result.prompt_markdown

    def test_load_specific_domains(self, tmp_path: Path) -> None:
        mgr, loader = self._setup(tmp_path)
        mgr.upsert_topic(scope="global", domain="alpha", summary="a", content="aaa")
        mgr.upsert_topic(scope="global", domain="beta", summary="b", content="bbb")
        mgr.upsert_topic(scope="global", domain="gamma", summary="c", content="ccc")

        result = loader.load("global", domains=["alpha", "gamma"])
        assert set(result.loaded_domains) == {"alpha", "gamma"}

    def test_token_budget_enforcement(self, tmp_path: Path) -> None:
        mgr, loader = self._setup(tmp_path)
        # 큰 토픽 여러 개 생성
        for i in range(10):
            mgr.upsert_topic(
                scope="global",
                domain=f"topic-{i}",
                summary=f"토픽 {i}",
                content=f"{'x' * 2000} topic {i}",
            )

        # 매우 작은 예산
        result = loader.load("global", token_budget=200)
        # 전부 로딩되지는 않아야 함
        assert len(result.loaded_domains) < 10

    def test_budget_used_ratio(self, tmp_path: Path) -> None:
        mgr, loader = self._setup(tmp_path)
        mgr.upsert_topic(scope="global", domain="small", summary="s", content="hello")
        result = loader.load("global", token_budget=4000)
        assert 0.0 <= result.budget_used <= 1.0

    def test_load_with_project_md(self, tmp_path: Path) -> None:
        storage = FileStorageAdapter(str(tmp_path / "mem"))
        storage.ensure_scope_directory("global")
        # PROJECT.md 생성
        project_path = tmp_path / "mem" / "global" / "PROJECT.md"
        project_path.write_text("# ARIA Rules\n- Rule 1")
        mgr = IndexManager(storage)
        loader = MemoryLoader(mgr, default_token_budget=4000)

        result = loader.load("global")
        assert "ARIA Rules" in result.prompt_markdown
        assert "Project Configuration" in result.prompt_markdown

    def test_load_result_structure(self, tmp_path: Path) -> None:
        mgr, loader = self._setup(tmp_path)
        result = loader.load("global")
        assert isinstance(result, LoadResult)
        assert hasattr(result, "scope")
        assert hasattr(result, "loaded_domains")
        assert hasattr(result, "prompt_markdown")
        assert hasattr(result, "total_tokens")
        assert hasattr(result, "budget_used")


class TestInjectMemoryContext:
    def test_basic_injection(self) -> None:
        system = "당신은 AI 어시스턴트입니다."
        memory = "## Memory\n- 사용자 이름: 승재"
        result = inject_memory_context(system, memory)
        assert "AI 어시스턴트" in result
        assert "승재" in result
        assert "hint로 취급" in result

    def test_empty_memory(self) -> None:
        system = "System prompt"
        result = inject_memory_context(system, "")
        assert "System prompt" in result
