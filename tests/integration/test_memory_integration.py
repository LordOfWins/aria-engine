"""ARIA Engine - Memory System Integration Tests

테스트 범위:
    - /v1/memory/{scope}/index 조회
    - /v1/memory/{scope}/topics/{domain} CRUD
    - /v1/memory/{scope}/load 로딩
    - VersionConflictError (409) 응답
    - 스코프 격리
    - 전체 파이프라인 (Storage → IndexManager → MemoryLoader)

테스트 격리:
    - 각 테스트마다 독립 임시 디렉터리
    - 메모리 글로벌 인스턴스 직접 주입 (Qdrant/LLM 불필요)
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def memory_client(tmp_path, monkeypatch):
    """메모리 전용 테스트 클라이언트 — Qdrant/LLM mock으로 lifespan 우회"""
    monkeypatch.setenv("ARIA_AUTH_DISABLED", "true")
    monkeypatch.setenv("ARIA_MEMORY_BASE_PATH", str(tmp_path / "mem"))
    monkeypatch.setenv("ARIA_MEMORY_TOKEN_BUDGET", "4000")

    from aria.core.config import get_config
    get_config.cache_clear()

    from aria.memory.file_storage import FileStorageAdapter
    from aria.memory.index_manager import IndexManager
    from aria.memory.memory_loader import MemoryLoader
    import aria.api.app as app_module

    # 메모리 컴포넌트 직접 초기화
    storage = FileStorageAdapter(str(tmp_path / "mem"))
    mgr = IndexManager(storage)
    loader = MemoryLoader(mgr, default_token_budget=4000)

    # 글로벌 인스턴스 직접 설정 (lifespan 우회)
    app_module.index_manager = mgr
    app_module.memory_loader = loader
    app_module.llm_provider = MagicMock()
    app_module.vector_store = MagicMock()
    app_module.react_agent = MagicMock()
    app_module.rate_limiter = MagicMock()
    app_module.rate_limiter.is_allowed.return_value = True

    from aria.api.app import app

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    # 정리
    app_module.index_manager = None
    app_module.memory_loader = None
    app_module.llm_provider = None
    app_module.vector_store = None
    app_module.react_agent = None
    app_module.rate_limiter = None
    get_config.cache_clear()


class TestMemoryIndexEndpoint:
    def test_empty_index(self, memory_client: TestClient) -> None:
        resp = memory_client.get("/v1/memory/global/index")
        assert resp.status_code == 200
        data = resp.json()
        assert data["scope"] == "global"
        assert data["entries"] == []

    def test_index_after_upsert(self, memory_client: TestClient) -> None:
        memory_client.put(
            "/v1/memory/global/topics/test-topic",
            json={"summary": "테스트 요약", "content": "# Test"},
        )
        resp = memory_client.get("/v1/memory/global/index")
        data = resp.json()
        assert len(data["entries"]) == 1
        assert data["entries"][0]["domain"] == "test-topic"

    def test_invalid_scope_returns_400(self, memory_client: TestClient) -> None:
        resp = memory_client.get("/v1/memory/invalid-scope/index")
        assert resp.status_code == 400
        assert resp.json()["error"] == "MEMORY_SCOPE_INVALID"


class TestMemoryTopicEndpoints:
    def test_create_topic(self, memory_client: TestClient) -> None:
        resp = memory_client.put(
            "/v1/memory/global/topics/user-profile",
            json={"summary": "프로필", "content": "# Profile\nName: 승재"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["domain"] == "user-profile"
        assert data["version"] == 1
        assert data["content"] == "# Profile\nName: 승재"

    def test_get_topic(self, memory_client: TestClient) -> None:
        memory_client.put(
            "/v1/memory/global/topics/my-topic",
            json={"summary": "요약", "content": "내용"},
        )
        resp = memory_client.get("/v1/memory/global/topics/my-topic")
        assert resp.status_code == 200
        assert resp.json()["domain"] == "my-topic"

    def test_get_nonexistent_returns_404(self, memory_client: TestClient) -> None:
        resp = memory_client.get("/v1/memory/global/topics/nonexistent")
        assert resp.status_code == 404

    def test_update_with_version(self, memory_client: TestClient) -> None:
        memory_client.put(
            "/v1/memory/global/topics/versioned",
            json={"summary": "v1", "content": "version 1"},
        )
        resp = memory_client.put(
            "/v1/memory/global/topics/versioned",
            json={"summary": "v2", "content": "version 2", "expected_version": 1},
        )
        assert resp.status_code == 200
        assert resp.json()["version"] == 2

    def test_version_conflict_returns_409(self, memory_client: TestClient) -> None:
        """read-before-write 위반 → 409 Conflict"""
        memory_client.put(
            "/v1/memory/global/topics/conflict-test",
            json={"summary": "v1", "content": "data"},
        )
        resp = memory_client.put(
            "/v1/memory/global/topics/conflict-test",
            json={"summary": "v2", "content": "new data", "expected_version": 999},
        )
        assert resp.status_code == 409
        assert resp.json()["error"] == "VERSION_CONFLICT"
        assert resp.json()["details"]["expected_version"] == 999
        assert resp.json()["details"]["actual_version"] == 1

    def test_delete_topic(self, memory_client: TestClient) -> None:
        memory_client.put(
            "/v1/memory/global/topics/to-delete",
            json={"summary": "bye", "content": "data"},
        )
        resp = memory_client.delete("/v1/memory/global/topics/to-delete")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"
        resp2 = memory_client.get("/v1/memory/global/topics/to-delete")
        assert resp2.status_code == 404

    def test_delete_nonexistent_returns_404(self, memory_client: TestClient) -> None:
        resp = memory_client.delete("/v1/memory/global/topics/nope")
        assert resp.status_code == 404


class TestMemoryLoadEndpoint:
    def test_load_empty(self, memory_client: TestClient) -> None:
        resp = memory_client.post("/v1/memory/global/load", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["loaded_domains"] == []

    def test_load_with_topics(self, memory_client: TestClient) -> None:
        memory_client.put(
            "/v1/memory/global/topics/alpha",
            json={"summary": "알파", "content": "Alpha content"},
        )
        memory_client.put(
            "/v1/memory/global/topics/beta",
            json={"summary": "베타", "content": "Beta content"},
        )
        resp = memory_client.post("/v1/memory/global/load", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["loaded_domains"]) == 2
        assert data["total_tokens"] > 0
        assert "Alpha content" in data["prompt_markdown"]

    def test_load_specific_domains(self, memory_client: TestClient) -> None:
        memory_client.put(
            "/v1/memory/global/topics/only-this",
            json={"summary": "s", "content": "selected"},
        )
        memory_client.put(
            "/v1/memory/global/topics/not-this",
            json={"summary": "s", "content": "excluded"},
        )
        resp = memory_client.post(
            "/v1/memory/global/load",
            json={"domains": ["only-this"]},
        )
        data = resp.json()
        assert data["loaded_domains"] == ["only-this"]
        assert "selected" in data["prompt_markdown"]
        assert "excluded" not in data["prompt_markdown"]


class TestScopeIsolation:
    def test_different_scopes_isolated(self, memory_client: TestClient) -> None:
        memory_client.put(
            "/v1/memory/global/topics/shared-name",
            json={"summary": "글로벌", "content": "global data"},
        )
        memory_client.put(
            "/v1/memory/testorum/topics/shared-name",
            json={"summary": "테스토럼", "content": "testorum data"},
        )
        g = memory_client.get("/v1/memory/global/topics/shared-name")
        t = memory_client.get("/v1/memory/testorum/topics/shared-name")
        assert g.json()["content"] == "global data"
        assert t.json()["content"] == "testorum data"


class TestFullPipeline:
    """전체 파이프라인 테스트: Create → Update → Load → Delete"""

    def test_crud_lifecycle(self, memory_client: TestClient) -> None:
        # 1. Create
        r1 = memory_client.put(
            "/v1/memory/global/topics/lifecycle",
            json={"summary": "라이프사이클 v1", "content": "# Phase 1"},
        )
        assert r1.status_code == 200
        assert r1.json()["version"] == 1

        # 2. Read
        r2 = memory_client.get("/v1/memory/global/topics/lifecycle")
        assert r2.json()["content"] == "# Phase 1"

        # 3. Update (read-before-write)
        r3 = memory_client.put(
            "/v1/memory/global/topics/lifecycle",
            json={
                "summary": "라이프사이클 v2",
                "content": "# Phase 2",
                "expected_version": 1,
            },
        )
        assert r3.json()["version"] == 2

        # 4. Load (프롬프트 마크다운)
        r4 = memory_client.post("/v1/memory/global/load", json={})
        assert "Phase 2" in r4.json()["prompt_markdown"]

        # 5. Delete
        r5 = memory_client.delete("/v1/memory/global/topics/lifecycle")
        assert r5.json()["status"] == "deleted"

        # 6. Verify deletion
        r6 = memory_client.get("/v1/memory/global/topics/lifecycle")
        assert r6.status_code == 404

        # 7. Index empty
        r7 = memory_client.get("/v1/memory/global/index")
        assert r7.json()["entries"] == []
