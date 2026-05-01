"""ARIA Engine - Event Collection Tests

이벤트 수집 시스템 단위 테스트
- EventSeverity / EventInput / Event 스키마
- EventStore 인입/조회/배치/cleanup
- EventQuery 필터링
- EventConfig 설정
- API 엔드포인트 통합
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from aria.events.types import (
    Event,
    EventInput,
    EventIngestRequest,
    EventIngestResponse,
    EventQuery,
    EventSeverity,
    VALID_SOURCES,
)
from aria.events.event_store import EventStore


# === Fixtures ===


@pytest.fixture
def tmp_events_dir(tmp_path):
    """임시 이벤트 디렉터리"""
    events_dir = tmp_path / "events"
    events_dir.mkdir()
    return str(events_dir)


@pytest.fixture
def store(tmp_events_dir):
    """EventStore 인스턴스"""
    return EventStore(
        base_path=tmp_events_dir,
        max_buffer_size=100,
        retention_days=30,
    )


@pytest.fixture
def sample_input():
    """기본 이벤트 입력"""
    return EventInput(
        event_type="test_completed",
        source="testorum",
        severity=EventSeverity.INFO,
        data={"test_id": "abc123", "score": 85},
    )


@pytest.fixture
def sample_inputs():
    """배치 이벤트 입력"""
    return [
        EventInput(
            event_type="test_completed",
            source="testorum",
            data={"test_id": f"test_{i}", "score": 50 + i},
        )
        for i in range(5)
    ]


# === EventSeverity Tests ===


class TestEventSeverity:
    """EventSeverity enum 테스트"""

    def test_info_value(self):
        assert EventSeverity.INFO == "info"

    def test_warning_value(self):
        assert EventSeverity.WARNING == "warning"

    def test_error_value(self):
        assert EventSeverity.ERROR == "error"

    def test_from_string(self):
        assert EventSeverity("info") == EventSeverity.INFO

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            EventSeverity("critical")


# === EventInput Validation Tests ===


class TestEventInputValidation:
    """EventInput 입력 검증 테스트"""

    def test_valid_input(self, sample_input):
        assert sample_input.event_type == "test_completed"
        assert sample_input.source == "testorum"
        assert sample_input.severity == EventSeverity.INFO

    def test_source_case_insensitive(self):
        inp = EventInput(event_type="test", source="Testorum")
        assert inp.source == "testorum"

    def test_all_valid_sources(self):
        for source in VALID_SOURCES:
            inp = EventInput(event_type="test", source=source)
            assert inp.source == source

    def test_invalid_source_rejects(self):
        with pytest.raises(ValueError, match="유효하지 않은 소스"):
            EventInput(event_type="test", source="unknown_app")

    def test_empty_source_rejects(self):
        with pytest.raises(ValueError):
            EventInput(event_type="test", source="")

    def test_event_type_normalization(self):
        inp = EventInput(event_type="  Test_Completed  ", source="testorum")
        assert inp.event_type == "test_completed"

    def test_event_type_with_hyphen(self):
        inp = EventInput(event_type="user-signup", source="testorum")
        assert inp.event_type == "user-signup"

    def test_event_type_invalid_chars(self):
        with pytest.raises(ValueError, match="유효하지 않은 event_type"):
            EventInput(event_type="test completed", source="testorum")

    def test_event_type_starts_with_underscore(self):
        with pytest.raises(ValueError, match="유효하지 않은 event_type"):
            EventInput(event_type="_test", source="testorum")

    def test_data_within_limit(self):
        data = {"key": "x" * 5000}
        inp = EventInput(event_type="test", source="testorum", data=data)
        assert len(json.dumps(inp.data)) > 0

    def test_data_exceeds_limit(self):
        data = {"key": "x" * 15000}
        with pytest.raises(ValueError, match="data 크기 초과"):
            EventInput(event_type="test", source="testorum", data=data)

    def test_valid_timestamp(self):
        ts = "2026-05-02T10:30:00+09:00"
        inp = EventInput(event_type="test", source="testorum", timestamp=ts)
        assert inp.timestamp == ts

    def test_utc_timestamp(self):
        ts = "2026-05-02T01:30:00Z"
        inp = EventInput(event_type="test", source="testorum", timestamp=ts)
        assert inp.timestamp == ts

    def test_invalid_timestamp(self):
        with pytest.raises(ValueError, match="유효하지 않은 타임스탬프"):
            EventInput(event_type="test", source="testorum", timestamp="not-a-date")

    def test_none_timestamp_accepted(self):
        inp = EventInput(event_type="test", source="testorum", timestamp=None)
        assert inp.timestamp is None

    def test_to_event_generates_id(self, sample_input):
        event = sample_input.to_event()
        assert event.event_id  # UUID4
        assert len(event.event_id) == 36  # UUID format

    def test_to_event_generates_timestamp(self, sample_input):
        event = sample_input.to_event()
        assert event.timestamp  # ISO format
        # 파싱 가능한지 확인
        datetime.fromisoformat(event.timestamp.replace("Z", "+00:00"))

    def test_to_event_preserves_fields(self, sample_input):
        event = sample_input.to_event()
        assert event.event_type == "test_completed"
        assert event.source == "testorum"
        assert event.data == {"test_id": "abc123", "score": 85}

    def test_to_event_uses_provided_timestamp(self):
        ts = "2026-05-01T00:00:00+00:00"
        inp = EventInput(event_type="test", source="testorum", timestamp=ts)
        event = inp.to_event()
        assert event.timestamp == ts


# === Event Model Tests ===


class TestEvent:
    """Event 모델 테스트"""

    def test_to_jsonl(self, sample_input):
        event = sample_input.to_event()
        jsonl = event.to_jsonl()
        parsed = json.loads(jsonl)
        assert parsed["event_type"] == "test_completed"
        assert parsed["source"] == "testorum"

    def test_roundtrip_jsonl(self, sample_input):
        event = sample_input.to_event()
        jsonl = event.to_jsonl()
        restored = Event.model_validate_json(jsonl)
        assert restored.event_id == event.event_id
        assert restored.data == event.data


# === EventIngestRequest Tests ===


class TestEventIngestRequest:
    """배치 인입 요청 테스트"""

    def test_valid_batch(self, sample_inputs):
        req = EventIngestRequest(events=sample_inputs)
        assert len(req.events) == 5

    def test_empty_batch_rejects(self):
        with pytest.raises(ValueError):
            EventIngestRequest(events=[])

    def test_max_batch_size(self):
        events = [
            EventInput(event_type="test", source="testorum")
            for _ in range(101)
        ]
        with pytest.raises(ValueError):
            EventIngestRequest(events=events)


# === EventQuery Tests ===


class TestEventQuery:
    """이벤트 조회 필터 테스트"""

    def test_default_query(self):
        q = EventQuery()
        assert q.limit == 50
        assert q.source is None

    def test_source_filter(self):
        q = EventQuery(source="testorum")
        assert q.source == "testorum"

    def test_invalid_source_rejects(self):
        with pytest.raises(ValueError):
            EventQuery(source="invalid_source")

    def test_limit_range(self):
        q = EventQuery(limit=500)
        assert q.limit == 500

    def test_limit_exceeds_max(self):
        with pytest.raises(ValueError):
            EventQuery(limit=501)


# === EventStore Tests ===


class TestEventStoreInit:
    """EventStore 초기화 테스트"""

    def test_creates_base_directory(self, tmp_path):
        base = str(tmp_path / "new_events")
        store = EventStore(base_path=base)
        assert Path(base).exists()

    def test_initial_stats(self, store):
        stats = store.get_stats()
        assert stats["total_ingested"] == 0
        assert stats["buffer_size"] == 0
        assert stats["retention_days"] == 30


class TestEventStoreIngest:
    """EventStore 인입 테스트"""

    def test_single_ingest(self, store, sample_input):
        event = store.ingest(sample_input)
        assert event.event_id
        assert event.event_type == "test_completed"
        assert store.total_ingested == 1
        assert store.buffer_size == 1

    def test_file_created(self, store, sample_input):
        event = store.ingest(sample_input)
        date_str = event.timestamp[:10]
        file_path = store.base_path / "testorum" / f"{date_str}.jsonl"
        assert file_path.exists()
        content = file_path.read_text()
        assert event.event_id in content

    def test_batch_ingest(self, store, sample_inputs):
        events = store.ingest_batch(sample_inputs)
        assert len(events) == 5
        assert store.total_ingested == 5
        assert store.buffer_size == 5

    def test_batch_groups_by_source(self, store):
        inputs = [
            EventInput(event_type="test", source="testorum"),
            EventInput(event_type="test", source="talksim"),
            EventInput(event_type="test", source="testorum"),
        ]
        events = store.ingest_batch(inputs)
        assert len(events) == 3

        # 각 소스 디렉터리에 파일 생성 확인
        assert (store.base_path / "testorum").exists()
        assert (store.base_path / "talksim").exists()

    def test_append_to_existing_file(self, store, sample_input):
        store.ingest(sample_input)
        store.ingest(sample_input)
        assert store.total_ingested == 2

        # 같은 파일에 2줄 있어야 함
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        file_path = store.base_path / "testorum" / f"{date_str}.jsonl"
        lines = file_path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_buffer_max_size(self, tmp_events_dir):
        store = EventStore(base_path=tmp_events_dir, max_buffer_size=3)
        for i in range(5):
            store.ingest(EventInput(event_type="test", source="testorum"))
        assert store.buffer_size == 3  # deque maxlen=3
        assert store.total_ingested == 5


# === EventStore Query Tests ===


class TestEventStoreQuery:
    """EventStore 조회 테스트"""

    def test_query_all(self, store, sample_inputs):
        store.ingest_batch(sample_inputs)
        events = store.query(EventQuery())
        assert len(events) == 5

    def test_query_by_source(self, store):
        store.ingest(EventInput(event_type="test", source="testorum"))
        store.ingest(EventInput(event_type="test", source="talksim"))
        events = store.query(EventQuery(source="testorum"))
        assert len(events) == 1
        assert events[0].source == "testorum"

    def test_query_by_event_type(self, store):
        store.ingest(EventInput(event_type="test_completed", source="testorum"))
        store.ingest(EventInput(event_type="user_signup", source="testorum"))
        events = store.query(EventQuery(event_type="test_completed"))
        assert len(events) == 1
        assert events[0].event_type == "test_completed"

    def test_query_by_severity(self, store):
        store.ingest(EventInput(event_type="test", source="testorum", severity=EventSeverity.ERROR))
        store.ingest(EventInput(event_type="test", source="testorum", severity=EventSeverity.INFO))
        events = store.query(EventQuery(severity=EventSeverity.ERROR))
        assert len(events) == 1
        assert events[0].severity == EventSeverity.ERROR

    def test_query_limit(self, store, sample_inputs):
        store.ingest_batch(sample_inputs)
        events = store.query(EventQuery(limit=2))
        assert len(events) == 2

    def test_query_returns_newest_first(self, store):
        ts1 = "2026-05-01T00:00:00+00:00"
        ts2 = "2026-05-02T00:00:00+00:00"
        store.ingest(EventInput(event_type="test", source="testorum", timestamp=ts1))
        store.ingest(EventInput(event_type="test", source="testorum", timestamp=ts2))
        events = store.query(EventQuery())
        assert events[0].timestamp == ts2
        assert events[1].timestamp == ts1

    def test_query_since_filter(self, store):
        ts_old = "2026-04-01T00:00:00+00:00"
        ts_new = "2026-05-02T00:00:00+00:00"
        store.ingest(EventInput(event_type="test", source="testorum", timestamp=ts_old))
        store.ingest(EventInput(event_type="test", source="testorum", timestamp=ts_new))
        events = store.query(EventQuery(since="2026-05-01T00:00:00+00:00"))
        assert len(events) == 1
        assert events[0].timestamp == ts_new

    def test_query_until_filter(self, store):
        ts_old = "2026-04-01T00:00:00+00:00"
        ts_new = "2026-05-02T00:00:00+00:00"
        store.ingest(EventInput(event_type="test", source="testorum", timestamp=ts_old))
        store.ingest(EventInput(event_type="test", source="testorum", timestamp=ts_new))
        events = store.query(EventQuery(until="2026-04-15T00:00:00+00:00"))
        assert len(events) == 1
        assert events[0].timestamp == ts_old

    def test_query_combined_filters(self, store):
        store.ingest(EventInput(event_type="test_completed", source="testorum", severity=EventSeverity.INFO))
        store.ingest(EventInput(event_type="user_signup", source="testorum", severity=EventSeverity.INFO))
        store.ingest(EventInput(event_type="test_completed", source="talksim", severity=EventSeverity.ERROR))
        events = store.query(EventQuery(
            source="testorum",
            event_type="test_completed",
        ))
        assert len(events) == 1
        assert events[0].source == "testorum"
        assert events[0].event_type == "test_completed"

    def test_query_empty_result(self, store, sample_inputs):
        store.ingest_batch(sample_inputs)
        events = store.query(EventQuery(source="talksim"))
        assert len(events) == 0

    def test_query_from_files_when_buffer_empty(self, tmp_events_dir):
        """버퍼가 비어도 파일에서 읽어야 함"""
        # 1. 작은 버퍼로 store 생성 (버퍼 3개)
        store = EventStore(base_path=tmp_events_dir, max_buffer_size=3)

        # 2. 5개 이벤트 인입 → 버퍼에는 마지막 3개만
        for i in range(5):
            store.ingest(EventInput(
                event_type=f"test_{i}",
                source="testorum",
            ))

        # 3. limit=5 조회 → 파일에서도 읽어야 5개 확보
        events = store.query(EventQuery(limit=5))
        assert len(events) == 5


# === EventStore Cleanup Tests ===


class TestEventStoreCleanup:
    """EventStore retention cleanup 테스트"""

    def test_cleanup_old_files(self, tmp_events_dir):
        store = EventStore(base_path=tmp_events_dir, retention_days=7)

        # 오래된 파일 수동 생성
        old_dir = Path(tmp_events_dir) / "testorum"
        old_dir.mkdir(parents=True, exist_ok=True)
        old_file = old_dir / "2020-01-01.jsonl"
        old_file.write_text('{"event_id":"old"}\n')

        # 최근 파일도 생성
        recent_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        recent_file = old_dir / f"{recent_date}.jsonl"
        recent_file.write_text('{"event_id":"recent"}\n')

        deleted = store.cleanup_old_files()
        assert deleted == 1
        assert not old_file.exists()
        assert recent_file.exists()

    def test_cleanup_removes_empty_dirs(self, tmp_events_dir):
        store = EventStore(base_path=tmp_events_dir, retention_days=1)

        # 오래된 파일만 있는 디렉터리
        old_dir = Path(tmp_events_dir) / "old_source"
        old_dir.mkdir(parents=True, exist_ok=True)
        old_file = old_dir / "2020-01-01.jsonl"
        old_file.write_text('{"event_id":"old"}\n')

        store.cleanup_old_files()
        assert not old_dir.exists()

    def test_cleanup_no_files(self, store):
        deleted = store.cleanup_old_files()
        assert deleted == 0


# === EventStore Stats Tests ===


class TestEventStoreStats:
    """EventStore 통계 테스트"""

    def test_stats_after_ingest(self, store):
        store.ingest(EventInput(event_type="test", source="testorum"))
        store.ingest(EventInput(event_type="test", source="talksim"))
        store.ingest(EventInput(event_type="test", source="testorum"))

        stats = store.get_stats()
        assert stats["total_ingested"] == 3
        assert stats["buffer_size"] == 3
        assert stats["buffer_by_source"]["testorum"] == 2
        assert stats["buffer_by_source"]["talksim"] == 1


# === EventConfig Tests ===


class TestEventConfig:
    """EventConfig 설정 테스트"""

    def test_default_values(self):
        os.environ.pop("ARIA_EVENT_BASE_PATH", None)
        os.environ.pop("ARIA_EVENT_MAX_BUFFER_SIZE", None)
        os.environ.pop("ARIA_EVENT_RETENTION_DAYS", None)
        from aria.core.config import EventConfig
        config = EventConfig()
        assert config.base_path == "./events"
        assert config.max_buffer_size == 1000
        assert config.retention_days == 30

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("ARIA_EVENT_BASE_PATH", "/tmp/custom_events")
        monkeypatch.setenv("ARIA_EVENT_MAX_BUFFER_SIZE", "500")
        monkeypatch.setenv("ARIA_EVENT_RETENTION_DAYS", "90")
        from aria.core.config import EventConfig
        config = EventConfig()
        assert config.base_path == "/tmp/custom_events"
        assert config.max_buffer_size == 500
        assert config.retention_days == 90


# === API Integration Tests (with TestClient) ===


class TestEventAPI:
    """이벤트 API 엔드포인트 테스트"""

    @pytest.fixture(autouse=True)
    def setup_app(self, tmp_events_dir, monkeypatch):
        """테스트용 앱 설정 (lifespan 우회 — 글로벌 직접 주입)"""
        monkeypatch.setenv("ARIA_AUTH_DISABLED", "true")
        monkeypatch.setenv("ARIA_ENV", "development")
        monkeypatch.setenv("ARIA_EVENT_BASE_PATH", tmp_events_dir)

        # config 캐시 초기화
        from aria.core.config import get_config
        get_config.cache_clear()

        import aria.api.app as app_module

        # 원본 글로벌 상태 저장
        _orig_event_store = app_module.event_store
        _orig_rate_limiter = app_module.rate_limiter

        # 글로벌 인스턴스 직접 주입
        app_module.event_store = EventStore(
            base_path=tmp_events_dir,
            max_buffer_size=100,
            retention_days=30,
        )
        app_module.rate_limiter = app_module.RateLimiter(max_requests=60, window_seconds=60)

        from fastapi.testclient import TestClient
        self.client = TestClient(app_module.app, raise_server_exceptions=False)

        yield

        # teardown — 원복
        app_module.event_store = _orig_event_store
        app_module.rate_limiter = _orig_rate_limiter
        get_config.cache_clear()

    def test_ingest_single_event(self):
        resp = self.client.post("/v1/events", json={
            "events": [
                {
                    "event_type": "test_completed",
                    "source": "testorum",
                    "data": {"test_id": "abc", "score": 90},
                }
            ]
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["ingested"] == 1
        assert len(body["event_ids"]) == 1

    def test_ingest_batch(self):
        resp = self.client.post("/v1/events", json={
            "events": [
                {"event_type": "test_completed", "source": "testorum"},
                {"event_type": "user_signup", "source": "testorum"},
                {"event_type": "analysis_done", "source": "talksim"},
            ]
        })
        assert resp.status_code == 200
        assert resp.json()["ingested"] == 3

    def test_ingest_invalid_source(self):
        resp = self.client.post("/v1/events", json={
            "events": [
                {"event_type": "test", "source": "invalid_app"},
            ]
        })
        assert resp.status_code == 422  # Pydantic validation error

    def test_ingest_empty_batch(self):
        resp = self.client.post("/v1/events", json={"events": []})
        assert resp.status_code == 422

    def test_query_events(self):
        # 인입 후 조회
        self.client.post("/v1/events", json={
            "events": [
                {"event_type": "test_completed", "source": "testorum"},
                {"event_type": "user_signup", "source": "testorum"},
            ]
        })
        resp = self.client.get("/v1/events")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 2

    def test_query_with_source_filter(self):
        self.client.post("/v1/events", json={
            "events": [
                {"event_type": "test", "source": "testorum"},
                {"event_type": "test", "source": "talksim"},
            ]
        })
        resp = self.client.get("/v1/events", params={"source": "testorum"})
        assert resp.status_code == 200
        assert resp.json()["count"] == 1

    def test_query_with_event_type_filter(self):
        self.client.post("/v1/events", json={
            "events": [
                {"event_type": "test_completed", "source": "testorum"},
                {"event_type": "user_signup", "source": "testorum"},
            ]
        })
        resp = self.client.get("/v1/events", params={"event_type": "test_completed"})
        assert resp.status_code == 200
        assert resp.json()["count"] == 1

    def test_query_with_severity_filter(self):
        self.client.post("/v1/events", json={
            "events": [
                {"event_type": "error_occurred", "source": "testorum", "severity": "error"},
                {"event_type": "test", "source": "testorum", "severity": "info"},
            ]
        })
        resp = self.client.get("/v1/events", params={"severity": "error"})
        assert resp.status_code == 200
        assert resp.json()["count"] == 1

    def test_query_invalid_severity(self):
        resp = self.client.get("/v1/events", params={"severity": "critical"})
        assert resp.status_code == 400
        assert "INVALID_SEVERITY" in resp.json()["error"]

    def test_query_with_limit(self):
        self.client.post("/v1/events", json={
            "events": [
                {"event_type": f"test_{i}", "source": "testorum"}
                for i in range(10)
            ]
        })
        resp = self.client.get("/v1/events", params={"limit": 3})
        assert resp.status_code == 200
        assert resp.json()["count"] == 3

    def test_event_stats(self):
        self.client.post("/v1/events", json={
            "events": [
                {"event_type": "test", "source": "testorum"},
                {"event_type": "test", "source": "testorum"},
            ]
        })
        resp = self.client.get("/v1/events/stats")
        assert resp.status_code == 200
        stats = resp.json()
        assert stats["total_ingested"] == 2
        assert stats["buffer_by_source"]["testorum"] == 2
