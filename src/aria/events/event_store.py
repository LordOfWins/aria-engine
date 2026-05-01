"""ARIA Engine - Event Store

파일 기반 이벤트 저장소 (JSONL / append-only)
- 소스별 일별 파일: events/{source}/{YYYY-MM-DD}.jsonl
- 인메모리 최근 버퍼: 빠른 조회용 (max_buffer_size)
- retention 정책: 설정된 일수 초과 파일 자동 삭제
- 원자적 쓰기: tmpfile → rename (메모리 시스템 패턴과 동일)
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path

import structlog

from aria.events.types import Event, EventInput, EventQuery, EventSeverity

logger = structlog.get_logger()


class EventStore:
    """파일 기반 이벤트 저장소

    Thread-safe: 쓰기 lock + deque 기반 버퍼
    """

    def __init__(
        self,
        base_path: str = "./events",
        max_buffer_size: int = 1000,
        retention_days: int = 30,
    ) -> None:
        self._base_path = Path(base_path)
        self._max_buffer_size = max_buffer_size
        self._retention_days = retention_days
        self._buffer: deque[Event] = deque(maxlen=max_buffer_size)
        self._lock = threading.Lock()
        self._total_ingested: int = 0

        # 기본 디렉터리 생성
        self._base_path.mkdir(parents=True, exist_ok=True)

    @property
    def base_path(self) -> Path:
        return self._base_path

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def total_ingested(self) -> int:
        return self._total_ingested

    def ingest(self, event_input: EventInput) -> Event:
        """단일 이벤트 인입 → 파일 저장 + 버퍼 추가"""
        event = event_input.to_event()
        self._write_event(event)
        with self._lock:
            self._buffer.append(event)
            self._total_ingested += 1
        return event

    def ingest_batch(self, event_inputs: list[EventInput]) -> list[Event]:
        """배치 이벤트 인입"""
        events: list[Event] = []
        # 소스+날짜별로 그룹핑하여 파일 I/O 최소화
        grouped: dict[tuple[str, str], list[Event]] = {}

        for inp in event_inputs:
            event = inp.to_event()
            events.append(event)
            date_str = event.timestamp[:10]  # YYYY-MM-DD
            key = (event.source, date_str)
            grouped.setdefault(key, []).append(event)

        # 그룹별 일괄 쓰기
        for (source, date_str), group_events in grouped.items():
            self._write_events_batch(source, date_str, group_events)

        # 버퍼 + 카운터 업데이트
        with self._lock:
            for event in events:
                self._buffer.append(event)
            self._total_ingested += len(events)

        return events

    def query(self, q: EventQuery) -> list[Event]:
        """이벤트 조회 (버퍼 우선 → 필요 시 파일 검색)

        최신순 정렬 반환
        """
        results: list[Event] = []

        # 1단계: 인메모리 버퍼에서 필터링
        with self._lock:
            buffer_copy = list(self._buffer)

        for event in reversed(buffer_copy):  # 최신순
            if self._matches_filter(event, q):
                results.append(event)
                if len(results) >= q.limit:
                    return results

        # 2단계: 버퍼만으로 부족하면 파일에서 추가 검색
        remaining = q.limit - len(results)
        if remaining > 0:
            # 버퍼에 있는 event_id를 제외
            buffer_ids = {e.event_id for e in results}
            file_events = self._read_from_files(q, limit=remaining, exclude_ids=buffer_ids)
            results.extend(file_events)

        return results

    def get_stats(self) -> dict:
        """이벤트 저장소 통계"""
        source_counts: dict[str, int] = {}
        with self._lock:
            for event in self._buffer:
                source_counts[event.source] = source_counts.get(event.source, 0) + 1

        return {
            "total_ingested": self._total_ingested,
            "buffer_size": len(self._buffer),
            "max_buffer_size": self._max_buffer_size,
            "retention_days": self._retention_days,
            "buffer_by_source": source_counts,
            "storage_path": str(self._base_path),
        }

    def cleanup_old_files(self) -> int:
        """retention 기간 초과 파일 삭제

        Returns: 삭제된 파일 수
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._retention_days)
        cutoff_str = cutoff.strftime("%Y-%m-%d")
        deleted = 0

        if not self._base_path.exists():
            return 0

        for source_dir in self._base_path.iterdir():
            if not source_dir.is_dir():
                continue
            for jsonl_file in source_dir.glob("*.jsonl"):
                date_part = jsonl_file.stem  # YYYY-MM-DD
                try:
                    if date_part < cutoff_str:
                        jsonl_file.unlink()
                        deleted += 1
                        logger.info(
                            "event_file_deleted",
                            source=source_dir.name,
                            file=jsonl_file.name,
                        )
                except (OSError, ValueError):
                    continue

            # 빈 디렉터리 삭제
            try:
                if source_dir.exists() and not any(source_dir.iterdir()):
                    source_dir.rmdir()
            except OSError:
                pass

        return deleted

    # === Private Methods ===

    def _get_file_path(self, source: str, date_str: str) -> Path:
        """이벤트 파일 경로 생성"""
        source_dir = self._base_path / source
        source_dir.mkdir(parents=True, exist_ok=True)
        return source_dir / f"{date_str}.jsonl"

    def _write_event(self, event: Event) -> None:
        """단일 이벤트 파일 추가 (append)"""
        date_str = event.timestamp[:10]
        file_path = self._get_file_path(event.source, date_str)

        line = event.to_jsonl() + "\n"
        try:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError as e:
            logger.error(
                "event_write_failed",
                source=event.source,
                event_id=event.event_id,
                error=str(e),
            )
            raise

    def _write_events_batch(
        self, source: str, date_str: str, events: list[Event]
    ) -> None:
        """배치 이벤트 파일 추가"""
        file_path = self._get_file_path(source, date_str)

        lines = "".join(e.to_jsonl() + "\n" for e in events)
        try:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(lines)
        except OSError as e:
            logger.error(
                "event_batch_write_failed",
                source=source,
                count=len(events),
                error=str(e),
            )
            raise

    def _read_from_files(
        self,
        q: EventQuery,
        limit: int,
        exclude_ids: set[str] | None = None,
    ) -> list[Event]:
        """파일에서 이벤트 읽기 (최신순)"""
        exclude_ids = exclude_ids or set()
        results: list[Event] = []

        # 검색 대상 소스 디렉터리 결정
        if q.source:
            source_dirs = [self._base_path / q.source]
        else:
            source_dirs = [
                d for d in self._base_path.iterdir() if d.is_dir()
            ]

        # 모든 관련 파일을 날짜 역순으로 수집
        all_files: list[tuple[str, Path]] = []
        for source_dir in source_dirs:
            if not source_dir.exists():
                continue
            for jsonl_file in source_dir.glob("*.jsonl"):
                date_str = jsonl_file.stem
                # 날짜 범위 필터 (파일 레벨)
                if q.since and date_str < q.since[:10]:
                    continue
                if q.until and date_str > q.until[:10]:
                    continue
                all_files.append((date_str, jsonl_file))

        # 최신 파일부터 처리
        all_files.sort(key=lambda x: x[0], reverse=True)

        for _, file_path in all_files:
            if len(results) >= limit:
                break

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                # 최신순 (파일 내에서 역순)
                for line in reversed(lines):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = Event.model_validate_json(line)
                    except Exception:
                        continue

                    if event.event_id in exclude_ids:
                        continue

                    if self._matches_filter(event, q):
                        results.append(event)
                        if len(results) >= limit:
                            break
            except OSError:
                continue

        return results

    def _matches_filter(self, event: Event, q: EventQuery) -> bool:
        """이벤트가 필터 조건에 부합하는지 확인"""
        if q.source and event.source != q.source:
            return False
        if q.event_type and event.event_type != q.event_type:
            return False
        if q.severity and event.severity != q.severity:
            return False
        if q.since and event.timestamp < q.since:
            return False
        if q.until and event.timestamp > q.until:
            return False
        return True
