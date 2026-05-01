"""ARIA Engine - Event Collection System

제품(Testorum/Talksim/AutoTube)에서 발생하는 이벤트를 수집하고
ARIA 메모리/분석/능동 알림에 활용하는 이벤트 파이프라인

- POST /v1/events → 이벤트 인입 (배치 지원)
- GET /v1/events → 이벤트 조회 (필터링)
"""

from aria.events.types import Event, EventInput, EventQuery, EventSeverity
from aria.events.event_store import EventStore

__all__ = [
    "Event",
    "EventInput",
    "EventQuery",
    "EventSeverity",
    "EventStore",
]
