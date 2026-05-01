"""ARIA Engine - Event Types

이벤트 스키마 정의 (Pydantic v2)
- EventSeverity: 이벤트 심각도 (info/warning/error)
- Event: 저장된 이벤트 (event_id + timestamp 포함)
- EventInput: 이벤트 인입 요청 (event_id/timestamp 자동 생성)
- EventIngestRequest: 배치 인입 요청
- EventIngestResponse: 배치 인입 응답
- EventQuery: 이벤트 조회 필터
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# === 허용된 소스 목록 ===
VALID_SOURCES = frozenset({"testorum", "talksim", "autotube", "trendbot", "aria"})


class EventSeverity(str, Enum):
    """이벤트 심각도"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Event(BaseModel):
    """저장된 이벤트 (불변)"""

    event_id: str = Field(description="고유 이벤트 ID (UUID4)")
    event_type: str = Field(description="이벤트 유형 (예: test_completed, user_signup)")
    source: str = Field(description="이벤트 소스 (testorum, talksim, autotube, trendbot, aria)")
    severity: EventSeverity = Field(default=EventSeverity.INFO)
    data: dict[str, Any] = Field(default_factory=dict, description="이벤트 페이로드")
    timestamp: str = Field(description="ISO 8601 타임스탬프 (UTC)")

    def to_jsonl(self) -> str:
        """JSONL 한 줄로 직렬화"""
        return self.model_dump_json()


class EventInput(BaseModel):
    """이벤트 인입 요청 (클라이언트 → ARIA)

    event_id와 timestamp는 서버에서 자동 생성
    """

    event_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="이벤트 유형 (예: test_completed)",
    )
    source: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="이벤트 소스",
    )
    severity: EventSeverity = Field(default=EventSeverity.INFO)
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="이벤트 페이로드 (최대 10KB)",
    )
    timestamp: str | None = Field(
        default=None,
        description="ISO 8601 타임스탬프 (미지정 시 서버 시간)",
    )

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        """허용된 소스만 수용"""
        v_lower = v.lower().strip()
        if v_lower not in VALID_SOURCES:
            raise ValueError(
                f"유효하지 않은 소스: '{v}'. "
                f"허용 목록: {', '.join(sorted(VALID_SOURCES))}"
            )
        return v_lower

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        """event_type 정규화 (소문자 + snake_case)"""
        v = v.strip().lower()
        # 영문/숫자/언더스코어/하이픈만 허용
        import re
        if not re.match(r"^[a-z0-9][a-z0-9_-]*$", v):
            raise ValueError(
                f"유효하지 않은 event_type: '{v}'. "
                "영문 소문자/숫자/언더스코어/하이픈만 허용 (첫 글자: 영문/숫자)"
            )
        return v

    @field_validator("data")
    @classmethod
    def validate_data_size(cls, v: dict) -> dict:
        """페이로드 크기 제한 (10KB)"""
        import json
        size = len(json.dumps(v, ensure_ascii=False).encode("utf-8"))
        if size > 10_240:
            raise ValueError(
                f"data 크기 초과: {size} bytes (최대 10,240 bytes)"
            )
        return v

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: str | None) -> str | None:
        """ISO 8601 형식 검증"""
        if v is None:
            return v
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            raise ValueError(f"유효하지 않은 타임스탬프: '{v}'. ISO 8601 형식을 사용하세요")
        return v

    def to_event(self) -> Event:
        """Event 객체로 변환 (event_id + timestamp 자동 생성)"""
        now = datetime.now(timezone.utc).isoformat()
        return Event(
            event_id=str(uuid.uuid4()),
            event_type=self.event_type,
            source=self.source,
            severity=self.severity,
            data=self.data,
            timestamp=self.timestamp or now,
        )


class EventIngestRequest(BaseModel):
    """배치 이벤트 인입 요청"""

    events: list[EventInput] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="인입할 이벤트 목록 (최대 100개)",
    )


class EventIngestResponse(BaseModel):
    """배치 이벤트 인입 응답"""

    status: str = "ok"
    ingested: int = Field(description="성공적으로 저장된 이벤트 수")
    event_ids: list[str] = Field(description="생성된 이벤트 ID 목록")


class EventQuery(BaseModel):
    """이벤트 조회 필터"""

    source: str | None = Field(default=None, description="소스 필터")
    event_type: str | None = Field(default=None, description="이벤트 유형 필터")
    severity: EventSeverity | None = Field(default=None, description="심각도 필터")
    since: str | None = Field(default=None, description="이 시점 이후 이벤트 (ISO 8601)")
    until: str | None = Field(default=None, description="이 시점 이전 이벤트 (ISO 8601)")
    limit: int = Field(default=50, ge=1, le=500, description="최대 반환 수")

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str | None) -> str | None:
        if v is not None:
            v = v.lower().strip()
            if v not in VALID_SOURCES:
                raise ValueError(f"유효하지 않은 소스: '{v}'")
        return v
