"""ARIA Engine - Memory System Type Definitions

3계층 메모리 아키텍처의 핵심 스키마 정의
- IndexEntry: 인덱스 한 줄 (포인터 — "directory, not diary")
- TopicFile: 도메인별 실제 지식 본문
- MemoryIndex: 전체 인덱스 컨테이너
- MemoryScope: 스코프 식별자 (프로젝트별 격리)
- API 요청/응답 모델

설계 원칙:
    "directory, not diary" — 인덱스는 포인터만 저장 (120자 이내)
    실제 지식은 Topic 파일에 저장
    read-before-write — 덮어쓰기 전 반드시 현재 버전 확인 (낙관적 락)
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator, model_validator


# === Domain Validation ===

# URL-safe: 소문자 + 숫자 + 하이픈만 허용 / 하이픈으로 시작/끝 불가
_DOMAIN_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")

# 허용 스코프 목록 (추후 확장 가능하도록 set 사용 — Literal 대신)
VALID_SCOPES: set[str] = {"global", "testorum", "talksim", "autotube"}

# 토픽 콘텐츠 최대 크기 (50KB)
MAX_CONTENT_BYTES = 50 * 1024

# 인덱스 요약 최대 길이
MAX_SUMMARY_LENGTH = 120

# 도메인 최대 길이
MAX_DOMAIN_LENGTH = 64


def validate_domain(value: str) -> str:
    """도메인명 유효성 검증

    규칙:
        - 1~64자
        - 소문자 + 숫자 + 하이픈만 허용
        - 하이픈으로 시작/끝 불가
        - URL-safe

    Raises:
        ValueError: 유효하지 않은 도메인명
    """
    if not value:
        raise ValueError("도메인명은 비어있을 수 없습니다")
    if len(value) > MAX_DOMAIN_LENGTH:
        raise ValueError(f"도메인명은 {MAX_DOMAIN_LENGTH}자 이하여야 합니다: {len(value)}자")
    if not _DOMAIN_PATTERN.match(value):
        raise ValueError(
            f"유효하지 않은 도메인명: '{value}' "
            "(소문자+숫자+하이픈만 허용 / 하이픈으로 시작·끝 불가)"
        )
    return value


def validate_scope(value: str) -> str:
    """스코프 유효성 검증

    Raises:
        ValueError: 허용되지 않은 스코프
    """
    if value not in VALID_SCOPES:
        raise ValueError(
            f"유효하지 않은 스코프: '{value}' (허용: {', '.join(sorted(VALID_SCOPES))})"
        )
    return value


# === Core Models ===


class IndexEntry(BaseModel):
    """인덱스 한 줄 — 포인터 (directory, not diary)

    인덱스는 실제 지식을 저장하지 않음
    도메인 + 요약 + 메타데이터만 보관하여 빠른 검색/로딩 지원

    Attributes:
        domain: URL-safe 도메인 식별자 (예: "user-profile" / "api-conventions")
        summary: 토픽 내용 요약 (120자 이내 — 인덱스 스캔용)
        updated_at: 마지막 업데이트 시각 (UTC)
        token_estimate: 토픽 파일 로딩 비용 추정 (토큰 수 / None이면 미측정)
    """

    domain: str = Field(
        ...,
        min_length=1,
        max_length=MAX_DOMAIN_LENGTH,
        description="URL-safe 도메인 식별자",
    )
    summary: str = Field(
        ...,
        min_length=1,
        max_length=MAX_SUMMARY_LENGTH,
        description="토픽 내용 요약 (120자 이내)",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="마지막 업데이트 시각 (UTC)",
    )
    token_estimate: int | None = Field(
        default=None,
        ge=0,
        description="토픽 파일 로딩 비용 추정 (토큰 수)",
    )

    @field_validator("domain")
    @classmethod
    def _validate_domain(cls, v: str) -> str:
        return validate_domain(v)



class TopicFile(BaseModel):
    """개별 토픽 본문 — 실제 지식 저장

    도메인별 하나의 마크다운 파일에 매핑
    낙관적 락(version 필드)으로 동시 수정 충돌 방지

    Attributes:
        domain: URL-safe 도메인 식별자
        scope: 스코프 (global / testorum / talksim / autotube)
        content: 실제 지식 본문 (마크다운 / 50KB 상한)
        version: 낙관적 락용 버전 번호 (1부터 시작 / 매 수정 시 +1)
        updated_at: 마지막 업데이트 시각 (UTC)
        created_at: 최초 생성 시각 (UTC)
    """

    domain: str = Field(
        ...,
        min_length=1,
        max_length=MAX_DOMAIN_LENGTH,
        description="URL-safe 도메인 식별자",
    )
    scope: str = Field(
        ...,
        description="스코프 식별자",
    )
    content: str = Field(
        ...,
        min_length=1,
        description="실제 지식 본문 (마크다운 / 50KB 상한)",
    )
    version: int = Field(
        default=1,
        ge=1,
        description="낙관적 락용 버전 번호 (1부터 시작)",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="마지막 업데이트 시각 (UTC)",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="최초 생성 시각 (UTC)",
    )

    @field_validator("domain")
    @classmethod
    def _validate_domain(cls, v: str) -> str:
        return validate_domain(v)

    @field_validator("scope")
    @classmethod
    def _validate_scope(cls, v: str) -> str:
        return validate_scope(v)

    @field_validator("content")
    @classmethod
    def _validate_content_size(cls, v: str) -> str:
        size = len(v.encode("utf-8"))
        if size > MAX_CONTENT_BYTES:
            raise ValueError(
                f"토픽 콘텐츠가 최대 크기를 초과합니다: "
                f"{size:,}bytes > {MAX_CONTENT_BYTES:,}bytes (50KB)"
            )
        return v



class MemoryIndex(BaseModel):
    """전체 인덱스 컨테이너

    스코프별 하나의 index.json에 매핑
    모든 IndexEntry를 보관하며 인덱스 전체를 원자적으로 읽기/쓰기

    Attributes:
        version: 인덱스 포맷 버전 (현재 1 고정)
        scope: 스코프 식별자
        entries: 인덱스 엔트리 목록
        updated_at: 마지막 업데이트 시각 (UTC)
    """

    version: int = Field(
        default=1,
        ge=1,
        description="인덱스 포맷 버전 (현재 1 고정)",
    )
    scope: str = Field(
        ...,
        description="스코프 식별자",
    )
    entries: list[IndexEntry] = Field(
        default_factory=list,
        description="인덱스 엔트리 목록",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="마지막 업데이트 시각 (UTC)",
    )

    @field_validator("scope")
    @classmethod
    def _validate_scope(cls, v: str) -> str:
        return validate_scope(v)

    def find_entry(self, domain: str) -> IndexEntry | None:
        """도메인으로 엔트리 검색

        Args:
            domain: 검색할 도메인명

        Returns:
            IndexEntry 또는 None (미존재)
        """
        for entry in self.entries:
            if entry.domain == domain:
                return entry
        return None

    def has_entry(self, domain: str) -> bool:
        """도메인 엔트리 존재 여부 확인"""
        return self.find_entry(domain) is not None



# === API Request/Response Models ===


class TopicUpsertRequest(BaseModel):
    """토픽 upsert 요청 (PUT /v1/memory/{scope}/topics/{domain})

    Attributes:
        summary: 인덱스 요약 (120자 이내)
        content: 토픽 본문 (마크다운)
        expected_version: 낙관적 락 — 현재 버전 (신규 생성 시 None)
    """

    summary: str = Field(
        ...,
        min_length=1,
        max_length=MAX_SUMMARY_LENGTH,
        description="인덱스 요약 (120자 이내)",
    )
    content: str = Field(
        ...,
        min_length=1,
        description="토픽 본문 (마크다운 / 50KB 상한)",
    )
    expected_version: int | None = Field(
        default=None,
        ge=1,
        description="낙관적 락 — 현재 알고 있는 버전 (신규 생성 시 None / 업데이트 시 필수)",
    )

    @field_validator("content")
    @classmethod
    def _validate_content_size(cls, v: str) -> str:
        size = len(v.encode("utf-8"))
        if size > MAX_CONTENT_BYTES:
            raise ValueError(
                f"토픽 콘텐츠가 최대 크기를 초과합니다: "
                f"{size:,}bytes > {MAX_CONTENT_BYTES:,}bytes (50KB)"
            )
        return v


class TopicResponse(BaseModel):
    """토픽 조회 응답

    Attributes:
        domain: 도메인 식별자
        scope: 스코프
        summary: 인덱스 요약
        content: 토픽 본문
        version: 현재 버전
        updated_at: 마지막 업데이트 시각
        created_at: 최초 생성 시각
        token_estimate: 토큰 수 추정
    """

    domain: str
    scope: str
    summary: str
    content: str
    version: int
    updated_at: datetime
    created_at: datetime
    token_estimate: int | None = None


class MemoryLoadRequest(BaseModel):
    """메모리 로딩 요청 (POST /v1/memory/{scope}/load)

    Attributes:
        domains: 로딩할 도메인 목록 (None이면 전체 인덱스 기반 자동 선택)
        token_budget: 토큰 예산 (None이면 설정 기본값 사용)
    """

    domains: list[str] | None = Field(
        default=None,
        description="로딩할 도메인 목록 (None이면 전체)",
    )
    token_budget: int | None = Field(
        default=None,
        ge=100,
        le=32000,
        description="토큰 예산 (None이면 설정 기본값)",
    )

    @field_validator("domains")
    @classmethod
    def _validate_domains(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            for domain in v:
                validate_domain(domain)
            if len(v) > 50:
                raise ValueError("도메인 목록은 최대 50개까지 지정 가능합니다")
        return v


class MemoryLoadResponse(BaseModel):
    """메모리 로딩 응답

    Attributes:
        scope: 로딩된 스코프
        loaded_domains: 실제 로딩된 도메인 목록
        prompt_markdown: 프롬프트 주입용 마크다운
        total_tokens: 총 토큰 수 추정
        budget_used: 예산 대비 사용 비율 (0.0~1.0)
    """

    scope: str
    loaded_domains: list[str]
    prompt_markdown: str
    total_tokens: int
    budget_used: float = Field(ge=0.0, le=1.0)
