"""ARIA Engine - Memory Index Manager

인덱스 CRUD + read-before-write 규율 + 낙관적 락
Strict Write Discipline: Topic 파일 쓰기 성공 후에만 인덱스 업데이트

사용법:
    storage = FileStorageAdapter("./memory")
    manager = IndexManager(storage)

    # 토픽 생성
    topic = manager.upsert_topic(
        scope="global",
        domain="user-profile",
        summary="사용자 프로필",
        content="# 프로필\n\n- 이름: 승재",
        expected_version=None,  # 신규 생성
    )

    # 토픽 업데이트 (read-before-write)
    existing = manager.get_topic("global", "user-profile")
    updated = manager.upsert_topic(
        scope="global",
        domain="user-profile",
        summary="사용자 프로필 (업데이트)",
        content="# 프로필\n\n- 이름: 승재\n- 역할: 개발자",
        expected_version=existing.version,  # 반드시 현재 버전 전달
    )
"""

from __future__ import annotations

from datetime import datetime, timezone

import structlog
import tiktoken

from aria.core.exceptions import (
    MemoryNotFoundError,
    MemoryScopeError,
    MemoryStorageError,
    VersionConflictError,
)
from aria.memory.storage_adapter import StorageAdapter
from aria.memory.types import (
    IndexEntry,
    MemoryIndex,
    TopicFile,
    validate_domain,
    validate_scope,
)

logger = structlog.get_logger()

# tiktoken 인코더 (토큰 수 추정용 — cl100k_base는 Claude/GPT 공통)
_ENCODING: tiktoken.Encoding | None = None


def _get_encoding() -> tiktoken.Encoding:
    """tiktoken 인코더 싱글톤"""
    global _ENCODING
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def estimate_tokens(text: str) -> int:
    """텍스트의 토큰 수 추정

    Args:
        text: 토큰 수를 추정할 텍스트

    Returns:
        추정 토큰 수
    """
    return len(_get_encoding().encode(text))


class IndexManager:
    """메모리 인덱스 관리자

    read-before-write 규율:
        - upsert 시 expected_version 필수 (신규 생성 시 None)
        - 기존 토픽 업데이트 시 expected_version != actual_version → VersionConflictError

    Strict Write Discipline:
        1. Topic 파일 쓰기 (storage.write_topic)
        2. 쓰기 성공 후 인덱스 업데이트 (storage.write_index)
        → Topic 쓰기 실패 시 인덱스는 변경되지 않음

    Args:
        storage: 저장소 어댑터 (FileStorageAdapter 등)
    """

    def __init__(self, storage: StorageAdapter) -> None:
        self._storage = storage

    def get_index(self, scope: str) -> MemoryIndex:
        """인덱스 조회

        Args:
            scope: 스코프 식별자

        Returns:
            MemoryIndex (미존재 시 빈 인덱스)
        """
        validate_scope(scope)
        return self._storage.read_index(scope)

    def get_entry(self, scope: str, domain: str) -> IndexEntry | None:
        """특정 도메인의 인덱스 엔트리 조회

        Args:
            scope: 스코프 식별자
            domain: 도메인 식별자

        Returns:
            IndexEntry 또는 None
        """
        index = self.get_index(scope)
        return index.find_entry(domain)

    def get_topic(self, scope: str, domain: str) -> TopicFile:
        """토픽 조회

        Args:
            scope: 스코프 식별자
            domain: 도메인 식별자

        Returns:
            TopicFile

        Raises:
            MemoryNotFoundError: 토픽 미존재
        """
        validate_scope(scope)
        validate_domain(domain)
        return self._storage.read_topic(scope, domain)

    def upsert_topic(
        self,
        *,
        scope: str,
        domain: str,
        summary: str,
        content: str,
        expected_version: int | None = None,
    ) -> TopicFile:
        """토픽 생성 또는 업데이트 (read-before-write 규율 적용)

        신규 생성:
            expected_version=None → version=1로 생성

        업데이트:
            expected_version 필수 → 현재 버전과 일치해야 함
            불일치 시 VersionConflictError 발생

        Strict Write Discipline:
            1. Topic 파일 쓰기
            2. 성공 후 인덱스 업데이트

        Args:
            scope: 스코프 식별자
            domain: 도메인 식별자
            summary: 인덱스 요약 (120자 이내)
            content: 토픽 본문
            expected_version: 기대 버전 (신규=None / 업데이트=현재 버전)

        Returns:
            생성/업데이트된 TopicFile

        Raises:
            VersionConflictError: 버전 불일치 (다른 곳에서 수정됨)
            MemoryStorageError: 저장 실패
            MemoryScopeError: 유효하지 않은 스코프
        """
        validate_scope(scope)
        validate_domain(domain)

        now = datetime.now(timezone.utc)
        self._storage.ensure_scope_directory(scope)

        existing_topic: TopicFile | None = None
        if self._storage.topic_exists(scope, domain):
            existing_topic = self._storage.read_topic(scope, domain)

        if existing_topic is not None:
            # === 업데이트 ===
            if expected_version is None:
                raise VersionConflictError(
                    scope=scope,
                    domain=domain,
                    expected_version=0,
                    actual_version=existing_topic.version,
                )
            if expected_version != existing_topic.version:
                raise VersionConflictError(
                    scope=scope,
                    domain=domain,
                    expected_version=expected_version,
                    actual_version=existing_topic.version,
                )

            new_version = existing_topic.version + 1
            topic = TopicFile(
                domain=domain,
                scope=scope,
                content=content,
                version=new_version,
                updated_at=now,
                created_at=existing_topic.created_at,
            )
        else:
            # === 신규 생성 ===
            if expected_version is not None:
                # 신규인데 버전이 지정된 경우 — 클라이언트가 존재한다고 착각
                logger.warning(
                    "upsert_expected_version_on_new_topic",
                    scope=scope,
                    domain=domain,
                    expected_version=expected_version,
                )
            topic = TopicFile(
                domain=domain,
                scope=scope,
                content=content,
                version=1,
                updated_at=now,
                created_at=now,
            )

        token_est = estimate_tokens(content)

        # === Strict Write Discipline ===
        # 1. Topic 파일 먼저 쓰기
        self._storage.write_topic(topic)

        # 2. Topic 쓰기 성공 후 인덱스 업데이트
        try:
            index = self._storage.read_index(scope)
            entry = IndexEntry(
                domain=domain,
                summary=summary,
                updated_at=now,
                token_estimate=token_est,
            )

            # 기존 엔트리 교체 또는 추가
            new_entries = [e for e in index.entries if e.domain != domain]
            new_entries.append(entry)

            updated_index = MemoryIndex(
                version=index.version,
                scope=scope,
                entries=new_entries,
                updated_at=now,
            )
            self._storage.write_index(updated_index)
        except Exception as e:
            # 인덱스 업데이트 실패 시 로깅 (Topic 파일은 이미 저장됨)
            # 다음 upsert 시 인덱스가 재구축됨
            logger.error(
                "index_update_failed_after_topic_write",
                scope=scope,
                domain=domain,
                error=str(e),
            )
            raise MemoryStorageError(
                f"토픽은 저장되었으나 인덱스 업데이트 실패: {domain} — {e}",
                scope=scope,
                domain=domain,
            ) from e

        logger.info(
            "topic_upserted",
            scope=scope,
            domain=domain,
            version=topic.version,
            token_estimate=token_est,
        )

        return topic

    def delete_topic(self, scope: str, domain: str) -> None:
        """토픽 + 인덱스 엔트리 삭제

        삭제 순서:
            1. 인덱스에서 엔트리 제거 + 인덱스 저장
            2. Topic 파일 삭제

        Args:
            scope: 스코프 식별자
            domain: 도메인 식별자

        Raises:
            MemoryNotFoundError: 토픽 미존재
            MemoryStorageError: 삭제 실패
        """
        validate_scope(scope)
        validate_domain(domain)

        if not self._storage.topic_exists(scope, domain):
            raise MemoryNotFoundError(scope=scope, domain=domain)

        # 1. 인덱스에서 제거
        now = datetime.now(timezone.utc)
        index = self._storage.read_index(scope)
        new_entries = [e for e in index.entries if e.domain != domain]
        updated_index = MemoryIndex(
            version=index.version,
            scope=scope,
            entries=new_entries,
            updated_at=now,
        )
        self._storage.write_index(updated_index)

        # 2. Topic 파일 삭제
        self._storage.delete_topic(scope, domain)

        logger.info("topic_deleted", scope=scope, domain=domain)

    def list_domains(self, scope: str) -> list[str]:
        """스코프 내 모든 도메인 목록

        Args:
            scope: 스코프 식별자

        Returns:
            도메인명 목록
        """
        index = self.get_index(scope)
        return [e.domain for e in index.entries]
