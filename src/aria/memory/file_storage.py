"""ARIA Engine - File-based Memory Storage

파일 시스템 기반 메모리 저장소 구현
- 인덱스: JSON (memory/{scope}/index.json)
- 토픽: Markdown (memory/{scope}/topics/{domain}.md) + JSON 메타데이터
- 정적 설정: Markdown (memory/{scope}/PROJECT.md)

설계 원칙:
    - Inspectable/Portable: 사람이 읽고 편집 가능한 평문 파일
    - git 버전관리 가능
    - 원자적 쓰기: 임시 파일 → rename (부분 쓰기 방지)
    - JSON 메타데이터와 MD 본문 분리 저장
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import structlog

from aria.core.exceptions import (
    MemoryNotFoundError,
    MemoryScopeError,
    MemoryStorageError,
)
from aria.memory.storage_adapter import StorageAdapter
from aria.memory.types import (
    MemoryIndex,
    TopicFile,
    validate_scope,
)

logger = structlog.get_logger()


class FileStorageAdapter(StorageAdapter):
    """파일 기반 메모리 저장소 구현

    디렉터리 구조:
        {base_path}/
        ├── global/
        │   ├── index.json
        │   ├── PROJECT.md
        │   └── topics/
        │       ├── {domain}.md       (본문)
        │       └── {domain}.meta.json (메타데이터)
        ├── testorum/
        │   ├── index.json
        │   └── topics/
        └── ...

    Args:
        base_path: 메모리 파일 루트 경로

    Raises:
        MemoryStorageError: 기본 디렉터리 생성 실패
    """

    def __init__(self, base_path: str) -> None:
        self._base_path = Path(base_path)
        try:
            self._base_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise MemoryStorageError(
                f"메모리 기본 디렉터리 생성 실패: {base_path} — {e}"
            ) from e
        logger.debug("file_storage_initialized", base_path=str(self._base_path))

    def _scope_path(self, scope: str) -> Path:
        """스코프 디렉터리 경로"""
        return self._base_path / scope

    def _index_path(self, scope: str) -> Path:
        """인덱스 파일 경로"""
        return self._scope_path(scope) / "index.json"

    def _topics_dir(self, scope: str) -> Path:
        """토픽 디렉터리 경로"""
        return self._scope_path(scope) / "topics"

    def _topic_content_path(self, scope: str, domain: str) -> Path:
        """토픽 본문 파일 경로 (.md)"""
        return self._topics_dir(scope) / f"{domain}.md"

    def _topic_meta_path(self, scope: str, domain: str) -> Path:
        """토픽 메타데이터 파일 경로 (.meta.json)"""
        return self._topics_dir(scope) / f"{domain}.meta.json"

    def _project_md_path(self, scope: str) -> Path:
        """PROJECT.md 경로"""
        return self._scope_path(scope) / "PROJECT.md"

    def _validate_scope(self, scope: str) -> None:
        """스코프 유효성 검증"""
        try:
            validate_scope(scope)
        except ValueError as e:
            raise MemoryScopeError(scope) from e

    def _atomic_write(self, path: Path, content: str) -> None:
        """원자적 파일 쓰기 (임시 파일 → rename)

        부분 쓰기 방지: 프로세스가 중단되어도 파일이 깨지지 않음

        Args:
            path: 대상 파일 경로
            content: 쓸 내용

        Raises:
            MemoryStorageError: 쓰기 실패
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # 같은 파일시스템에 임시 파일 생성 (rename 원자성 보장)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent),
                prefix=f".{path.stem}_",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, str(path))
            except Exception:
                # 실패 시 임시 파일 정리
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except MemoryStorageError:
            raise
        except OSError as e:
            raise MemoryStorageError(
                f"파일 쓰기 실패: {path} — {e}"
            ) from e

    def read_index(self, scope: str) -> MemoryIndex:
        """인덱스 읽기 — 파일 미존재 시 빈 인덱스 반환"""
        self._validate_scope(scope)

        index_path = self._index_path(scope)
        if not index_path.exists():
            logger.debug("index_not_found_creating_empty", scope=scope)
            return MemoryIndex(scope=scope, entries=[])

        try:
            raw = index_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            return MemoryIndex.model_validate(data)
        except json.JSONDecodeError as e:
            raise MemoryStorageError(
                f"인덱스 JSON 파싱 실패: {index_path} — {e}",
                scope=scope,
            ) from e
        except Exception as e:
            raise MemoryStorageError(
                f"인덱스 읽기 실패: {index_path} — {e}",
                scope=scope,
            ) from e

    def write_index(self, index: MemoryIndex) -> None:
        """인덱스 쓰기 — 원자적 교체"""
        index_path = self._index_path(index.scope)
        data = index.model_dump(mode="json")
        content = json.dumps(data, ensure_ascii=False, indent=2, default=str)
        self._atomic_write(index_path, content)
        logger.debug(
            "index_written",
            scope=index.scope,
            entries_count=len(index.entries),
        )

    def read_topic(self, scope: str, domain: str) -> TopicFile:
        """토픽 읽기 — 메타데이터 JSON + 본문 MD 결합"""
        self._validate_scope(scope)

        meta_path = self._topic_meta_path(scope, domain)
        content_path = self._topic_content_path(scope, domain)

        if not meta_path.exists() or not content_path.exists():
            raise MemoryNotFoundError(scope=scope, domain=domain)

        try:
            meta_raw = meta_path.read_text(encoding="utf-8")
            meta = json.loads(meta_raw)
            content = content_path.read_text(encoding="utf-8")

            return TopicFile(
                domain=meta["domain"],
                scope=meta["scope"],
                content=content,
                version=meta["version"],
                updated_at=meta["updated_at"],
                created_at=meta["created_at"],
            )
        except (json.JSONDecodeError, KeyError) as e:
            raise MemoryStorageError(
                f"토픽 메타데이터 파싱 실패: {meta_path} — {e}",
                scope=scope,
                domain=domain,
            ) from e
        except Exception as e:
            raise MemoryStorageError(
                f"토픽 읽기 실패: {domain} (scope={scope}) — {e}",
                scope=scope,
                domain=domain,
            ) from e

    def write_topic(self, topic: TopicFile) -> None:
        """토픽 쓰기 — 메타데이터 JSON + 본문 MD 분리 저장

        Strict Write Discipline:
            1. 본문(.md) 먼저 쓰기
            2. 메타데이터(.meta.json) 쓰기
            → 이후 caller가 인덱스 업데이트
        """
        meta_path = self._topic_meta_path(topic.scope, topic.domain)
        content_path = self._topic_content_path(topic.scope, topic.domain)

        meta = {
            "domain": topic.domain,
            "scope": topic.scope,
            "version": topic.version,
            "updated_at": topic.updated_at.isoformat(),
            "created_at": topic.created_at.isoformat(),
        }

        # 1. 본문 먼저 (더 큰 파일 → 실패 가능성 높은 작업 먼저)
        self._atomic_write(content_path, topic.content)
        # 2. 메타데이터
        meta_json = json.dumps(meta, ensure_ascii=False, indent=2)
        self._atomic_write(meta_path, meta_json)

        logger.debug(
            "topic_written",
            scope=topic.scope,
            domain=topic.domain,
            version=topic.version,
        )

    def delete_topic(self, scope: str, domain: str) -> None:
        """토픽 삭제 — 본문 + 메타데이터 모두 삭제"""
        self._validate_scope(scope)

        meta_path = self._topic_meta_path(scope, domain)
        content_path = self._topic_content_path(scope, domain)

        if not meta_path.exists() and not content_path.exists():
            raise MemoryNotFoundError(scope=scope, domain=domain)

        try:
            if content_path.exists():
                content_path.unlink()
            if meta_path.exists():
                meta_path.unlink()
        except OSError as e:
            raise MemoryStorageError(
                f"토픽 삭제 실패: {domain} (scope={scope}) — {e}",
                scope=scope,
                domain=domain,
            ) from e

        logger.debug("topic_deleted", scope=scope, domain=domain)

    def topic_exists(self, scope: str, domain: str) -> bool:
        """토픽 존재 여부 확인"""
        meta_path = self._topic_meta_path(scope, domain)
        content_path = self._topic_content_path(scope, domain)
        return meta_path.exists() and content_path.exists()

    def ensure_scope_directory(self, scope: str) -> None:
        """스코프 디렉터리 + topics 하위 디렉터리 생성"""
        self._validate_scope(scope)
        try:
            self._topics_dir(scope).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise MemoryStorageError(
                f"스코프 디렉터리 생성 실패: {scope} — {e}",
                scope=scope,
            ) from e

    def read_static_config(self, scope: str) -> str | None:
        """Layer 3 정적 설정 (PROJECT.md) 읽기"""
        self._validate_scope(scope)
        project_path = self._project_md_path(scope)
        if not project_path.exists():
            return None
        try:
            return project_path.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning(
                "project_md_read_failed",
                scope=scope,
                error=str(e),
            )
            return None
