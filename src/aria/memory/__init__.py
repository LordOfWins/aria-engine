"""ARIA Engine - 3계층 메모리 아키텍처

Layer 1: In-Context Memory (런타임 토큰 예산 관리)
Layer 2: Index + Topic Files (파일 기반 포인터 인덱스 + 도메인별 지식)
Layer 3: Static Config (PROJECT.md 프로젝트 헌법)

설계 원칙:
    - "directory, not diary" — 인덱스는 포인터만 저장
    - Strict Write Discipline — Topic 파일 쓰기 성공 후에만 인덱스 업데이트
    - read-before-write — 낙관적 락으로 동시 수정 충돌 방지
    - Inspectable/Portable — 사람이 읽고 편집 가능한 평문 파일
"""

from aria.memory.types import (
    VALID_SCOPES,
    IndexEntry,
    MemoryIndex,
    MemoryLoadRequest,
    MemoryLoadResponse,
    TopicFile,
    TopicResponse,
    TopicUpsertRequest,
)
from aria.memory.storage_adapter import StorageAdapter
from aria.memory.file_storage import FileStorageAdapter
from aria.memory.index_manager import IndexManager
from aria.memory.memory_loader import MemoryLoader, LoadResult, inject_memory_context

__all__ = [
    # Types
    "VALID_SCOPES",
    "IndexEntry",
    "MemoryIndex",
    "MemoryLoadRequest",
    "MemoryLoadResponse",
    "TopicFile",
    "TopicResponse",
    "TopicUpsertRequest",
    # Storage
    "StorageAdapter",
    "FileStorageAdapter",
    # Manager
    "IndexManager",
    # Loader
    "MemoryLoader",
    "LoadResult",
    "inject_memory_context",
]
