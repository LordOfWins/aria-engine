"""ARIA Engine - Memory Storage Adapter (Abstract Base Class)

메모리 저장소의 추상 인터페이스 정의
구현체가 파일 기반 / DB 기반 / 클라우드 기반 등 자유롭게 교체 가능

설계 원칙:
    - Strict Write Discipline: Topic 파일 쓰기 성공 후에만 인덱스 업데이트
    - read-before-write: 덮어쓰기 전 반드시 현재 버전 확인
    - 구현체에 독립적인 인터페이스
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from aria.memory.types import MemoryIndex, TopicFile


class StorageAdapter(ABC):
    """메모리 저장소 추상 인터페이스

    모든 메모리 저장소 구현체는 이 인터페이스를 상속해야 함
    파일 기반(FileStorageAdapter) / 추후 DB 기반 구현체 교체 가능

    Raises:
        MemoryStorageError: 저장소 I/O 실패
        MemoryNotFoundError: 요청한 리소스 미존재
        VersionConflictError: 낙관적 락 충돌
        MemoryScopeError: 유효하지 않은 스코프
    """

    @abstractmethod
    def read_index(self, scope: str) -> MemoryIndex:
        """인덱스 읽기

        Args:
            scope: 스코프 식별자 (global / testorum / talksim / autotube)

        Returns:
            MemoryIndex (파일 미존재 시 빈 인덱스 반환)

        Raises:
            MemoryStorageError: 읽기 실패 (파싱 에러 등)
            MemoryScopeError: 유효하지 않은 스코프
        """
        ...

    @abstractmethod
    def write_index(self, index: MemoryIndex) -> None:
        """인덱스 쓰기 (원자적 교체)

        Args:
            index: 저장할 인덱스

        Raises:
            MemoryStorageError: 쓰기 실패
        """
        ...

    @abstractmethod
    def read_topic(self, scope: str, domain: str) -> TopicFile:
        """토픽 파일 읽기

        Args:
            scope: 스코프 식별자
            domain: 도메인 식별자

        Returns:
            TopicFile

        Raises:
            MemoryNotFoundError: 토픽 파일 미존재
            MemoryStorageError: 읽기 실패
        """
        ...

    @abstractmethod
    def write_topic(self, topic: TopicFile) -> None:
        """토픽 파일 쓰기

        Args:
            topic: 저장할 토픽

        Raises:
            MemoryStorageError: 쓰기 실패
        """
        ...

    @abstractmethod
    def delete_topic(self, scope: str, domain: str) -> None:
        """토픽 파일 삭제

        Args:
            scope: 스코프 식별자
            domain: 도메인 식별자

        Raises:
            MemoryNotFoundError: 토픽 파일 미존재
            MemoryStorageError: 삭제 실패
        """
        ...

    @abstractmethod
    def topic_exists(self, scope: str, domain: str) -> bool:
        """토픽 파일 존재 여부 확인

        Args:
            scope: 스코프 식별자
            domain: 도메인 식별자

        Returns:
            존재하면 True
        """
        ...

    @abstractmethod
    def ensure_scope_directory(self, scope: str) -> None:
        """스코프 디렉터리 생성 (없으면)

        Args:
            scope: 스코프 식별자

        Raises:
            MemoryStorageError: 디렉터리 생성 실패
        """
        ...

    @abstractmethod
    def read_static_config(self, scope: str) -> str | None:
        """Layer 3 정적 설정 (PROJECT.md) 읽기

        Args:
            scope: 스코프 식별자

        Returns:
            PROJECT.md 내용 또는 None (미존재)
        """
        ...
