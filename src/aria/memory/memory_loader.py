"""ARIA Engine - Memory Loader (Layer 1: In-Context Memory)

토큰 예산 내에서 필요한 토픽만 선택 로딩하여 프롬프트에 주입
- TokenBudgetManager: 예산 초과 시 우선순위 낮은 토픽 제외
- ContextInjector: ReAct 에이전트 시스템 프롬프트에 메모리 마크다운 주입

설계 원칙:
    - 토큰 예산 엄수 (기본 4000 토큰)
    - 최신 업데이트 우선 (updated_at 내림차순)
    - 예산 초과 시 graceful degradation (에러 없이 가능한 만큼만 로딩)
"""

from __future__ import annotations

from datetime import datetime

import structlog

from aria.memory.index_manager import IndexManager, estimate_tokens
from aria.memory.types import IndexEntry, MemoryIndex

logger = structlog.get_logger()


class MemoryLoader:
    """Layer 1 메모리 로더

    토큰 예산 내에서 토픽을 선택적으로 로딩하고
    프롬프트 주입용 마크다운으로 렌더링

    Args:
        index_manager: 인덱스 매니저
        default_token_budget: 기본 토큰 예산
    """

    def __init__(
        self,
        index_manager: IndexManager,
        default_token_budget: int = 4000,
    ) -> None:
        self._manager = index_manager
        self._default_budget = default_token_budget

    def load(
        self,
        scope: str,
        domains: list[str] | None = None,
        token_budget: int | None = None,
    ) -> LoadResult:
        """메모리 로딩

        Args:
            scope: 스코프 식별자
            domains: 로딩할 도메인 목록 (None이면 전체 인덱스 기반)
            token_budget: 토큰 예산 (None이면 기본값)

        Returns:
            LoadResult (로딩된 도메인 + 마크다운 + 토큰 정보)
        """
        budget = token_budget or self._default_budget
        index = self._manager.get_index(scope)

        # 대상 엔트리 결정
        if domains:
            target_entries = [e for e in index.entries if e.domain in domains]
        else:
            target_entries = list(index.entries)

        # 우선순위 정렬: 최신 업데이트 우선
        target_entries.sort(key=lambda e: e.updated_at, reverse=True)

        # Layer 3 정적 설정 (PROJECT.md)
        static_config = self._manager._storage.read_static_config(scope)
        static_tokens = 0
        if static_config:
            static_tokens = estimate_tokens(static_config)
            # 정적 설정은 무조건 포함 (예산에서 먼저 차감)
            budget -= static_tokens

        # 토큰 예산 내에서 토픽 선택
        loaded_topics: list[_LoadedTopic] = []
        used_tokens = 0

        for entry in target_entries:
            # 토큰 추정값 확인 (없으면 기본 추정)
            est = entry.token_estimate or 200
            if used_tokens + est > budget:
                logger.debug(
                    "topic_skipped_budget",
                    domain=entry.domain,
                    token_estimate=est,
                    remaining_budget=budget - used_tokens,
                )
                continue

            try:
                topic = self._manager.get_topic(scope, entry.domain)
                actual_tokens = estimate_tokens(topic.content)

                # 실제 토큰이 예산 초과면 스킵
                if used_tokens + actual_tokens > budget:
                    logger.debug(
                        "topic_skipped_actual_tokens",
                        domain=entry.domain,
                        actual_tokens=actual_tokens,
                        remaining_budget=budget - used_tokens,
                    )
                    continue

                loaded_topics.append(
                    _LoadedTopic(
                        domain=entry.domain,
                        summary=entry.summary,
                        content=topic.content,
                        tokens=actual_tokens,
                    )
                )
                used_tokens += actual_tokens
            except Exception as e:
                # 개별 토픽 로딩 실패는 무시 (다른 토픽 계속 로딩)
                logger.warning(
                    "topic_load_failed",
                    domain=entry.domain,
                    error=str(e),
                )
                continue

        # 마크다운 렌더링
        prompt_md = self._render_markdown(
            scope=scope,
            static_config=static_config,
            topics=loaded_topics,
        )

        total_tokens = used_tokens + static_tokens
        budget_total = (token_budget or self._default_budget)

        logger.info(
            "memory_loaded",
            scope=scope,
            domains_loaded=len(loaded_topics),
            total_tokens=total_tokens,
            budget_used=round(total_tokens / budget_total, 3) if budget_total > 0 else 0,
        )

        return LoadResult(
            scope=scope,
            loaded_domains=[t.domain for t in loaded_topics],
            prompt_markdown=prompt_md,
            total_tokens=total_tokens,
            budget_used=min(total_tokens / budget_total, 1.0) if budget_total > 0 else 0.0,
        )

    def _render_markdown(
        self,
        scope: str,
        static_config: str | None,
        topics: list[_LoadedTopic],
    ) -> str:
        """프롬프트 주입용 마크다운 렌더링

        구조:
            ## ARIA Memory Context
            ### Project Configuration (Layer 3)
            {PROJECT.md 내용}
            ### Knowledge Base (Layer 2)
            #### {domain}: {summary}
            {토픽 본문}
        """
        sections: list[str] = []
        sections.append(f"## ARIA Memory Context (scope: {scope})")
        sections.append("")

        # Layer 3: 정적 설정
        if static_config:
            sections.append("### Project Configuration")
            sections.append(static_config.strip())
            sections.append("")

        # Layer 2: 동적 토픽
        if topics:
            sections.append("### Knowledge Base")
            sections.append("")
            for topic in topics:
                sections.append(f"#### {topic.domain}: {topic.summary}")
                sections.append(topic.content.strip())
                sections.append("")

        if not static_config and not topics:
            sections.append("_메모리에 저장된 정보가 없습니다._")
            sections.append("")

        return "\n".join(sections)


class _LoadedTopic:
    """내부용 로딩된 토픽 데이터"""

    __slots__ = ("domain", "summary", "content", "tokens")

    def __init__(self, domain: str, summary: str, content: str, tokens: int) -> None:
        self.domain = domain
        self.summary = summary
        self.content = content
        self.tokens = tokens


class LoadResult:
    """메모리 로딩 결과

    Attributes:
        scope: 로딩된 스코프
        loaded_domains: 실제 로딩된 도메인 목록
        prompt_markdown: 프롬프트 주입용 마크다운
        total_tokens: 총 토큰 수 추정
        budget_used: 예산 대비 사용 비율 (0.0~1.0)
    """

    __slots__ = ("scope", "loaded_domains", "prompt_markdown", "total_tokens", "budget_used")

    def __init__(
        self,
        scope: str,
        loaded_domains: list[str],
        prompt_markdown: str,
        total_tokens: int,
        budget_used: float,
    ) -> None:
        self.scope = scope
        self.loaded_domains = loaded_domains
        self.prompt_markdown = prompt_markdown
        self.total_tokens = total_tokens
        self.budget_used = budget_used


def inject_memory_context(
    system_prompt: str,
    memory_markdown: str,
) -> str:
    """시스템 프롬프트에 메모리 컨텍스트 주입

    메모리를 hint로 취급 — 실제 코드/데이터에서 검증하라는 안내 포함

    Args:
        system_prompt: 기존 시스템 프롬프트
        memory_markdown: 메모리 로딩 결과 마크다운

    Returns:
        메모리가 주입된 시스템 프롬프트
    """
    memory_section = (
        "\n\n---\n"
        "아래는 이전 세션에서 축적된 메모리입니다. "
        "이 정보는 hint로 취급하세요 — 실제 코드베이스나 데이터와 충돌하면 "
        "실제 소스를 우선하세요.\n\n"
        f"{memory_markdown}\n"
        "---\n"
    )

    return system_prompt + memory_section
