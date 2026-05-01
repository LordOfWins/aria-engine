"""ARIA Engine - Daily Briefing

매일 아침 텔레그램으로 능동 보고 전송
- ARIA 비용 현황 (일/월)
- 메모리 인덱스 요약 (스코프별 토픽 수)
- 서버 상태

python-telegram-bot의 JobQueue(APScheduler)를 활용하여
별도 의존성 없이 스케줄링

스케줄링: KST 06:00 = UTC 21:00 (전날)
"""

from __future__ import annotations

from datetime import time, timezone, timedelta
from typing import Any

import structlog
from telegram.ext import Application, ContextTypes

from aria.telegram.client import ARIAClient
from aria.telegram.notifier import send_message

logger = structlog.get_logger()

# KST = UTC+9
KST = timezone(timedelta(hours=9))


async def build_briefing(client: ARIAClient) -> str:
    """브리핑 메시지 생성

    ARIA API에서 비용/메모리/헬스 데이터를 수집하여
    텔레그램용 Markdown 브리핑 텍스트 반환

    Args:
        client: ARIAClient 인스턴스

    Returns:
        브리핑 텍스트 (Markdown)
    """
    sections: list[str] = ["☀️ *ARIA 모닝 브리핑*\n"]

    # 1. 서버 상태
    health = await client.health_check()
    if "error" in health:
        sections.append("🔴 *서버 상태*: 연결 실패")
    else:
        status = health.get("status", "unknown")
        version = health.get("version", "?")
        emoji = "🟢" if status == "ok" else "🟡"
        sections.append(f"{emoji} *서버*: ARIA {version} — {status}")

    # 2. 비용 현황
    cost = await client.get_cost()
    if "error" not in cost:
        daily = cost.get("daily_cost_usd", 0)
        monthly = cost.get("monthly_cost_usd", 0)
        daily_limit = cost.get("daily_limit_usd", 0)
        monthly_limit = cost.get("monthly_limit_usd", 0)
        total_req = cost.get("total_requests", 0)
        cached = cost.get("total_cached_tokens", 0)

        sections.append(
            f"\n💰 *비용*\n"
            f"  어제: ${daily:.4f} / ${daily_limit}\n"
            f"  이번달: ${monthly:.4f} / ${monthly_limit}\n"
            f"  총 요청: {total_req}회 | 캐시: {cached:,}tok"
        )

    # 3. 메모리 요약 (주요 스코프)
    scopes = ["global", "testorum", "talksim", "autotube"]
    memory_lines: list[str] = []
    for scope in scopes:
        idx = await client.get_memory_index(scope)
        if "error" not in idx:
            entries = idx.get("entries", [])
            if entries:
                memory_lines.append(f"  {scope}: {len(entries)}개 토픽")

    if memory_lines:
        sections.append(f"\n🧠 *메모리*\n" + "\n".join(memory_lines))

    # 4. 마무리
    sections.append("\n_좋은 하루 되세요!_ 🚀")

    return "\n".join(sections)


async def send_daily_briefing(
    bot_token: str,
    chat_id: str,
    client: ARIAClient,
) -> dict[str, Any]:
    """브리핑 생성 후 텔레그램 전송

    Args:
        bot_token: 텔레그램 봇 토큰
        chat_id: 수신자 채팅 ID
        client: ARIAClient 인스턴스

    Returns:
        텔레그램 API 응답
    """
    try:
        text = await build_briefing(client)
    except Exception as e:
        logger.error("briefing_build_failed", error=str(e)[:200])
        text = f"⚠️ 브리핑 생성 실패: {str(e)[:200]}"

    logger.info("briefing_sending", text_length=len(text))
    result = await send_message(bot_token, chat_id, text)
    logger.info("briefing_sent", ok=result.get("ok", False))
    return result


async def _briefing_job_callback(context: ContextTypes.DEFAULT_TYPE) -> None:
    """JobQueue 콜백 — 스케줄된 시간에 실행"""
    job_data = context.job.data or {}
    bot_token: str = job_data.get("bot_token", "")
    chat_id: str = job_data.get("chat_id", "")
    client: ARIAClient | None = job_data.get("client")

    if not bot_token or not chat_id or client is None:
        logger.error("briefing_job_missing_data")
        return

    await send_daily_briefing(bot_token, chat_id, client)


def schedule_daily_briefing(
    app: Application,
    bot_token: str,
    chat_id: str,
    client: ARIAClient,
    hour: int = 6,
    minute: int = 0,
) -> None:
    """매일 브리핑 스케줄 등록

    python-telegram-bot의 JobQueue 사용 (APScheduler 내장)

    Args:
        app: telegram Application 인스턴스
        bot_token: 봇 토큰
        chat_id: 수신자 채팅 ID
        client: ARIAClient 인스턴스
        hour: 브리핑 시간 (KST 기준 / 기본 06시)
        minute: 브리핑 분 (기본 0분)
    """
    job_queue = app.job_queue
    if job_queue is None:
        logger.error("briefing_schedule_failed", reason="JobQueue not available")
        return

    briefing_time = time(hour=hour, minute=minute, tzinfo=KST)

    job_queue.run_daily(
        _briefing_job_callback,
        time=briefing_time,
        name="daily_briefing",
        data={
            "bot_token": bot_token,
            "chat_id": chat_id,
            "client": client,
        },
    )

    logger.info(
        "briefing_scheduled",
        time_kst=f"{hour:02d}:{minute:02d}",
        timezone="KST (UTC+9)",
    )
