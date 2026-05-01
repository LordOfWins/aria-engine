"""ARIA Engine - Telegram Notifier

ARIA가 능동적으로 텔레그램에 메시지를 발송하는 유틸
- 매일 아침 브리핑 (Phase 4)
- HITL 확인 요청 (인라인 키보드)
- TrendBot 결과 알림
- 이상 탐지 알림

다른 모듈에서 import하여 사용:
    from aria.telegram.notifier import send_message, send_confirmation
    await send_message("오늘의 브리핑입니다...")
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog

logger = structlog.get_logger()

# 텔레그램 Bot API 기본 URL
_TG_API_BASE = "https://api.telegram.org/bot{token}"

# 메시지 길이 제한 (텔레그램 API 한도)
_MAX_MESSAGE_LENGTH = 4096


async def send_message(
    bot_token: str,
    chat_id: str,
    text: str,
    *,
    parse_mode: str = "Markdown",
    disable_preview: bool = True,
) -> dict[str, Any]:
    """텔레그램 메시지 발송

    Args:
        bot_token: 봇 토큰
        chat_id: 채팅 ID
        text: 메시지 텍스트
        parse_mode: 포맷 ("Markdown" / "HTML" / None)
        disable_preview: 링크 미리보기 비활성화

    Returns:
        텔레그램 API 응답 또는 에러 dict
    """
    if not bot_token or not chat_id:
        logger.error("notifier_missing_config", has_token=bool(bot_token), has_chat_id=bool(chat_id))
        return {"ok": False, "error": "봇 토큰 또는 채팅 ID 미설정"}

    # 긴 메시지 분할
    chunks = _split_message(text)

    results = []
    for chunk in chunks:
        result = await _send_single(
            bot_token, chat_id, chunk,
            parse_mode=parse_mode,
            disable_preview=disable_preview,
        )
        results.append(result)
        if not result.get("ok"):
            # parse_mode 실패 시 plain text로 재시도
            if parse_mode:
                logger.warning("notifier_parse_mode_failed", parse_mode=parse_mode)
                result = await _send_single(
                    bot_token, chat_id, chunk,
                    parse_mode=None,
                    disable_preview=disable_preview,
                )
                results[-1] = result

    return results[0] if len(results) == 1 else {"ok": True, "chunks_sent": len(results)}


async def send_confirmation(
    bot_token: str,
    chat_id: str,
    confirmation_id: str,
    tool_name: str,
    description: str,
) -> dict[str, Any]:
    """HITL 확인 요청 — 인라인 키보드 전송

    Args:
        confirmation_id: 확인 ID
        tool_name: 도구 이름
        description: 실행 내용 설명

    Returns:
        텔레그램 API 응답
    """
    text = (
        f"🔐 *도구 실행 확인 요청*\n\n"
        f"도구: `{tool_name}`\n"
        f"내용: {description}\n\n"
        f"실행하시겠습니까?"
    )

    # 인라인 키보드 (승인/거부)
    reply_markup = {
        "inline_keyboard": [
            [
                {"text": "✅ 승인", "callback_data": f"confirm:{confirmation_id}:approve"},
                {"text": "❌ 거부", "callback_data": f"confirm:{confirmation_id}:deny"},
            ]
        ]
    }

    url = f"{_TG_API_BASE.format(token=bot_token)}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "reply_markup": reply_markup,
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, json=payload)
            data = response.json()
            if not data.get("ok"):
                logger.error("notifier_confirmation_failed", error=data.get("description", ""))
            return data
    except Exception as e:
        logger.error("notifier_confirmation_error", error=str(e)[:200])
        return {"ok": False, "error": str(e)[:300]}


def _split_message(text: str) -> list[str]:
    """긴 메시지를 텔레그램 한도(4096자) 내로 분할"""
    if len(text) <= _MAX_MESSAGE_LENGTH:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= _MAX_MESSAGE_LENGTH:
            chunks.append(text)
            break

        # 줄바꿈 기준으로 자르기
        cut_at = text.rfind("\n", 0, _MAX_MESSAGE_LENGTH)
        if cut_at <= 0:
            cut_at = _MAX_MESSAGE_LENGTH

        chunks.append(text[:cut_at])
        text = text[cut_at:].lstrip("\n")

    return chunks


async def _send_single(
    bot_token: str,
    chat_id: str,
    text: str,
    *,
    parse_mode: str | None = "Markdown",
    disable_preview: bool = True,
) -> dict[str, Any]:
    """단일 메시지 발송"""
    url = f"{_TG_API_BASE.format(token=bot_token)}/sendMessage"
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": disable_preview,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, json=payload)
            data = response.json()
            if not data.get("ok"):
                logger.error(
                    "notifier_send_failed",
                    status=response.status_code,
                    error=data.get("description", "")[:200],
                )
            return data
    except Exception as e:
        logger.error("notifier_send_error", error=str(e)[:200])
        return {"ok": False, "error": str(e)[:300]}
