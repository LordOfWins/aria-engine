#!/usr/bin/env python3
"""ARIA Telegram Bot 실행 스크립트

ARIA 서버(run.py)와 별도로 실행:
    # 1. ARIA 서버 시작
    python run.py

    # 2. 텔레그램 봇 시작 (별도 터미널)
    python run_telegram.py

필수 환경변수 (.env):
    ARIA_TELEGRAM_BOT_TOKEN=your-bot-token
    ARIA_TELEGRAM_CHAT_ID=your-chat-id
    ARIA_TELEGRAM_ARIA_API_KEY=your-aria-api-key

선택 환경변수:
    ARIA_TELEGRAM_ARIA_BASE_URL=http://localhost:8100  (기본값)
    ARIA_TELEGRAM_DEFAULT_SCOPE=global  (기본값)
    ARIA_TELEGRAM_DEFAULT_COLLECTION=default  (기본값)
    ARIA_TELEGRAM_REQUEST_TIMEOUT=120  (기본값)
"""

from __future__ import annotations

import sys

import structlog

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
)

logger = structlog.get_logger()


def main() -> None:
    from aria.core.config import TelegramConfig
    from aria.telegram.bot import run_bot

    config = TelegramConfig()

    if not config.bot_token:
        logger.error("missing_bot_token", hint="ARIA_TELEGRAM_BOT_TOKEN을 .env에 설정하세요")
        sys.exit(1)

    if not config.chat_id:
        logger.error("missing_chat_id", hint="ARIA_TELEGRAM_CHAT_ID를 .env에 설정하세요")
        sys.exit(1)

    logger.info(
        "aria_telegram_bot_config",
        aria_url=config.aria_base_url,
        chat_id=config.chat_id[:4] + "****",
        scope=config.default_scope,
        timeout=config.request_timeout,
    )

    run_bot(config)


if __name__ == "__main__":
    main()
