"""ARIA Engine - Telegram Integration

텔레그램 양방향 통합 모듈
- 명령 수신: 텍스트 메시지 → ARIA 에이전트 호출 → 응답 전송
- 능동 보고: ARIA → 텔레그램 발송 (알림/브리핑)
- HITL: 도구 실행 확인 요청 → 인라인 키보드 승인/거부
"""

from aria.telegram.bot import create_bot, run_bot
from aria.telegram.client import ARIAClient
from aria.telegram.notifier import send_message, send_confirmation

__all__ = [
    "ARIAClient",
    "create_bot",
    "run_bot",
    "send_confirmation",
    "send_message",
]
