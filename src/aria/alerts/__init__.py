"""ARIA Engine - Proactive Alert System

ARIA가 스스로 판단하여 필요할 때만 대표님에게 텔레그램 알림
- 비용 임계치 도달 (70% / 90%)
- KillSwitch 발동
- 연속 에러 발생
- 낮은 confidence 응답
- 메모리 버전 충돌
"""

from aria.alerts.alert_types import Alert, AlertLevel, AlertType
from aria.alerts.alert_manager import AlertManager

__all__ = [
    "Alert",
    "AlertLevel",
    "AlertType",
    "AlertManager",
]
