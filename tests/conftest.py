"""ARIA Engine 테스트 공통 설정

config.py는 모듈 로드 시 ARIA_ENV_FILE 환경변수를 읽어 _ENV_FILE을 결정함
테스트에서는 conftest.py가 모듈 로드보다 먼저 실행되므로
여기서 ARIA_ENV_FILE=""을 설정하면 모든 Config 클래스가 .env 파일을 읽지 않음

주의: conftest.py는 pytest 수집 단계에서 실행되므로
      config.py의 import보다 반드시 먼저 실행됨
"""

from __future__ import annotations

import os

# config.py 모듈이 import되기 전에 설정해야 함
os.environ["ARIA_ENV_FILE"] = ""

import pytest

_ENV_VARS_TO_CLEAR = [
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "DEEPSEEK_API_KEY",
    "ARIA_DEFAULT_MODEL",
    "ARIA_FALLBACK_MODEL",
    "ARIA_CHEAP_MODEL",
    "ARIA_EMBEDDING_MODEL",
    "ARIA_HOST",
    "ARIA_PORT",
    "ARIA_ENV",
    "ARIA_LOG_LEVEL",
    "ARIA_API_KEY",
    "ARIA_AUTH_DISABLED",
    "ARIA_RATE_LIMIT_PER_MINUTE",
    "ARIA_RATE_LIMIT_BURST",
    "ARIA_MAX_TOKENS_PER_REQUEST",
    "ARIA_DAILY_COST_LIMIT_USD",
    "ARIA_MONTHLY_COST_LIMIT_USD",
    "QDRANT_HOST",
    "QDRANT_PORT",
    "QDRANT_API_KEY",
    "QDRANT_URL",
]


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """테스트 실행 전 모든 ARIA/API 키 환경변수 클리어

    ARIA_ENV_FILE=""은 conftest.py 최상단에서 이미 설정됨 (.env 파일 읽기 차단)
    이 fixture는 os.environ에 남아있을 수 있는 환경변수를 추가로 클리어
    """
    for var in _ENV_VARS_TO_CLEAR:
        monkeypatch.delenv(var, raising=False)
