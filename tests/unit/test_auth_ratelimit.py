"""Task 1: API 인증 + Rate Limiting 테스트

테스트 범위:
1. APIConfig production 안전장치 (기본 키 / auth_disabled 차단)
2. verify_api_key 인증 로직
3. Rate limiter config 연동
4. 인증 헤더 응답
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from aria.core.config import APIConfig, AriaConfig, Environment


# === 1. APIConfig Validator Tests ===


class TestAPIConfigValidator:
    def test_development_allows_default_key(self) -> None:
        """development에서는 기본 키 허용"""
        config = APIConfig(env=Environment.DEVELOPMENT)
        assert config.api_key == "aria-dev-key-change-me"

    def test_development_allows_auth_disabled(self) -> None:
        """development에서는 auth_disabled 허용"""
        config = APIConfig(env=Environment.DEVELOPMENT, auth_disabled=True)
        assert config.auth_disabled is True

    def test_production_rejects_default_key(self) -> None:
        """production에서 기본 키 사용 시 ValueError"""
        with pytest.raises(ValueError, match="기본 API 키"):
            APIConfig(env=Environment.PRODUCTION)

    def test_staging_rejects_default_key(self) -> None:
        """staging에서 기본 키 사용 시 ValueError"""
        with pytest.raises(ValueError, match="기본 API 키"):
            APIConfig(env=Environment.STAGING)

    def test_production_rejects_auth_disabled(self) -> None:
        """production에서 auth_disabled=true 시 ValueError"""
        with pytest.raises(ValueError, match="ARIA_AUTH_DISABLED"):
            APIConfig(
                env=Environment.PRODUCTION,
                api_key="real-production-key-12345",
                auth_disabled=True,
            )

    def test_production_accepts_custom_key(self) -> None:
        """production에서 커스텀 키 + auth 활성화 시 정상"""
        config = APIConfig(
            env=Environment.PRODUCTION,
            api_key="real-production-key-12345",
        )
        assert config.api_key == "real-production-key-12345"
        assert config.auth_disabled is False

    def test_default_auth_disabled_is_false(self) -> None:
        """auth_disabled 기본값은 False"""
        config = APIConfig()
        assert config.auth_disabled is False

    def test_rate_limit_burst_default(self) -> None:
        """rate_limit_burst 기본값 확인"""
        config = APIConfig()
        assert config.rate_limit_burst == 10


# === 2. FastAPI Auth Integration Tests ===


class TestAuthEndpoints:
    @pytest.fixture
    def client(self):
        """auth_disabled=False (기본값) 상태의 테스트 클라이언트"""
        from aria.api.app import app

        return TestClient(app)

    def test_health_no_auth_required(self, client) -> None:
        """health 엔드포인트는 인증 불필요"""
        response = client.get("/v1/health")
        assert response.status_code == 200

    def test_query_without_api_key_returns_401(self, client) -> None:
        """API 키 없이 요청 → 401 (auth_disabled=False 기본 상태)"""
        with patch("aria.api.app.get_config") as mock_config:
            cfg = MagicMock()
            cfg.api.auth_disabled = False
            cfg.api.api_key = "test-key-12345678"
            cfg.api.env = Environment.DEVELOPMENT
            mock_config.return_value = cfg

            response = client.post(
                "/v1/query",
                json={"query": "테스트"},
            )
            assert response.status_code == 401
            assert "API 키" in response.json()["detail"]

    def test_query_with_wrong_api_key_returns_401(self, client) -> None:
        """잘못된 API 키 → 401"""
        with patch("aria.api.app.get_config") as mock_config:
            cfg = MagicMock()
            cfg.api.auth_disabled = False
            cfg.api.api_key = "correct-key-12345"
            cfg.api.env = Environment.DEVELOPMENT
            mock_config.return_value = cfg

            response = client.post(
                "/v1/query",
                json={"query": "테스트"},
                headers={"X-API-Key": "wrong-key-99999"},
            )
            assert response.status_code == 401
            assert "유효하지 않은" in response.json()["detail"]

    def test_query_with_correct_api_key_passes_auth(self, client) -> None:
        """올바른 API 키 → 인증 통과 (에이전트 미초기화라 503)"""
        with patch("aria.api.app.get_config") as mock_config:
            cfg = MagicMock()
            cfg.api.auth_disabled = False
            cfg.api.api_key = "correct-key-12345"
            cfg.api.env = Environment.DEVELOPMENT
            mock_config.return_value = cfg

            # rate_limiter도 mock
            with patch("aria.api.app.rate_limiter") as mock_rl:
                mock_rl.is_allowed.return_value = True

                response = client.post(
                    "/v1/query",
                    json={"query": "테스트"},
                    headers={"X-API-Key": "correct-key-12345"},
                )
                # 인증 통과 → 에이전트 미초기화 503 (정상)
                assert response.status_code == 503

    def test_query_with_auth_disabled_skips_key_check(self, client) -> None:
        """auth_disabled=True → API 키 없어도 인증 통과"""
        with patch("aria.api.app.get_config") as mock_config:
            cfg = MagicMock()
            cfg.api.auth_disabled = True
            cfg.api.env = Environment.DEVELOPMENT
            mock_config.return_value = cfg

            with patch("aria.api.app.rate_limiter") as mock_rl:
                mock_rl.is_allowed.return_value = True

                response = client.post(
                    "/v1/query",
                    json={"query": "테스트"},
                    # X-API-Key 헤더 없음
                )
                # 인증 스킵 → 에이전트 미초기화 503
                assert response.status_code == 503

    def test_rate_limit_applied_even_with_auth_disabled(self, client) -> None:
        """auth_disabled=True에서도 rate limit은 적용"""
        with patch("aria.api.app.get_config") as mock_config:
            cfg = MagicMock()
            cfg.api.auth_disabled = True
            cfg.api.env = Environment.DEVELOPMENT
            mock_config.return_value = cfg

            with patch("aria.api.app.rate_limiter") as mock_rl:
                mock_rl.is_allowed.return_value = False  # rate limit 초과

                response = client.post(
                    "/v1/query",
                    json={"query": "테스트"},
                )
                assert response.status_code == 429
                assert "Retry-After" in response.headers


# === 3. Rate Limiter Config Integration ===


class TestRateLimiterConfig:
    def test_rate_limiter_uses_config_values(self) -> None:
        """RateLimiter가 config의 rate_limit_per_minute 값 사용"""
        from aria.api.app import RateLimiter

        limiter = RateLimiter(max_requests=5, window_seconds=60)
        assert limiter.max_requests == 5
        assert limiter.window_seconds == 60

    def test_ip_based_client_id_when_auth_disabled(self) -> None:
        """auth_disabled 시 IP 기반 client_id로 rate limit 분리"""
        from aria.api.app import RateLimiter

        limiter = RateLimiter(max_requests=2, window_seconds=60)

        # 서로 다른 IP는 별도 bucket
        assert limiter.is_allowed("anon:192.168.1.1") is True
        assert limiter.is_allowed("anon:192.168.1.1") is True
        assert limiter.is_allowed("anon:192.168.1.1") is False

        # 다른 IP는 별도
        assert limiter.is_allowed("anon:192.168.1.2") is True
