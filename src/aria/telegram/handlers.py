"""ARIA Engine - Telegram Handlers

텔레그램 봇 메시지/명령어 핸들러
- /start, /help — 안내
- /cost — ARIA 비용 현황
- /memory — 메모리 인덱스 조회
- /health — ARIA 서버 상태
- 일반 텍스트 → ARIA /v1/query 호출 → 응답 전송

인증: CHAT_ID 기반 (승재 전용 — 다른 사용자 차단)
"""

from __future__ import annotations

from typing import Any

import structlog
from telegram import Update
from telegram.ext import ContextTypes

from aria.telegram.client import ARIAClient

logger = structlog.get_logger()


class ARIAHandlers:
    """텔레그램 핸들러 모음

    Args:
        aria_client: ARIA API 클라이언트
        allowed_chat_id: 허용된 채팅 ID (승재 전용)
        default_scope: 기본 메모리 스코프
        default_collection: 기본 검색 컬렉션
    """

    def __init__(
        self,
        aria_client: ARIAClient,
        allowed_chat_id: str,
        default_scope: str = "global",
        default_collection: str = "default",
    ) -> None:
        self.client = aria_client
        self.allowed_chat_id = allowed_chat_id
        self.default_scope = default_scope
        self.default_collection = default_collection

    def _is_authorized(self, update: Update) -> bool:
        """채팅 ID 기반 인증"""
        if not update.effective_chat:
            return False
        chat_id = str(update.effective_chat.id)
        if chat_id != self.allowed_chat_id:
            logger.warning(
                "telegram_unauthorized",
                chat_id=chat_id,
                allowed=self.allowed_chat_id,
            )
            return False
        return True

    # === 명령어 핸들러 ===

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/start 명령어"""
        if not self._is_authorized(update):
            if update.message:
                await update.message.reply_text("⛔ 접근 권한이 없습니다.")
            return

        await update.message.reply_text(
            "🧠 *ARIA Engine 준비 완료*\n\n"
            "사용 가능한 명령어:\n"
            "/help — 도움말\n"
            "/cost — API 비용 현황\n"
            "/memory — 메모리 인덱스\n"
            "/health — 서버 상태\n\n"
            "자유롭게 질문하세요!",
            parse_mode="Markdown",
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/help 명령어"""
        if not self._is_authorized(update):
            return

        await update.message.reply_text(
            "🧠 *ARIA 사용 가이드*\n\n"
            "*일반 질문*\n"
            "텍스트를 보내면 ARIA가 추론하여 답변합니다\n"
            "메모리와 지식 베이스를 자동으로 활용합니다\n\n"
            "*명령어*\n"
            "`/cost` — 오늘/이번 달 API 비용\n"
            "`/memory` — 저장된 메모리 토픽 목록\n"
            "`/memory testorum` — 특정 스코프 조회\n"
            "`/health` — ARIA 서버 상태 확인\n\n"
            "*스코프 지정*\n"
            "`@testorum 질문` — Testorum 스코프로 질문\n"
            "`@talksim 질문` — Talksim 스코프로 질문",
            parse_mode="Markdown",
        )

    async def cost(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/cost — API 비용 현황"""
        if not self._is_authorized(update):
            return

        data = await self.client.get_cost()
        if "error" in data:
            await update.message.reply_text(f"❌ 비용 조회 실패: {data['message']}")
            return

        daily = data.get("daily_cost_usd", 0)
        monthly = data.get("monthly_cost_usd", 0)
        daily_limit = data.get("daily_limit_usd", 0)
        monthly_limit = data.get("monthly_limit_usd", 0)
        total_requests = data.get("total_requests", 0)
        cached_tokens = data.get("total_cached_tokens", 0)

        await update.message.reply_text(
            f"💰 *ARIA 비용 현황*\n\n"
            f"오늘: ${daily:.4f} / ${daily_limit}\n"
            f"이번 달: ${monthly:.4f} / ${monthly_limit}\n"
            f"총 요청: {total_requests}회\n"
            f"캐시 토큰: {cached_tokens:,}",
            parse_mode="Markdown",
        )

    async def memory(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/memory [scope] — 메모리 인덱스 조회"""
        if not self._is_authorized(update):
            return

        # /memory testorum → scope=testorum
        scope = self.default_scope
        if context.args:
            scope = context.args[0]

        data = await self.client.get_memory_index(scope)
        if "error" in data:
            await update.message.reply_text(f"❌ 메모리 조회 실패: {data['message']}")
            return

        entries = data.get("entries", [])
        if not entries:
            await update.message.reply_text(f"📭 `{scope}` 스코프에 저장된 메모리가 없습니다.", parse_mode="Markdown")
            return

        lines = [f"📋 *메모리 인덱스* (`{scope}`)\n"]
        for entry in entries:
            domain = entry.get("domain", "?")
            summary = entry.get("summary", "")[:60]
            tokens = entry.get("token_estimate", "?")
            lines.append(f"• `{domain}` — {summary} ({tokens}tok)")

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    async def health(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/health — ARIA 서버 상태"""
        if not self._is_authorized(update):
            return

        data = await self.client.health_check()
        if "error" in data:
            await update.message.reply_text(f"❌ 서버 연결 실패: {data['message']}")
            return

        status = data.get("status", "unknown")
        version = data.get("version", "?")
        emoji = "✅" if status == "ok" else "⚠️"
        await update.message.reply_text(f"{emoji} ARIA {version} — {status}")

    # === 일반 메시지 핸들러 ===

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """일반 텍스트 메시지 → ARIA 질의"""
        if not self._is_authorized(update):
            return
        if not update.message or not update.message.text:
            return

        text = update.message.text.strip()
        if not text:
            return

        # 스코프 지정 (@testorum, @talksim 등)
        scope = self.default_scope
        if text.startswith("@"):
            parts = text.split(" ", 1)
            if len(parts) == 2:
                scope = parts[0][1:]  # @ 제거
                text = parts[1].strip()

        # "생각 중" 표시
        thinking_msg = await update.message.reply_text("🤔 생각 중...")

        # ARIA API 호출
        data = await self.client.query(
            text,
            scope=scope,
            collection=self.default_collection,
        )

        # "생각 중" 메시지 삭제
        try:
            await thinking_msg.delete()
        except Exception:
            pass

        if "error" in data:
            await update.message.reply_text(f"❌ {data['message']}")
            return

        # 응답 포맷
        answer = data.get("answer") or "응답을 생성하지 못했습니다."
        tool_calls = data.get("tool_calls_made", 0)
        confidence = data.get("confidence", 0)

        # 메타 정보 (짧게)
        meta_parts: list[str] = []
        if tool_calls > 0:
            meta_parts.append(f"🔧{tool_calls}")
        if confidence > 0:
            meta_parts.append(f"📊{confidence:.0%}")
        meta = " ".join(meta_parts)

        response_text = answer
        if meta:
            response_text += f"\n\n_{meta}_"

        # 응답 전송 (Markdown 실패 시 plain text 재시도)
        try:
            await update.message.reply_text(response_text, parse_mode="Markdown")
        except Exception:
            try:
                await update.message.reply_text(answer)
            except Exception as send_err:
                logger.error("telegram_send_failed", error=str(send_err)[:200])
                await update.message.reply_text("⚠️ 응답 전송 중 오류가 발생했습니다.")

    # === HITL 콜백 핸들러 ===

    async def handle_confirmation_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """인라인 키보드 콜백 처리 (승인/거부)

        callback_data 형식: "confirm:{confirmation_id}:{approve|deny}"
        """
        query = update.callback_query
        if not query or not query.data:
            return

        # 인증
        if not self._is_authorized(update):
            await query.answer("⛔ 권한 없음")
            return

        data = query.data
        if not data.startswith("confirm:"):
            return

        parts = data.split(":")
        if len(parts) != 3:
            await query.answer("❌ 잘못된 요청")
            return

        confirmation_id = parts[1]
        action = parts[2]

        if action == "approve":
            # TODO: Phase 3 후속 — pending store에서 도구 실행
            await query.answer("✅ 승인됨")
            await query.edit_message_text(
                f"✅ 승인 완료 (ID: `{confirmation_id}`)\n\n"
                f"_도구 실행은 다음 업데이트에서 구현됩니다_",
                parse_mode="Markdown",
            )
            logger.info("hitl_approved", confirmation_id=confirmation_id)
        elif action == "deny":
            await query.answer("❌ 거부됨")
            await query.edit_message_text(
                f"❌ 거부됨 (ID: `{confirmation_id}`)",
                parse_mode="Markdown",
            )
            logger.info("hitl_denied", confirmation_id=confirmation_id)
        else:
            await query.answer("❌ 알 수 없는 액션")
