import asyncio
import json
import logging
from datetime import datetime, timezone
from enum import Enum

import aiohttp

import config

logger = logging.getLogger(__name__)


class OpenClawConnectionState(Enum):
    NOT_CONFIGURED = "not_configured"
    CHECKING = "checking"
    CONNECTED = "connected"
    UNREACHABLE = "unreachable"


class OpenClawBridge:
    """HTTP client for the local OpenClaw gateway."""

    def __init__(self):
        self.connection_state = OpenClawConnectionState.NOT_CONFIGURED
        self.last_tool_status = "idle"
        self._session_key = self._new_session_key()
        self._conversation_history = []
        self._max_history_turns = 10
        self.on_state_changed = None
        self._http_session: aiohttp.ClientSession | None = None  # I10: reuse session

    def _set_state(self, state):
        self.connection_state = state
        if self.on_state_changed:
            self.on_state_changed(state)

    @property
    def _base_url(self):
        return f"{config.OPENCLAW_HOST}:{config.OPENCLAW_PORT}"

    @staticmethod
    def _new_session_key():
        ts = datetime.now(timezone.utc).isoformat()
        return f"agent:main:aria:{ts}"

    def reset_session(self):
        self._session_key = self._new_session_key()
        self._conversation_history = []
        logger.info("OpenClaw new session: %s", self._session_key)

    async def _get_session(self, timeout_total: int = 120) -> aiohttp.ClientSession:
        """Get or create a reusable HTTP session."""
        if self._http_session is None or self._http_session.closed:
            timeout = aiohttp.ClientTimeout(total=timeout_total)
            self._http_session = aiohttp.ClientSession(timeout=timeout)
        return self._http_session

    async def close(self):
        """Close the HTTP session."""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
            self._http_session = None

    async def check_connection(self):
        if not config.OPENCLAW_TOKEN:
            self._set_state(OpenClawConnectionState.NOT_CONFIGURED)
            return

        self._set_state(OpenClawConnectionState.CHECKING)
        url = f"{self._base_url}/v1/chat/completions"

        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                headers = {"Authorization": f"Bearer {config.OPENCLAW_TOKEN}"}
                async with session.get(url, headers=headers) as resp:
                    # M8: only treat 2xx and 405 as connected
                    if 200 <= resp.status <= 299 or resp.status == 405:
                        self._set_state(OpenClawConnectionState.CONNECTED)
                        logger.info("OpenClaw reachable (HTTP %d)", resp.status)
                    else:
                        self._set_state(OpenClawConnectionState.UNREACHABLE)
                        logger.warning("OpenClaw returned HTTP %d", resp.status)
        except Exception as e:
            self._set_state(OpenClawConnectionState.UNREACHABLE)
            logger.warning("OpenClaw unreachable: %s", e)

    async def delegate_task(self, task: str, tool_name: str = "execute") -> dict:
        """Send a task to OpenClaw and return {"result": ...} or {"error": ...}."""
        self.last_tool_status = f"executing:{tool_name}"
        url = f"{self._base_url}/v1/chat/completions"

        self._conversation_history.append({"role": "user", "content": task})
        if len(self._conversation_history) > self._max_history_turns * 2:
            self._conversation_history = self._conversation_history[
                -(self._max_history_turns * 2) :
            ]

        headers = {
            "Authorization": f"Bearer {config.OPENCLAW_TOKEN}",
            "Content-Type": "application/json",
            "x-openclaw-session-key": self._session_key,
        }
        body = {
            "model": "openclaw",
            "messages": self._conversation_history,
            "stream": False,
        }

        logger.info("OpenClaw sending %d messages", len(self._conversation_history))

        try:
            session = await self._get_session()
            async with session.post(url, headers=headers, json=body) as resp:
                # C5: read body once as text, then parse
                raw_text = await resp.text()

                if not (200 <= resp.status <= 299):
                    logger.error(
                        "OpenClaw failed: HTTP %d - %s",
                        resp.status,
                        raw_text[:200],
                    )
                    self.last_tool_status = f"failed:{tool_name}"
                    return {"error": f"HTTP {resp.status}"}

                try:
                    data = json.loads(raw_text)
                except json.JSONDecodeError:
                    data = {}

                choices = data.get("choices", [])
                if choices:
                    content = (
                        choices[0].get("message", {}).get("content", "")
                    )
                    self._conversation_history.append(
                        {"role": "assistant", "content": content}
                    )
                    logger.info("OpenClaw result: %s", content[:200])
                    self.last_tool_status = f"completed:{tool_name}"
                    return {"result": content}

                self._conversation_history.append(
                    {"role": "assistant", "content": raw_text}
                )
                self.last_tool_status = f"completed:{tool_name}"
                return {"result": raw_text}

        except Exception as e:
            logger.error("OpenClaw error: %s", e)
            self.last_tool_status = f"failed:{tool_name}"
            return {"error": str(e)}
