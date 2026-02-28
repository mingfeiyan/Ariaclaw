import asyncio
import base64
import json
import logging
import time
from enum import Enum

import websockets

import config

logger = logging.getLogger(__name__)


class GeminiConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    SETTING_UP = "setting_up"
    READY = "ready"
    ERROR = "error"


class GeminiLiveService:
    """WebSocket client for Gemini Live bidirectional streaming API."""

    def __init__(self):
        self.connection_state = GeminiConnectionState.DISCONNECTED
        self.is_model_speaking = False
        self._ws = None
        self._receive_task = None

        # Callbacks
        self.on_audio_received = None  # (bytes) -> None
        self.on_turn_complete = None  # () -> None
        self.on_interrupted = None  # () -> None
        self.on_disconnected = None  # (str) -> None
        self.on_input_transcription = None  # (str) -> None
        self.on_output_transcription = None  # (str) -> None
        self.on_tool_call = None  # (dict) -> None
        self.on_tool_call_cancellation = None  # (list[str]) -> None
        self.on_state_changed = None  # (GeminiConnectionState) -> None

        # Latency tracking
        self._last_user_speech_end = None
        self._response_latency_logged = False

        # Debug counters
        self._audio_send_count = 0
        self._recv_count = 0

    def _set_state(self, state):
        self.connection_state = state
        if self.on_state_changed:
            self.on_state_changed(state)

    async def connect(self) -> bool:
        if not config.GEMINI_API_KEY:
            self._set_state(GeminiConnectionState.ERROR)
            return False

        self._set_state(GeminiConnectionState.CONNECTING)
        url = f"{config.GEMINI_WS_URL}?key={config.GEMINI_API_KEY}"

        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(url, max_size=None, close_timeout=5),
                timeout=15,
            )
            self._set_state(GeminiConnectionState.SETTING_UP)
            await self._send_setup_message()  # C1: await instead of fire-and-forget
            self._receive_task = asyncio.create_task(self._receive_loop())
            return True
        except asyncio.TimeoutError:
            self._set_state(GeminiConnectionState.ERROR)
            return False
        except Exception as e:
            logger.error("Gemini connect failed: %s", e)
            self._set_state(GeminiConnectionState.ERROR)
            return False

    async def disconnect(self):
        # Clear on_disconnected before cancelling to prevent the finally block
        # in _receive_loop from triggering auto-reconnect on intentional disconnect
        self.on_disconnected = None
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None
        if self._ws:
            await self._ws.close()
            self._ws = None
        self.is_model_speaking = False
        self._set_state(GeminiConnectionState.DISCONNECTED)

    async def send_audio(self, pcm_data: bytes):
        if self.connection_state != GeminiConnectionState.READY:
            return
        self._audio_send_count += 1
        if self._audio_send_count <= 5 or self._audio_send_count % 50 == 0:
            logger.info("Gemini send_audio #%d: %d bytes, state=%s",
                        self._audio_send_count, len(pcm_data), self.connection_state.value)
        b64 = base64.b64encode(pcm_data).decode("ascii")
        msg = {
            "realtimeInput": {
                "audio": {"mimeType": "audio/pcm;rate=16000", "data": b64}
            }
        }
        await self._send_json(msg)

    async def send_video_frame(self, jpeg_base64: str):
        if self.connection_state != GeminiConnectionState.READY:
            return
        msg = {
            "realtimeInput": {
                "video": {"mimeType": "image/jpeg", "data": jpeg_base64}
            }
        }
        await self._send_json(msg)

    async def send_activity_start(self):
        if self.connection_state != GeminiConnectionState.READY:
            return
        await self._send_json({"realtimeInput": {"activityStart": {}}})

    async def send_activity_end(self):
        if self.connection_state != GeminiConnectionState.READY:
            return
        await self._send_json({"realtimeInput": {"activityEnd": {}}})

    async def send_tool_response(self, call_id: str, name: str, result: dict):
        msg = {
            "toolResponse": {
                "functionResponses": [
                    {"id": call_id, "name": name, "response": result}
                ]
            }
        }
        await self._send_json(msg)

    # -- Private --

    async def _send_setup_message(self):  # C1: now async
        setup = {
            "setup": {
                "model": config.GEMINI_MODEL,
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "thinkingConfig": {"thinkingBudget": 0},
                },
                "systemInstruction": {
                    "parts": [{"text": config.SYSTEM_INSTRUCTION}]
                },
                "tools": [{"functionDeclarations": [_EXECUTE_TOOL]}],
                "realtimeInputConfig": {
                    "automaticActivityDetection": {"disabled": True},
                    "activityHandling": "START_OF_ACTIVITY_INTERRUPTS",
                    "turnCoverage": "TURN_INCLUDES_ALL_INPUT",
                },
                "inputAudioTranscription": {},
                "outputAudioTranscription": {},
            }
        }
        await self._send_json(setup)  # C1: await directly

    async def _send_json(self, obj: dict):
        if self._ws is None:
            return
        try:
            await self._ws.send(json.dumps(obj))
        except Exception as e:
            # Sanitize error message to avoid leaking API key from WebSocket URL
            import re
            msg = re.sub(r'key=[^&\s\'"]+', 'key=***', str(e))
            logger.error("Gemini send error: %s: %s", type(e).__name__, msg)

    async def _receive_loop(self):
        try:
            async for raw in self._ws:
                self._recv_count += 1
                data = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode())
                if self._recv_count <= 10:
                    # Log first few messages (truncated) for debugging
                    msg_str = str(data)[:300]
                    logger.info("Gemini recv #%d: %s", self._recv_count, msg_str)
                self._handle_message(data)
        except websockets.ConnectionClosed as e:
            logger.info("Gemini connection closed: %s", e)
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error("Gemini receive error: %s", e)
        finally:
            self.is_model_speaking = False
            self._set_state(GeminiConnectionState.DISCONNECTED)
            if self.on_disconnected:
                self.on_disconnected("Connection closed")

    def _handle_message(self, data: dict):
        # Setup complete
        if "setupComplete" in data:
            self._set_state(GeminiConnectionState.READY)
            logger.info("Gemini setup complete")
            return

        # GoAway
        if "goAway" in data:
            time_left = data["goAway"].get("timeLeft", {}).get("seconds", 0)
            logger.warning("Gemini goAway: %ds left", time_left)
            self.is_model_speaking = False
            self._set_state(GeminiConnectionState.DISCONNECTED)
            if self.on_disconnected:
                self.on_disconnected(f"Server closing ({time_left}s left)")
            return

        # Tool call
        if "toolCall" in data:
            calls = data["toolCall"].get("functionCalls", [])
            logger.info("Gemini tool call: %d function(s)", len(calls))
            if self.on_tool_call:
                self.on_tool_call(data)
            return

        # Tool call cancellation
        if "toolCallCancellation" in data:
            ids = data["toolCallCancellation"].get("ids", [])
            logger.info("Gemini tool call cancellation: %s", ids)
            if self.on_tool_call_cancellation:
                self.on_tool_call_cancellation(ids)
            return

        # Server content
        server_content = data.get("serverContent")
        if not server_content:
            return

        if server_content.get("interrupted"):
            self.is_model_speaking = False
            if self.on_interrupted:
                self.on_interrupted()
            return

        model_turn = server_content.get("modelTurn", {})
        for part in model_turn.get("parts", []):
            inline_data = part.get("inlineData")
            if inline_data:
                mime = inline_data.get("mimeType", "")
                if mime.startswith("audio/pcm") and inline_data.get("data"):
                    audio_bytes = base64.b64decode(inline_data["data"])
                    if not self.is_model_speaking:
                        self.is_model_speaking = True
                        if self._last_user_speech_end and not self._response_latency_logged:
                            latency_ms = (time.time() - self._last_user_speech_end) * 1000
                            logger.info("Latency: %.0fms (speech end -> first audio)", latency_ms)
                            self._response_latency_logged = True
                    if self.on_audio_received:
                        self.on_audio_received(audio_bytes)
            elif "text" in part:
                logger.info("Gemini text: %s", part["text"])

        if server_content.get("turnComplete"):
            self.is_model_speaking = False
            self._response_latency_logged = False
            if self.on_turn_complete:
                self.on_turn_complete()

        input_tx = server_content.get("inputTranscription", {})
        if input_tx.get("text"):
            text = input_tx["text"]
            logger.info("You: %s", text)
            self._last_user_speech_end = time.time()
            self._response_latency_logged = False
            if self.on_input_transcription:
                self.on_input_transcription(text)

        output_tx = server_content.get("outputTranscription", {})
        if output_tx.get("text"):
            text = output_tx["text"]
            logger.info("AI: %s", text)
            if self.on_output_transcription:
                self.on_output_transcription(text)


# Tool declaration for OpenClaw execute
_EXECUTE_TOOL = {
    "name": "execute",
    "description": (
        "Your only way to take action. You have no memory, storage, or ability "
        "to do anything on your own -- use this tool for everything: sending "
        "messages, searching the web, adding to lists, setting reminders, "
        "creating notes, research, drafts, scheduling, smart home control, "
        "app interactions, or any request that goes beyond answering a question. "
        "When in doubt, use this tool."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": (
                    "Clear, detailed description of what to do. Include all "
                    "relevant context: names, content, platforms, quantities, etc."
                ),
            }
        },
        "required": ["task"],
    },
    "behavior": "BLOCKING",
}
