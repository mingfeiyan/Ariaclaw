#!/usr/bin/env python3
"""AriaClaw - Aria Gen 2 glasses + Gemini Live AI assistant."""

import asyncio
import logging
import signal
import sys
import threading

import uvicorn

import config
import dashboard
from aria_stream import AriaConnectionState, AriaStream
from audio_playback import AudioPlayback
from gemini_service import GeminiConnectionState, GeminiLiveService
from context.context_pipeline import ContextPipeline
from openclaw_bridge import OpenClawBridge, OpenClawConnectionState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ariaclaw")


class AriaClaw:
    """Main coordinator wiring Aria, Gemini, AudioPlayback, OpenClaw, and Dashboard."""

    def __init__(self):
        self.aria = AriaStream()
        self.gemini = GeminiLiveService()
        self.audio = AudioPlayback()
        self.openclaw = OpenClawBridge()
        self.context = ContextPipeline(
            output_dir=config.OPENCLAW_WORKSPACE,
            whisper_model=config.WHISPER_MODEL,
        ) if config.CONTEXT_ENABLED else None
        self._loop = None
        self._session_lock = asyncio.Lock()  # I1: prevent double-start race
        self._session_active = False
        self._uvicorn_server = None
        self._reconnect_delay = 2.0  # exponential backoff for Gemini reconnect
        self._max_reconnect_delay = 60.0

    async def run(self):
        self._loop = asyncio.get_running_loop()  # M1: use get_running_loop

        # Wire up state change callbacks (dashboard functions are thread-safe, no coroutine needed)
        self.aria.on_state_changed = lambda s: dashboard.update_status("aria", s.value)
        self.gemini.on_state_changed = lambda s: dashboard.update_status("gemini", s.value)
        self.openclaw.on_state_changed = lambda s: dashboard.update_status("openclaw", s.value)

        # I6: pass the main event loop to dashboard
        dashboard.configure(
            main_loop=self._loop,
            get_latest_jpeg=lambda: self.aria.latest_jpeg,
            on_start_session=self._start_session,
            on_stop_session=self._stop_session,
        )

        # I5: start uvicorn with graceful shutdown support
        server_config = uvicorn.Config(
            dashboard.app,
            host=config.DASHBOARD_HOST,
            port=config.DASHBOARD_PORT,
            log_level="warning",
        )
        self._uvicorn_server = uvicorn.Server(server_config)
        server_thread = threading.Thread(
            target=self._uvicorn_server.run, daemon=True
        )
        server_thread.start()
        logger.info(
            "Dashboard running at http://localhost:%d", config.DASHBOARD_PORT
        )

        # C4: connect to Aria without blocking the event loop
        try:
            await asyncio.to_thread(self._connect_aria_blocking)
        except Exception as e:
            logger.warning("Aria connection failed: %s - continuing without glasses", e)

        # Check OpenClaw
        await self.openclaw.check_connection()

        # Start context pipeline (always-on, independent of Gemini session)
        if self.context:
            self.context.on_dashboard_event = (
                lambda et, content, ts: dashboard.send_context_event(et, content, ts)
            )
            self.context.on_heart_rate_update = (
                lambda bpm: dashboard.send_heart_rate(bpm)
            )
            self.context.start()
            logger.info("Context persistence enabled (output: %s)", config.OPENCLAW_WORKSPACE)

        logger.info("Ready. Open http://localhost:%d and click Start Session.", config.DASHBOARD_PORT)

        # Keep running until interrupted
        stop_event = asyncio.Event()

        def handle_signal():
            logger.info("Shutting down...")
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            self._loop.add_signal_handler(sig, handle_signal)

        await stop_event.wait()

        # Graceful shutdown
        await self._stop_session()
        if self.context:
            self.context.stop()
        self.aria.stop()  # I8: stop Aria streaming on shutdown
        await self.openclaw.close()  # I10: close reusable HTTP session
        if self._uvicorn_server:  # I5: signal uvicorn to stop
            self._uvicorn_server.should_exit = True

    def _connect_aria_blocking(self):
        """Connect to Aria — runs in a thread via asyncio.to_thread."""
        try:
            self.aria.connect_and_start()
        except Exception as e:
            logger.warning(
                "Aria not streaming (state: %s): %s",
                self.aria.connection_state.value,
                e,
            )

    async def _start_session(self):
        """Start the Gemini session and wire up all pipelines."""
        async with self._session_lock:  # I1: prevent double-start
            if self._session_active:
                return
            self._session_active = True

        logger.info("Starting session...")

        # Start audio playback
        self.audio.start()

        # Connect to Gemini
        success = await self.gemini.connect()
        if not success:
            logger.error("Failed to connect to Gemini")
            self.audio.stop()
            async with self._session_lock:
                self._session_active = False
            return

        # Reset OpenClaw session
        self.openclaw.reset_session()

        # Wire Aria video → Gemini
        self.aria.on_video_frame = lambda b64: asyncio.run_coroutine_threadsafe(
            self.gemini.send_video_frame(b64), self._loop
        )

        # Wire Aria audio → Gemini (and context pipeline)
        def _on_audio_chunk(chunk):
            asyncio.run_coroutine_threadsafe(
                self.gemini.send_audio(chunk), self._loop
            )
            if self.context:
                # Use VAD state as proxy for speaker: if VAD detected speech,
                # user is talking (self). Contact mic integration is pending
                # separate Aria SDK callback registration.
                is_self = self.aria._vad_active
                self.context.on_audio_chunk(chunk, is_contact_mic=is_self)
        self.aria.on_audio_chunk = _on_audio_chunk

        # Wire Aria video → context pipeline (raw frames for scene detection)
        if self.context:
            self.aria.on_video_frame_raw = lambda frame: self.context.on_video_frame(frame)

        # Wire Aria PPG → context pipeline (heart rate extraction)
        if self.context:
            self.aria.on_ppg_data = lambda signal, rate: self.context.on_ppg_data(signal, rate)

        # Wire Aria GPS → context pipeline (location tracking)
        if self.context:
            self.aria.on_gps_data = lambda lat, lon, alt, acc: self.context.on_gps_data(lat, lon, alt, acc)

        # Wire Aria VAD → Gemini activity signals
        self.aria.on_activity_start = lambda: asyncio.run_coroutine_threadsafe(
            self.gemini.send_activity_start(), self._loop
        )
        self.aria.on_activity_end = lambda: asyncio.run_coroutine_threadsafe(
            self.gemini.send_activity_end(), self._loop
        )

        # Suppress VAD while AI is speaking (prevents AI audio from triggering self-interrupt)
        def _on_audio_received(data):
            self.aria._vad_suppressed = True
            self.audio.play(data)
        self.gemini.on_audio_received = _on_audio_received

        # Wire Gemini transcriptions → dashboard
        self.gemini.on_input_transcription = (
            lambda text: dashboard.send_transcript("user", text)
        )
        self.gemini.on_output_transcription = (
            lambda text: dashboard.send_transcript("model", text)
        )

        # Wire Gemini turn complete → mark audio done + unsuppress VAD
        def _on_turn_complete():
            self.audio.is_speaking = False
            self.aria._vad_suppressed = False
        self.gemini.on_turn_complete = _on_turn_complete

        # I2: Wire Gemini interruptions → flush audio buffer + unsuppress VAD
        def _on_interrupted():
            self.audio.stop_playback()
            self.aria._vad_suppressed = False
        self.gemini.on_interrupted = _on_interrupted

        # Wire Gemini tool calls → OpenClaw
        self.gemini.on_tool_call = lambda data: asyncio.run_coroutine_threadsafe(
            self._handle_tool_call(data), self._loop
        )

        # Wire Gemini disconnection
        self.gemini.on_disconnected = lambda reason: asyncio.run_coroutine_threadsafe(
            self._on_gemini_disconnected(reason), self._loop
        )

        self._reconnect_delay = 2.0  # reset backoff on successful connect
        logger.info("Session started")

    async def _stop_session(self):
        """Stop the Gemini session and clean up."""
        async with self._session_lock:  # I1: prevent race
            if not self._session_active:
                return
            self._session_active = False

        logger.info("Stopping session...")

        # Disconnect callbacks
        self.aria.on_video_frame = None
        self.aria.on_video_frame_raw = None
        self.aria.on_audio_chunk = None
        self.aria.on_ppg_data = None
        self.aria.on_gps_data = None
        self.aria.on_activity_start = None
        self.aria.on_activity_end = None

        # Disconnect Gemini
        await self.gemini.disconnect()

        # Stop audio
        self.audio.stop()

        logger.info("Session stopped")

    async def _handle_tool_call(self, data: dict):
        """Route a Gemini tool call to OpenClaw."""
        calls = data.get("toolCall", {}).get("functionCalls", [])
        for call in calls:
            call_id = call.get("id", "")
            name = call.get("name", "")
            args = call.get("args", {})
            task_desc = args.get("task", str(args))

            logger.info("Tool call: %s (id: %s) - %s", name, call_id, task_desc[:100])
            dashboard.send_tool_event("tool_call", task_desc)

            result = await self.openclaw.delegate_task(task_desc, name)
            result_text = result.get("result", result.get("error", "Unknown"))
            dashboard.send_tool_event("tool_result", result_text[:200])

            await self.gemini.send_tool_response(call_id, name, result)

    async def _on_gemini_disconnected(self, reason: str):
        logger.warning("Gemini disconnected: %s", reason)
        async with self._session_lock:
            if not self._session_active:
                return
            self._session_active = False
        self.audio.stop()
        self.aria._vad_suppressed = False

        # Auto-reconnect with exponential backoff
        delay = self._reconnect_delay
        self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)
        logger.info("Auto-reconnecting Gemini in %.0f seconds...", delay)
        await asyncio.sleep(delay)
        await self._start_session()


def main():
    if not config.GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not set. Set it in config.py or as an env var.")
        sys.exit(1)

    app = AriaClaw()
    asyncio.run(app.run())


if __name__ == "__main__":
    main()
