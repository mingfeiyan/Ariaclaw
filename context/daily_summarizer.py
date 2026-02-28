"""End-of-day summarization: compresses daily logs into MEMORY.md."""
import glob
import logging
import os
import threading

logger = logging.getLogger(__name__)


class DailySummarizer:
    """Reads daily context logs and produces long-term memory summaries."""

    def __init__(self, workspace_dir: str):
        self._workspace_dir = workspace_dir
        self._memory_dir = os.path.join(workspace_dir, "memory")
        self._memory_md_path = os.path.join(workspace_dir, "MEMORY.md")
        self._write_lock = threading.Lock()

    def list_daily_logs(self) -> list[str]:
        pattern = os.path.join(self._memory_dir, "????-??-??.md")
        files = sorted(glob.glob(pattern))
        return [os.path.splitext(os.path.basename(f))[0] for f in files]

    def _read_daily_log(self, date_str: str) -> str:
        path = os.path.join(self._memory_dir, f"{date_str}.md")
        if not os.path.exists(path):
            return ""
        with open(path) as f:
            return f.read()

    def _append_to_memory_md(self, date_str: str, summary: str):
        header = f"\n## {date_str}\n\n"
        with self._write_lock:
            with open(self._memory_md_path, "a") as f:
                f.write(header)
                f.write(summary)
                f.write("\n")
        logger.info("Appended summary for %s to MEMORY.md", date_str)

    async def summarize_day(self, date_str: str, summarize_fn=None):
        daily_log = self._read_daily_log(date_str)
        if not daily_log.strip():
            logger.info("No content for %s, skipping", date_str)
            return
        if summarize_fn:
            summary = await summarize_fn(daily_log)
        else:
            summary = daily_log
        self._append_to_memory_md(date_str, summary)
