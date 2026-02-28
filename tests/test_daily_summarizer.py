"""Tests for DailySummarizer - compresses daily logs into MEMORY.md summaries."""
import os
import shutil
import tempfile

import pytest

from context.daily_summarizer import DailySummarizer


class TestDailySummarizer:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.memory_dir = os.path.join(self.tmpdir, "memory")
        os.makedirs(self.memory_dir)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_reads_daily_log(self):
        log_path = os.path.join(self.memory_dir, "2026-02-25.md")
        with open(log_path, "w") as f:
            f.write("# 2026-02-25\n\n- **09:15** — **You said**: \"Hello\"\n")
        summarizer = DailySummarizer(workspace_dir=self.tmpdir)
        content = summarizer._read_daily_log("2026-02-25")
        assert "Hello" in content

    def test_writes_to_memory_md(self):
        summarizer = DailySummarizer(workspace_dir=self.tmpdir)
        summarizer._append_to_memory_md("2026-02-25", "Had a meeting about deadlines.")
        memory_path = os.path.join(self.tmpdir, "MEMORY.md")
        assert os.path.exists(memory_path)
        content = open(memory_path).read()
        assert "2026-02-25" in content
        assert "deadlines" in content

    def test_appends_multiple_days(self):
        summarizer = DailySummarizer(workspace_dir=self.tmpdir)
        summarizer._append_to_memory_md("2026-02-24", "Day one summary.")
        summarizer._append_to_memory_md("2026-02-25", "Day two summary.")
        memory_path = os.path.join(self.tmpdir, "MEMORY.md")
        content = open(memory_path).read()
        assert "Day one" in content
        assert "Day two" in content

    def test_lists_daily_logs(self):
        for date in ["2026-02-23", "2026-02-24", "2026-02-25"]:
            with open(os.path.join(self.memory_dir, f"{date}.md"), "w") as f:
                f.write(f"# {date}\n")
        summarizer = DailySummarizer(workspace_dir=self.tmpdir)
        logs = summarizer.list_daily_logs()
        assert len(logs) == 3
