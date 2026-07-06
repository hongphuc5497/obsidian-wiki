from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# extract-jsonl.py has a hyphen, so import it by file path.
_SPEC = importlib.util.spec_from_file_location(
    "extract_jsonl",
    Path(__file__).resolve().parent.parent / "scripts" / "extract-jsonl.py",
)
extract_jsonl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(extract_jsonl)  # type: ignore[union-attr]


class ExtractStalenessTest(unittest.TestCase):
    """The extractor must be idempotent based on output-file staleness.

    Regression guard: after the first run, a session whose output is missing or
    whose source is newer than its output must be (re)extracted, without the
    caller having to pass the right --since. A previous version keyed staleness
    off a wallclock manifest, so a session that landed after the first run (or
    whose output was deleted) was silently treated as already-done and never
    re-extracted (real case: session 2d83ed0d reported missing downstream).
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.history = root / ".claude"
        self.output = root / "out"
        self.project = "-Users-x-app"
        self.source = self.history / "projects" / self.project / "sess1.jsonl"
        self.source.parent.mkdir(parents=True)
        self.source.write_text(
            '{"type":"user","timestamp":"2026-06-01T10:00:00Z","cwd":"/Users/x/app",'
            '"sessionId":"sess1","message":{"role":"user","content":"hello"}}\n'
            '{"type":"assistant","timestamp":"2026-06-01T10:00:05Z","sessionId":"sess1",'
            '"message":{"role":"assistant","content":[{"type":"text","text":"hi there"}]}}\n'
        )
        self.out_file = self.output / self.project / "sess1.json"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, *extra: str) -> str:
        buf = io.StringIO()
        argv = [
            "--history-path", str(self.history),
            "--output-dir", str(self.output),
            *extra,
        ]
        with contextlib.redirect_stdout(buf):
            rc = extract_jsonl.main(argv)
        self.assertEqual(rc, 0)
        return buf.getvalue()

    def _set_mtimes(self, source_mtime: float, output_mtime: float) -> None:
        os.utime(self.source, (source_mtime, source_mtime))
        os.utime(self.out_file, (output_mtime, output_mtime))

    def test_first_run_extracts(self) -> None:
        self._run()
        self.assertTrue(self.out_file.exists())
        data = json.loads(self.out_file.read_text())
        self.assertEqual(data["session_id"], "sess1")

    def test_missing_output_is_reextracted(self) -> None:
        # The reported bug: output gone but source unchanged -> must re-extract.
        self._run()
        self.out_file.unlink()
        self._run()
        self.assertTrue(self.out_file.exists())

    def test_source_newer_is_reextracted(self) -> None:
        self._run()
        # Source strictly newer than its output -> re-extract.
        self._set_mtimes(source_mtime=2000, output_mtime=1000)
        out = self._run("--verbose")
        self.assertIn("EXTRACT", out)
        self.assertNotIn("SKIP(unchanged)", out)

    def test_missing_output_reextracted_despite_since_gate(self) -> None:
        # Missing output must (re)extract even when --since would exclude the
        # (older) source; --since only narrows files that already have output.
        os.utime(self.source, (1000, 1000))  # source well in the past
        out = self._run("--since", "2099-01-01", "--verbose")
        self.assertIn("EXTRACT", out)
        self.assertTrue(self.out_file.exists())

    def test_output_newer_is_skipped(self) -> None:
        self._run()
        # Output at least as new as source -> skip.
        self._set_mtimes(source_mtime=1000, output_mtime=2000)
        out = self._run("--verbose")
        self.assertIn("SKIP(unchanged)", out)
        self.assertNotIn("EXTRACT ", out)


if __name__ == "__main__":
    sys.exit(unittest.main())
