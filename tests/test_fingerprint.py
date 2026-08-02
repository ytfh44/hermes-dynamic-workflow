"""Edge tests for the content fingerprint resolver (engine/fingerprint.py)."""

from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hermes_dynamic_workflows.engine.fingerprint import resolve_fingerprint


def _digest(*parts: str) -> str:
    blob = "\u0000".join(parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class FingerprintResolverTests(unittest.TestCase):
    def test_empty_entries_do_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = resolve_fingerprint([], tmp)
        self.assertEqual(result, hashlib.sha256(b"").hexdigest())

    def test_git_head_in_non_git_dir_yields_deterministic_sentinel(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = resolve_fingerprint(["git:HEAD"], tmp)
            second = resolve_fingerprint(["git:HEAD"], tmp)
        self.assertEqual(first, second)
        self.assertEqual(first, _digest("git:HEAD=<no-git>"))

    def test_git_ref_starting_with_dash_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = resolve_fingerprint(["git:-o-whatever"], tmp)
        self.assertEqual(result, _digest("git:-o-whatever=<invalid-ref>"))

    def test_git_entries_with_different_refs_differ(self):
        with tempfile.TemporaryDirectory() as tmp:
            head = resolve_fingerprint(["git:HEAD"], tmp)
            main = resolve_fingerprint(["git:main"], tmp)
        self.assertNotEqual(head, main)

    def test_file_missing_yields_sentinel(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = resolve_fingerprint(["file:missing.txt"], tmp)
        self.assertEqual(result, _digest("file:missing.txt=<missing>"))

    def test_file_present_is_stable_and_content_sensitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.txt"
            path.write_text("hello", encoding="utf-8")
            first = resolve_fingerprint(["file:data.txt"], tmp)
            repeated = resolve_fingerprint(["file:data.txt"], tmp)
            expected = _digest(
                "file:data.txt=" + hashlib.sha256(b"hello").hexdigest()
            )
            path.write_text("world", encoding="utf-8")
            changed = resolve_fingerprint(["file:data.txt"], tmp)
        self.assertEqual(first, repeated)
        self.assertEqual(first, expected)
        self.assertNotEqual(first, changed)

    def test_env_set_and_unset_are_deterministic(self):
        with patch.dict(os.environ, {"FP_TEST": "abc"}, clear=False):
            set_value = resolve_fingerprint(["env:FP_TEST"], ".")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FP_TEST", None)
            unset_value = resolve_fingerprint(["env:FP_TEST"], ".")
        self.assertEqual(set_value, _digest("env:FP_TEST=abc"))
        self.assertEqual(unset_value, _digest("env:FP_TEST="))
        self.assertNotEqual(set_value, unset_value)

    def test_glob_matching_is_deterministic_and_pattern_sensitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.txt").write_text("a", encoding="utf-8")
            (Path(tmp) / "b.txt").write_text("b", encoding="utf-8")
            first = resolve_fingerprint(["glob:*.txt"], tmp)
            repeated = resolve_fingerprint(["glob:*.txt"], tmp)
            no_match = resolve_fingerprint(["glob:*.nope"], tmp)
            other_pattern = resolve_fingerprint(["glob:b.*"], tmp)
            bang = resolve_fingerprint(["glob!:*.txt"], tmp)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, no_match)
        self.assertNotEqual(first, other_pattern)
        self.assertNotEqual(bang, resolve_fingerprint(["glob!:*.nope"], tmp))

    def test_glob_is_content_aware(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.txt"
            path.write_text("one", encoding="utf-8")
            before = resolve_fingerprint(["glob!:*.txt"], tmp)
            path.write_text("two", encoding="utf-8")
            after = resolve_fingerprint(["glob!:*.txt"], tmp)
        self.assertNotEqual(before, after)

    def test_entry_order_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.txt").write_text("x", encoding="utf-8")
            forward = resolve_fingerprint(["file:a.txt", "env:FP_ORDER"], tmp)
            reverse = resolve_fingerprint(["env:FP_ORDER", "file:a.txt"], tmp)
        self.assertNotEqual(forward, reverse)

    def test_unknown_prefix_yields_deterministic_sentinel(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = resolve_fingerprint(["bogus:thing"], tmp)
            repeated = resolve_fingerprint(["bogus:thing"], tmp)
        self.assertEqual(first, repeated)
        self.assertEqual(first, _digest("<unknown>:bogus:thing"))

    def test_mixed_entries_do_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.txt").write_text("x", encoding="utf-8")
            result = resolve_fingerprint(
                ["file:a.txt", "env:FP_MIXED", "glob:*.txt", "git:HEAD"],
                tmp,
            )
        self.assertIsInstance(result, str)
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
