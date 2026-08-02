"""Tests for the cross-run content-addressed cache (storage/cache_store.py).

Integration tests run real workflows via ``run_workflow`` + a FakeRunner,
mirroring tests/test_runtime.py, to prove the cache integrates into the
``agent()`` execution path. Unit tests cover the CacheStore filesystem
semantics directly.
"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from hermes_dynamic_workflows.core.errors import WorkflowRuntimeError
from hermes_dynamic_workflows.core.types import (
    ChildAgentRequest,
    ChildAgentRunner,
)
from hermes_dynamic_workflows.engine.cache import ResumeCache
from hermes_dynamic_workflows.engine.runtime import WorkflowOptions, run_workflow
from hermes_dynamic_workflows.storage.cache_store import CacheStore

CACHE_SCRIPT = """
meta = {"name": "cache-store", "description": "Test workflow"}

return await agent("same prompt", {"fingerprint": ["file:data.txt"], "label": "agent"})
"""


class FakeRunner(ChildAgentRunner):
    def __init__(self, responses=None):
        self.requests: list[ChildAgentRequest] = []
        self.responses = list(responses or [])

    def run(self, request: ChildAgentRequest):
        self.requests.append(request)
        if self.responses:
            return self.responses.pop(0)
        return f"{request.label}:{request.prompt}"


class CacheStoreIntegrationTests(unittest.TestCase):
    def test_cross_run_hit_when_resources_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "data.txt").write_text("v1", encoding="utf-8")
            store = CacheStore(Path(tmp) / "store")
            run_workflow(
                CACHE_SCRIPT,
                WorkflowOptions(cwd=tmp, child_runner=FakeRunner(), cache_store=store),
            )
            runner2 = FakeRunner()
            result = run_workflow(
                CACHE_SCRIPT,
                WorkflowOptions(cwd=tmp, child_runner=runner2, cache_store=store),
            )
            self.assertEqual(runner2.requests, [])
            self.assertEqual(result.value, "agent:same prompt")

    def test_changed_resource_invalidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data.txt"
            data.write_text("v1", encoding="utf-8")
            store = CacheStore(Path(tmp) / "store")
            run_workflow(
                CACHE_SCRIPT,
                WorkflowOptions(cwd=tmp, child_runner=FakeRunner(), cache_store=store),
            )
            data.write_text("v2", encoding="utf-8")
            runner2 = FakeRunner()
            run_workflow(
                CACHE_SCRIPT,
                WorkflowOptions(cwd=tmp, child_runner=runner2, cache_store=store),
            )
            self.assertEqual(len(runner2.requests), 1)

    def test_cache_store_untouched_without_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CacheStore(Path(tmp) / "store")
            script = """
meta = {"name": "no-fp", "description": "Test workflow"}

return await agent("plain")
"""
            runner = FakeRunner()
            run_workflow(
                script,
                WorkflowOptions(cwd=tmp, child_runner=runner, cache_store=store),
            )
            self.assertEqual(len(runner.requests), 1)
            self.assertEqual(list(store.cache_dir.glob("*.json")), [])

    def test_same_declaration_different_prompt_is_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "data.txt").write_text("v1", encoding="utf-8")
            store = CacheStore(Path(tmp) / "store")
            script_one = """
meta = {"name": "fp-a", "description": "Test workflow"}

return await agent("prompt one", {"fingerprint": ["file:data.txt"]})
"""
            script_two = """
meta = {"name": "fp-b", "description": "Test workflow"}

return await agent("prompt two", {"fingerprint": ["file:data.txt"]})
"""
            run_workflow(
                script_one,
                WorkflowOptions(cwd=tmp, child_runner=FakeRunner(), cache_store=store),
            )
            runner2 = FakeRunner()
            run_workflow(
                script_two,
                WorkflowOptions(cwd=tmp, child_runner=runner2, cache_store=store),
            )
            self.assertEqual(len(runner2.requests), 1)

    def test_corrupt_entry_in_cache_dir_does_not_break_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CacheStore(Path(tmp) / "store")
            corrupt = store.cache_dir / "aaaaaaaa.json"
            corrupt.write_text("{not json", encoding="utf-8")
            runner = FakeRunner()
            result = run_workflow(
                CACHE_SCRIPT,
                WorkflowOptions(cwd=tmp, child_runner=runner, cache_store=store),
            )
            self.assertEqual(len(runner.requests), 1)
            self.assertEqual(result.value, "agent:same prompt")
            # The corrupt entry was reaped by the write's opportunistic cleanup.
            self.assertFalse(corrupt.exists())

    def test_parallel_same_fingerprint_no_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "data.txt").write_text("v1", encoding="utf-8")
            store = CacheStore(Path(tmp) / "store")
            script = """
meta = {"name": "fp-par", "description": "Test workflow"}

results = await parallel([
    lambda: agent("same", {"fingerprint": ["file:data.txt"], "label": "a"}),
    lambda: agent("same", {"fingerprint": ["file:data.txt"], "label": "b"}),
])
return results
"""
            runner = FakeRunner()
            result = run_workflow(
                script,
                WorkflowOptions(cwd=tmp, child_runner=runner, cache_store=store),
            )
            self.assertIn(len(runner.requests), (1, 2))
            # Both agents share the same fingerprint, so under content
            # addressing their results are interchangeable: either both ran
            # (a:same + b:same) or the second hit the first's cache entry.
            self.assertEqual(len(result.value), 2)
            self.assertEqual(result.value[0], result.value[1])
            self.assertIn(result.value[0], ("a:same", "b:same"))
            self.assertEqual(list(store.cache_dir.glob("*.tmp")), [])

    def test_cache_store_hit_precedes_and_spares_resume_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "data.txt").write_text("v1", encoding="utf-8")
            store = CacheStore(Path(tmp) / "store")
            first_cache = ResumeCache()
            run_workflow(
                CACHE_SCRIPT,
                WorkflowOptions(
                    cwd=tmp,
                    child_runner=FakeRunner(),
                    resume_cache=first_cache,
                    cache_store=store,
                ),
            )
            # Override the stored value so a cache_store hit is distinguishable
            # from a resume_cache hit (which would return the original value and
            # consume the FIFO bucket).
            files = list(store.cache_dir.glob("*.json"))
            self.assertEqual(len(files), 1)
            entry = json.loads(files[0].read_text(encoding="utf-8"))
            entry["result"] = "OVERRIDDEN"
            files[0].write_text(json.dumps(entry), encoding="utf-8")
            runner2 = FakeRunner()
            result = run_workflow(
                CACHE_SCRIPT,
                WorkflowOptions(
                    cwd=tmp,
                    child_runner=runner2,
                    resume_cache=ResumeCache(first_cache.current),
                    cache_store=store,
                ),
            )
            self.assertEqual(runner2.requests, [])
            self.assertEqual(result.value, "OVERRIDDEN")


class CacheStoreUnitTests(unittest.TestCase):
    def test_invalid_key_get_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CacheStore(Path(tmp) / "store")
            self.assertIsNone(store.get("INVALID!"))
            self.assertIsNone(store.get("short"))

    def test_invalid_key_put_noops(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CacheStore(Path(tmp) / "store")
            store.put({"key": "bad key!", "createdAt": time.time() * 1000, "result": "x"})
            self.assertEqual(list(store.cache_dir.glob("*.json")), [])

    def test_roundtrip_put_get(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CacheStore(Path(tmp) / "store")
            key = "a" * 64
            store.put({"key": key, "createdAt": time.time() * 1000, "result": {"answer": 42}})
            self.assertEqual(store.get(key), {"answer": 42})

    def test_ttl_expiry_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CacheStore(Path(tmp) / "store")
            key = "b" * 64
            store.put({"key": key, "createdAt": time.time() * 1000 - 10_000, "result": "stale"})
            self.assertIsNone(store.get(key, ttl_ms=1000))
            self.assertEqual(store.get(key, ttl_ms=60_000), "stale")

    def test_default_max_age_expiry(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CacheStore(Path(tmp) / "store")
            key = "c" * 64
            stale = time.time() * 1000 - 31 * 24 * 3600 * 1000
            (store.cache_dir / f"{key}.json").write_text(
                json.dumps({"key": key, "createdAt": stale, "result": "old"}),
                encoding="utf-8",
            )
            self.assertIsNone(store.get(key))

    def test_corrupt_file_get_returns_none_and_deletes(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CacheStore(Path(tmp) / "store")
            key = "d" * 64
            path = store.cache_dir / f"{key}.json"
            path.write_text("{corrupt", encoding="utf-8")
            self.assertIsNone(store.get(key))
            self.assertFalse(path.exists())

    def test_missing_result_field_is_a_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CacheStore(Path(tmp) / "store")
            key = "e" * 64
            path = store.cache_dir / f"{key}.json"
            path.write_text(
                json.dumps({"key": key, "createdAt": time.time() * 1000}),
                encoding="utf-8",
            )
            self.assertIsNone(store.get(key))

    def test_cleanup_evicts_oldest_beyond_max(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CacheStore(Path(tmp) / "store")
            now = time.time() * 1000
            for i in range(5):
                store.put({"key": f"{i:064x}", "createdAt": now + i, "result": i})
            store.DEFAULT_MAX_ENTRIES = 3
            store._cleanup()
            remaining = sorted(int(p.stem, 16) for p in store.cache_dir.glob("*.json"))
            self.assertEqual(remaining, [2, 3, 4])

    def test_cleanup_removes_expired_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CacheStore(Path(tmp) / "store")
            stale = time.time() * 1000 - 31 * 24 * 3600 * 1000
            (store.cache_dir / f"f{'0' * 63}.json").write_text(
                json.dumps({"key": f"f{'0' * 63}", "createdAt": stale, "result": "old"}),
                encoding="utf-8",
            )
            store._cleanup()
            self.assertEqual(list(store.cache_dir.glob("*.json")), [])

    def test_clear_removes_all_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CacheStore(Path(tmp) / "store")
            for i in range(3):
                store.put({"key": f"{i:064x}", "createdAt": time.time() * 1000, "result": i})
            store.clear()
            self.assertEqual(list(store.cache_dir.glob("*.json")), [])


class FingerprintOptionValidationTests(unittest.TestCase):
    def _run(self, script):
        return run_workflow(script, WorkflowOptions(child_runner=FakeRunner()))

    def test_rejects_fingerprint_not_a_list(self):
        script = """
meta = {"name": "v", "description": "Test workflow"}
return await agent("p", {"fingerprint": "file:x"})
"""
        with self.assertRaises(WorkflowRuntimeError):
            self._run(script)

    def test_rejects_empty_fingerprint_list(self):
        script = """
meta = {"name": "v", "description": "Test workflow"}
return await agent("p", {"fingerprint": []})
"""
        with self.assertRaises(WorkflowRuntimeError):
            self._run(script)

    def test_rejects_non_string_fingerprint_entry(self):
        script = """
meta = {"name": "v", "description": "Test workflow"}
return await agent("p", {"fingerprint": [123]})
"""
        with self.assertRaises(WorkflowRuntimeError):
            self._run(script)

    def test_rejects_blank_fingerprint_entry(self):
        script = """
meta = {"name": "v", "description": "Test workflow"}
return await agent("p", {"fingerprint": ["   "]})
"""
        with self.assertRaises(WorkflowRuntimeError):
            self._run(script)

    def test_rejects_unknown_fingerprint_prefix(self):
        script = """
meta = {"name": "v", "description": "Test workflow"}
return await agent("p", {"fingerprint": ["bogus:x"]})
"""
        with self.assertRaises(WorkflowRuntimeError):
            self._run(script)

    def test_rejects_non_positive_ttl(self):
        for bad in (0, -5, "10", True, None):
            script = f"""
meta = {{"name": "v", "description": "Test workflow"}}
return await agent("p", {{"fingerprint": ["env:X"], "ttl_seconds": {bad!r}}})
"""
            with self.assertRaises(WorkflowRuntimeError):
                self._run(script)


if __name__ == "__main__":
    unittest.main()
