"""Content-addressed disk cache for cross-run agent results.

A ``CacheStore`` persists agent results keyed by a content fingerprint so a
later workflow run in the same directory (or a different process altogether)
can reuse a result without re-running the child agent. It complements
``ResumeCache``, which is an in-memory FIFO per-fingerprint list bounded to one
process lifetime: a CacheStore holds a single result per key on disk and
survives across runs.

Failure contract
----------------
The store is best-effort storage for cache hits. Every public method is
wrapped so a corrupt file, a filesystem error, or a concurrent writer never
propagates an exception into the workflow run: reads that cannot be satisfied
are reported as a miss, writes that fail are dropped, and stale entries are
reaped opportunistically.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

_REPLACE_RETRIES = 3
_REPLACE_RETRY_DELAY_SECONDS = 0.01


def _write_json_atomic(path: Path, data: Any) -> None:
    """Atomically write JSON to ``path`` via a uniquely-named tmp + rename.

    The unique tmp name keeps concurrent writers of the same path from
    clobbering each other's in-flight tmp file; if the write or rename fails,
    the leftover tmp is removed so no residue remains. The rename is retried a
    few times because on Windows a concurrent replace of the same target can
    transiently fail with ERROR_ACCESS_DENIED.
    """
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        for attempt in range(_REPLACE_RETRIES):
            try:
                tmp.replace(path)
                break
            except OSError:
                if attempt == _REPLACE_RETRIES - 1:
                    raise
                time.sleep(_REPLACE_RETRY_DELAY_SECONDS)
    finally:
        tmp.unlink(missing_ok=True)


class CacheStore:
    """Disk-backed, content-addressed, single-result-per-key agent cache.

    Keys are sha256 fingerprints (``[0-9a-f]{8,64}``) so the cache directory is
    flat and attacker/typo-proof: a key that does not match the pattern is
    rejected before any filesystem access. Entries carry ``createdAt`` (epoch
    milliseconds) and a JSON-serializable ``result``.
    """

    KEY_RE = re.compile(r"^[0-9a-f]{8,64}$")
    DEFAULT_MAX_AGE_SECONDS = 30 * 24 * 3600
    DEFAULT_MAX_ENTRIES = 1000

    def __init__(self, root: Path | None = None) -> None:
        """Create the cache directory under ``root`` (``<root>/cache``).

        Args:
            root: Base directory for the cache. When ``None`` the current
                directory is used; the ``cache`` subdirectory is created on
                demand (``mkdir parents=True exist_ok=True``).
        """
        base = Path(root).expanduser() if root is not None else Path(".")
        self.cache_dir = base / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def get(self, key: str, ttl_ms: int | None = None) -> Any | None:
        """Return the stored result for ``key``, or ``None`` on any miss.

        A key that fails the ``KEY_RE`` pattern is an immediate miss. A missing,
        corrupt, or stale file is also a miss: corrupt files (unparseable JSON,
        missing fields, non-numeric ``createdAt``) and entries beyond
        ``DEFAULT_MAX_AGE_SECONDS`` are deleted before returning ``None``. An
        entry within the default window but older than a given ``ttl_ms`` is
        reported as a miss but left in place, so a later read with a wider
        window can still hit it.

        Args:
            key: The cache key (a sha256 hex fingerprint).
            ttl_ms: Optional per-read freshness window in milliseconds; entries
                older than this are reported as a miss.

        Returns:
            The stored result value, or ``None`` if the key is invalid, missing,
            corrupt, or expired. Never raises.
        """
        try:
            if not isinstance(key, str) or not self.KEY_RE.match(key):
                return None
            path = self.cache_dir / f"{key}.json"
            if not path.exists():
                return None
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                path.unlink(missing_ok=True)
                return None
            if not isinstance(data, dict):
                path.unlink(missing_ok=True)
                return None
            created = data.get("createdAt")
            if isinstance(created, bool) or not isinstance(created, (int, float)):
                path.unlink(missing_ok=True)
                return None
            if data.get("key") != key or "result" not in data:
                path.unlink(missing_ok=True)
                return None
            now = time.time() * 1000
            age = now - created
            if age > self.DEFAULT_MAX_AGE_SECONDS * 1000:
                path.unlink(missing_ok=True)
                return None
            if ttl_ms is not None and age > ttl_ms:
                return None
            return data["result"]
        except Exception:
            return None

    def put(self, entry: dict) -> None:
        """Persist one result under its key.

        ``entry`` must be a dict with ``key`` (a ``KEY_RE``-matching string),
        ``createdAt`` (epoch milliseconds as a number), and ``result`` (any
        JSON-serializable value). The file is written atomically via a
        uniquely-named tmp + replace with a small retry loop, so concurrent
        writers of the same key never produce a torn file. A key that fails the
        pattern is a no-op. After the write, ``_cleanup`` runs opportunistically.

        Args:
            entry: The ``{"key", "createdAt", "result"}`` record to persist.

        Never raises: write and cleanup failures are swallowed.
        """
        try:
            if not isinstance(entry, dict):
                return
            key = entry.get("key")
            if not isinstance(key, str) or not self.KEY_RE.match(key):
                return
            payload = {
                "key": key,
                "createdAt": entry.get("createdAt"),
                "result": entry.get("result"),
            }
            with self._lock:
                _write_json_atomic(self.cache_dir / f"{key}.json", payload)
            self._cleanup()
        except Exception:
            pass

    def clear(self) -> None:
        """Delete every ``*.json`` entry in the cache directory.

        Tolerates individual unlink failures and missing directories.
        """
        try:
            with self._lock:
                for path in self.cache_dir.glob("*.json"):
                    try:
                        path.unlink(missing_ok=True)
                    except Exception:
                        pass
        except Exception:
            pass

    def _cleanup(self) -> None:
        """Reap expired and over-capacity entries from the cache directory.

        Deletes every entry older than ``DEFAULT_MAX_AGE_SECONDS`` (judged by
        its ``createdAt`` field; entries without a numeric ``createdAt`` are
        treated as expired). If the remaining count exceeds
        ``DEFAULT_MAX_ENTRIES``, the oldest entries are deleted (by
        ``createdAt``) until the cap holds. Fully tolerant: a scan or unlink
        failure is swallowed.
        """
        try:
            with self._lock:
                now = time.time() * 1000
                max_age_ms = self.DEFAULT_MAX_AGE_SECONDS * 1000
                entries: list[tuple[float, Path]] = []
                for path in self.cache_dir.glob("*.json"):
                    try:
                        data = json.loads(path.read_text(encoding="utf-8"))
                        created = data.get("createdAt") if isinstance(data, dict) else None
                    except Exception:
                        created = None
                    if isinstance(created, bool) or not isinstance(created, (int, float)):
                        path.unlink(missing_ok=True)
                        continue
                    if now - created > max_age_ms:
                        path.unlink(missing_ok=True)
                        continue
                    entries.append((float(created), path))
                entries.sort(key=lambda item: item[0])
                while len(entries) > self.DEFAULT_MAX_ENTRIES:
                    _, path = entries.pop(0)
                    try:
                        path.unlink(missing_ok=True)
                    except Exception:
                        pass
        except Exception:
            pass
