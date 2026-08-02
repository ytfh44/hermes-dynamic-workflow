"""Content fingerprint resolver for the cross-run agent cache.

Mirrors the fingerprint entry syntaxes used by taskflow. A fingerprint is a
list of entries that each name a resource (git ref, file, glob, environment
variable); ``resolve_fingerprint`` turns them into one stable digest so two
``agent()`` calls can be told apart by the content they depend on rather than
by execution order.

Resolver contract
-----------------
* Every entry resolves to a stable string, never raises, and degrades to a
  deterministic sentinel on any failure (missing resource, non-repo, timeout,
  unreadable file, oversized file, unknown prefix).
* The ordered resolved strings are joined with the NUL character and hashed
  with sha256; the resulting hex digest is the fingerprint.
* Entry order is significant: the same entries in a different order produce a
  different digest.
* The empty entry list yields the sha256 digest of the empty string — a fixed,
  deterministic value that must not crash.

The layer that accepts user input validates entry prefixes before calling this
module. An entry with an unknown prefix resolves here to a deterministic
``<unknown>:<entry>`` sentinel rather than raising, so ``resolve_fingerprint``
is total, but garbage prefixes are still rejected upstream so a typo cannot
silently cache under a bogus key.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

_GIT_TIMEOUT_SECONDS = 30
_MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MiB
_NUL = "\u0000"


def resolve_fingerprint(entries: list[str], cwd: str) -> str:
    """Resolve ordered fingerprint entries to a stable content digest.

    Args:
        entries: Ordered list of fingerprint entries. Supported syntaxes:

            - ``git:<ref>`` — resolve ``<ref>`` via ``git rev-parse`` in
              ``cwd`` with a 30s timeout; sentinel on any failure.
            - ``file:<path>`` — sha256 of the file's bytes (resolved relative
              to ``cwd``, 10 MiB limit); sentinels for missing/oversized.
            - ``glob:<pattern>`` — sha256 of the sorted list of matched paths
              under ``cwd``.
            - ``glob!:<pattern>`` — like glob, but additionally hashes the
              content of each matched readable file under the size limit.
            - ``env:<name>`` — the environment variable's value (``""`` when
              unset).
        cwd: Base directory for relative ``file:``/``glob:`` paths and the
            working directory for ``git rev-parse``.

    Returns:
        The sha256 hex digest of the ordered resolved entries joined with NUL.

    Never raises: every entry resolution is wrapped so a failure yields a
    deterministic sentinel instead of an exception. Unknown entry prefixes are
    the validation layer's responsibility; here they degrade to
    ``<unknown>:<entry>``.
    """
    resolved = [_resolve_entry(str(entry), cwd) for entry in (entries or [])]
    return _digest(resolved)


def _resolve_entry(entry: str, cwd: str) -> str:
    try:
        if entry.startswith("git:"):
            return _resolve_git(entry[len("git:") :], cwd)
        if entry.startswith("glob!:"):
            return _resolve_glob(entry[len("glob!:") :], cwd, contents=True)
        if entry.startswith("glob:"):
            return _resolve_glob(entry[len("glob:") :], cwd, contents=False)
        if entry.startswith("file:"):
            return _resolve_file(entry[len("file:") :], cwd)
        if entry.startswith("env:"):
            return _resolve_env(entry[len("env:") :])
        return _unknown(entry)
    except Exception:
        return _unknown(entry)


def _digest(parts: list[str]) -> str:
    return hashlib.sha256(_NUL.join(parts).encode("utf-8")).hexdigest()


def _unknown(entry: str) -> str:
    return f"<unknown>:{entry}"


def _resolve_git(ref: str, cwd: str) -> str:
    key = f"git:{ref}"
    if ref.startswith("-"):
        return f"{key}=<invalid-ref>"
    try:
        proc = subprocess.run(
            ["git", "rev-parse", ref],
            cwd=cwd,
            capture_output=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"{key}=<timeout>"
    except Exception:
        return f"{key}=<no-git>"
    if proc.returncode != 0:
        return f"{key}=<no-git>"
    sha = proc.stdout.decode("utf-8", errors="replace").strip()
    return f"{key}={sha}"


def _resolve_file(path: str, cwd: str) -> str:
    key = f"file:{path}"
    try:
        resolved = Path(path).expanduser()
        if not resolved.is_absolute():
            resolved = Path(cwd) / resolved
        if not resolved.is_file():
            return f"{key}=<missing>"
        if resolved.stat().st_size > _MAX_FILE_BYTES:
            return f"{key}=<skip>"
        data = resolved.read_bytes()
    except Exception:
        return f"{key}=<missing>"
    return f"{key}={hashlib.sha256(data).hexdigest()}"


def _resolve_glob(pattern: str, cwd: str, contents: bool) -> str:
    prefix = "glob!:" if contents else "glob:"
    try:
        matches = sorted(Path(cwd).glob(pattern))
        rel_paths = [str(path.relative_to(cwd)) for path in matches]
        if not contents:
            return f"{prefix}{pattern}={_digest(rel_paths)}"
        payload = list(rel_paths)
        for path in matches:
            if not path.is_file():
                continue
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if len(data) <= _MAX_FILE_BYTES:
                payload.append(hashlib.sha256(data).hexdigest())
        return f"{prefix}{pattern}={_digest(payload)}"
    except Exception:
        return f"{prefix}{pattern}=<glob-error>"


def _resolve_env(name: str) -> str:
    return f"env:{name}={os.environ.get(name, '')}"
