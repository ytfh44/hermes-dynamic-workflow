"""Pure gate-check logic, shared by the live ``gate()`` API and deterministic replay.

The functions here are the zero-token half of ``gate()``: they decide, from a
validated ``check`` config and an agent result, whether the result satisfies the
check. They are module-level pure functions with no access to API state, so
``run/replay.py`` can reuse them to re-judge a recorded gate without running any
child agent or touching the engine.

Every function mirrors the behavior of the originals in ``engine/api.py``; the
live API imports them from here to keep a single source of truth.
"""

from __future__ import annotations

import json
import re
from typing import Any


def _gate_text(result: Any) -> str:
    """Render an agent result as text for the text-based zero-token checks.

    Strings pass through; structured results are JSON-serialized so
    ``contains`` / ``not_contains`` / ``regex`` can inspect them.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, (dict, list)):
        return json.dumps(result, ensure_ascii=False, sort_keys=True)
    return str(result)


def _gate_checks_pass(check: dict[str, Any], result: Any) -> bool:
    """Return True when ``result`` satisfies the (already validated) ``check``.

    ``contains`` / ``not_contains`` / ``regex`` match against the text form of
    the result; ``field`` + ``equals`` compares a key of a structured-output
    dict. A missing field or a non-dict result fails the check.
    """
    if "contains" in check:
        return check["contains"] in _gate_text(result)
    if "not_contains" in check:
        return check["not_contains"] not in _gate_text(result)
    if "regex" in check:
        return re.search(check["regex"], _gate_text(result)) is not None
    return (
        isinstance(result, dict)
        and check["field"] in result
        and result[check["field"]] == check["equals"]
    )


def _gate_check_description(check: dict[str, Any]) -> str:
    """Human-readable summary of a check, used in the ``GateBlocked`` message."""
    if "contains" in check:
        return f"contains {check['contains']!r}"
    if "not_contains" in check:
        return f"not_contains {check['not_contains']!r}"
    if "regex" in check:
        return f"regex {check['regex']!r}"
    return f"field {check['field']!r} equals {check['equals']!r}"
