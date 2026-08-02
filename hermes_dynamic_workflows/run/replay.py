"""Deterministic replay of a recorded run's journal events.

``replay_run`` folds a journal event list (the same events the run manager
appends to a run's ``journalFile``) into a per-agent report WITHOUT running any
child agent: outcomes are read back from the recorded ``result`` / ``error``
events, gates are re-judged with the zero-token ``_gate_checks_pass`` when an
override is supplied, and budget overruns are recomputed from the recorded
agent order. The function never raises: malformed or unknown events are
skipped and a degenerate input yields an empty report.

Journal event shapes consumed (see ``engine/api.py`` for the emitting sites):

* ``{"type": "started", "key", "agentId"}`` — marks an agent as begun; used to
  establish agent presence and ordering.
* ``{"type": "result", "key", "agentId", "result", "cached"?: true}`` — final
  outcome; ``skipped: true`` marks a skip.
* ``{"type": "error", "key", "agentId", "error"}`` — final failure, outranks
  any ``result`` event for the same agent.
* ``{"type": "gate", "key", "agentId", "verdict", "mode", "reason", "check"}``
  — attaches a gate verdict to the governed agent (the ``agentId`` of the
  child whose result was gated).

``activity`` / ``approval`` / other event types are ignored.

Judge agents: an LLM-mode gate re-runs the prompt as a fresh child agent whose
``agent()`` call carries no ``_on_agent_id``, so it records its own
``started``/``result`` events with no ``gate`` event pointing at it. Replay
keeps those events as ordinary agents (they are indistinguishable from a
plain ``agent()`` call in the journal); the ``gate`` field is only attached to
agents a ``gate`` event names.
"""

from __future__ import annotations

from typing import Any

from ..engine.gate_logic import _gate_checks_pass
from ..core.text import preview


def replay_run(events: Any, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fold recorded journal events into a per-agent replay report.

    Args:
        events: The journal event list (each a dict with at least ``type`` and
            usually ``agentId``). Malformed, unknown, and unrelated events are
            skipped. Typed as ``Any`` because a real journal is read back from
            disk and may be corrupt; ``replay_run`` must never raise on it.
        overrides:         Optional replay overrides, ``None`` disables all of them.

            - ``gate_checks``: ``dict`` mapping a ``gate:``-prefixed key or an
              agentId to a new check config. For each matching gate, the check
              is re-judged against the agent's recorded result with the
              zero-token ``_gate_checks_pass`` and the outcome is reported as
              ``replayed_outcome`` ``"would-pass"`` / ``"would-block"``.
            - ``budget_max_tokens``: positive int. Agents are charged in
              recorded order, 1 per agent whose result was not ``cached``; the
              first agent whose cumulative charge reaches the cap is marked
              ``replayed_outcome`` ``"would-exceed"`` (a cached result is free,
              so an all-cached run never exceeds). Applied after gate
              re-judging, so it may overwrite a ``replayed_outcome``.

    Returns:
        A report dict::

            {
                "totals": {
                    "agents", "succeeded", "failed", "skipped", "blocked",
                    "cached", "calls",
                },
                "agents": [
                    {
                        "id", "status", "result_preview", "cached",
                        "gate"?, "replayed_outcome"?, "reason"?,
                    },
                    ...
                ],
            }

        ``status`` is ``"done"``, ``"error"``, or ``"skipped"`` (or ``None``
        when an agent only has a ``started`` event); ``cached`` is True when
        the recorded result came from a cache hit. ``gate`` (when present)
        mirrors the gate event. ``succeeded`` excludes cached agents (a cache
        hit means no model call actually executed); ``blocked`` covers done
        agents whose gate verdict was ``BLOCK``. Never raises.

    Never raises: every failure mode degrades to a partial or empty report.
    """
    if not isinstance(events, list):
        return {"totals": _empty_totals(), "agents": []}
    if not isinstance(overrides, dict):
        overrides = {}

    gate_checks = overrides.get("gate_checks")
    if not isinstance(gate_checks, dict):
        gate_checks = {}
    budget = overrides.get("budget_max_tokens")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        budget = None

    agents: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for event in events:
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        if etype not in ("started", "result", "error", "gate"):
            continue
        agent_id = event.get("agentId")
        if agent_id is None:
            continue
        key = str(agent_id)
        if etype == "gate":
            if key in agents:
                agents[key]["gate"] = {
                    "key": event.get("key"),
                    "verdict": event.get("verdict"),
                    "mode": event.get("mode"),
                    "reason": event.get("reason", ""),
                    "check": event.get("check") or {},
                }
            continue
        if etype == "result" and "result" not in event:
            continue
        if key not in agents:
            agents[key] = {
                "id": key,
                "status": None,
                "result_preview": None,
                "cached": False,
            }
            order.append(key)
        agent = agents[key]
        if etype == "started":
            continue
        if etype == "result":
            raw = event.get("result")
            agent["status"] = "skipped" if event.get("skipped") else "done"
            agent["result_preview"] = None if raw is None else preview(raw)
            if event.get("cached"):
                agent["cached"] = True
            agent["_raw_result"] = raw
        else:  # error
            agent["status"] = "error"
            agent["result_preview"] = None
            agent["reason"] = event.get("error")

    # Gate re-judging: zero-token replay of the original check against the
    # recorded result, when an override names the gate (by key or agentId).
    for key in order:
        agent = agents[key]
        gate_info = agent.get("gate")
        if gate_info is None or "_raw_result" not in agent:
            continue
        new_check = None
        gate_key = gate_info.get("key")
        if gate_key is not None and gate_key in gate_checks:
            new_check = gate_checks[gate_key]
        elif key in gate_checks:
            new_check = gate_checks[key]
        if new_check is None:
            continue
        try:
            passed = _gate_checks_pass(new_check, agent["_raw_result"])
        except Exception:
            passed = False  # corrupt override check degrades to a block
        agent["replayed_outcome"] = "would-pass" if passed else "would-block"

    # Budget: charge non-cached agents in recorded order until the cap.
    calls = 0
    for key in order:
        if agents[key].get("cached"):
            continue
        calls += 1
        if budget is not None and calls >= budget:
            agents[key]["replayed_outcome"] = "would-exceed"
            break

    totals: dict[str, int] = {
        "agents": len(order),
        "succeeded": 0,
        "failed": 0,
        "skipped": 0,
        "blocked": 0,
        "cached": 0,
        "calls": calls,
    }
    for key in order:
        agent = agents[key]
        if agent.get("cached"):
            totals["cached"] += 1
        status = agent.get("status")
        if status == "done":
            if agent.get("gate", {}).get("verdict") == "BLOCK":
                totals["blocked"] += 1
            elif not agent.get("cached"):
                totals["succeeded"] += 1
        elif status == "error":
            totals["failed"] += 1
        elif status == "skipped":
            totals["skipped"] += 1

    return {
        "totals": totals,
        "agents": [
            {k: v for k, v in agents[key].items() if not k.startswith("_")}
            for key in order
        ],
    }


def _empty_totals() -> dict[str, int]:
    """Zeroed totals for an empty or unreadable event list."""
    return {
        "agents": 0,
        "succeeded": 0,
        "failed": 0,
        "skipped": 0,
        "blocked": 0,
        "cached": 0,
        "calls": 0,
    }
