"""Tests for deterministic replay (run/replay.py).

Replay folds a journal event list into a per-agent report with zero model
calls. Tests construct event dicts directly (mirroring the shapes emitted by
``engine/api.py``) and assert on the folded report. ``replay_run`` must never
raise, so every malformed-input test asserts graceful degradation instead of
exceptions.
"""

from __future__ import annotations

import unittest

from hermes_dynamic_workflows.run.replay import replay_run


def started(agent_id, key="v2:fp"):
    return {"type": "started", "key": key, "agentId": str(agent_id)}


def result(agent_id, value, key="v2:fp", cached=False, skipped=False):
    event = {"type": "result", "key": key, "agentId": str(agent_id), "result": value}
    if cached:
        event["cached"] = True
    if skipped:
        event["skipped"] = True
    return event


def error(agent_id, message, key="v2:fp"):
    return {"type": "error", "key": key, "agentId": str(agent_id), "error": message}


def gate(agent_id, verdict, mode="check", check=None, reason="", key="gate:fp"):
    return {
        "type": "gate",
        "key": key,
        "agentId": str(agent_id),
        "verdict": verdict,
        "mode": mode,
        "reason": reason,
        "check": check or {},
    }


class ReplayBaselineFoldTests(unittest.TestCase):
    def test_started_plus_result_folds_one_done_agent(self):
        report = replay_run([started(1), result(1, "hello")])
        self.assertEqual(report["totals"]["agents"], 1)
        agent = report["agents"][0]
        self.assertEqual(agent["id"], "1")
        self.assertEqual(agent["status"], "done")
        self.assertEqual(agent["result_preview"], "hello")
        self.assertFalse(agent["cached"])
        self.assertNotIn("gate", agent)

    def test_error_outranks_result_for_same_agent(self):
        report = replay_run([started(1), result(1, "ok"), error(1, "boom")])
        self.assertEqual(report["agents"][0]["status"], "error")
        self.assertEqual(report["totals"]["failed"], 1)
        self.assertEqual(report["totals"]["succeeded"], 0)

    def test_last_terminal_event_wins(self):
        report = replay_run([started(1), result(1, "first"), result(1, "second")])
        self.assertEqual(report["agents"][0]["result_preview"], "second")

    def test_skipped_result_marks_skipped_status(self):
        report = replay_run([started(1), result(1, None, skipped=True)])
        agent = report["agents"][0]
        self.assertEqual(agent["status"], "skipped")
        self.assertEqual(report["totals"]["skipped"], 1)

    def test_cached_result_marks_cached_flag(self):
        report = replay_run([started(1), result(1, "from cache", cached=True)])
        self.assertTrue(report["agents"][0]["cached"])
        self.assertEqual(report["totals"]["cached"], 1)

    def test_multiple_agents_keep_emission_order(self):
        report = replay_run(
            [started(1), result(1, "a"), started(2), result(2, "b"), started(3), result(3, "c")]
        )
        self.assertEqual([agent["id"] for agent in report["agents"]], ["1", "2", "3"])

    def test_result_without_started_still_folds(self):
        report = replay_run([result(1, "no start")])
        self.assertEqual(report["totals"]["agents"], 1)
        self.assertEqual(report["agents"][0]["status"], "done")


class ReplayGateAssociationTests(unittest.TestCase):
    def test_gate_attaches_to_named_agent(self):
        report = replay_run(
            [
                started(1),
                result(1, "report body"),
                gate(1, "BLOCK", mode="check", check={"contains": "xyz"}, reason="expected contains 'xyz'"),
            ]
        )
        agent = report["agents"][0]
        self.assertEqual(agent["gate"]["verdict"], "BLOCK")
        self.assertEqual(agent["gate"]["mode"], "check")
        self.assertEqual(agent["gate"]["check"], {"contains": "xyz"})
        self.assertEqual(report["totals"]["blocked"], 1)

    def test_pass_gate_not_blocked(self):
        report = replay_run([started(1), result(1, "fine"), gate(1, "PASS", mode="check")])
        self.assertEqual(report["totals"]["blocked"], 0)

    def test_llm_mode_gate_attaches(self):
        report = replay_run(
            [started(1), result(1, "x"), gate(1, "PASS", mode="llm", reason="good")]
        )
        self.assertEqual(report["agents"][0]["gate"]["mode"], "llm")

    def test_agent_without_gate_has_no_gate_field(self):
        report = replay_run([started(1), result(1, "x"), gate(2, "BLOCK")])
        self.assertNotIn("gate", report["agents"][0])

    def test_gate_without_matching_agent_does_not_crash(self):
        report = replay_run([gate(99, "BLOCK")])
        self.assertEqual(report["totals"]["agents"], 0)

    def test_judge_agent_event_stays_ordinary_agent(self):
        # LLM-mode gates spawn a judge child whose events carry no gate event.
        report = replay_run(
            [
                started(1),
                result(1, "governed result"),
                gate(1, "PASS", mode="llm"),
                started(2),
                result(2, {"verdict": "PASS", "reason": "looks good"}),
            ]
        )
        self.assertEqual(report["totals"]["agents"], 2)
        self.assertIn("gate", report["agents"][0])
        self.assertNotIn("gate", report["agents"][1])
        self.assertEqual(report["agents"][1]["status"], "done")


class ReplayGateChecksOverrideTests(unittest.TestCase):
    def test_override_flips_block_to_pass(self):
        events = [started(1), result(1, "needs keyword"), gate(1, "BLOCK", check={"contains": "other"})]
        report = replay_run(events, {"gate_checks": {"1": {"contains": "keyword"}}})
        agent = report["agents"][0]
        self.assertEqual(agent["replayed_outcome"], "would-pass")
        self.assertEqual(agent["gate"]["verdict"], "BLOCK")

    def test_override_flips_pass_to_block(self):
        events = [started(1), result(1, "clean"), gate(1, "PASS", check={"contains": "bad"})]
        report = replay_run(events, {"gate_checks": {"1": {"contains": "poison"}}})
        self.assertEqual(report["agents"][0]["replayed_outcome"], "would-block")

    def test_override_by_gate_key(self):
        events = [
            started(1),
            result(1, "abc"),
            gate(1, "BLOCK", check={"contains": "zzz"}, key="gate:fp1"),
        ]
        report = replay_run(events, {"gate_checks": {"gate:fp1": {"contains": "abc"}}})
        self.assertEqual(report["agents"][0]["replayed_outcome"], "would-pass")

    def test_override_matching_unknown_gate_is_skipped(self):
        events = [started(1), result(1, "x"), gate(1, "BLOCK")]
        report = replay_run(events, {"gate_checks": {"gate:unknown": {"contains": "x"}}})
        self.assertNotIn("replayed_outcome", report["agents"][0])

    def test_override_on_field_equals(self):
        events = [
            started(1),
            result(1, {"answer": 42}),
            gate(1, "BLOCK", check={"field": "answer", "equals": 0}),
        ]
        report = replay_run(events, {"gate_checks": {"1": {"field": "answer", "equals": 42}}})
        self.assertEqual(report["agents"][0]["replayed_outcome"], "would-pass")

    def test_no_override_leaves_no_replayed_outcome(self):
        report = replay_run([started(1), result(1, "x"), gate(1, "BLOCK")])
        self.assertNotIn("replayed_outcome", report["agents"][0])


class ReplayBudgetOverrideTests(unittest.TestCase):
    def test_exceeding_budget_marks_would_exceed(self):
        events = [started(1), result(1, "a"), started(2), result(2, "b")]
        report = replay_run(events, {"budget_max_tokens": 1})
        self.assertEqual(report["agents"][0]["replayed_outcome"], "would-exceed")
        self.assertNotIn("replayed_outcome", report["agents"][1])

    def test_within_budget_no_marker(self):
        events = [started(1), result(1, "a"), started(2), result(2, "b")]
        report = replay_run(events, {"budget_max_tokens": 5})
        for agent in report["agents"]:
            self.assertNotIn("replayed_outcome", agent)

    def test_cached_agents_are_free(self):
        events = [started(1), result(1, "cached", cached=True), started(2), result(2, "live")]
        report = replay_run(events, {"budget_max_tokens": 1})
        self.assertNotIn("replayed_outcome", report["agents"][0])
        self.assertEqual(report["agents"][1]["replayed_outcome"], "would-exceed")

    def test_all_cached_never_exceeds(self):
        events = [started(1), result(1, "a", cached=True), started(2), result(2, "b", cached=True)]
        report = replay_run(events, {"budget_max_tokens": 1})
        for agent in report["agents"]:
            self.assertNotIn("replayed_outcome", agent)

    def test_non_positive_budget_ignored(self):
        for bad in (0, -1, 0.5, "x", None):
            report = replay_run([started(1), result(1, "a")], {"budget_max_tokens": bad})
            self.assertNotIn("replayed_outcome", report["agents"][0])

    def test_totals_reports_calls(self):
        events = [
            started(1), result(1, "a", cached=True),
            started(2), result(2, "b"),
            started(3), error(3, "boom"),
        ]
        report = replay_run(events)
        self.assertEqual(report["totals"]["calls"], 2)  # cached excluded


class ReplayToleranceTests(unittest.TestCase):
    def test_empty_events_yields_empty_report(self):
        report = replay_run([])
        self.assertEqual(report["totals"]["agents"], 0)
        self.assertEqual(report["agents"], [])

    def test_non_dict_events_skipped(self):
        report = replay_run([None, "junk", 42, {"type": "started", "agentId": "1"}])
        self.assertEqual(report["totals"]["agents"], 1)

    def test_event_missing_type_skipped(self):
        report = replay_run([{"agentId": "1"}])
        self.assertEqual(report["totals"]["agents"], 0)

    def test_unknown_event_types_ignored(self):
        report = replay_run(
            [
                {"type": "activity", "agentId": "1", "activity": "working"},
                {"type": "approval", "agentId": "1"},
                {"type": "phase", "title": "Review"},
                {"type": "started", "agentId": "1"},
                {"type": "result", "agentId": "1", "result": "x"},
            ]
        )
        self.assertEqual(report["totals"]["agents"], 1)

    def test_missing_agent_id_event_skipped(self):
        report = replay_run([{"type": "started"}, {"type": "result", "agentId": "1", "result": "x"}])
        self.assertEqual(report["totals"]["agents"], 1)

    def test_missing_result_field_skips_agent(self):
        report = replay_run([{"type": "result", "agentId": "1"}])
        self.assertEqual(report["totals"]["agents"], 0)

    def test_non_list_events_never_raises(self):
        report = replay_run(None)
        self.assertEqual(report["totals"]["agents"], 0)
        report = replay_run("not a list")
        self.assertEqual(report["totals"]["agents"], 0)

    def test_totals_breakdown(self):
        events = [
            started(1), result(1, "ok"),
            started(2), result(2, "ok"),
            started(3), error(3, "bad"),
            started(4), result(4, None, skipped=True),
            started(5), result(5, "cached", cached=True),
            started(6), result(6, "blocked"), gate(6, "BLOCK"),
        ]
        report = replay_run(events)
        totals = report["totals"]
        self.assertEqual(totals["agents"], 6)
        self.assertEqual(totals["succeeded"], 2)   # done, not blocked
        self.assertEqual(totals["failed"], 1)
        self.assertEqual(totals["skipped"], 1)
        self.assertEqual(totals["blocked"], 1)
        self.assertEqual(totals["cached"], 1)
        self.assertEqual(totals["calls"], 5)       # all but the cached one


if __name__ == "__main__":
    unittest.main()
