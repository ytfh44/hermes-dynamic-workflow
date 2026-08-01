from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from hermes_dynamic_workflows.core.config import PluginConfig
from hermes_dynamic_workflows.core.errors import (
    ChildAgentError,
    DynamicWorkflowError,
    GateBlocked,
    WorkflowHalt,
    WorkflowRuntimeError,
)
from hermes_dynamic_workflows.core.types import ChildAgentRequest, ChildAgentResult, ChildAgentRunner
from hermes_dynamic_workflows.engine.api import _GATE_JUDGE_SCHEMA
from hermes_dynamic_workflows.engine.cache import ResumeCache, agent_fingerprint, is_cache_miss
from hermes_dynamic_workflows.engine.runtime import WorkflowOptions, run_workflow

_BOOL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
}


class ContentRunner(ChildAgentRunner):
    """Returns fixed content, optionally as a captured structured result."""

    def __init__(self, content="done", structured=None, tokens=7):
        self.requests: list[ChildAgentRequest] = []
        self.content = content
        self.structured = structured
        self.tokens = tokens

    def run(self, request: ChildAgentRequest):
        self.requests.append(request)
        metadata = {"tokens": self.tokens}
        if self.structured is not None:
            metadata.update(
                {
                    "structured_captured": True,
                    "structured_result": self.structured,
                    "structured_attempts": 1,
                }
            )
        return ChildAgentResult(content=self.content, metadata=metadata)


class TextRunner(ChildAgentRunner):
    """Returns f"{label}:{prompt}" — mirrors test_runtime.FakeRunner's default."""

    def __init__(self):
        self.requests: list[ChildAgentRequest] = []

    def run(self, request: ChildAgentRequest):
        self.requests.append(request)
        return f"{request.label}:{request.prompt}"


class LabelRunner(ChildAgentRunner):
    """Returns content derived from the request label, for parallel attribution."""

    def __init__(self):
        self.requests: list[ChildAgentRequest] = []

    def run(self, request: ChildAgentRequest):
        self.requests.append(request)
        return f"OUT-{request.label}"


class FailSecondRunner(ChildAgentRunner):
    """Succeeds on the first (main) call, crashes on the second (judge) call."""

    def __init__(self):
        self.requests: list[ChildAgentRequest] = []

    def run(self, request: ChildAgentRequest):
        self.requests.append(request)
        if len(self.requests) == 2:
            raise RuntimeError("judge crashed")
        return ChildAgentResult(content="done", metadata={"tokens": 7})


def _run_script(script, runner, events, resume_cache=None):
    """Run a workflow with a fake child runner, capturing journal events."""
    with patch("hermes_dynamic_workflows.child.runner._discoverable_child_toolsets", return_value=[]):
        return run_workflow(
            script,
            WorkflowOptions(
                config=PluginConfig(),
                child_runner=runner,
                on_journal=events.append,
                resume_cache=resume_cache,
            ),
        )


def _gate_events(events):
    return [event for event in events if event.get("type") == "gate"]


def _gate_key(prompt, check):
    """Replicate the gate: fingerprint gate() computes for a gate invocation."""
    return "gate:" + agent_fingerprint(
        prompt,
        {
            "gate": True,
            "check": check,
            "model": None,
            "agentType": None,
            "isolation": None,
        },
    )


_JUDGE_SCRIPT = (
    'meta = {"name": "judge", "description": "j"}\n'
    'return await gate("work", {"judge": True, "label": "j"})'
)


class GateTests(unittest.TestCase):
    def test_check_contains_pass_returns_result_and_journals_pass(self):
        runner = ContentRunner("the quality is excellent, score 95")
        events = []
        result = _run_script(
            'meta = {"name": "contains", "description": "c"}\n'
            'return await gate("work", {"label": "g1", "check": {"contains": "excellent"}})',
            runner,
            events,
        )

        self.assertEqual(result.value, "the quality is excellent, score 95")
        self.assertEqual(result.agent_count, 1)
        gate_events = _gate_events(events)
        self.assertEqual(len(gate_events), 1)
        self.assertEqual(
            gate_events[0],
            {
                "type": "gate",
                "key": _gate_key("work", {"contains": "excellent"}),
                "agentId": str(runner.requests[0].id),
                "verdict": "PASS",
                "mode": "check",
                "reason": "contains 'excellent'",
                "check": {"contains": "excellent"},
            },
        )

    def test_check_contains_fail_raises_gateblocked_with_block_event(self):
        runner = ContentRunner("the quality is excellent, score 95")
        events = []
        with self.assertRaises(GateBlocked) as ctx:
            _run_script(
                'meta = {"name": "contains-fail", "description": "c"}\n'
                'return await gate("work", {"label": "g1", "check": {"contains": "terrible"}})',
                runner,
                events,
            )

        self.assertEqual(str(ctx.exception), "gate() check failed: expected contains 'terrible'")
        self.assertEqual(len(runner.requests), 1)
        gate_events = _gate_events(events)
        self.assertEqual(len(gate_events), 1)
        self.assertEqual(gate_events[0]["verdict"], "BLOCK")
        self.assertEqual(gate_events[0]["mode"], "check")
        self.assertEqual(gate_events[0]["reason"], "expected contains 'terrible'")
        self.assertEqual(gate_events[0]["agentId"], str(runner.requests[0].id))

    def test_check_not_contains_pass_and_fail(self):
        pass_runner = ContentRunner("the quality is excellent, score 95")
        pass_events = []
        result = _run_script(
            'meta = {"name": "not-contains", "description": "n"}\n'
            'return await gate("work", {"label": "g", "check": {"not_contains": "terrible"}})',
            pass_runner,
            pass_events,
        )
        self.assertEqual(result.value, "the quality is excellent, score 95")
        self.assertEqual(_gate_events(pass_events)[0]["verdict"], "PASS")

        fail_runner = ContentRunner("the quality is excellent, score 95")
        fail_events = []
        with self.assertRaises(GateBlocked) as ctx:
            _run_script(
                'meta = {"name": "not-contains-fail", "description": "n"}\n'
                'return await gate("work", {"label": "g", "check": {"not_contains": "excellent"}})',
                fail_runner,
                fail_events,
            )
        self.assertEqual(str(ctx.exception), "gate() check failed: expected not_contains 'excellent'")
        self.assertEqual(_gate_events(fail_events)[0]["verdict"], "BLOCK")

    def test_check_regex_pass_and_fail(self):
        pass_runner = ContentRunner("the quality is excellent, score 95")
        pass_events = []
        result = _run_script(
            'meta = {"name": "regex", "description": "r"}\n'
            'return await gate("work", {"label": "g", "check": {"regex": r"score \\d+"}})',
            pass_runner,
            pass_events,
        )
        self.assertEqual(result.value, "the quality is excellent, score 95")
        self.assertEqual(_gate_events(pass_events)[0]["verdict"], "PASS")

        fail_runner = ContentRunner("the quality is excellent, score 95")
        fail_events = []
        with self.assertRaises(GateBlocked) as ctx:
            _run_script(
                'meta = {"name": "regex-fail", "description": "r"}\n'
                'return await gate("work", {"label": "g", "check": {"regex": r"score \\d{4}"}})',
                fail_runner,
                fail_events,
            )
        self.assertIn("expected regex 'score", str(ctx.exception))
        self.assertEqual(_gate_events(fail_events)[0]["verdict"], "BLOCK")

    def test_check_field_equals_pass_on_structured_result(self):
        runner = ContentRunner(structured={"ok": True})
        events = []
        result = _run_script(
            'meta = {"name": "field", "description": "f"}\n'
            'return await gate("work", {"label": "f", "schema": '
            + repr(_BOOL_SCHEMA)
            + ', "check": {"field": "ok", "equals": True}})',
            runner,
            events,
        )

        self.assertEqual(result.value, {"ok": True})
        self.assertEqual(result.agent_count, 1)
        gate_events = _gate_events(events)
        self.assertEqual(gate_events[0]["verdict"], "PASS")
        self.assertEqual(gate_events[0]["mode"], "check")
        self.assertEqual(gate_events[0]["reason"], "field 'ok' equals True")

    def test_check_field_equals_fail_on_value_mismatch(self):
        runner = ContentRunner(structured={"ok": True})
        events = []
        with self.assertRaises(GateBlocked) as ctx:
            _run_script(
                'meta = {"name": "field-fail", "description": "f"}\n'
                'return await gate("work", {"label": "f", "schema": '
                + repr(_BOOL_SCHEMA)
                + ', "check": {"field": "ok", "equals": False}})',
                runner,
                events,
            )
        self.assertEqual(str(ctx.exception), "gate() check failed: expected field 'ok' equals False")
        self.assertEqual(_gate_events(events)[0]["verdict"], "BLOCK")

    def test_check_field_fails_on_missing_field_and_non_dict_result(self):
        missing_runner = ContentRunner(structured={"ok": True})
        missing_events = []
        with self.assertRaises(GateBlocked):
            _run_script(
                'meta = {"name": "field-missing", "description": "f"}\n'
                'return await gate("work", {"label": "f", "schema": '
                + repr(_BOOL_SCHEMA)
                + ', "check": {"field": "verdict", "equals": "PASS"}})',
                missing_runner,
                missing_events,
            )
        self.assertEqual(_gate_events(missing_events)[0]["verdict"], "BLOCK")

        non_dict_runner = ContentRunner("plain text result")
        non_dict_events = []
        with self.assertRaises(GateBlocked):
            _run_script(
                'meta = {"name": "field-nondict", "description": "f"}\n'
                'return await gate("work", {"label": "f", "check": {"field": "ok", "equals": True}})',
                non_dict_runner,
                non_dict_events,
            )
        self.assertEqual(_gate_events(non_dict_events)[0]["verdict"], "BLOCK")

    def test_invalid_check_configs_fail_fast_with_zero_child_calls(self):
        invalid_checks = [
            "not-a-dict",
            {"contains": "a", "not_contains": "b"},
            {"field": "x"},
            {"equals": 1},
            {"contains": ""},
            {"contains": 42},
            {"not_contains": "   "},
            {"regex": "["},
            {"nope": "x"},
            {"field": "", "equals": 1},
            {"field": "x", "equals": 1, "contains": "a"},
        ]
        for invalid in invalid_checks:
            with self.subTest(check=invalid):
                runner = ContentRunner("any content")
                events = []
                script = (
                    'meta = {"name": "bad-check", "description": "b"}\n'
                    "return await gate('work', {'label': 'x', 'check': "
                    + json.dumps(invalid)
                    + "})"
                )
                with self.assertRaises(WorkflowRuntimeError):
                    _run_script(script, runner, events)
                self.assertEqual(runner.requests, [])
                self.assertEqual(_gate_events(events), [])

    def test_judge_option_and_prompt_opts_must_be_valid(self):
        cases = [
            ('return await gate("work", {"judge": "yes"})', "judge option must be True"),
            ('return await gate("   ", {"judge": True})', "non-empty prompt"),
            ('return await gate("work", "not-a-dict")', "options must be a dict"),
        ]
        for body, fragment in cases:
            with self.subTest(body=body):
                runner = ContentRunner()
                events = []
                script = f'meta = {{"name": "misuse", "description": "m"}}\n{body}'
                with self.assertRaises(WorkflowRuntimeError) as ctx:
                    _run_script(script, runner, events)
                self.assertIn(fragment, str(ctx.exception))
                self.assertEqual(runner.requests, [])

    def test_judge_pass_journals_llm_pass_and_runs_two_agents(self):
        runner = ContentRunner(structured={"verdict": "PASS", "reason": "quality ok"})
        events = []
        result = _run_script(_JUDGE_SCRIPT, runner, events)

        self.assertEqual(result.value, "done")
        self.assertEqual(result.agent_count, 2)
        gate_events = _gate_events(events)
        self.assertEqual(len(gate_events), 1)
        self.assertEqual(
            gate_events[0],
            {
                "type": "gate",
                "key": _gate_key("work", None),
                "agentId": str(runner.requests[0].id),
                "verdict": "PASS",
                "mode": "llm",
                "reason": "quality ok",
                "check": None,
            },
        )

    def test_judge_block_raises_gateblocked(self):
        runner = ContentRunner(structured={"verdict": "BLOCK", "reason": "irrelevant"})
        events = []
        with self.assertRaises(GateBlocked) as ctx:
            _run_script(_JUDGE_SCRIPT, runner, events)

        self.assertEqual(str(ctx.exception), "gate() judge blocked the result: irrelevant")
        self.assertEqual(len(runner.requests), 2)
        gate_events = _gate_events(events)
        self.assertEqual(len(gate_events), 1)
        self.assertEqual(gate_events[0]["verdict"], "BLOCK")
        self.assertEqual(gate_events[0]["mode"], "llm")
        self.assertEqual(gate_events[0]["reason"], "irrelevant")
        self.assertEqual(gate_events[0]["agentId"], str(runner.requests[0].id))

    def test_judge_fail_closed_when_no_valid_verdict(self):
        cases = [
            (None, "judge could not obtain a verdict"),
            ({"reason": "missing verdict key"}, "missing verdict key"),
            ("not-a-dict", "verdict is not PASS"),
        ]
        for structured, fragment in cases:
            with self.subTest(structured=structured):
                runner = ContentRunner(structured=structured)
                events = []
                with self.assertRaises(GateBlocked) as ctx:
                    _run_script(_JUDGE_SCRIPT, runner, events)
                self.assertIn(fragment, str(ctx.exception))
                gate_events = _gate_events(events)
                self.assertEqual(len(gate_events), 1)
                self.assertEqual(gate_events[0]["verdict"], "BLOCK")
                self.assertEqual(gate_events[0]["mode"], "llm")

    def test_judge_fail_closed_when_child_raises(self):
        runner = FailSecondRunner()
        events = []
        with self.assertRaises(GateBlocked) as ctx:
            _run_script(_JUDGE_SCRIPT, runner, events)

        self.assertIn("judge could not obtain a verdict: judge crashed", str(ctx.exception))
        self.assertEqual(len(runner.requests), 2)
        gate_events = _gate_events(events)
        self.assertEqual(len(gate_events), 1)
        self.assertEqual(gate_events[0]["verdict"], "BLOCK")
        self.assertEqual(gate_events[0]["mode"], "llm")
        self.assertIn("judge could not obtain a verdict", gate_events[0]["reason"])

    def test_judge_verdict_is_cached_and_judge_not_rerun(self):
        script = (
            'meta = {"name": "judge-cache", "description": "c"}\n'
            'first = await gate("review X", {"judge": True, "label": "j"})\n'
            'second = await gate("review X", {"judge": True, "label": "j"})\n'
            "return [first, second]"
        )
        # Run 1 warms the resume cache (fresh cache: every call runs).
        warmup_runner = ContentRunner(structured={"verdict": "PASS", "reason": "ok"})
        warmup_cache = ResumeCache()
        warmup = _run_script(script, warmup_runner, [], resume_cache=warmup_cache)
        self.assertEqual(warmup.value, ["done", "done"])
        self.assertEqual(len(warmup_runner.requests), 4)  # 2 main agents + 2 judges
        judge_requests = [
            request for request in warmup_runner.requests if request.schema is not None
        ]
        self.assertEqual(len(judge_requests), 2)
        self.assertEqual(judge_requests[0].schema, _GATE_JUDGE_SCHEMA)

        # Run 2 resumes from run 1's produced cache: main children and judge
        # verdicts are served from the cache — the runner is never invoked.
        resume_runner = ContentRunner(structured={"verdict": "PASS", "reason": "ok"})
        events = []
        result = _run_script(
            script,
            resume_runner,
            events,
            resume_cache=ResumeCache(previous=warmup_cache.current),
        )
        self.assertEqual(result.value, ["done", "done"])
        self.assertEqual(len(resume_runner.requests), 0)
        # The gate: fingerprint cache held the judge's verdicts.
        gate_key = _gate_key("review X", None)
        self.assertFalse(is_cache_miss(warmup_cache.current.get(gate_key)))
        self.assertEqual(
            warmup_cache.current.get(gate_key),
            [{"verdict": "PASS", "reason": "ok"}] * 2,
        )
        gate_events = _gate_events(events)
        self.assertEqual(len(gate_events), 2)
        self.assertEqual([event["verdict"] for event in gate_events], ["PASS", "PASS"])
        self.assertEqual([event["mode"] for event in gate_events], ["llm", "llm"])
        self.assertEqual({event["key"] for event in gate_events}, {gate_key})

    def test_check_revalidates_cached_result_without_rerunning_child(self):
        script = (
            'meta = {"name": "cached-check", "description": "c"}\n'
            'first = await gate("mission report", {"label": "r", "check": {"contains": "GOOD"}})\n'
            "try:\n"
            '    second = await gate("mission report", {"label": "r", "check": {"contains": "NOPE"}})\n'
            '    return "second passed"\n'
            "except Exception:\n"
            '    return "second blocked"'
        )
        # Run 1 warms the resume cache (fresh cache: every child runs).
        warmup_runner = ContentRunner("GOOD report")
        warmup_cache = ResumeCache()
        warmup = _run_script(script, warmup_runner, [], resume_cache=warmup_cache)
        self.assertEqual(warmup.value, "second blocked")
        self.assertEqual(len(warmup_runner.requests), 2)

        # Run 2 resumes from run 1's produced cache: the runner is never invoked,
        # yet the second gate's different check still re-validates the cached
        # content and blocks it.
        resume_runner = ContentRunner("GOOD report")
        events = []
        result = _run_script(
            script,
            resume_runner,
            events,
            resume_cache=ResumeCache(previous=warmup_cache.current),
        )
        self.assertEqual(result.value, "second blocked")
        self.assertEqual(len(resume_runner.requests), 0)
        gate_events = _gate_events(events)
        self.assertEqual(len(gate_events), 2)
        self.assertEqual([event["verdict"] for event in gate_events], ["PASS", "BLOCK"])
        self.assertEqual([event["mode"] for event in gate_events], ["check", "check"])
        self.assertNotEqual(gate_events[0]["agentId"], gate_events[1]["agentId"])
        for event in gate_events:
            self.assertNotEqual(event["agentId"], "?")

    def test_gate_token_accounting(self):
        check_runner = ContentRunner("the quality is excellent", tokens=7)
        check_events = []
        check_result = _run_script(
            'meta = {"name": "tokens", "description": "t"}\n'
            'return await gate("work", {"label": "t", "check": {"contains": "excellent"}})',
            check_runner,
            check_events,
        )
        self.assertEqual(check_result.agent_count, 1)
        self.assertEqual(check_result.state.snapshot()["agents"][0]["tokens"], 7)

        judge_runner = ContentRunner(structured={"verdict": "PASS", "reason": "ok"}, tokens=7)
        judge_events = []
        judge_result = _run_script(_JUDGE_SCRIPT, judge_runner, judge_events)
        self.assertEqual(judge_result.agent_count, 2)
        agents = judge_result.state.snapshot()["agents"]
        self.assertEqual(sum(agent["tokens"] for agent in agents), 14)

    def test_parallel_gates_attributed_to_their_own_agents(self):
        runner = LabelRunner()
        events = []
        result = _run_script(
            'meta = {"name": "parallel-gates", "description": "pg"}\n'
            "return await parallel([\n"
            '    lambda: gate("alpha", {"label": "alpha", "check": {"contains": "OUT-alpha"}}),\n'
            '    lambda: gate("beta", {"label": "beta", "check": {"contains": "OUT-beta"}}),\n'
            "])",
            runner,
            events,
        )

        self.assertEqual(result.value, ["OUT-alpha", "OUT-beta"])
        by_label = {request.label: str(request.id) for request in runner.requests}
        self.assertEqual(set(by_label), {"alpha", "beta"})
        gate_events = _gate_events(events)
        self.assertEqual(len(gate_events), 2)
        for event in gate_events:
            self.assertEqual(event["verdict"], "PASS")
            self.assertEqual(event["mode"], "check")
            self.assertNotEqual(event["agentId"], "?")
            check_value = event["check"]["contains"]
            self.assertEqual(event["agentId"], by_label[check_value.removeprefix("OUT-")])
        self.assertNotEqual(by_label["alpha"], by_label["beta"])

    def test_gateblocked_recoverable_via_except_exception(self):
        runner = ContentRunner("bad output")
        events = []
        result = _run_script(
            'meta = {"name": "recover", "description": "r"}\n'
            "try:\n"
            '    out = await gate("bad output", {"label": "g", "check": {"contains": "GOOD"}})\n'
            '    return "passed"\n'
            "except Exception as exc:\n"
            '    return f"degraded:{exc}"',
            runner,
            events,
        )

        self.assertEqual(
            result.value, "degraded:gate() check failed: expected contains 'GOOD'"
        )
        self.assertEqual(_gate_events(events)[0]["verdict"], "BLOCK")

    def test_gateblocked_is_dynamic_workflow_error_not_halt(self):
        self.assertTrue(issubclass(GateBlocked, DynamicWorkflowError))
        self.assertTrue(issubclass(GateBlocked, Exception))
        self.assertFalse(issubclass(GateBlocked, WorkflowHalt))
        self.assertFalse(issubclass(GateBlocked, ChildAgentError))

    def test_gate_without_check_or_judge_is_plain_agent_passthrough(self):
        runner = TextRunner()
        events = []
        result = _run_script(
            'meta = {"name": "passthrough", "description": "p"}\n'
            'return await gate("plain task", {"label": "p1"})',
            runner,
            events,
        )

        self.assertEqual(result.value, "p1:plain task")
        self.assertEqual(result.agent_count, 1)
        self.assertEqual(_gate_events(events), [])


if __name__ == "__main__":
    unittest.main()
