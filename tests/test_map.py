from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from hermes_dynamic_workflows.core.config import PluginConfig
from hermes_dynamic_workflows.core.errors import WorkflowRuntimeError
from hermes_dynamic_workflows.core.types import ChildAgentRequest, ChildAgentRunner
from hermes_dynamic_workflows.engine.runtime import WorkflowOptions, run_workflow


class LabelRunner(ChildAgentRunner):
    """Returns content derived from the request label, for parallel attribution."""

    def __init__(self):
        self.requests: list[ChildAgentRequest] = []

    def run(self, request: ChildAgentRequest):
        self.requests.append(request)
        return f"OUT-{request.label}"


class SlowFastRunner(ChildAgentRunner):
    """Completes 'fast' requests before 'slow' ones despite input order."""

    def __init__(self):
        self.requests: list[ChildAgentRequest] = []

    def run(self, request: ChildAgentRequest):
        self.requests.append(request)
        if request.label == "slow":
            time.sleep(0.15)
        return f"OUT-{request.label}"


class FailLabelRunner(ChildAgentRunner):
    """Raises for requests whose label starts with 'boom'."""

    def __init__(self):
        self.requests: list[ChildAgentRequest] = []

    def run(self, request: ChildAgentRequest):
        self.requests.append(request)
        if request.label.startswith("boom"):
            raise RuntimeError(f"{request.label} crashed")
        return f"OUT-{request.label}"


def _run_script(script, runner, events):
    """Run a workflow with a fake child runner, capturing journal events."""
    with patch("hermes_dynamic_workflows.child.runner._discoverable_child_toolsets", return_value=[]):
        return run_workflow(
            script,
            WorkflowOptions(
                config=PluginConfig(),
                child_runner=runner,
                on_journal=events.append,
            ),
        )


class MapTests(unittest.TestCase):
    def test_map_returns_results_in_input_order_when_completion_is_out_of_order(self):
        runner = SlowFastRunner()
        events = []
        result = _run_script(
            'meta = {"name": "order", "description": "o"}\n'
            'return await map(["slow", "fast"], lambda item, idx: agent(f"task {item}", {"label": item}))',
            runner,
            events,
        )

        self.assertEqual(result.value, ["OUT-slow", "OUT-fast"])
        self.assertEqual({request.label for request in runner.requests}, {"slow", "fast"})
        self.assertEqual(result.agent_count, 2)

    def test_map_passes_item_and_index_to_thunk_and_passes_through_values(self):
        runner = LabelRunner()
        events = []
        result = _run_script(
            'meta = {"name": "item-idx", "description": "i"}\n'
            'return await map(["a", "b", "c"], lambda item, idx: f"{idx}:{item}")',
            runner,
            events,
        )

        self.assertEqual(result.value, ["0:a", "1:b", "2:c"])
        self.assertEqual(runner.requests, [])
        self.assertEqual(result.agent_count, 0)

    def test_map_empty_list_returns_empty_without_launching(self):
        runner = LabelRunner()
        events = []
        result = _run_script(
            'meta = {"name": "empty", "description": "e"}\n'
            'return await map([], lambda item, idx: agent(str(item), {"label": str(idx)}))',
            runner,
            events,
        )

        self.assertEqual(result.value, [])
        self.assertEqual(runner.requests, [])
        self.assertEqual(result.agent_count, 0)

    def test_map_non_iterable_items_raise_workflow_runtime_error_without_launching(self):
        runner = LabelRunner()
        events = []
        with self.assertRaises(WorkflowRuntimeError) as ctx:
            _run_script(
                'meta = {"name": "noniter", "description": "n"}\n'
                'return await map(42, lambda item, idx: agent(str(item), {"label": str(idx)}))',
                runner,
                events,
            )
        self.assertIn("iterable", str(ctx.exception))
        self.assertEqual(runner.requests, [])

    def test_map_non_callable_thunk_raises_workflow_runtime_error_without_launching(self):
        runner = LabelRunner()
        events = []
        with self.assertRaises(WorkflowRuntimeError) as ctx:
            _run_script(
                'meta = {"name": "nocall", "description": "n"}\n'
                'return await map(["a"], "not-a-callable")',
                runner,
                events,
            )
        self.assertIn("callable", str(ctx.exception))
        self.assertEqual(runner.requests, [])

    def test_map_single_failure_returns_none_and_keeps_other_results(self):
        runner = FailLabelRunner()
        events = []
        result = _run_script(
            'meta = {"name": "fail", "description": "f"}\n'
            'return await map(["ok", "boom"], lambda item, idx: agent(f"task {item}", {"label": item}))',
            runner,
            events,
        )

        self.assertEqual(result.value, ["OUT-ok", None])
        self.assertEqual(result.agent_count, 2)
        self.assertEqual(len(runner.requests), 2)

    def test_map_with_agent_counts_and_collects_labels(self):
        runner = LabelRunner()
        events = []
        result = _run_script(
            'meta = {"name": "agents", "description": "a"}\n'
            'return await map(["x", "y", "z"], lambda item, idx: agent(f"task {item}", {"label": f"m{idx}"}))',
            runner,
            events,
        )

        self.assertEqual(result.value, ["OUT-m0", "OUT-m1", "OUT-m2"])
        self.assertEqual(result.agent_count, 3)
        self.assertEqual({request.label for request in runner.requests}, {"m0", "m1", "m2"})

    def test_map_with_gates_attributes_events_to_their_own_agents(self):
        runner = LabelRunner()
        events = []
        result = _run_script(
            'meta = {"name": "map-gates", "description": "mg"}\n'
            "return await map([\"alpha\", \"beta\"], lambda item, idx: gate(item, {\"label\": item, \"check\": {\"contains\": \"OUT-\" + item}}))",
            runner,
            events,
        )

        self.assertEqual(result.value, ["OUT-alpha", "OUT-beta"])
        gate_events = [event for event in events if event.get("type") == "gate"]
        self.assertEqual(len(gate_events), 2)
        by_label = {request.label: str(request.id) for request in runner.requests}
        self.assertEqual(set(by_label), {"alpha", "beta"})
        for event in gate_events:
            self.assertEqual(event["verdict"], "PASS")
            self.assertEqual(event["mode"], "check")
            self.assertNotEqual(event["agentId"], "?")
            check_value = event["check"]["contains"]
            self.assertEqual(event["agentId"], by_label[check_value.removeprefix("OUT-")])
        self.assertNotEqual(by_label["alpha"], by_label["beta"])


if __name__ == "__main__":
    unittest.main()
