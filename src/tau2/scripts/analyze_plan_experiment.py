"""Summarize plan experiments from one or more tau2 results files."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
from tau2.data_model.simulation import Results, SimulationRun

RETRIEVAL_TOOLS = {
    "KB_search",
    "KB_search_bm25",
    "KB_search_dense",
    "KB_search_next",
    "grep",
    "shell",
}


@dataclass
class SimulationSummary:
    """Stable per-simulation metrics for incremental plan experiments."""

    source: str
    task_id: str
    trial: int | None
    reward: float | None
    termination_reason: str
    duration: float
    agent_cost: float | None
    user_cost: float | None
    conversation_messages: int
    tool_calls: int
    retrieval_calls: int
    write_actions: int
    last_completed_action: str | None
    shadow_plan_status: str | None
    shadow_plan_task_mode: str | None


def _load_results(path: Path) -> Results:
    if path.is_dir():
        path = path / "results.json"
    return Results.load(path)


def _tool_calls(simulation: SimulationRun) -> list[tuple[str, dict[str, Any]]]:
    calls = []
    for message in simulation.get_messages():
        if (
            not isinstance(message, (AssistantMessage, UserMessage))
            or not message.tool_calls
        ):
            continue
        calls.extend((call.name, call.arguments) for call in message.tool_calls)
    return calls


def _effective_call_name(name: str, arguments: dict[str, Any]) -> str:
    """Expose the business tool behind discoverable-tool wrapper calls."""
    if name in {"call_discoverable_agent_tool", "give_discoverable_user_tool"}:
        return str(
            arguments.get("agent_tool_name")
            or arguments.get("discoverable_tool_name")
            or name
        )
    return name


def _last_tool_result_name(simulation: SimulationRun) -> str | None:
    pending_names: list[str] = []
    last_name = None
    for message in simulation.get_messages():
        if isinstance(message, (AssistantMessage, UserMessage)) and message.tool_calls:
            pending_names.extend(
                _effective_call_name(call.name, call.arguments)
                for call in message.tool_calls
            )
        elif isinstance(message, ToolMessage) and pending_names:
            last_name = pending_names.pop(0)
    return last_name


def summarize_simulation(source: str, simulation: SimulationRun) -> SimulationSummary:
    """Create an experiment summary without benchmark-specific task logic."""
    calls = _tool_calls(simulation)
    effective_calls = [
        (_effective_call_name(name, arguments), arguments) for name, arguments in calls
    ]
    reward = simulation.reward_info.reward if simulation.reward_info else None
    diagnostics = (simulation.info or {}).get("shadow_plan") or {}
    action_checks = (
        simulation.reward_info.action_checks if simulation.reward_info else None
    ) or []
    write_names = {
        check.action.name for check in action_checks if check.tool_type == "write"
    }
    return SimulationSummary(
        source=source,
        task_id=str(simulation.task_id),
        trial=simulation.trial,
        reward=reward,
        termination_reason=str(simulation.termination_reason.value),
        duration=round(simulation.duration, 2),
        agent_cost=simulation.agent_cost,
        user_cost=simulation.user_cost,
        conversation_messages=len(simulation.get_messages()),
        tool_calls=len(calls),
        retrieval_calls=sum(name in RETRIEVAL_TOOLS for name, _ in calls),
        write_actions=sum(name in write_names for name, _ in effective_calls),
        last_completed_action=_last_tool_result_name(simulation),
        shadow_plan_status=diagnostics.get("status"),
        shadow_plan_task_mode=(diagnostics.get("plan") or {}).get("task_mode"),
    )


def analyze_paths(paths: Iterable[Path]) -> list[SimulationSummary]:
    """Load and summarize simulations from all supplied result paths."""
    summaries = []
    for path in paths:
        results = _load_results(path)
        source = str(path)
        summaries.extend(
            summarize_simulation(source, simulation)
            for simulation in results.simulations
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    rows = [asdict(summary) for summary in analyze_paths(args.paths)]
    if args.as_json:
        print(json.dumps(rows, indent=2))
        return

    columns = [
        "task_id",
        "reward",
        "termination_reason",
        "duration",
        "agent_cost",
        "tool_calls",
        "retrieval_calls",
        "write_actions",
        "last_completed_action",
        "shadow_plan_task_mode",
    ]
    print("\t".join(columns))
    for row in rows:
        print("\t".join(str(row[column]) for column in columns))


if __name__ == "__main__":
    main()
