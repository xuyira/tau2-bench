from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.scripts.analyze_plan_experiment import summarize_simulation


def test_summarize_simulation_counts_calls_and_shadow_plan():
    simulation = SimulationRun(
        id="sim",
        task_id="task_001",
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:00:01",
        duration=1.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=[
            UserMessage(role="user", content="Help"),
            AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(id="1", name="KB_search", arguments={"query": "q"})
                ],
            ),
            ToolMessage(role="tool", id="1", content="result"),
        ],
        info={
            "shadow_plan": {
                "status": "success",
                "plan": {"task_mode": "recommendation"},
            }
        },
    )

    summary = summarize_simulation("results.json", simulation)

    assert summary.tool_calls == 1
    assert summary.retrieval_calls == 1
    assert summary.last_completed_action == "KB_search"
    assert summary.shadow_plan_status == "success"
    assert summary.shadow_plan_task_mode == "recommendation"
