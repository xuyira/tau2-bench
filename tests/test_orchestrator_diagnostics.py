from unittest.mock import Mock

from tau2.data_model.simulation import TerminationReason
from tau2.orchestrator.orchestrator import Orchestrator


def test_finalize_collects_agent_diagnostics():
    orchestrator = object.__new__(Orchestrator)
    orchestrator.termination_reason = TerminationReason.USER_STOP
    orchestrator.to_role = None
    orchestrator.agent_state = None
    orchestrator.user_state = None
    orchestrator.agent = Mock()
    orchestrator.agent.get_simulation_diagnostics.return_value = {
        "shadow_plan": {"status": "success"}
    }
    orchestrator.user = Mock()
    orchestrator.user.voice_settings = None
    orchestrator.environment = Mock()
    orchestrator.task = Mock(id="task")
    orchestrator._run_start_perf = 0.0
    orchestrator._run_start_time = "2026-01-01T00:00:00"
    orchestrator.simulation_id = "sim"
    orchestrator.seed = 300
    orchestrator.mode = Mock(value="half_duplex")
    orchestrator.get_trajectory = Mock(return_value=[])
    orchestrator._finalize_voice_metadata = Mock()

    simulation = orchestrator._finalize()

    assert simulation.info == {"shadow_plan": {"status": "success"}}
