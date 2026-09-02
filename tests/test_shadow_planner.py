from unittest.mock import patch

from tau2.agent.observations import Observation
from tau2.agent.plan import PlanState, RetrievalPlan, SelectionPlan
from tau2.agent.shadow_planner import ShadowPlanningLLMAgent
from tau2.data_model.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import as_tool


def KB_search(
    query: str,
    product_category: str | None = None,
    product_names: list[str] | None = None,
    coverage: str = "relevance",
) -> str:
    """Search a test knowledge base."""
    return query


KB_SEARCH_TOOL = as_tool(KB_search)


def valid_plan_json() -> str:
    return PlanState(
        goal="Help the customer",
        capabilities={"workflow"},
        success_conditions=[
            {"id": "resolved", "description": "The request is resolved"}
        ],
        steps=[
            {
                "id": "understand",
                "kind": "ask_user",
                "description": "Clarify the request",
                "status": "ready",
            }
        ],
        current_step_id="understand",
    ).model_dump_json()


def planner_response(plan: PlanState | None = None) -> AssistantMessage:
    if plan is None:
        plan = PlanState.model_validate_json(valid_plan_json())
    return AssistantMessage(
        role="assistant",
        tool_calls=[
            ToolCall(
                id="plan",
                name="submit_shadow_plan",
                arguments={"plan": plan.model_dump(mode="json")},
            )
        ],
    )


def test_retrieval_plan_round_trip():
    selection = SelectionPlan(
        candidate_scope="checking accounts",
        objective="maximize",
        objective_expression="combined referral bonus",
        required_attributes=["referrer bonus", "referred bonus"],
        retrieval=RetrievalPlan(
            resource_type="product",
            product_categories=["checking_account"],
            document_intents=["referral_program"],
            coverage="all_products",
            queries=["checking account referral bonus"],
        ),
    )
    assert SelectionPlan.model_validate_json(selection.model_dump_json()) == selection


def test_current_ready_step_uses_dependency_order():
    plan = PlanState(
        goal="Resolve request",
        capabilities={"workflow"},
        success_conditions=[{"id": "done", "description": "Done"}],
        steps=[
            {"id": "first", "kind": "read", "description": "Read", "status": "ready"},
            {
                "id": "second",
                "kind": "write",
                "description": "Write",
                "depends_on": ["first"],
            },
        ],
    )
    assert ShadowPlanningLLMAgent._current_ready_step(plan).id == "first"
    plan.transition_step("first", "completed")
    assert ShadowPlanningLLMAgent._current_ready_step(plan).id == "second"


def test_step_tool_names_are_explicit_only():
    plan = PlanState(
        goal="Resolve request",
        capabilities={"workflow"},
        success_conditions=[{"id": "done", "description": "Done"}],
        steps=[
            {
                "id": "read",
                "kind": "read",
                "description": "Read",
                "completion_evidence": {"tool_names": ["get_referrals_by_user"]},
            }
        ],
    )
    assert ShadowPlanningLLMAgent._step_tool_names(plan.steps[0]) == {
        "get_referrals_by_user"
    }


def test_observation_normalizes_wrapper_tool_name():
    observation = Observation(
        event_id="call-1",
        event_type="tool_call",
        tool_name="get_referrals_by_user",
        wrapper_name="call_discoverable_agent_tool",
        arguments={"agent_tool_name": "get_referrals_by_user"},
        success=True,
    )
    assert observation.tool_name == "get_referrals_by_user"
    assert observation.wrapper_name == "call_discoverable_agent_tool"


def make_agent(get_environment):
    environment = get_environment()
    return ShadowPlanningLLMAgent(
        llm="test-model",
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
    )


def test_shadow_planner_uses_runtime_metadata_catalog(get_environment):
    environment = get_environment()
    agent = ShadowPlanningLLMAgent(
        llm="test-model",
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        metadata_catalog={
            "checking_account": {
                "Blue Account": ["doc_checking_accounts_blue_account_010"]
            }
        },
    )

    with patch(
        "tau2.agent.shadow_planner.generate", return_value=planner_response()
    ) as planner_generate:
        agent._run_shadow_planner(UserMessage(role="user", content="Compare accounts"))

    prompt = planner_generate.call_args.kwargs["messages"][0].content
    assert '"inventory"' in prompt
    assert '"checking_account"' in prompt
    assert '"Blue Account"' in prompt
    assert "doc_checking_accounts_blue_account_010" in prompt


def test_bootstrap_retrieval_runs_before_planner_and_is_injected(get_environment):
    environment = get_environment()
    search_tool = KB_SEARCH_TOOL.model_copy(deep=True)
    search_tool._func = lambda **kwargs: "Retrieved credit-limit workflow evidence"
    agent = ShadowPlanningLLMAgent(
        llm="test-model",
        tools=environment.get_tools() + [search_tool],
        domain_policy=environment.get_policy(),
    )
    state = agent.get_init_state()

    with (
        patch.object(search_tool, "_call", wraps=search_tool._call) as bootstrap_search,
        patch(
            "tau2.agent.shadow_planner.generate", return_value=planner_response()
        ) as planner_generate,
        patch(
            "tau2.agent.llm_agent.generate",
            return_value=AssistantMessage(role="assistant", content="Response"),
        ),
    ):
        agent.generate_next_message(
            UserMessage(role="user", content="Increase my credit limit"), state
        )

    bootstrap_search.assert_called_once_with(
        query="Increase my credit limit", coverage="relevance"
    )
    planner_prompt = planner_generate.call_args.kwargs["messages"][0].content
    assert "<bootstrap_evidence>" in planner_prompt
    assert "Retrieved credit-limit workflow evidence" in planner_prompt
    assert state.bootstrap_evidence == "Retrieved credit-limit workflow evidence"


def test_bootstrap_retrieval_failure_does_not_block_planner(get_environment):
    environment = get_environment()
    search_tool = KB_SEARCH_TOOL.model_copy(deep=True)
    search_tool._func = lambda **kwargs: (_ for _ in ()).throw(
        RuntimeError("retrieval unavailable")
    )
    agent = ShadowPlanningLLMAgent(
        llm="test-model",
        tools=environment.get_tools() + [search_tool],
        domain_policy=environment.get_policy(),
    )
    state = agent.get_init_state()

    with (
        patch("tau2.agent.shadow_planner.generate", return_value=planner_response()),
        patch(
            "tau2.agent.llm_agent.generate",
            return_value=AssistantMessage(role="assistant", content="Response"),
        ),
    ):
        response, state = agent.generate_next_message(
            UserMessage(role="user", content="Help"), state
        )

    assert response.content == "Response"
    assert state.bootstrap_evidence is None
    assert state.bootstrap_retrieval_error == "RuntimeError: retrieval unavailable"
    assert state.plan is not None


def test_shadow_plan_is_readonly_executor_context(get_environment):
    agent = make_agent(get_environment)
    state = agent.get_init_state()
    user_message = UserMessage(role="user", content="Please help")
    executor_response = AssistantMessage(role="assistant", content="How can I help?")
    shadow_response = planner_response()

    with (
        patch(
            "tau2.agent.shadow_planner.generate",
            side_effect=[shadow_response],
        ) as planner_generate,
        patch(
            "tau2.agent.llm_agent.generate",
            return_value=executor_response,
        ) as executor_generate,
    ):
        response, state = agent.generate_next_message(user_message, state)

    assert response == executor_response
    assert state.messages == [user_message, executor_response]
    assert planner_generate.call_count == 1
    executor_messages = executor_generate.call_args.kwargs["messages"]
    assert state.messages == [user_message, executor_response]
    assert len(state.readonly_context) == 1
    assert "<execution_plan>" in state.readonly_context[0].content
    assert "Help the customer" in state.readonly_context[0].content
    assert executor_messages == (
        state.system_messages + state.readonly_context + [user_message]
    )
    assert "submit_shadow_plan" not in state.readonly_context[0].content


def test_non_selection_executor_schema_only_allows_relevance(get_environment):
    agent = make_agent(get_environment)
    agent.tools.append(KB_SEARCH_TOOL.model_copy(deep=True))
    state = agent.get_init_state()

    with (
        patch("tau2.agent.shadow_planner.generate", return_value=planner_response()),
        patch(
            "tau2.agent.llm_agent.generate",
            return_value=AssistantMessage(role="assistant", content="Response"),
        ),
    ):
        agent.generate_next_message(UserMessage(role="user", content="Help"), state)

    schema = next(tool for tool in agent.tools if tool.name == "KB_search").params
    properties = schema.model_json_schema()["properties"]
    assert properties["coverage"]["const"] == "relevance"
    assert "product_category" not in properties
    assert "product_names" not in properties


def test_mixed_plan_schema_follows_current_retrieval_request(get_environment):
    agent = make_agent(get_environment)
    agent.tools.append(KB_SEARCH_TOOL.model_copy(deep=True))
    plan = PlanState(
        goal="Compare accounts and find the referral workflow",
        capabilities={"selection", "workflow"},
        selection={
            "candidate_scope": "accounts",
            "objective": "maximize",
            "objective_expression": "bonus",
            "required_attributes": ["bonus"],
        },
        retrieval_requests=[
            {
                "id": "workflow",
                "purpose": "Find referral workflow",
                "query": "referral workflow",
                "mode": "relevance",
            },
            {
                "id": "products",
                "purpose": "Compare products",
                "query": "account referral bonuses",
                "mode": "all_products",
                "resource_type": "product",
                "product_category": "checking_account",
                "depends_on": ["workflow"],
            },
        ],
        success_conditions=[{"id": "done", "description": "Resolved"}],
        steps=[
            {
                "id": "retrieve",
                "kind": "retrieve",
                "description": "Retrieve evidence",
                "retrieval_request_ids": ["workflow", "products"],
                "status": "ready",
            }
        ],
        current_step_id="retrieve",
    )

    agent._configure_executor_retrieval_schema(plan)
    properties = KB_SEARCH_TOOL.params.model_json_schema()["properties"]
    configured = next(tool for tool in agent.tools if tool.name == "KB_search")
    properties = configured.params.model_json_schema()["properties"]
    assert properties["coverage"]["const"] == "relevance"
    assert "product_category" not in properties

    plan.retrieval_requests[0].status = "completed"
    agent._configure_executor_retrieval_schema(plan)
    properties = configured.params.model_json_schema()["properties"]
    assert properties["coverage"]["const"] == "all_products"
    assert properties["product_category"]["const"] == "checking_account"


def test_selection_executor_schema_allows_all_products(get_environment):
    environment = get_environment()
    agent = ShadowPlanningLLMAgent(
        llm="test-model",
        tools=environment.get_tools() + [KB_SEARCH_TOOL.model_copy(deep=True)],
        domain_policy=environment.get_policy(),
        metadata_catalog={
            "checking_account": {
                "Blue Account": ["doc_checking_accounts_blue_account_010"]
            }
        },
    )
    plan = PlanState(
        goal="Choose the best account",
        capabilities={"selection"},
        selection={
            "candidate_scope": "checking accounts",
            "objective": "maximize",
            "objective_expression": "combined bonus",
            "required_attributes": ["bonus"],
            "retrieval": {
                "resource_type": "product",
                "product_categories": ["checking_account"],
                "coverage": "all_products",
                "queries": ["checking account bonus"],
            },
        },
        success_conditions=[{"id": "chosen", "description": "Choice made"}],
        steps=[
            {
                "id": "retrieve",
                "kind": "retrieve",
                "description": "Retrieve candidates",
                "status": "ready",
            }
        ],
        current_step_id="retrieve",
    )

    agent._configure_executor_retrieval_schema(plan)

    schema = next(tool for tool in agent.tools if tool.name == "KB_search").params
    properties = schema.model_json_schema()["properties"]
    assert properties["coverage"]["const"] == "all_products"
    assert properties["product_category"]["const"] == "checking_account"


def test_top_level_selection_request_controls_coverage_gate(get_environment):
    agent = make_agent(get_environment)
    state = agent.get_init_state()
    state.plan = PlanState(
        goal="Compare every checking account",
        capabilities={"selection"},
        selection={
            "candidate_scope": "checking accounts",
            "objective": "maximize",
            "objective_expression": "combined bonus",
            "required_attributes": ["bonus"],
        },
        retrieval_requests=[
            {
                "id": "checking_products",
                "purpose": "Retrieve every checking account",
                "query": "checking account bonus",
                "mode": "all_products",
                "resource_type": "product",
                "product_category": "checking_account",
            }
        ],
        success_conditions=[{"id": "chosen", "description": "Choice made"}],
        steps=[
            {
                "id": "retrieve",
                "kind": "retrieve",
                "description": "Retrieve candidates",
                "retrieval_request_ids": ["checking_products"],
                "status": "ready",
            }
        ],
        current_step_id="retrieve",
    )

    assert agent._retrieval_complete(state) is False
    request = state.plan.retrieval_requests[0]
    request.status = "completed"
    request.progress.coverage_complete = True
    assert agent._retrieval_complete(state) is True


def test_kb_search_updates_top_level_product_coverage_progress(get_environment):
    agent = make_agent(get_environment)
    state = agent.get_init_state()
    state.plan = PlanState(
        goal="Compare checking accounts",
        capabilities={"selection"},
        selection={
            "candidate_scope": "checking accounts",
            "objective": "maximize",
            "objective_expression": "bonus",
            "required_attributes": ["bonus"],
        },
        retrieval_requests=[
            {
                "id": "products",
                "purpose": "Retrieve every account",
                "query": "checking account bonus",
                "mode": "all_products",
                "resource_type": "product",
                "product_category": "checking_account",
            }
        ],
        success_conditions=[{"id": "chosen", "description": "Choice made"}],
        steps=[
            {
                "id": "retrieve",
                "kind": "retrieve",
                "description": "Retrieve products",
                "retrieval_request_ids": ["products"],
                "status": "ready",
            }
        ],
        current_step_id="retrieve",
    )

    agent._record_retrieval_observation(
        state,
        {
            "query": "checking account bonus",
            "product_category": "checking_account",
            "coverage": "all_products",
        },
        """[Product Coverage]
{"covered_products":["Blue Account"],"missing_products":["Green Account"],"coverage_complete":false}

1. Blue Account""",
    )
    request = state.plan.retrieval_requests[0]
    assert request.status == "in_progress"
    assert request.progress.attempts == 1
    assert request.progress.covered_product_names == ["Blue Account"]
    assert request.progress.missing_product_names == ["Green Account"]
    assert agent._retrieval_complete(state) is False

    agent._record_retrieval_observation(
        state,
        {
            "query": "checking account bonus",
            "product_category": "checking_account",
            "product_names": ["Green Account"],
            "coverage": "all_products",
        },
        """[Product Coverage]
{"covered_products":["Green Account"],"missing_products":[],"coverage_complete":true}

1. Green Account""",
    )
    assert request.status == "completed"
    assert request.progress.attempts == 2
    assert request.progress.covered_product_names == [
        "Blue Account",
        "Green Account",
    ]
    assert request.progress.missing_product_names == []
    assert request.progress.coverage_complete is True
    assert request.progress.stop_reason == "coverage_complete"
    assert state.covered_product_categories == ["checking_account"]
    assert agent._retrieval_complete(state) is True


def test_kb_search_updates_matching_relevance_request(get_environment):
    agent = make_agent(get_environment)
    state = agent.get_init_state()
    state.plan = PlanState(
        goal="Resolve a cash back issue",
        capabilities={"investigation"},
        retrieval_requests=[
            {
                "id": "reward_rules",
                "purpose": "Find reward rules",
                "query": "Silver Rewards cash back rules",
                "mode": "relevance",
            },
            {
                "id": "dispute_workflow",
                "purpose": "Find dispute workflow",
                "query": "cash back dispute workflow",
                "mode": "relevance",
            },
        ],
        success_conditions=[{"id": "resolved", "description": "Issue resolved"}],
        steps=[
            {
                "id": "retrieve",
                "kind": "retrieve",
                "description": "Retrieve evidence",
                "retrieval_request_ids": ["reward_rules", "dispute_workflow"],
                "status": "ready",
            }
        ],
        current_step_id="retrieve",
    )

    agent._record_retrieval_observation(
        state,
        {"query": "cash back dispute workflow", "coverage": "relevance"},
        "Relevant workflow document",
    )

    reward_rules, dispute_workflow = state.plan.retrieval_requests
    assert reward_rules.status == "pending"
    assert dispute_workflow.status == "completed"
    assert dispute_workflow.progress.attempts == 1
    assert dispute_workflow.progress.query_history == ["cash back dispute workflow"]


def test_retrieval_step_waits_for_every_linked_relevance_request(get_environment):
    agent = make_agent(get_environment)
    state = agent.get_init_state()
    state.plan = PlanState(
        goal="Resolve a cash back issue",
        capabilities={"investigation"},
        retrieval_requests=[
            {"id": "rules", "purpose": "Find rules", "query": "reward rules"},
            {
                "id": "workflow",
                "purpose": "Find workflow",
                "query": "dispute workflow",
            },
        ],
        success_conditions=[{"id": "done", "description": "Issue resolved"}],
        steps=[
            {
                "id": "retrieve",
                "kind": "retrieve",
                "description": "Retrieve all required evidence",
                "retrieval_request_ids": ["rules", "workflow"],
                "status": "ready",
            },
            {
                "id": "analyze",
                "kind": "analyze",
                "description": "Analyze evidence",
                "depends_on": ["retrieve"],
            },
        ],
        current_step_id="retrieve",
    )

    first, second = state.plan.retrieval_requests
    first.status = "completed"
    agent._complete_ready_retrieval_step(state)
    assert state.plan.steps[0].status == "ready"

    second.status = "completed"
    agent._complete_ready_retrieval_step(state)
    assert state.plan.steps[0].status == "completed"


def test_controller_executes_ready_relevance_requests(get_environment):
    search_tool = KB_SEARCH_TOOL.model_copy(deep=True)
    calls = []

    def search(**kwargs):
        calls.append(kwargs)
        return f"Evidence for {kwargs['query']}"

    search_tool._func = search
    agent = ShadowPlanningLLMAgent(
        llm="test-model",
        tools=[search_tool],
        domain_policy="policy",
    )
    state = agent.get_init_state()
    state.plan = PlanState(
        goal="Investigate rewards",
        capabilities={"investigation"},
        retrieval_requests=[
            {"id": "rules", "purpose": "Find rules", "query": "reward rules"},
            {
                "id": "workflow",
                "purpose": "Find workflow",
                "query": "dispute workflow",
            },
        ],
        success_conditions=[{"id": "done", "description": "Resolved"}],
        steps=[
            {
                "id": "retrieve",
                "kind": "retrieve",
                "description": "Retrieve evidence",
                "retrieval_request_ids": ["rules", "workflow"],
                "status": "ready",
            },
            {
                "id": "answer",
                "kind": "confirm",
                "description": "Answer",
                "depends_on": ["retrieve"],
            },
        ],
        current_step_id="retrieve",
    )

    agent._execute_ready_retrieval_requests(state)

    assert calls == [
        {"query": "reward rules", "coverage": "relevance"},
        {"query": "dispute workflow", "coverage": "relevance"},
    ]
    assert all(
        request.status == "completed" for request in state.plan.retrieval_requests
    )
    assert state.plan.steps[0].status == "completed"
    assert state.plan.current_step_id == "answer"
    assert state.retrieval_evidence == {
        "rules": ["Evidence for reward rules"],
        "workflow": ["Evidence for dispute workflow"],
    }


def test_controller_continues_all_products_with_missing_names(get_environment):
    search_tool = KB_SEARCH_TOOL.model_copy(deep=True)
    calls = []

    def search(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return (
                "[Product Coverage]\n"
                '{"covered_products":["Blue"],"missing_products":["Green"],'
                '"coverage_complete":false}'
            )
        return (
            "[Product Coverage]\n"
            '{"covered_products":["Green"],"missing_products":[],'
            '"coverage_complete":true}'
        )

    search_tool._func = search
    agent = ShadowPlanningLLMAgent(
        llm="test-model",
        tools=[search_tool],
        domain_policy="policy",
    )
    state = agent.get_init_state()
    state.plan = PlanState(
        goal="Compare accounts",
        capabilities={"selection"},
        selection={
            "candidate_scope": "accounts",
            "objective": "maximize",
            "objective_expression": "bonus",
            "required_attributes": ["bonus"],
        },
        retrieval_requests=[
            {
                "id": "products",
                "purpose": "Compare account bonuses",
                "query": "account bonuses",
                "mode": "all_products",
                "resource_type": "product",
                "product_category": "checking_account",
            }
        ],
        success_conditions=[{"id": "done", "description": "Compared"}],
        steps=[
            {
                "id": "retrieve",
                "kind": "retrieve",
                "description": "Retrieve products",
                "retrieval_request_ids": ["products"],
                "status": "ready",
            },
            {
                "id": "answer",
                "kind": "confirm",
                "description": "Answer",
                "depends_on": ["retrieve"],
            },
        ],
        current_step_id="retrieve",
    )

    agent._execute_ready_retrieval_requests(state)

    assert calls == [
        {
            "query": "account bonuses",
            "product_category": "checking_account",
            "coverage": "all_products",
        },
        {
            "query": "account bonuses",
            "product_category": "checking_account",
            "product_names": ["Green"],
            "coverage": "all_products",
        },
    ]
    request = state.plan.retrieval_requests[0]
    assert request.status == "completed"
    assert request.progress.attempts == 2
    assert request.progress.covered_product_names == ["Blue", "Green"]
    assert state.plan.current_step_id == "answer"


def test_controller_stops_all_products_after_one_stalled_rewrite(get_environment):
    search_tool = KB_SEARCH_TOOL.model_copy(deep=True)
    calls = []

    def search(**kwargs):
        calls.append(kwargs)
        return (
            "[Product Coverage]\n"
            '{"covered_products":[],"missing_products":["Blue"],'
            '"coverage_complete":false}'
        )

    search_tool._func = search
    agent = ShadowPlanningLLMAgent(
        llm="test-model", tools=[search_tool], domain_policy="policy"
    )
    state = agent.get_init_state()
    state.plan = PlanState(
        goal="Compare accounts",
        capabilities={"selection"},
        selection={
            "candidate_scope": "accounts",
            "objective": "maximize",
            "objective_expression": "bonus",
            "required_attributes": ["bonus"],
        },
        retrieval_requests=[
            {
                "id": "products",
                "purpose": "Compare account bonuses",
                "query": "account bonuses",
                "mode": "all_products",
                "resource_type": "product",
                "product_category": "checking_account",
            }
        ],
        success_conditions=[{"id": "done", "description": "Compared"}],
        steps=[
            {
                "id": "retrieve",
                "kind": "retrieve",
                "description": "Retrieve products",
                "retrieval_request_ids": ["products"],
                "status": "ready",
            }
        ],
        current_step_id="retrieve",
    )

    agent._execute_ready_retrieval_requests(state)

    request = state.plan.retrieval_requests[0]
    assert len(calls) == 2
    assert calls[0]["query"] == "account bonuses"
    assert calls[1]["query"].startswith("Compare account bonuses.")
    assert calls[1]["product_names"] == ["Blue"]
    assert request.status == "incomplete"
    assert request.progress.query_rewrite_attempted is True
    assert request.progress.stop_reason == "no_progress"
    assert state.plan.steps[0].status == "failed"


def test_controller_retrieval_failure_is_recorded_without_tool_history(
    get_environment,
):
    search_tool = KB_SEARCH_TOOL.model_copy(deep=True)
    search_tool._func = lambda **kwargs: (_ for _ in ()).throw(
        RuntimeError("retrieval unavailable")
    )
    agent = ShadowPlanningLLMAgent(
        llm="test-model", tools=[search_tool], domain_policy="policy"
    )
    state = agent.get_init_state()
    state.plan = PlanState(
        goal="Find a workflow",
        capabilities={"workflow"},
        retrieval_requests=[
            {"id": "workflow", "purpose": "Find workflow", "query": "workflow"}
        ],
        success_conditions=[{"id": "done", "description": "Resolved"}],
        steps=[
            {
                "id": "retrieve",
                "kind": "retrieve",
                "description": "Retrieve workflow",
                "retrieval_request_ids": ["workflow"],
                "status": "ready",
            }
        ],
        current_step_id="retrieve",
    )

    agent._execute_ready_retrieval_requests(state)
    agent._attach_execution_context(state)

    request = state.plan.retrieval_requests[0]
    assert request.status == "failed"
    assert request.progress.stop_reason == "error"
    assert state.messages == []
    assert "Retrieval failed: RuntimeError" in state.retrieval_evidence["workflow"][0]
    assert "<retrieval_evidence>" in state.readonly_context[0].content


def test_retrieval_evidence_is_included_in_plan_diagnostics(get_environment):
    agent = make_agent(get_environment)
    state = agent.get_init_state()
    state.plan = PlanState.model_validate_json(valid_plan_json())
    state.retrieval_evidence = {"workflow": ["Retrieved workflow evidence"]}
    agent._shadow_plan_diagnostics = {"status": "success"}

    agent._sync_plan_diagnostics(state)

    execution = agent._shadow_plan_diagnostics["execution"]
    assert execution["retrieval_evidence"] == {
        "workflow": ["Retrieved workflow evidence"]
    }


def test_plan_state_advances_only_after_selection_coverage_and_user_state(
    get_environment,
):
    environment = get_environment()
    agent = ShadowPlanningLLMAgent(
        llm="test-model",
        tools=environment.get_tools() + [KB_SEARCH_TOOL.model_copy(deep=True)],
        domain_policy=environment.get_policy(),
        metadata_catalog={
            "checking_account": {
                "Blue Account": ["doc_checking_accounts_blue_account_010"]
            }
        },
    )
    state = agent.get_init_state()
    plan = PlanState(
        goal="Recommend an eligible account",
        capabilities={"selection", "workflow"},
        selection={
            "candidate_scope": "checking accounts",
            "objective": "maximize",
            "objective_expression": "combined bonus",
            "required_attributes": ["bonus"],
            "requires_user_state": True,
            "retrieval": {
                "resource_type": "product",
                "product_categories": ["checking_account"],
                "coverage": "all_products",
                "queries": ["checking referral bonus"],
            },
        },
        success_conditions=[{"id": "chosen", "description": "Choice made"}],
        steps=[
            {
                "id": "retrieve",
                "kind": "retrieve",
                "description": "Retrieve all products",
                "status": "ready",
            },
            {
                "id": "verify",
                "kind": "verify",
                "description": "Verify identity",
                "depends_on": ["retrieve"],
                "completion_evidence": {
                    "event": "tool_call",
                    "tool_names": ["log_verification"],
                    "state_updates": ["identity_verified"],
                },
            },
            {
                "id": "read_state",
                "kind": "read",
                "description": "Read referral history",
                "depends_on": ["verify"],
                "completion_evidence": {
                    "event": "tool_call",
                    "tool_names": ["get_referrals_by_user"],
                    "state_updates": ["customer_state_read"],
                },
            },
            {
                "id": "recommend",
                "kind": "confirm",
                "description": "Recommend an eligible account",
                "depends_on": ["read_state"],
            },
        ],
        current_step_id="retrieve",
    )
    responses = [
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="search",
                    name="KB_search",
                    arguments={
                        "query": "checking referral bonus",
                        "product_category": "checking_account",
                        "coverage": "all_products",
                    },
                )
            ],
        ),
        AssistantMessage(
            role="assistant",
            tool_calls=[ToolCall(id="verify", name="log_verification", arguments={})],
        ),
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(id="referrals", name="get_referrals_by_user", arguments={})
            ],
        ),
        AssistantMessage(role="assistant", content="Recommendation"),
    ]

    with (
        patch(
            "tau2.agent.shadow_planner.generate", return_value=planner_response(plan)
        ),
        patch.object(agent, "_execute_ready_retrieval_requests"),
        patch("tau2.agent.llm_agent.generate", side_effect=responses),
    ):
        agent.generate_next_message(UserMessage(role="user", content="Choose"), state)
        assert state.plan.current_step_id == "retrieve"
        agent.generate_next_message(
            ToolMessage(
                id="search",
                role="tool",
                content=(
                    "[Product Coverage]\n"
                    '{"covered_products":["Blue Account"],'
                    '"missing_products":[],"coverage_complete":true}'
                ),
            ),
            state,
        )
        assert state.plan.current_step_id == "verify"
        assert "HARD EXECUTION GATE" in state.readonly_context[0].content
        agent.generate_next_message(
            ToolMessage(id="verify", role="tool", content="verified"), state
        )
        assert state.identity_verified is True
        assert state.customer_state_read is False
        assert state.plan.current_step_id == "read_state"
        agent.generate_next_message(
            ToolMessage(id="referrals", role="tool", content="no referrals"), state
        )

    assert state.customer_state_read is True
    assert state.customer_state_tools_read == ["get_referrals_by_user"]
    assert state.plan.current_step_id == "recommend"
    assert "HARD EXECUTION GATE" not in state.readonly_context[0].content


def test_one_observed_event_is_not_reused_across_dependent_steps(
    get_environment,
):
    agent = make_agent(get_environment)
    plan = PlanState(
        goal="Read state twice",
        capabilities={"workflow"},
        success_conditions=[{"id": "done", "description": "Resolved"}],
        steps=[
            {
                "id": "first",
                "kind": "read",
                "description": "First read",
                "completion_evidence": {
                    "tool_names": ["get_account"],
                },
                "status": "ready",
            },
            {
                "id": "second",
                "kind": "read",
                "description": "Second read",
                "depends_on": ["first"],
                "completion_evidence": {
                    "tool_names": ["get_account"],
                },
            },
        ],
        current_step_id="first",
    )

    agent._complete_steps_from_evidence(
        plan,
        observed_calls=[ToolCall(id="read", name="get_account", arguments={})],
        user_message_observed=False,
    )

    assert plan.steps[0].status == "completed"
    assert plan.steps[1].status == "pending"

    later = PlanState(
        goal="Do a later read",
        capabilities={"workflow"},
        success_conditions=[{"id": "done", "description": "Resolved"}],
        steps=[
            {
                "id": "later",
                "kind": "read",
                "description": "Later read",
                "completion_evidence": {"tool_names": ["get_account"]},
                "status": "ready",
            }
        ],
        current_step_id="later",
    )
    agent._complete_steps_from_evidence(
        later,
        observed_calls=[],
        user_message_observed=False,
    )
    assert later.steps[0].status == "ready"


def test_analyze_and_determine_steps_do_not_auto_complete(get_environment):
    agent = make_agent(get_environment)
    plan = PlanState(
        goal="Investigate",
        capabilities={"investigation"},
        success_conditions=[{"id": "done", "description": "Resolved"}],
        steps=[
            {
                "id": "analyze",
                "kind": "analyze",
                "description": "Analyze evidence",
                "status": "ready",
            },
            {
                "id": "determine",
                "kind": "verify",
                "description": "Determine the resolution",
                "depends_on": ["analyze"],
            },
        ],
        current_step_id="analyze",
    )

    agent._complete_steps_from_evidence(
        plan,
        observed_calls=[],
        user_message_observed=False,
    )

    assert plan.steps[0].status == "ready"
    assert plan.steps[1].status == "pending"


def test_executor_output_completes_only_one_declared_step(get_environment):
    agent = make_agent(get_environment)
    plan = PlanState(
        goal="Analyze and confirm",
        capabilities={"investigation"},
        success_conditions=[{"id": "done", "description": "Resolved"}],
        steps=[
            {
                "id": "analyze",
                "kind": "analyze",
                "description": "Analyze evidence",
                "completion_evidence": {
                    "event": "assistant_message",
                    "assistant_output": "any",
                },
                "status": "ready",
            },
            {
                "id": "confirm",
                "kind": "confirm",
                "description": "Confirm outcome",
                "depends_on": ["analyze"],
                "completion_evidence": {
                    "event": "assistant_message",
                    "assistant_output": "text",
                },
            },
        ],
        current_step_id="analyze",
    )
    state = agent.get_init_state()
    state.plan = plan

    agent._record_executor_response(
        AssistantMessage(role="assistant", content="Analysis result"), state
    )

    assert plan.steps[0].status == "completed"
    assert plan.steps[1].status == "ready"


def test_structured_state_updates_replace_tool_name_heuristics(get_environment):
    agent = make_agent(get_environment)
    plan = PlanState(
        goal="Verify and read required state",
        capabilities={"workflow"},
        success_conditions=[{"id": "done", "description": "Resolved"}],
        steps=[
            {
                "id": "verify",
                "kind": "verify",
                "description": "Log verification",
                "completion_evidence": {
                    "tool_names": ["log_verification"],
                    "state_updates": ["identity_verified"],
                },
                "status": "ready",
            },
            {
                "id": "read",
                "kind": "read",
                "description": "Read required account state",
                "depends_on": ["verify"],
                "completion_evidence": {
                    "tool_names": ["read_any_named_tool"],
                    "state_updates": ["customer_state_read"],
                },
            },
        ],
        current_step_id="verify",
    )

    assert agent._required_state_tools(plan) == ["read_any_named_tool"]
    updates = agent._complete_steps_from_evidence(
        plan,
        observed_calls=[ToolCall(id="verify", name="log_verification", arguments={})],
        user_message_observed=False,
    )
    assert updates == {"identity_verified"}
    assert plan.steps[0].status == "completed"
    assert plan.steps[1].status == "pending"


def test_completion_evidence_matches_discoverable_tool_arguments(get_environment):
    agent = make_agent(get_environment)
    plan = PlanState(
        goal="Apply the correct update",
        capabilities={"workflow"},
        success_conditions=[{"id": "done", "description": "Updated"}],
        steps=[
            {
                "id": "update_rewards",
                "kind": "write",
                "description": "Update transaction rewards",
                "completion_evidence": {
                    "tool_calls": [
                        {
                            "tool_name": "call_discoverable_agent_tool",
                            "arguments": {
                                "agent_tool_name": "update_transaction_rewards_3847"
                            },
                        }
                    ]
                },
                "status": "ready",
            }
        ],
        current_step_id="update_rewards",
    )

    agent._complete_steps_from_evidence(
        plan,
        observed_calls=[
            ToolCall(
                id="wrong",
                name="call_discoverable_agent_tool",
                arguments={"agent_tool_name": "some_other_tool"},
            )
        ],
        user_message_observed=False,
    )
    assert plan.steps[0].status == "ready"

    agent._complete_steps_from_evidence(
        plan,
        observed_calls=[
            ToolCall(
                id="right",
                name="call_discoverable_agent_tool",
                arguments={
                    "agent_tool_name": "update_transaction_rewards_3847",
                    "arguments": "{}",
                },
            )
        ],
        user_message_observed=False,
    )
    assert plan.steps[0].status == "completed"


def test_selection_user_state_gate_restricts_executor_tools(get_environment):
    environment = get_environment()
    control_tool_names = [
        "get_current_time",
        "get_user_information_by_id",
        "get_user_information_by_name",
        "get_user_information_by_email",
        "log_verification",
        "get_referrals_by_user",
        "give_discoverable_user_tool",
    ]
    control_tools = []
    for name in control_tool_names:
        tool = KB_SEARCH_TOOL.model_copy(deep=True)
        tool.name = name
        control_tools.append(tool)
    agent = ShadowPlanningLLMAgent(
        llm="test-model",
        tools=(
            environment.get_tools()
            + [KB_SEARCH_TOOL.model_copy(deep=True)]
            + control_tools
        ),
        domain_policy=environment.get_policy(),
    )
    state = agent.get_init_state()
    state.plan = PlanState(
        goal="Recommend after checking eligibility",
        capabilities={"selection", "workflow"},
        selection={
            "candidate_scope": "accounts",
            "objective": "maximize",
            "objective_expression": "bonus",
            "required_attributes": ["bonus"],
            "requires_user_state": True,
            "retrieval": {
                "resource_type": "product",
                "product_categories": ["checking_account"],
                "coverage": "all_products",
                "queries": ["bonus"],
            },
        },
        success_conditions=[{"id": "done", "description": "done"}],
        steps=[
            {
                "id": "retrieve",
                "kind": "retrieve",
                "description": "retrieve",
                "status": "ready",
            }
        ],
        current_step_id="retrieve",
    )

    assert [tool.name for tool in agent._get_executor_tools(state)] == ["KB_search"]
    request = state.plan.retrieval_requests[0]
    request.status = "completed"
    request.progress.coverage_complete = True
    assert {tool.name for tool in agent._get_executor_tools(state)} == {
        "get_current_time",
        "get_user_information_by_id",
        "get_user_information_by_name",
        "get_user_information_by_email",
        "log_verification",
    }
    state.identity_verified = True
    state.required_customer_state_tools = ["get_referrals_by_user"]
    names = {tool.name for tool in agent._get_executor_tools(state)}
    assert names == {"get_referrals_by_user"}
    state.customer_state_read = True
    assert {tool.name for tool in agent._get_executor_tools(state)} == {
        tool.name for tool in agent.tools
    }


def test_non_selection_plan_tracks_state_without_hiding_workflow_tools(
    get_environment,
):
    environment = get_environment()
    agent = ShadowPlanningLLMAgent(
        llm="test-model",
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
    )
    state = agent.get_init_state()
    state.plan = PlanState.model_validate_json(valid_plan_json())

    assert {tool.name for tool in agent._get_executor_tools(state)} == {
        tool.name for tool in agent.tools
    }


def test_shadow_plan_runs_only_once(get_environment):
    agent = make_agent(get_environment)
    state = agent.get_init_state()
    shadow_response = planner_response()
    executor_response = AssistantMessage(role="assistant", content="Response")

    with (
        patch(
            "tau2.agent.shadow_planner.generate", return_value=shadow_response
        ) as planner_generate,
        patch("tau2.agent.llm_agent.generate", return_value=executor_response),
    ):
        _, state = agent.generate_next_message(
            UserMessage(role="user", content="First"), state
        )
        agent.generate_next_message(UserMessage(role="user", content="Second"), state)

    assert planner_generate.call_count == 1


def test_shadow_plan_fails_open(get_environment):
    agent = make_agent(get_environment)
    state = agent.get_init_state()
    executor_response = AssistantMessage(role="assistant", content="Still works")

    with (
        patch(
            "tau2.agent.shadow_planner.generate",
            side_effect=RuntimeError("planner down"),
        ),
        patch("tau2.agent.llm_agent.generate", return_value=executor_response),
    ):
        response, _ = agent.generate_next_message(
            UserMessage(role="user", content="Please help"), state
        )

    assert response == executor_response
    diagnostics = agent.get_simulation_diagnostics()["shadow_plan"]
    assert diagnostics["status"] == "error"
    assert diagnostics["error_type"] == "RuntimeError"
    assert state.readonly_context == []
