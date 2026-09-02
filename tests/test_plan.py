import pytest
from pydantic import ValidationError

from tau2.agent.plan import (
    PlanState,
    PlanStep,
    RetrievalPlan,
    RetrievalRequest,
    SuccessCondition,
    TaskMode,
)


def make_plan(**overrides):
    data = {
        "goal": "Resolve the customer's request",
        "capabilities": {TaskMode.WORKFLOW},
        "success_conditions": [
            SuccessCondition(id="resolved", description="The request is resolved")
        ],
        "steps": [
            PlanStep(
                id="retrieve",
                kind="retrieve",
                description="Find the applicable process",
                status="ready",
            ),
            PlanStep(
                id="act",
                kind="write",
                description="Perform the process",
                depends_on=["retrieve"],
            ),
        ],
        "current_step_id": "retrieve",
    }
    data.update(overrides)
    return PlanState(**data)


def test_plan_round_trip():
    plan = make_plan()
    assert PlanState.model_validate_json(plan.model_dump_json()) == plan


def test_task_mode_removed_from_schema():
    assert "task_mode" not in PlanState.model_fields


def test_assistant_completion_evidence_cannot_include_tool_patterns():
    with pytest.raises(ValidationError, match="cannot use tool call patterns"):
        PlanStep(
            id="analyze",
            kind="analyze",
            description="Analyze",
            completion_evidence={
                "event": "assistant_message",
                "tool_names": ["some_tool"],
            },
        )


def test_non_assistant_evidence_ignores_assistant_output():
    step = PlanStep(
        id="ask",
        kind="ask_user",
        description="Ask",
        completion_evidence={
            "event": "user_message",
            "assistant_output": "text",
        },
    )
    assert step.completion_evidence.assistant_output == "any"


def test_plan_rejects_duplicate_step_ids():
    with pytest.raises(ValidationError, match="step IDs must be unique"):
        make_plan(
            steps=[
                PlanStep(id="same", kind="read", description="First"),
                PlanStep(id="same", kind="write", description="Second"),
            ]
        )


def test_plan_rejects_unknown_dependency():
    with pytest.raises(ValidationError, match="unknown dependencies"):
        make_plan(
            steps=[
                PlanStep(
                    id="act",
                    kind="write",
                    description="Act",
                    depends_on=["missing"],
                )
            ],
            current_step_id=None,
        )


def test_plan_rejects_cycles():
    with pytest.raises(ValidationError, match="acyclic"):
        make_plan(
            steps=[
                PlanStep(id="a", kind="read", description="A", depends_on=["b"]),
                PlanStep(id="b", kind="write", description="B", depends_on=["a"]),
            ],
            current_step_id=None,
        )


def test_selection_requires_structured_selection_details():
    with pytest.raises(ValidationError, match="selection details are required"):
        make_plan(
            capabilities={TaskMode.SELECTION},
        )


def test_selection_can_also_require_workflow():
    plan = make_plan(
        capabilities={TaskMode.SELECTION, TaskMode.WORKFLOW},
        selection={
            "candidate_scope": "checking account referral programs",
            "objective": "maximize",
            "objective_expression": "referrer bonus + referred bonus",
            "constraints": ["minimum deposit <= customer budget"],
            "required_attributes": [
                "referrer bonus",
                "referred bonus",
                "minimum deposit",
            ],
            "requires_user_state": True,
            "retrieval": RetrievalPlan(
                resource_type="product",
                product_categories=["checking_account"],
                document_intents=["referral_program"],
                coverage="all_products",
                queries=["checking account referral bonus"],
            ),
        },
        steps=[
            PlanStep(
                id="verify",
                kind="verify",
                description="Log identity verification",
                completion_evidence={
                    "tool_names": ["log_verification"],
                    "state_updates": ["identity_verified"],
                },
                status="ready",
            ),
            PlanStep(
                id="read_referrals",
                kind="read",
                description="Read referral history",
                depends_on=["verify"],
                completion_evidence={
                    "tool_names": ["get_referrals_by_user"],
                    "state_updates": ["customer_state_read"],
                },
            ),
        ],
        current_step_id="verify",
    )

    assert plan.capabilities == {TaskMode.SELECTION, TaskMode.WORKFLOW}
    assert plan.selection is not None
    assert plan.selection.objective == "maximize"
    assert plan.selection.retrieval is not None
    assert plan.selection.retrieval.coverage == "all_products"
    assert len(plan.retrieval_requests) == 1
    assert plan.retrieval_requests[0].mode == "all_products"
    assert plan.retrieval_requests[0].product_category == "checking_account"
    assert plan.retrieval_requests[0].id == "legacy_selection_coverage"


def test_legacy_selection_retrieval_does_not_duplicate_top_level_request():
    plan = make_plan(
        capabilities={TaskMode.SELECTION},
        selection={
            "candidate_scope": "checking accounts",
            "objective": "maximize",
            "objective_expression": "combined bonus",
            "required_attributes": ["bonus"],
            "retrieval": {
                "resource_type": "product",
                "product_categories": ["checking_account"],
                "coverage": "all_products",
                "queries": ["legacy checking account query"],
            },
        },
        retrieval_requests=[
            RetrievalRequest(
                id="checking_products",
                purpose="Retrieve checking accounts",
                query="new checking account query",
                mode="all_products",
                resource_type="product",
                product_category="checking_account",
            )
        ],
    )

    assert [request.id for request in plan.retrieval_requests] == ["checking_products"]


def test_legacy_selection_relevance_promotes_each_query():
    plan = make_plan(
        capabilities={TaskMode.SELECTION},
        selection={
            "candidate_scope": "accounts",
            "objective": "compare_only",
            "objective_expression": "fees and eligibility",
            "required_attributes": ["fees", "eligibility"],
            "requires_exhaustive_comparison": False,
            "retrieval": {
                "resource_type": "topic",
                "coverage": "relevance",
                "queries": ["account fees", "account eligibility"],
            },
        },
    )

    assert [request.query for request in plan.retrieval_requests] == [
        "account fees",
        "account eligibility",
    ]
    assert all(request.mode == "relevance" for request in plan.retrieval_requests)


def test_top_level_retrieval_requests_support_selection_and_workflow_evidence():
    plan = make_plan(
        capabilities={TaskMode.SELECTION, TaskMode.WORKFLOW},
        selection={
            "candidate_scope": "checking and savings accounts",
            "objective": "maximize",
            "objective_expression": "combined referral bonus",
            "required_attributes": ["referrer bonus", "referred bonus"],
        },
        retrieval_requests=[
            RetrievalRequest(
                id="checking_products",
                purpose="Compare every checking account referral offer",
                query="checking account referral bonus",
                mode="all_products",
                resource_type="product",
                product_category="checking_account",
            ),
            RetrievalRequest(
                id="referral_workflow",
                purpose="Find the referral eligibility workflow",
                query="referral eligibility and submission workflow",
                mode="relevance",
                resource_type="service",
            ),
        ],
        steps=[
            PlanStep(
                id="retrieve",
                kind="retrieve",
                description="Collect product and workflow evidence",
                retrieval_request_ids=[
                    "checking_products",
                    "referral_workflow",
                ],
                status="ready",
            )
        ],
        current_step_id="retrieve",
    )

    assert plan.selection is not None
    assert plan.selection.retrieval is None
    assert [request.mode for request in plan.retrieval_requests] == [
        "all_products",
        "relevance",
    ]


def test_relevance_retrieval_discards_product_scope():
    request = RetrievalRequest(
        id="scoped_relevance",
        purpose="Find a workflow",
        query="referral workflow",
        mode="relevance",
        product_category="checking_account",
        target_product_names=["Blue Account"],
    )
    assert request.product_category is None
    assert request.target_product_names == []


def test_plan_rejects_unknown_retrieval_request_reference():
    with pytest.raises(ValidationError, match="unknown retrieval requests"):
        make_plan(
            steps=[
                PlanStep(
                    id="retrieve",
                    kind="retrieve",
                    description="Find evidence",
                    retrieval_request_ids=["missing"],
                )
            ],
            current_step_id=None,
        )


def test_request_step_dependency_moves_to_referencing_retrieve_step():
    plan = make_plan(
        retrieval_requests=[
            RetrievalRequest(
                id="card_rules",
                purpose="Find rules for the customer's cards",
                query="customer card reward rules",
                depends_on=["read_cards"],
            )
        ],
        steps=[
            PlanStep(
                id="read_cards",
                kind="read",
                description="Read the customer's cards",
                status="ready",
            ),
            PlanStep(
                id="retrieve_rules",
                kind="retrieve",
                description="Retrieve applicable card rules",
                retrieval_request_ids=["card_rules"],
            ),
        ],
        current_step_id="read_cards",
    )

    assert plan.retrieval_requests[0].depends_on == []
    assert plan.steps[1].depends_on == ["read_cards"]


def test_request_unknown_dependency_still_fails():
    with pytest.raises(ValidationError, match="unknown dependencies"):
        make_plan(
            retrieval_requests=[
                RetrievalRequest(
                    id="rules",
                    purpose="Find rules",
                    query="rules",
                    depends_on=["not_a_request_or_step"],
                )
            ]
        )


def test_mixed_is_not_a_capability():
    with pytest.raises(ValidationError, match="not a capability"):
        make_plan(capabilities={TaskMode.MIXED})


@pytest.mark.parametrize(
    ("planner_kind", "normalized_kind"),
    [
        ("investigation", "analyze"),
        ("selection", "analyze"),
        ("exception_handling", "analyze"),
        ("mixed", "analyze"),
        ("workflow", "write"),
    ],
)
def test_plan_normalizes_task_modes_used_as_step_kinds(planner_kind, normalized_kind):
    step = PlanStep(id="work", kind=planner_kind, description="Do the work")
    assert step.kind == normalized_kind
