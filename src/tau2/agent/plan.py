"""Data models for observable, stateful agent plans."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class TaskMode(str, Enum):
    """High-level shape of a customer-service task."""

    SELECTION = "selection"
    WORKFLOW = "workflow"
    INVESTIGATION = "investigation"
    EXCEPTION_HANDLING = "exception_handling"
    MIXED = "mixed"


class RetrievalPlan(BaseModel):
    """Legacy selection-scoped retrieval requirements.

    New plans should use ``PlanState.retrieval_requests``. This model remains
    during the migration so previously recorded plans can still be loaded.
    """

    resource_type: Literal["product", "service", "topic", "unknown"] = "unknown"
    product_categories: list[str] = Field(default_factory=list)
    document_intents: list[str] = Field(default_factory=list)
    coverage: Literal["relevance", "all_products"] = "relevance"
    queries: list[str] = Field(default_factory=list)


class RetrievalProgress(BaseModel):
    """Controller-owned execution progress for one retrieval requirement."""

    attempts: int = Field(default=0, ge=0)
    covered_product_names: list[str] = Field(default_factory=list)
    missing_product_names: list[str] = Field(default_factory=list)
    coverage_complete: bool = False
    query_history: list[str] = Field(default_factory=list)
    query_rewrite_attempted: bool = False
    stop_reason: Optional[
        Literal["coverage_complete", "no_progress", "budget", "error"]
    ] = None

    @model_validator(mode="after")
    def validate_coverage(self) -> "RetrievalProgress":
        """Keep terminal coverage state internally consistent."""
        if self.coverage_complete and self.missing_product_names:
            raise ValueError(
                "coverage_complete cannot be true while products are missing"
            )
        if self.stop_reason == "coverage_complete" and not self.coverage_complete:
            raise ValueError("coverage_complete stop reason requires complete coverage")
        return self


class RetrievalRequest(BaseModel):
    """A top-level evidence requirement shared by every task mode."""

    id: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    query: str = Field(min_length=1)
    mode: Literal["relevance", "all_products"] = "relevance"
    resource_type: Literal["product", "service", "topic", "unknown"] = "unknown"
    document_intents: list[str] = Field(default_factory=list)
    product_category: Optional[str] = None
    target_product_names: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    status: Literal["pending", "in_progress", "completed", "incomplete", "failed"] = (
        "pending"
    )
    progress: RetrievalProgress = Field(default_factory=RetrievalProgress)

    @model_validator(mode="after")
    def validate_scope(self) -> "RetrievalRequest":
        """Enforce the binary relevance versus product-coverage contract."""
        if self.mode == "all_products":
            if self.resource_type != "product":
                raise ValueError(
                    "all_products retrieval requires resource_type=product"
                )
            if not self.product_category:
                raise ValueError("all_products retrieval requires product_category")
        elif self.product_category or self.target_product_names:
            self.product_category = None
            self.target_product_names = []
        if self.target_product_names and not self.product_category:
            raise ValueError("target_product_names requires product_category")
        return self


class SelectionPlan(BaseModel):
    """An evidence-backed choice among candidate products or options."""

    candidate_scope: str = Field(min_length=1)
    objective: Literal["maximize", "minimize", "best_match", "compare_only"]
    objective_expression: str = Field(min_length=1)
    constraints: list[str] = Field(default_factory=list)
    required_attributes: list[str] = Field(min_length=1)
    requires_exhaustive_comparison: bool = True
    requires_user_state: bool = False
    retrieval: Optional[RetrievalPlan] = None


class SuccessCondition(BaseModel):
    """An observable outcome that must hold before the goal is complete."""

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    status: Literal["pending", "completed", "failed"] = "pending"


class ToolCallEvidence(BaseModel):
    """A tool call pattern that can be checked without reading prose."""

    tool_name: str = Field(min_length=1)
    arguments: dict[str, object] = Field(default_factory=dict)


class CompletionEvidence(BaseModel):
    """Observable evidence required to complete a plan step."""

    event: Literal["tool_call", "user_message", "assistant_message"] = "tool_call"
    assistant_output: Literal["any", "text", "tool_call"] = "any"
    tool_calls: list[ToolCallEvidence] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)
    require_all: bool = False
    state_updates: set[Literal["identity_verified", "customer_state_read"]] = Field(
        default_factory=set
    )

    @model_validator(mode="after")
    def validate_evidence(self) -> "CompletionEvidence":
        if self.event == "tool_call" and not (self.tool_calls or self.tool_names):
            raise ValueError(
                "tool_call completion evidence requires tool_calls or tool_names"
            )
        if self.event != "tool_call" and (self.tool_calls or self.tool_names):
            raise ValueError(
                "non-tool completion evidence cannot use tool call patterns"
            )
        if self.event != "assistant_message":
            self.assistant_output = "any"
        return self


class PlanStep(BaseModel):
    """A coarse-grained unit of work in a plan dependency graph."""

    id: str = Field(min_length=1)
    kind: Literal[
        "ask_user",
        "retrieve",
        "analyze",
        "verify",
        "read",
        "write",
        "user_action",
        "wait_user",
        "confirm",
        "transfer",
    ]
    description: str = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    retrieval_request_ids: list[str] = Field(default_factory=list)
    completion_evidence: Optional[CompletionEvidence] = None
    status: Literal[
        "pending",
        "ready",
        "in_progress",
        "waiting_user",
        "completed",
        "failed",
        "blocked",
        "skipped",
    ] = "pending"

    @field_validator("kind", mode="before")
    @classmethod
    def normalize_task_modes_used_as_step_kinds(cls, value: str) -> str:
        """Recover when a planner confuses task modes with executable actions."""
        return {
            "investigation": "analyze",
            "workflow": "write",
            "exception_handling": "analyze",
            "selection": "analyze",
            "mixed": "analyze",
        }.get(value, value)


class PlanState(BaseModel):
    """One persistent plan whose state may evolve throughout a simulation."""

    version: int = Field(default=1, ge=1)
    goal: str = Field(min_length=1)
    capabilities: set[TaskMode] = Field(min_length=1)
    selection: Optional[SelectionPlan] = None
    retrieval_requests: list[RetrievalRequest] = Field(default_factory=list)
    success_conditions: list[SuccessCondition] = Field(min_length=1)
    steps: list[PlanStep] = Field(min_length=1)
    unknowns: list[str] = Field(default_factory=list)
    current_step_id: Optional[str] = None

    @model_validator(mode="after")
    def normalize_legacy_selection_retrieval(self) -> "PlanState":
        """Promote legacy selection retrieval into top-level requests."""
        retrieval = self.selection.retrieval if self.selection else None
        if retrieval is None:
            return self

        existing_ids = {request.id for request in self.retrieval_requests}

        def next_id(prefix: str) -> str:
            index = 1
            candidate = prefix
            while candidate in existing_ids:
                index += 1
                candidate = f"{prefix}_{index}"
            existing_ids.add(candidate)
            return candidate

        intents = ", ".join(retrieval.document_intents)
        promoted_ids: list[str] = []
        if retrieval.coverage == "all_products":
            covered_categories = {
                request.product_category
                for request in self.retrieval_requests
                if request.mode == "all_products"
            }
            for index, category in enumerate(retrieval.product_categories):
                if category in covered_categories:
                    continue
                query = (
                    retrieval.queries[index]
                    if index < len(retrieval.queries)
                    else retrieval.queries[0]
                    if retrieval.queries
                    else f"{category} {intents or 'product comparison'}"
                )
                request_id = next_id("legacy_selection_coverage")
                self.retrieval_requests.append(
                    RetrievalRequest(
                        id=request_id,
                        purpose=f"Retrieve complete product coverage for {category}",
                        query=query,
                        mode="all_products",
                        resource_type="product",
                        document_intents=retrieval.document_intents,
                        product_category=category,
                    )
                )
                promoted_ids.append(request_id)
            self._attach_promoted_requests(promoted_ids)
            return self

        existing_queries = {
            request.query
            for request in self.retrieval_requests
            if request.mode == "relevance"
        }
        queries = retrieval.queries or [
            f"{self.selection.candidate_scope} {intents or 'selection criteria'}"
        ]
        for query in queries:
            if query in existing_queries:
                continue
            request_id = next_id("legacy_selection_relevance")
            self.retrieval_requests.append(
                RetrievalRequest(
                    id=request_id,
                    purpose="Retrieve evidence required for product selection",
                    query=query,
                    mode="relevance",
                    resource_type=retrieval.resource_type,
                    document_intents=retrieval.document_intents,
                )
            )
            promoted_ids.append(request_id)
        self._attach_promoted_requests(promoted_ids)
        return self

    def _attach_promoted_requests(self, request_ids: list[str]) -> None:
        """Bind legacy promoted requests to the first unbound retrieve step."""
        if not request_ids:
            return
        step = next(
            (
                step
                for step in self.steps
                if step.kind == "retrieve" and not step.retrieval_request_ids
            ),
            None,
        )
        if step is not None:
            step.retrieval_request_ids.extend(request_ids)

    @model_validator(mode="after")
    def normalize_retrieval_step_dependencies(self) -> "PlanState":
        """Move planner-supplied step prerequisites onto retrieval steps."""
        known_steps = {step.id for step in self.steps}
        for request in self.retrieval_requests:
            step_dependencies = [
                dependency
                for dependency in request.depends_on
                if dependency in known_steps
            ]
            if not step_dependencies:
                continue
            referencing_steps = [
                step for step in self.steps if request.id in step.retrieval_request_ids
            ]
            if not referencing_steps:
                continue
            for step in referencing_steps:
                for dependency in step_dependencies:
                    if dependency not in step.depends_on:
                        step.depends_on.append(dependency)
            request.depends_on = [
                dependency
                for dependency in request.depends_on
                if dependency not in known_steps
            ]
        return self

    @model_validator(mode="after")
    def validate_graph(self) -> "PlanState":
        """Reject ambiguous references and cyclic dependency graphs."""
        if TaskMode.MIXED in self.capabilities:
            raise ValueError("mixed is a primary-mode fallback, not a capability")
        if TaskMode.SELECTION in self.capabilities and self.selection is None:
            raise ValueError("selection details are required for selection tasks")
        if TaskMode.SELECTION not in self.capabilities and self.selection is not None:
            raise ValueError("selection details require the selection capability")

        step_ids = [step.id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("Plan step IDs must be unique")

        condition_ids = [condition.id for condition in self.success_conditions]
        if len(condition_ids) != len(set(condition_ids)):
            raise ValueError("Success condition IDs must be unique")

        request_ids = [request.id for request in self.retrieval_requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("Retrieval request IDs must be unique")

        known_requests = set(request_ids)
        for request in self.retrieval_requests:
            unknown_dependencies = set(request.depends_on) - known_requests
            if unknown_dependencies:
                unknown = ", ".join(sorted(unknown_dependencies))
                raise ValueError(
                    f"Retrieval request {request.id!r} has unknown dependencies: "
                    f"{unknown}"
                )
            if request.id in request.depends_on:
                raise ValueError(
                    f"Retrieval request {request.id!r} cannot depend on itself"
                )

        known_steps = set(step_ids)
        for step in self.steps:
            unknown_dependencies = set(step.depends_on) - known_steps
            if unknown_dependencies:
                unknown = ", ".join(sorted(unknown_dependencies))
                raise ValueError(
                    f"Step {step.id!r} has unknown dependencies: {unknown}"
                )
            if step.id in step.depends_on:
                raise ValueError(f"Step {step.id!r} cannot depend on itself")
            unknown_requests = set(step.retrieval_request_ids) - known_requests
            if unknown_requests:
                unknown = ", ".join(sorted(unknown_requests))
                raise ValueError(
                    f"Step {step.id!r} references unknown retrieval requests: {unknown}"
                )

        if self.current_step_id is not None and self.current_step_id not in known_steps:
            raise ValueError("current_step_id must reference an existing step")

        dependencies = {step.id: step.depends_on for step in self.steps}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(step_id: str) -> None:
            if step_id in visiting:
                raise ValueError("Plan step dependencies must be acyclic")
            if step_id in visited:
                return
            visiting.add(step_id)
            for dependency in dependencies[step_id]:
                visit(dependency)
            visiting.remove(step_id)
            visited.add(step_id)

        for step_id in step_ids:
            visit(step_id)

        request_dependencies = {
            request.id: request.depends_on for request in self.retrieval_requests
        }
        visiting.clear()
        visited.clear()

        def visit_request(request_id: str) -> None:
            if request_id in visiting:
                raise ValueError("Retrieval request dependencies must be acyclic")
            if request_id in visited:
                return
            visiting.add(request_id)
            for dependency in request_dependencies[request_id]:
                visit_request(dependency)
            visiting.remove(request_id)
            visited.add(request_id)

        for request_id in request_ids:
            visit_request(request_id)
        return self

    def ready_steps(self) -> list[PlanStep]:
        """Return pending steps whose dependencies have completed."""
        completed = {step.id for step in self.steps if step.status == "completed"}
        return [
            step
            for step in self.steps
            if step.status in {"pending", "ready"} and set(step.depends_on) <= completed
        ]

    def transition_step(self, step_id: str, status: str) -> PlanStep:
        """Apply a controlled step transition and return the updated step.

        Completed steps are terminal. This method intentionally does not enforce
        semantic correctness; runtime observations and a future replan decide
        whether a pending step should be retried or blocked.
        """
        step = next(
            (candidate for candidate in self.steps if candidate.id == step_id), None
        )
        if step is None:
            raise ValueError(f"Unknown plan step: {step_id}")
        if step.status == "completed" and status != "completed":
            raise ValueError(f"Completed plan step cannot transition: {step_id}")
        allowed = {
            "pending",
            "ready",
            "in_progress",
            "waiting_user",
            "completed",
            "failed",
            "blocked",
            "skipped",
        }
        if status not in allowed:
            raise ValueError(f"Unknown plan step status: {status}")
        step.status = status
        self.current_step_id = (
            step_id
            if status in {"ready", "in_progress", "waiting_user"}
            else self.current_step_id
        )
        return step
