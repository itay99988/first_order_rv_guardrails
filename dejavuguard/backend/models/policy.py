"""
Pydantic models for predicates, policies, and monitor verdicts.

Defines the core data structures used by the grounding engine,
DejaVu runtime verification engine, and monitor orchestrator.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

GROUNDING_SCOPE_SINGLE_MESSAGE = "single_message"
GROUNDING_SCOPE_CONVERSATION_HISTORY = "conversation_history"
GROUNDING_SCOPES = {
    GROUNDING_SCOPE_SINGLE_MESSAGE,
    GROUNDING_SCOPE_CONVERSATION_HISTORY,
}


class Proposition(BaseModel):
    """A predicate for semantic grounding and DejaVu runtime verification.

    Attributes:
        prop_id: Predicate name / unique identifier (e.g., "p_fraud", "p_transfer").
        description: Canonical description for grounding (delta_p).
        role: Which message role this predicate applies to ("user" or "assistant").
        arity: Number of arguments (0 = Boolean, >0 = first-order with data).
               e.g., p_fraud has arity 0, p_transfer(account, dest, amount) has arity 3.
    """

    prop_id: str
    description: str
    role: str  # "user" | "assistant"
    grounding_scope: str = GROUNDING_SCOPE_SINGLE_MESSAGE
    arity: int = 0
    arg_descriptions: list[str] = Field(default_factory=list)
    few_shot_positive: list[str] = Field(default_factory=list)
    few_shot_negative: list[str] = Field(default_factory=list)
    few_shot_examples: list[dict[str, Any]] = Field(default_factory=list)
    few_shot_generated_at: str | None = None


class Policy(BaseModel):
    """A temporal safety policy referencing predicates.

    Attributes:
        policy_id: Unique identifier for the policy.
        name: Human-readable name.
        formula_str: Past-time temporal logic formula string (DejaVu syntax).
        propositions: List of prop_ids referenced in the formula.
        enabled: Whether the policy is actively monitored.
    """

    policy_id: str
    name: str
    formula_str: str
    propositions: list[str] = Field(default_factory=list)
    enabled: bool = True


class ViolationInfo(BaseModel):
    """Details about a policy violation.

    Attributes:
        policy_id: Which policy was violated.
        policy_name: Human-readable name of the violated policy.
        formula_str: The temporal logic formula that was violated.
        violated_at_index: Trace index where violation occurred.
        labeling: Predicate truth values at the violating step.
        grounding_details: Reasoning from the grounding engine.
    """

    policy_id: str
    policy_name: str
    formula_str: str
    violated_at_index: int
    labeling: dict[str, bool] = Field(default_factory=dict)
    grounding_details: list[dict] = Field(default_factory=list)


class MonitorVerdict(BaseModel):
    """Result of processing a message through the monitor pipeline.

    Attributes:
        passed: True if all policies are satisfied, False if any violated.
        per_policy: Verdict for each policy (policy_id -> bool).
        labeling: Predicate truth values at this step.
        grounding_details: Detailed grounding results per predicate.
        trace_index: Position in the conversation trace.
        violations: List of violation details (empty if passed).
        verified: Whether DejaVu actually evaluated this step. False means the
            monitor failed open -- ``passed`` carries no verification weight and
            must not be read as "this step was checked and was clean".
        monitor_error: Why verification did not happen, when verified is False.
        composite_event: The event list actually sent to DejaVu at this step,
            after numeric coercion. Reported rather than reconstructed so
            debugging surfaces show the real payload.
    """

    passed: bool
    per_policy: dict[str, bool] = Field(default_factory=dict)
    labeling: dict[str, bool] = Field(default_factory=dict)
    grounding_details: list[dict] = Field(default_factory=list)
    trace_index: int = 0
    violations: list[ViolationInfo] = Field(default_factory=list)
    verified: bool = True
    monitor_error: str | None = None
    composite_event: list[dict] = Field(default_factory=list)
