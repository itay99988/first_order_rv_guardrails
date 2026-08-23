"""Pydantic models for scenario JSON files.

A scenario describes a complete monitored conversation:
- predicates (and their few-shot examples)
- policies (formulas)
- model configuration for grounding
- a sequence of pre-recorded user/assistant messages with optional
  per-message expected verdicts
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ScenarioModel(BaseModel):
    """Model configuration for a scenario run.

    Only the grounding model is exercised — the chat LLM is bypassed
    because assistant messages come from the scenario itself.
    """

    grounding_provider: str
    grounding_model: str
    few_shot_provider: str | None = None
    few_shot_model: str | None = None

    model_config = ConfigDict(extra="forbid")


class ScenarioObject(BaseModel):
    """One predicate object slot (e.g., person, location)."""

    object_id: str
    description: str
    entity_type: str

    model_config = ConfigDict(extra="forbid")


class ScenarioPredicate(BaseModel):
    """A predicate definition embedded in a scenario.

    arg_descriptions, when supplied, is derived from `objects` if absent,
    matching how the backend Proposition model is populated.
    """

    prop_id: str
    description: str
    role: str
    grounding_scope: str = "single_message"
    arity: int = 0
    arg_descriptions: list[str] = Field(default_factory=list)
    objects: list[ScenarioObject] = Field(default_factory=list)
    few_shot_examples: list[dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @field_validator("role")
    @classmethod
    def _role_is_known(cls, v: str) -> str:
        if v not in {"user", "assistant"}:
            raise ValueError(f"role must be 'user' or 'assistant', got '{v}'")
        return v

    @field_validator("grounding_scope")
    @classmethod
    def _grounding_scope_is_known(cls, v: str) -> str:
        if v not in {"single_message", "conversation_history"}:
            raise ValueError(
                "grounding_scope must be 'single_message' or "
                f"'conversation_history', got '{v}'"
            )
        return v

    @model_validator(mode="after")
    def _arity_matches_objects(self) -> ScenarioPredicate:
        if self.objects and self.arity and self.arity != len(self.objects):
            raise ValueError(
                f"predicate {self.prop_id}: arity={self.arity} disagrees with "
                f"len(objects)={len(self.objects)}"
            )
        if self.arity == 0 and self.objects:
            self.arity = len(self.objects)
        if not self.arg_descriptions and self.objects:
            self.arg_descriptions = [o.description for o in self.objects]
        return self


class ScenarioPolicy(BaseModel):
    """A policy definition embedded in a scenario."""

    policy_id: str
    name: str
    formula_str: str
    enabled: bool = True

    model_config = ConfigDict(extra="forbid")


class ScenarioRelatedObjects(BaseModel):
    """Declares which object slots of different predicates refer to the
    same conceptual entity, scoped to a single policy.

    `pairs` is a list of two-element string lists in the form
    ["prop_id.object_id", "prop_id.object_id"]. Each pair is expanded to
    a bidirectional relation in the DB so the grounding engine surfaces
    canonical-form history across both predicates.

    Example:
        {
          "policy_id": "car-booking-order",
          "pairs": [
            ["user_car.o1", "assistant_car.o1"],
            ["user_car.o2", "assistant_car.o2"]
          ]
        }
    """

    policy_id: str
    pairs: list[list[str]]

    model_config = ConfigDict(extra="forbid")

    @field_validator("pairs")
    @classmethod
    def _validate_pair_shape(cls, v: list[list[str]]) -> list[list[str]]:
        for i, pair in enumerate(v):
            if len(pair) != 2:
                raise ValueError(
                    f"pair {i} must have exactly two elements, got {len(pair)}"
                )
            for endpoint in pair:
                if "." not in endpoint:
                    raise ValueError(
                        f"pair {i} endpoint {endpoint!r} must be 'prop_id.object_id'"
                    )
        return v


class ScenarioMessage(BaseModel):
    """One message in the recorded conversation.

    expected_verdict maps policy_id -> expected per-policy verdict
    (True = passing, False = violated) after this message is processed.
    expected_playbook_state / expected_guidance are only meaningful in
    playbook mode; each is checked only when supplied.
    """

    role: str
    text: str
    expected_verdict: dict[str, bool] | None = None
    expected_playbook_state: str | None = None
    expected_guidance: list[str] | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("role")
    @classmethod
    def _role_is_known(cls, v: str) -> str:
        if v not in {"user", "assistant"}:
            raise ValueError(f"message role must be 'user' or 'assistant', got '{v}'")
        return v


class ScenarioMonitoring(BaseModel):
    """Which monitoring mode the scenario runs under."""

    mode: str = "policies"
    playbook_id: str | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("mode")
    @classmethod
    def _mode_is_known(cls, v: str) -> str:
        if v not in {"policies", "playbook"}:
            raise ValueError(f"mode must be 'policies' or 'playbook', got '{v}'")
        return v


class ScenarioPlaybookMember(BaseModel):
    policy_id: str
    position: int = 0
    fires_on: bool = False
    guidance: str = ""

    model_config = ConfigDict(extra="forbid")


class ScenarioPlaybookState(BaseModel):
    state_key: str
    rule_refs: list[dict] | None = None
    flagged: bool = False
    label: str | None = None

    model_config = ConfigDict(extra="forbid")


class ScenarioPlaybook(BaseModel):
    playbook_id: str
    name: str
    members: list[ScenarioPlaybookMember] = Field(default_factory=list)
    globals: list[dict] = Field(default_factory=list)
    states: list[ScenarioPlaybookState] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class Scenario(BaseModel):
    """A complete scenario specification."""

    scenario_id: str
    description: str = ""
    model: ScenarioModel
    predicates: list[ScenarioPredicate] = Field(default_factory=list)
    policies: list[ScenarioPolicy] = Field(default_factory=list)
    related_objects: list[ScenarioRelatedObjects] = Field(default_factory=list)
    monitoring: ScenarioMonitoring = Field(default_factory=ScenarioMonitoring)
    playbooks: list[ScenarioPlaybook] = Field(default_factory=list)
    messages: list[ScenarioMessage] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _expected_verdicts_reference_known_policies(self) -> Scenario:
        known = {p.policy_id for p in self.policies}
        for idx, msg in enumerate(self.messages):
            if not msg.expected_verdict:
                continue
            unknown = set(msg.expected_verdict) - known
            if unknown:
                raise ValueError(
                    f"message {idx}: expected_verdict references unknown "
                    f"policy_ids {sorted(unknown)}; known: {sorted(known)}"
                )
        return self

    @model_validator(mode="after")
    def _related_objects_reference_known_predicates(self) -> Scenario:
        known_policies = {p.policy_id for p in self.policies}
        known_predicates = {p.prop_id for p in self.predicates}
        for entry in self.related_objects:
            if entry.policy_id not in known_policies:
                raise ValueError(
                    f"related_objects: unknown policy_id {entry.policy_id!r}; "
                    f"known: {sorted(known_policies)}"
                )
            for pair in entry.pairs:
                for endpoint in pair:
                    prop_id, _ = endpoint.split(".", 1)
                    if prop_id not in known_predicates:
                        raise ValueError(
                            f"related_objects {entry.policy_id}: "
                            f"endpoint {endpoint!r} references unknown "
                            f"predicate {prop_id!r}"
                        )
        return self

    @model_validator(mode="after")
    def _unique_ids(self) -> Scenario:
        pred_ids = [p.prop_id for p in self.predicates]
        if len(pred_ids) != len(set(pred_ids)):
            raise ValueError("predicate prop_ids must be unique")
        policy_ids = [p.policy_id for p in self.policies]
        if len(policy_ids) != len(set(policy_ids)):
            raise ValueError("policy policy_ids must be unique")
        return self


def load_scenario(path: str | Path) -> Scenario:
    """Load and validate a scenario from a JSON file on disk."""
    text = Path(path).read_text(encoding="utf-8")
    raw = json.loads(text)
    return Scenario.model_validate(raw)
