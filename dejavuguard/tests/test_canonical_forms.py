from __future__ import annotations

import pytest

from backend.engine.dejavu_client import DejaVuVerdict
from backend.engine.grounding import (
    GroundingMethod,
    GroundingResult,
    LLMGrounding,
    build_grounding_prompts,
)
from backend.engine.monitor import ConversationMonitor
from backend.engine.trace import MessageEvent
from backend.models.policy import Policy, Proposition
from backend.routers.policies import _extract_related_object_relations
from backend.routers.settings import _load_settings
from backend.store.db import DatabaseStore


class RecordingGrounding(GroundingMethod):
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def evaluate(
        self,
        message: MessageEvent,
        proposition: Proposition,
        related_object_context_block: str = "NONE",
        related_object_history_block: str = "NONE",
    ) -> GroundingResult:
        self.calls.append({
            "role": message.role,
            "prop_id": proposition.prop_id,
            "context": related_object_context_block,
            "history": related_object_history_block,
        })
        if proposition.prop_id == "p_user_account":
            return GroundingResult(
                match=True,
                confidence=1.0,
                reasoning="user account",
                method="test",
                prop_id=proposition.prop_id,
                object_mentions=[{
                    "object_id": "o1",
                    "mention": "account 123",
                    "canonical_form": "acct-123",
                }],
            )
        return GroundingResult(
            match=True,
            confidence=1.0,
            reasoning="assistant account",
            method="test",
            prop_id=proposition.prop_id,
            object_mentions=[{
                "object_id": "o1",
                "mention": "that account",
                "canonical_form": "acct-123",
            }],
        )


class RecordingDejaVuClient:
    def __init__(self) -> None:
        self.events: list[list[dict]] = []

    async def create_session(self, spec: str) -> tuple[str, list[str]]:
        return "dejavu-session", ["pol_policy_1"]

    async def send_events(self, session_id: str, events: list[dict]) -> DejaVuVerdict:
        self.events.append(events)
        return DejaVuVerdict(
            event_number=len(self.events),
            verdicts={"pol_policy_1": True},
            violations=[],
        )

    async def delete_session(self, session_id: str) -> bool:
        return True


class MultiInstanceGrounding(GroundingMethod):
    async def evaluate(
        self,
        message: MessageEvent,
        proposition: Proposition,
        related_object_context_block: str = "NONE",
        related_object_history_block: str = "NONE",
    ) -> GroundingResult:
        return GroundingResult(
            match=True,
            confidence=1.0,
            reasoning="two instances",
            method="test",
            prop_id=proposition.prop_id,
            instances=[
                {
                    "instance_id": "i1",
                    "object_mentions": [
                        {
                            "object_id": "o1",
                            "mention": "Toyota",
                            "canonical_form": "Toyota",
                        },
                        {
                            "object_id": "o2",
                            "mention": "12000$",
                            "canonical_form": "12000 USD",
                        },
                    ],
                },
                {
                    "instance_id": "i2",
                    "object_mentions": [
                        {
                            "object_id": "o1",
                            "mention": "Skoda",
                            "canonical_form": "Skoda",
                        },
                        {
                            "object_id": "o2",
                            "mention": "12500$",
                            "canonical_form": "12500 USD",
                        },
                    ],
                },
            ],
        )


@pytest.fixture
async def db():
    store = DatabaseStore(":memory:")
    await store.initialize()
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_policy_variable_relations_are_extracted_and_persisted(db):
    await db.create_proposition(
        "p_user_account",
        "the user provides an account",
        "user",
        arity=1,
        arg_descriptions=["account"],
    )
    await db.create_proposition(
        "q_assistant_account",
        "the assistant references an account",
        "assistant",
        arity=1,
        arg_descriptions=["account"],
    )
    await db.create_policy(
        "policy-1",
        "Account continuity",
        "Forall x . (p_user_account(x) -> P(q_assistant_account(x)))",
    )

    relations = await _extract_related_object_relations(
        db,
        "Forall x . (p_user_account(x) -> P(q_assistant_account(x)))",
    )
    await db.set_policy_related_objects("policy-1", relations)
    stored = await db.list_related_objects()

    assert {
        (r["prop_id"], r["object_id"], r["related_prop_id"], r["related_object_id"])
        for r in stored
    } == {
        ("p_user_account", "o1", "q_assistant_account", "o1"),
        ("q_assistant_account", "o1", "p_user_account", "o1"),
    }

    await db.delete_policy("policy-1")
    assert await db.list_related_objects() == []


@pytest.mark.asyncio
async def test_compared_policy_variables_are_related(db):
    await db.create_proposition(
        "assistant_car",
        "the assistant provides a car model and price",
        "assistant",
        arity=2,
        arg_descriptions=["model", "price"],
    )
    await db.create_proposition(
        "user_car",
        "the user provides a car model and budget",
        "user",
        arity=2,
        arg_descriptions=["model", "budget"],
    )

    relations = await _extract_related_object_relations(
        db,
        (
            "forall m . forall p . "
            "(assistant_car(m,p) -> exists b . (P user_car(m,b) & !(b < p)))"
        ),
    )

    assert {
        (r["prop_id"], r["object_id"], r["related_prop_id"], r["related_object_id"])
        for r in relations
    } == {
        ("assistant_car", "o1", "user_car", "o1"),
        ("user_car", "o1", "assistant_car", "o1"),
        ("assistant_car", "o2", "user_car", "o2"),
        ("user_car", "o2", "assistant_car", "o2"),
    }


@pytest.mark.asyncio
async def test_monitor_uses_related_history_and_sends_canonical_forms_to_dejavu():
    propositions = [
        Proposition(
            prop_id="p_user_account",
            description="the user provides an account",
            role="user",
            arity=1,
            arg_descriptions=["account"],
        ),
        Proposition(
            prop_id="q_assistant_account",
            description="the assistant references an account",
            role="assistant",
            arity=1,
            arg_descriptions=["account"],
        ),
    ]
    policies = [
        Policy(
            policy_id="policy-1",
            name="Account continuity",
            formula_str="Forall x . (p_user_account(x) -> q_assistant_account(x))",
            propositions=["p_user_account", "q_assistant_account"],
            enabled=True,
        )
    ]
    related_objects = [
        {
            "policy_id": "policy-1",
            "prop_id": "p_user_account",
            "object_id": "o1",
            "related_prop_id": "q_assistant_account",
            "related_object_id": "o1",
        },
        {
            "policy_id": "policy-1",
            "prop_id": "q_assistant_account",
            "object_id": "o1",
            "related_prop_id": "p_user_account",
            "related_object_id": "o1",
        },
    ]
    grounding = RecordingGrounding()
    dejavu_client = RecordingDejaVuClient()
    monitor = ConversationMonitor(
        policies=policies,
        propositions=propositions,
        grounding=grounding,
        dejavu_client=dejavu_client,  # type: ignore[arg-type]
        session_id="session-1",
        related_objects=related_objects,
    )

    await monitor.process_message("user", "Use account 123.")
    await monitor.process_message("assistant", "I will use that account.")

    assistant_call = next(c for c in grounding.calls if c["prop_id"] == "q_assistant_account")
    assert "p_user_account" in assistant_call["context"]
    assert "acct-123" in assistant_call["history"]
    assert dejavu_client.events[-1][0]["args"] == ["acct-123"]


@pytest.mark.asyncio
async def test_monitor_sends_same_predicate_instances_in_one_dejavu_composite_call():
    proposition = Proposition(
        prop_id="p_car",
        description="the user requests a car brand under a maximum price",
        role="user",
        arity=2,
        arg_descriptions=["car brand", "maximum price"],
    )
    policy = Policy(
        policy_id="policy-1",
        name="Car requests",
        formula_str="exists brand . exists price . p_car(brand, price)",
        propositions=["p_car"],
        enabled=True,
    )
    dejavu_client = RecordingDejaVuClient()
    monitor = ConversationMonitor(
        policies=[policy],
        propositions=[proposition],
        grounding=MultiInstanceGrounding(),
        dejavu_client=dejavu_client,  # type: ignore[arg-type]
        session_id="session-1",
    )

    verdict = await monitor.process_message(
        "user",
        "I'm considering Toyota under 12000$ and Skoda under 12500$.",
    )

    assert verdict.grounding_details[0]["instances"][0]["instance_id"] == "i1"
    assert len(dejavu_client.events) == 1
    assert dejavu_client.events[-1] == [
        {"name": "p_car", "args": ["Toyota", "12000 USD"]},
        {"name": "p_car", "args": ["Skoda", "12500 USD"]},
    ]


def test_lora_style_response_parses_canonical_form():
    grounding = LLMGrounding(client=None)  # type: ignore[arg-type]
    result = grounding._parse_response(
        (
            '{"found": true, "reasoning": "ok", '
            '"object_mentions": [{"object_id": "o1", "mention": "IBM", '
            '"canonical_form": "International Business Machines"}]}'
        ),
        "p_company",
    )

    assert result.object_mentions == [{
        "object_id": "o1",
        "mention": "IBM",
        "canonical_form": "International Business Machines",
    }]
    assert result.instances == [{
        "instance_id": "i1",
        "object_mentions": [{
            "object_id": "o1",
            "mention": "IBM",
            "canonical_form": "International Business Machines",
        }],
    }]


def test_multi_instance_response_parses_all_instances():
    grounding = LLMGrounding(client=None)  # type: ignore[arg-type]
    result = grounding._parse_response(
        (
            '{"found": true, "reasoning": "two cars", "instances": ['
            '{"instance_id": "i1", "object_mentions": ['
            '{"object_id": "o1", "mention": "Toyota", "canonical_form": "Toyota"}, '
            '{"object_id": "o2", "mention": "12000$", "canonical_form": "12000 USD"}]}, '
            '{"instance_id": "i2", "object_mentions": ['
            '{"object_id": "o1", "mention": "Skoda", "canonical_form": "Skoda"}, '
            '{"object_id": "o2", "mention": "12500$", "canonical_form": "12500 USD"}]}'
            "]}"
        ),
        "p_car",
    )

    assert result.match is True
    assert len(result.instances) == 2
    assert result.instances[0]["object_mentions"][0]["canonical_form"] == "Toyota"
    assert result.instances[1]["object_mentions"][1]["canonical_form"] == "12500 USD"


def test_prompt_preview_can_preserve_related_object_placeholders():
    _, prompt = build_grounding_prompts(
        proposition=Proposition(
            prop_id="p_account",
            description="the user provides an account",
            role="user",
            arity=1,
            arg_descriptions=["account"],
        ),
        message_role="user",
        message_text="<MESSAGE_TEXT_GOES_HERE>",
        system_prompt="system",
        user_prompt_template_user=(
            "Predicate: {proposition_description}\n"
            "{objects_section}"
            "Related object context:\n"
            "{related_object_context_block}\n"
            "Related object mention and canonical history:\n"
            "{related_object_history_block}\n"
        ),
        user_prompt_template_assistant="unused",
        related_object_context_block="{{RELATED_OBJECT_CONTEXT_BLOCK}}",
        related_object_history_block="{{RELATED_OBJECT_HISTORY_BLOCK}}",
    )

    assert "{{RELATED_OBJECT_CONTEXT_BLOCK}}" in prompt
    assert "{{RELATED_OBJECT_HISTORY_BLOCK}}" in prompt
    assert "Related object context:\nNONE" not in prompt


@pytest.mark.asyncio
async def test_settings_upgrade_persisted_old_prompts_to_canonical_defaults(db):
    await db.set_setting("grounding_user_prompt_template_user", "old {message_text}")
    await db.set_setting("grounding_user_prompt_template_assistant", "old {message_text}")

    settings = await _load_settings(db)

    assert "canonical_form" in settings.grounding.user_prompt_template_user
    assert "instances" in settings.grounding.user_prompt_template_user
    assert "RELATED_OBJECT_CONTEXT_BLOCK" in settings.grounding.user_prompt_template_user
    assert "canonical_form" in settings.grounding.user_prompt_template_assistant
    assert "instances" in settings.grounding.user_prompt_template_assistant
    assert await db.get_setting("grounding_prompt_version") == "multi_instances_v1"
    assert await db.get_setting("grounding_user_prompt_template") is None
