from __future__ import annotations

import json

import pytest

from backend.engine.dejavu_client import DejaVuVerdict
from backend.engine.grounding import (
    ConversationSummaryUpdater,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT_TEMPLATE_ASSISTANT,
    DEFAULT_USER_PROMPT_TEMPLATE_USER,
    GroundingMethod,
    GroundingResult,
    LLMGrounding,
    build_grounding_prompts,
)
from backend.engine.monitor import ConversationMonitor
from backend.engine.trace import MessageEvent
from backend.models.policy import Policy, Proposition
from backend.routers.policies import _extract_related_object_relations, _parse_few_shot_examples
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


class RejectingDejaVuClient(RecordingDejaVuClient):
    async def send_events(self, session_id: str, events: list[dict]) -> DejaVuVerdict:
        self.events.append(events)
        return DejaVuVerdict(
            event_number=len(self.events),
            verdicts={"pol_policy_1": False},
            violations=[],
        )


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


class SummaryRecordingGrounding(GroundingMethod):
    def __init__(self, match: bool = True) -> None:
        self.match = match
        self.calls: list[dict] = []

    async def evaluate(
        self,
        message: MessageEvent,
        proposition: Proposition,
        related_object_context_block: str = "NONE",
        related_object_history_block: str = "NONE",
        conversation_summary_block: str = "NONE",
        grounding_scope: str | None = None,
    ) -> GroundingResult:
        self.calls.append({
            "prop_id": proposition.prop_id,
            "summary": conversation_summary_block,
            "grounding_scope": grounding_scope,
        })
        return GroundingResult(
            match=self.match,
            confidence=1.0 if self.match else 0.0,
            reasoning="summary test",
            method="test",
            prop_id=proposition.prop_id,
        )


class FakeSummaryUpdater:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    async def update(self, previous_summary: str, role: str, text: str) -> str:
        self.calls.append((previous_summary, role, text))
        return f"{previous_summary}|{role}: {text}".strip("|")


class FakeSummaryClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    async def chat(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.response

    async def health_check(self) -> bool:
        return True

    async def list_models(self) -> list[str]:
        return []


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
    assert "the user provides an account" in assistant_call["context"]
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
        {"name": "user_turn", "args": []},
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


def test_grounding_prompt_renders_conversation_summary_separately_from_message_text():
    _, prompt = build_grounding_prompts(
        proposition=Proposition(
            prop_id="p_account",
            description="the user provides an account",
            role="user",
            arity=1,
            arg_descriptions=["account"],
        ),
        message_role="user",
        message_text="Use account 123.",
        system_prompt="system",
        user_prompt_template_user=DEFAULT_USER_PROMPT_TEMPLATE_USER,
        user_prompt_template_assistant=DEFAULT_USER_PROMPT_TEMPLATE_ASSISTANT,
        conversation_summary_block="Earlier: the user discussed account ABC.",
        include_conversation_summary=True,
    )

    assert "Conversation summary before the current message:" in prompt
    assert "Earlier: the user discussed account ABC." in prompt
    assert "Message text:\nUse account 123." in prompt


def test_single_message_grounding_prompt_omits_conversation_summary():
    _, prompt = build_grounding_prompts(
        proposition=Proposition(
            prop_id="p_account",
            description="the user provides an account",
            role="user",
        ),
        message_role="user",
        message_text="Use account 123.",
        system_prompt="system",
        user_prompt_template_user=DEFAULT_USER_PROMPT_TEMPLATE_USER,
        user_prompt_template_assistant=DEFAULT_USER_PROMPT_TEMPLATE_ASSISTANT,
        conversation_summary_block="Earlier: the user discussed account ABC.",
        include_conversation_summary=False,
    )

    assert "Conversation summary before the current message:" not in prompt
    assert "Earlier: the user discussed account ABC." not in prompt
    assert "Message text:\nUse account 123." in prompt


@pytest.mark.asyncio
async def test_summary_updater_parses_valid_json_and_fails_open():
    updater = ConversationSummaryUpdater(
        client=FakeSummaryClient('{"summary": "User mentioned account 123."}'),  # type: ignore[arg-type]
        system_prompt="system",
        user_prompt_template="{conversation_summary}\n{role}: {text}",
    )
    updated = await updater.update("", "user", "Use account 123.")
    assert updated == "User mentioned account 123."

    bad_updater = ConversationSummaryUpdater(
        client=FakeSummaryClient("not json"),  # type: ignore[arg-type]
        system_prompt="system",
        user_prompt_template="{conversation_summary}\n{role}: {text}",
    )
    assert await bad_updater.update("old summary", "assistant", "ok") == "old summary"


@pytest.mark.asyncio
async def test_summary_updater_injects_previous_summary_for_stale_template():
    client = FakeSummaryClient('{"summary": "Updated natural language summary."}')
    updater = ConversationSummaryUpdater(
        client=client,  # type: ignore[arg-type]
        system_prompt="system",
        user_prompt_template='Return {"summary": "..."}',
    )

    await updater.update("Previous natural language summary.", "user", "New message.")

    sent_prompt = client.calls[0][1]
    assert "Previous conversation summary:" in sent_prompt
    assert "Previous natural language summary." in sent_prompt
    assert "New delivered message:" in sent_prompt
    assert "user: New message." in sent_prompt


def test_grounding_prompt_injects_summary_for_stale_template():
    _, prompt = build_grounding_prompts(
        proposition=Proposition(
            prop_id="p_account",
            description="the user provides an account",
            role="user",
        ),
        message_role="user",
        message_text="Use that account.",
        system_prompt="system",
        user_prompt_template_user="Predicate: {predicate_description}\nMessage text:\n{text}",
        user_prompt_template_assistant="unused",
        conversation_summary_block="Earlier, the user identified ACCT-123.",
        include_conversation_summary=True,
    )

    assert "Conversation summary before the current message:" in prompt
    assert "Earlier, the user identified ACCT-123." in prompt
    assert "Message text:\nUse that account." in prompt


@pytest.mark.asyncio
async def test_monitor_passes_previous_summary_and_updates_only_after_pass():
    proposition = Proposition(
        prop_id="p_any",
        description="the user says anything",
        role="user",
        grounding_scope="conversation_history",
    )
    policy = Policy(
        policy_id="policy-1",
        name="Always true",
        formula_str="true",
        propositions=["p_any"],
        enabled=True,
    )
    grounding = SummaryRecordingGrounding(match=True)
    updater = FakeSummaryUpdater()
    monitor = ConversationMonitor(
        policies=[policy],
        propositions=[proposition],
        grounding=grounding,
        dejavu_client=RecordingDejaVuClient(),  # type: ignore[arg-type]
        session_id="session-1",
        conversation_summary="Previous summary",
        summary_last_trace_index=4,
        summary_updater=updater,  # type: ignore[arg-type]
    )

    verdict = await monitor.process_message("user", "New message.")

    assert verdict.passed is True
    assert grounding.calls[0]["summary"] == "Previous summary"
    assert updater.calls == [("Previous summary", "user", "New message.")]
    assert monitor.conversation_summary == "Previous summary|user: New message."
    assert monitor.summary_last_trace_index == 0


@pytest.mark.asyncio
async def test_monitor_does_not_update_summary_for_blocked_messages():
    proposition = Proposition(
        prop_id="p_any",
        description="the user says anything",
        role="user",
        grounding_scope="conversation_history",
    )
    policy = Policy(
        policy_id="policy-1",
        name="Reject",
        formula_str="p_any",
        propositions=["p_any"],
        enabled=True,
    )
    updater = FakeSummaryUpdater()
    monitor = ConversationMonitor(
        policies=[policy],
        propositions=[proposition],
        grounding=SummaryRecordingGrounding(match=True),
        dejavu_client=RejectingDejaVuClient(),  # type: ignore[arg-type]
        session_id="session-1",
        conversation_summary="Previous summary",
        summary_updater=updater,  # type: ignore[arg-type]
    )

    verdict = await monitor.process_message("user", "Blocked message.")

    assert verdict.passed is False
    assert updater.calls == []
    assert monitor.conversation_summary == "Previous summary"


def test_optimized_prompt_renders_structured_few_shot_examples_and_objects():
    system_prompt, prompt = build_grounding_prompts(
        proposition=Proposition(
            prop_id="p_car",
            description="the user requests a car under a maximum price",
            role="user",
            arity=2,
            arg_descriptions=["car brand", "maximum price"],
            few_shot_examples=[{
                "text": "Toyota under 12000$",
                "role": "user",
                "found": True,
                "instances": [{
                    "instance_id": "i1",
                    "object_mentions": [
                        {
                            "object_id": "o1",
                            "mention": "Toyota",
                            "canonical_form": "Toyota",
                            "canonical_source": {"type": "new"},
                        },
                        {
                            "object_id": "o2",
                            "mention": "12000$",
                            "canonical_form": "12000 USD",
                            "canonical_source": {"type": "new"},
                        },
                    ],
                }],
            }],
        ),
        message_role="user",
        message_text="Honda under 11000$",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        user_prompt_template_user=DEFAULT_USER_PROMPT_TEMPLATE_USER,
        user_prompt_template_assistant=DEFAULT_USER_PROMPT_TEMPLATE_ASSISTANT,
        related_object_context_block=(
            "- p_allergy: the user mentions a specific allergy; "
            "related object o1 (allergen)"
        ),
        related_object_history_block="[]",
    )

    assert "strict JSON-only extraction model" in system_prompt
    assert '"description": "car brand"' in prompt
    assert '"text": "Toyota under 12000$"' in prompt
    assert '"canonical_source"' in prompt
    assert "the user mentions a specific allergy" in prompt
    assert "Honda under 11000$" in prompt


def test_structured_few_shot_generation_output_validation():
    positives = []
    for index in range(3):
        source = {"type": "new"}
        history = []
        canonical_form = "ACCT-7"
        if index == 0:
            source = {"type": "history", "matched_history_index": 0}
            history = [{"mention": "Account Seven", "canonical_form": "ACCT-7"}]
        positives.append({
            "text": "Use account seven for this request.",
            "role": "user",
            "related_object_context": [],
            "related_object_history": history,
            "found": True,
            "instances": [{
                "instance_id": "i1",
                "object_mentions": [{
                    "object_id": "o1",
                    "mention": "account seven",
                    "canonical_form": canonical_form,
                    "canonical_source": source,
                }],
            }],
        })
    negatives = [{
        "text": "Where can I find my account number?",
        "role": "user",
        "related_object_context": [],
        "related_object_history": [],
        "found": False,
    } for _ in range(3)]

    examples = _parse_few_shot_examples(
        json.dumps({"examples": [*positives, *negatives]}),
        "user",
        ["o1"],
    )

    assert len(examples) == 6
    assert examples[0]["instances"][0]["object_mentions"][0]["canonical_source"]["type"] == "history"


@pytest.mark.asyncio
async def test_settings_upgrade_persisted_old_prompts_to_canonical_defaults(db):
    await db.set_setting("grounding_user_prompt_template_user", "old {message_text}")
    await db.set_setting("grounding_user_prompt_template_assistant", "old {message_text}")

    settings = await _load_settings(db)

    assert "few_shot_block" in settings.grounding.single_user_prompt_template_user
    assert "predicate_block" in settings.grounding.single_user_prompt_template_user
    assert "related_object_context" in settings.grounding.single_user_prompt_template_user
    assert "related_object_history" in settings.grounding.single_user_prompt_template_user
    assert "conversation_summary" not in settings.grounding.single_user_prompt_template_user
    assert "few_shot_block" in settings.grounding.history_user_prompt_template_assistant
    assert "instances" in settings.grounding.history_user_prompt_template_assistant
    assert "conversation_summary" in settings.grounding.history_user_prompt_template_user
    assert settings.grounding.summary_system_prompt
    assert settings.grounding.summary_user_prompt_template
    assert await db.get_setting("grounding_prompt_version") == "grounding_scope_split_v1"
    assert await db.get_setting("grounding_user_prompt_template") is None




def test_zero_arity_prompt_does_not_ask_for_object_mentions():
    """A predicate with no objects must not be shown an object-bearing template.

    The generation prompt was hardcoded for object-bearing predicates: it said
    every instance "includes every required object exactly once" and showed a
    template containing object_id "o1". For a 0-arity predicate the model
    dutifully invented an object, validation rejected the example, and the whole
    generation was discarded -- so every 0-arity predicate on a fresh install
    silently fell back to zero-shot.
    """
    from backend.routers.policies import _few_shot_generation_prompt

    prompt = _few_shot_generation_prompt(
        "sa_pressure_a",
        "The assistant uses urgency or scarcity language to push the user to buy now.",
        "assistant",
        [],
    )

    assert '"object_id"' not in prompt
    assert '"instances": []' in prompt


def test_zero_arity_positive_example_survives_validation():
    """The shape the fixed prompt asks for must actually validate."""
    positives = [{
        "text": "Only two left in stock -- order today!",
        "role": "assistant",
        "related_object_context": [],
        "related_object_history": [],
        "found": True,
        "instances": [],
    } for _ in range(3)]
    negatives = [{
        "text": "The model is available in three colours.",
        "role": "assistant",
        "related_object_context": [],
        "related_object_history": [],
        "found": False,
    } for _ in range(3)]

    examples = _parse_few_shot_examples(
        json.dumps({"examples": [*positives, *negatives]}),
        "assistant",
        [],
    )

    assert len(examples) == 6
