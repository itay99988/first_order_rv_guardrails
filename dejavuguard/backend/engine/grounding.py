"""
Semantic grounding engine.

Evaluates whether a message matches a predicate using LLM-as-judge.
Grounding results feed into the DejaVu runtime verification engine,
which evaluates first-order past-time temporal logic properties.
The GroundingMethod ABC allows future extension with cosine/NLI/hybrid methods.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from backend.engine.trace import MessageEvent
from backend.models.policy import Proposition
from backend.models.settings import (
    DEFAULT_GROUNDING_SYSTEM_PROMPT,
    DEFAULT_GROUNDING_USER_PROMPT_TEMPLATE_ASSISTANT,
    DEFAULT_GROUNDING_USER_PROMPT_TEMPLATE_USER,
)
from backend.prompts.optimized_grounding import INSTANCE_RULES
from backend.services.grounding_client import GroundingClientProtocol

# Default Prompts

DEFAULT_SYSTEM_PROMPT = DEFAULT_GROUNDING_SYSTEM_PROMPT
DEFAULT_USER_PROMPT_TEMPLATE_USER = DEFAULT_GROUNDING_USER_PROMPT_TEMPLATE_USER
DEFAULT_USER_PROMPT_TEMPLATE_ASSISTANT = DEFAULT_GROUNDING_USER_PROMPT_TEMPLATE_ASSISTANT


# GroundingResult


@dataclass
class GroundingResult:
    """Result of evaluating a message against a predicate.

    The ``match`` field maps to ``found`` in the LLM response format.
    ``instances`` carries every complete predicate occurrence in the message.
    Each instance contains verbatim extracted object mentions for that
    occurrence. ``object_mentions`` is a flattened convenience view used for
    display and history indexing.
    """

    match: bool
    confidence: float
    reasoning: str
    method: str  # "llm" | "cosine" | "nli" | "hybrid"
    prop_id: str = ""
    instances: list[dict] = field(default_factory=list)
    object_mentions: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Keep old test doubles and direct constructions usable."""
        if self.instances and not self.object_mentions:
            for instance in self.instances:
                if isinstance(instance, dict):
                    mentions = instance.get("object_mentions", [])
                    if isinstance(mentions, list):
                        self.object_mentions.extend(
                            m for m in mentions if isinstance(m, dict)
                        )
        elif self.object_mentions and not self.instances:
            self.instances = [{
                "instance_id": "i1",
                "object_mentions": self.object_mentions,
            }]

    def to_dict(self) -> dict:
        """Convert to a serializable dictionary."""
        return {
            "match": self.match,
            "found": self.match,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "method": self.method,
            "prop_id": self.prop_id,
            "instances": self.instances,
            "object_mentions": self.object_mentions,
        }


# GroundingMethod ABC


class GroundingMethod(ABC):
    """Abstract base for semantic grounding methods.

    Each method evaluates whether a message matches a predicate.
    This interface enables future extension with cosine, NLI, hybrid, etc.
    """

    @abstractmethod
    async def evaluate(
        self,
        message: MessageEvent,
        proposition: Proposition,
        related_object_context_block: str = "NONE",
        related_object_history_block: str = "NONE",
    ) -> GroundingResult:
        """Evaluate whether message matches predicate.

        Returns:
            GroundingResult with match (bool), confidence (float), reasoning (str).
        """
        ...


# LLMGrounding


def _extract_json(text: str) -> dict | None:
    """Try to extract a JSON object from text, handling markdown code blocks."""
    # Strip markdown code blocks
    text = text.strip()
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if code_block:
        text = code_block.group(1).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object in the text (support nested objects)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    return None


class LLMGrounding(GroundingMethod):
    """LLM-as-judge grounding (current implementation).

    Calls a local LLM via any supported provider (Ollama, LM Studio, vLLM,
    or any OpenAI-compatible server) to classify messages against predicates.

    Fail-open: on any error, returns match=False (never blocks the conversation).
    """

    def __init__(
        self,
        client: GroundingClientProtocol,
        system_prompt: str = "",
        user_prompt_template_user: str = "",
        user_prompt_template_assistant: str = "",
    ) -> None:
        self._client = client
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.user_prompt_template_user = (
            user_prompt_template_user or DEFAULT_USER_PROMPT_TEMPLATE_USER
        )
        self.user_prompt_template_assistant = (
            user_prompt_template_assistant or DEFAULT_USER_PROMPT_TEMPLATE_ASSISTANT
        )

    async def evaluate(
        self,
        message: MessageEvent,
        proposition: Proposition,
        related_object_context_block: str = "NONE",
        related_object_history_block: str = "NONE",
    ) -> GroundingResult:
        """Evaluate whether message matches predicate using LLM.

        Fail-open: on any error (connection, JSON parse, etc.),
        returns match=False with confidence=0.0.
        """
        try:
            system_prompt, user_prompt = build_grounding_prompts(
                proposition=proposition,
                message_role=message.role,
                message_text=message.text,
                system_prompt=self.system_prompt,
                user_prompt_template_user=self.user_prompt_template_user,
                user_prompt_template_assistant=self.user_prompt_template_assistant,
                related_object_context_block=related_object_context_block,
                related_object_history_block=related_object_history_block,
            )

            print(  # noqa: T201 - user-facing debug visibility for grounding prompts.
                "\n[Grounding] RELATED_OBJECT_CONTEXT_BLOCK "
                f"for {proposition.prop_id}:\n{related_object_context_block}",
                flush=True,
            )
            print(  # noqa: T201 - user-facing debug visibility for grounding prompts.
                "[Grounding] RELATED_OBJECT_HISTORY_BLOCK "
                f"for {proposition.prop_id}:\n{related_object_history_block}\n",
                flush=True,
            )

            response_text = await self._client.chat(system_prompt, user_prompt)
            return self._parse_response(response_text, proposition.prop_id)

        except Exception as e:
            return GroundingResult(
                match=False,
                confidence=0.0,
                reasoning=f"Grounding failed with error: {e}",
                method="llm",
                prop_id=proposition.prop_id,
            )

    def _parse_response(self, response_text: str, prop_id: str) -> GroundingResult:
        """Parse the LLM's JSON response into a GroundingResult.

        Supports both new format (``found``) and old format (``match``).
        Fail-open: on parse errors, returns match=False.
        """
        data = _extract_json(response_text)

        if data is None:
            return GroundingResult(
                match=False,
                confidence=0.0,
                reasoning=f"Failed to parse LLM response as JSON: {response_text[:200]}",
                method="llm",
                prop_id=prop_id,
            )

        # Support both "found" (new) and "match" (old) field names
        if "found" in data:
            match_val = data["found"]
        elif "match" in data:
            match_val = data["match"]
        else:
            match_val = None

        if not isinstance(match_val, bool):
            return GroundingResult(
                match=False,
                confidence=0.0,
                reasoning=f"'found'/'match' field is not a boolean: {match_val}",
                method="llm",
                prop_id=prop_id,
            )

        confidence_raw = data.get("confidence")
        if isinstance(confidence_raw, int | float):
            confidence = float(confidence_raw)
        else:
            confidence = 1.0 if match_val else 0.0

        reasoning = data.get("reasoning", "")
        if not isinstance(reasoning, str):
            reasoning = str(reasoning)

        # Parse instances when found=True. Legacy flat object_mentions responses
        # are normalized to a single instance so old test doubles still work.
        instances: list[dict] = []
        object_mentions: list[dict] = []
        if match_val:
            raw_instances = data.get("instances", [])
            if isinstance(raw_instances, list):
                for idx, raw_instance in enumerate(raw_instances, start=1):
                    if not isinstance(raw_instance, dict):
                        continue
                    mentions = self._parse_object_mentions(
                        raw_instance.get("object_mentions", [])
                    )
                    if not mentions:
                        continue
                    instance_id = str(raw_instance.get("instance_id") or f"i{idx}")
                    instance = {
                        "instance_id": instance_id,
                        "object_mentions": mentions,
                    }
                    instances.append(instance)
                    object_mentions.extend(mentions)

            if not instances:
                mentions = self._parse_object_mentions(data.get("object_mentions", []))
                if mentions:
                    instances.append({
                        "instance_id": "i1",
                        "object_mentions": mentions,
                    })
                    object_mentions.extend(mentions)

        return GroundingResult(
            match=match_val,
            confidence=confidence,
            reasoning=reasoning,
            method="llm",
            prop_id=prop_id,
            instances=instances,
            object_mentions=object_mentions,
        )

    @staticmethod
    def _parse_object_mentions(raw_mentions: object) -> list[dict]:
        mentions: list[dict] = []
        if not isinstance(raw_mentions, list):
            return mentions

        for item in raw_mentions:
            if not isinstance(item, dict):
                continue
            if "object_id" not in item or "mention" not in item:
                continue
            mention = str(item["mention"])
            canonical_form = item.get("canonical_form")
            if canonical_form is None or not str(canonical_form).strip():
                canonical_form = mention
            mentions.append({
                "object_id": str(item["object_id"]),
                "mention": mention,
                "canonical_form": str(canonical_form),
                **(
                    {"canonical_source": item["canonical_source"]}
                    if isinstance(item.get("canonical_source"), dict)
                    else {}
                ),
            })
        return mentions


def render_few_shots(proposition: Proposition, role: str) -> str:
    """Render structured predicate examples as used by the optimized evaluator."""
    examples = proposition.few_shot_examples or []
    if not examples:
        return "[]"

    def instance_count(example: dict) -> int:
        return len(example.get("instances") or []) if example.get("found") else 0

    compact: list[dict] = []
    for example in sorted(examples, key=instance_count, reverse=True)[:6]:
        found = bool(example.get("found"))
        output: dict = {"found": found}
        if found:
            output["instances"] = example.get("instances", [])
        compact.append({
            "input": {
                "text": example.get("text"),
                "role": example.get("role", role),
            },
            "output": output,
        })
    return json.dumps(compact, ensure_ascii=False, indent=2)


def _build_objects_block(proposition: Proposition) -> str:
    """Build only the o1..oN object lines for templates that provide their own header."""
    if proposition.arity <= 0:
        return "NONE"

    descriptions = proposition.arg_descriptions or []
    lines: list[str] = []
    for i in range(proposition.arity):
        desc = descriptions[i] if i < len(descriptions) else f"argument {i + 1}"
        lines.append(f"  - o{i + 1}: {desc}")
    return "\n".join(lines)


def _build_objects_section(proposition: Proposition) -> str:
    """Build the Objects section for the grounding prompt.

    When arity > 0, creates an objects list mapping o1..oN to argument
    descriptions.  When arity is 0, returns an empty string so the prompt
    uses a simplified Boolean-only format.
    """
    if proposition.arity <= 0:
        return ""

    return "Objects:\n" + _build_objects_block(proposition) + "\n"


def _build_predicate_block(proposition: Proposition) -> str:
    """Build the predicate-and-objects JSON block used by the optimized prompt."""
    objects: list[dict[str, str]] = []
    for i in range(proposition.arity):
        description = (
            proposition.arg_descriptions[i]
            if i < len(proposition.arg_descriptions)
            else f"argument {i + 1}"
        )
        objects.append({"object_id": f"o{i + 1}", "description": description})
    return json.dumps(
        {"predicate_description": proposition.description, "objects": objects},
        ensure_ascii=False,
        indent=2,
    )


def _replace_prompt_template_aliases(rendered_prompt: str, alias_values: dict[str, str]) -> str:
    """Replace known placeholders without interpreting JSON braces.

    The active grounding prompts contain JSON examples. Using ``str.format``
    on those templates is brittle because JSON object braces look like Python
    format fields. This renderer intentionally performs literal replacement
    only for known placeholders.
    """
    for key, value in alias_values.items():
        rendered_prompt = rendered_prompt.replace("{{" + key + "}}", value)
        rendered_prompt = rendered_prompt.replace("{" + key + "}", value)
    return rendered_prompt


def build_grounding_prompts(
    proposition: Proposition,
    message_role: str,
    message_text: str,
    system_prompt: str,
    user_prompt_template_user: str,
    user_prompt_template_assistant: str,
    related_object_context_block: str = "NONE",
    related_object_history_block: str = "NONE",
) -> tuple[str, str]:
    """Build system/user prompts for a predicate-message pair."""
    role = (proposition.role or message_role or "user").strip().lower()
    if role == "assistant":
        final_system_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        user_template = (
            user_prompt_template_assistant or DEFAULT_USER_PROMPT_TEMPLATE_ASSISTANT
        )
    else:
        final_system_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        user_template = user_prompt_template_user or DEFAULT_USER_PROMPT_TEMPLATE_USER

    few_shot_examples = render_few_shots(proposition, role)
    objects_section = _build_objects_section(proposition)
    objects_block = _build_objects_block(proposition)
    predicate_block = _build_predicate_block(proposition)
    alias_values = {
        "TEXT": message_text,
        "PREDICATE_DESCRIPTION": proposition.description,
        "PREDICATE_ROLE": proposition.role,
        "MESSAGE_ROLE": message_role,
        "MESSAGE_ROLE_UPPER": (message_role or "user").strip().upper(),
        "OBJECTS_BLOCK": objects_block,
        "OBJECTS_SECTION": objects_section,
        "FEW_SHOT_EXAMPLES": few_shot_examples,
        "RELATED_OBJECT_CONTEXT_BLOCK": related_object_context_block,
        "RELATED_OBJECT_HISTORY_BLOCK": related_object_history_block,
        "proposition_description": proposition.description,
        "proposition_role": proposition.role,
        "message_role": message_role,
        "message_role_upper": (message_role or "user").strip().upper(),
        "message_text": message_text,
        "few_shot_examples": few_shot_examples,
        "objects_section": objects_section,
        "objects_block": objects_block,
        "related_object_context_block": related_object_context_block,
        "related_object_history_block": related_object_history_block,
        "predicate_block": predicate_block,
        "related_object_context": related_object_context_block,
        "related_object_history": related_object_history_block,
        "few_shot_block": few_shot_examples,
        "instance_rules": INSTANCE_RULES,
        "predicate_description": proposition.description,
        "role": message_role,
        "text": message_text,
    }

    user_prompt = _replace_prompt_template_aliases(user_template, alias_values)
    return final_system_prompt, user_prompt
