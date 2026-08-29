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
    DEFAULT_GROUNDING_HISTORY_SYSTEM_PROMPT,
    DEFAULT_GROUNDING_HISTORY_USER_PROMPT_TEMPLATE_ASSISTANT,
    DEFAULT_GROUNDING_HISTORY_USER_PROMPT_TEMPLATE_USER,
    DEFAULT_GROUNDING_SINGLE_SYSTEM_PROMPT,
    DEFAULT_GROUNDING_SINGLE_USER_PROMPT_TEMPLATE_ASSISTANT,
    DEFAULT_GROUNDING_SINGLE_USER_PROMPT_TEMPLATE_USER,
    DEFAULT_GROUNDING_SUMMARY_SYSTEM_PROMPT,
    DEFAULT_GROUNDING_SUMMARY_USER_PROMPT_TEMPLATE,
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
DEFAULT_SINGLE_SYSTEM_PROMPT = DEFAULT_GROUNDING_SINGLE_SYSTEM_PROMPT
DEFAULT_SINGLE_USER_PROMPT_TEMPLATE_USER = DEFAULT_GROUNDING_SINGLE_USER_PROMPT_TEMPLATE_USER
DEFAULT_SINGLE_USER_PROMPT_TEMPLATE_ASSISTANT = (
    DEFAULT_GROUNDING_SINGLE_USER_PROMPT_TEMPLATE_ASSISTANT
)
DEFAULT_HISTORY_SYSTEM_PROMPT = DEFAULT_GROUNDING_HISTORY_SYSTEM_PROMPT
DEFAULT_HISTORY_USER_PROMPT_TEMPLATE_USER = DEFAULT_GROUNDING_HISTORY_USER_PROMPT_TEMPLATE_USER
DEFAULT_HISTORY_USER_PROMPT_TEMPLATE_ASSISTANT = (
    DEFAULT_GROUNDING_HISTORY_USER_PROMPT_TEMPLATE_ASSISTANT
)
DEFAULT_SUMMARY_SYSTEM_PROMPT = DEFAULT_GROUNDING_SUMMARY_SYSTEM_PROMPT
DEFAULT_SUMMARY_USER_PROMPT_TEMPLATE = DEFAULT_GROUNDING_SUMMARY_USER_PROMPT_TEMPLATE
GROUNDING_SUMMARY_HEADER = "Conversation summary before the current message:"
SUMMARY_PREVIOUS_HEADER = "Previous conversation summary:"
SUMMARY_NEW_MESSAGE_HEADER = "New delivered message:"


# Predicates whose grounding prompt has already been dumped to the batch log
# this process run. We write one complete prompt per predicate (system + the
# first user prompt, which embeds the few-shot block) the first time that
# predicate is grounded — i.e. once, for the whole batch.
_PROMPT_SEEN: set[str] = set()

# Recorded in place of the justification when a grounding answer arrives
# without one, so the gap is visible rather than silent.
MISSING_REASONING = "The grounding model gave no reasoning for this verdict."


def _save_grounding_prompt(prop_id: str, system_prompt: str, user_prompt: str) -> None:
    """If GROUNDING_PROMPT_DIR is set (the runner points it at the batch log
    folder), save the grounding prompt for this predicate to <prop_id>.txt.

    Written exactly once per predicate per batch: the system prompt plus the
    first user prompt (which already embeds the generated few-shot examples).
    Subsequent messages and scenarios are skipped, so the file stays a single
    clean representative prompt rather than one entry per message."""
    import os
    d = os.environ.get("GROUNDING_PROMPT_DIR")
    if not d or prop_id in _PROMPT_SEEN:
        return
    _PROMPT_SEEN.add(prop_id)
    try:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{prop_id}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + f"\nGROUNDING PROMPT — predicate '{prop_id}'\n")
            f.write("(saved once per batch, from the first scenario that grounds "
                    "this predicate; few-shot examples are embedded in the user "
                    "prompt)\n" + "=" * 80)
            f.write("\n\n----- SYSTEM PROMPT -----\n" + system_prompt + "\n")
            f.write("\n----- USER PROMPT -----\n" + user_prompt + "\n")
    except Exception:
        pass  # prompt logging must never break grounding


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
    # True when the predicate could not be evaluated at all -- a dead provider,
    # a refused key, an unparseable reply. Distinct from match=False, which
    # asserts the predicate genuinely did not occur. Collapsing the two lets a
    # broken grounder report every policy as satisfied.
    unavailable: bool = False
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
            "unavailable": self.unavailable,
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
        conversation_summary_block: str = "NONE",
        grounding_scope: str | None = None,
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


def _parse_confidence(raw: object) -> float:
    """Read the model's confidence in its own verdict, clamped to 0..1.

    A missing or non-numeric confidence becomes 0.0. The model told us nothing
    about how sure it is, which is not the same as being certain, and a
    fabricated 1.0 is exactly what made past grounding records unreadable.
    """
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return 0.0
    return max(0.0, min(1.0, float(raw)))


def _parse_reasoning(raw: object) -> str:
    """Read the one-sentence justification the grounding prompt asks for.

    An answer that arrives without one is labelled as such, so a person
    auditing a blocked message can tell an unexplained verdict from an
    explained one instead of reading a blank line.
    """
    if raw is None:
        return MISSING_REASONING
    if isinstance(raw, str):
        return raw.strip() or MISSING_REASONING
    return str(raw)


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
        single_system_prompt: str = "",
        single_user_prompt_template_user: str = "",
        single_user_prompt_template_assistant: str = "",
        history_system_prompt: str = "",
        history_user_prompt_template_user: str = "",
        history_user_prompt_template_assistant: str = "",
    ) -> None:
        self._client = client
        self.single_system_prompt = (
            single_system_prompt or system_prompt or DEFAULT_SINGLE_SYSTEM_PROMPT
        )
        self.single_user_prompt_template_user = (
            single_user_prompt_template_user
            or user_prompt_template_user
            or DEFAULT_SINGLE_USER_PROMPT_TEMPLATE_USER
        )
        self.single_user_prompt_template_assistant = (
            single_user_prompt_template_assistant
            or user_prompt_template_assistant
            or DEFAULT_SINGLE_USER_PROMPT_TEMPLATE_ASSISTANT
        )
        self.history_system_prompt = (
            history_system_prompt or system_prompt or DEFAULT_HISTORY_SYSTEM_PROMPT
        )
        self.history_user_prompt_template_user = (
            history_user_prompt_template_user
            or user_prompt_template_user
            or DEFAULT_HISTORY_USER_PROMPT_TEMPLATE_USER
        )
        self.history_user_prompt_template_assistant = (
            history_user_prompt_template_assistant
            or user_prompt_template_assistant
            or DEFAULT_HISTORY_USER_PROMPT_TEMPLATE_ASSISTANT
        )
        # Legacy attributes retained for tests/custom callers that inspect them.
        self.system_prompt = self.history_system_prompt
        self.user_prompt_template_user = self.history_user_prompt_template_user
        self.user_prompt_template_assistant = self.history_user_prompt_template_assistant

    async def evaluate(
        self,
        message: MessageEvent,
        proposition: Proposition,
        related_object_context_block: str = "NONE",
        related_object_history_block: str = "NONE",
        conversation_summary_block: str = "NONE",
        grounding_scope: str | None = None,
    ) -> GroundingResult:
        """Evaluate whether message matches predicate using LLM.

        Fail-open: on any error (connection, JSON parse, etc.),
        returns match=False with confidence=0.0.
        """
        try:
            effective_scope = grounding_scope or proposition.grounding_scope
            history_aware = effective_scope == "conversation_history"
            system_prompt = (
                self.history_system_prompt
                if history_aware
                else self.single_system_prompt
            )
            user_prompt_template_user = (
                self.history_user_prompt_template_user
                if history_aware
                else self.single_user_prompt_template_user
            )
            user_prompt_template_assistant = (
                self.history_user_prompt_template_assistant
                if history_aware
                else self.single_user_prompt_template_assistant
            )
            system_prompt, user_prompt = build_grounding_prompts(
                proposition=proposition,
                message_role=message.role,
                message_text=message.text,
                system_prompt=system_prompt,
                user_prompt_template_user=user_prompt_template_user,
                user_prompt_template_assistant=user_prompt_template_assistant,
                related_object_context_block=related_object_context_block,
                related_object_history_block=related_object_history_block,
                conversation_summary_block=conversation_summary_block,
                include_conversation_summary=history_aware,
            )

            _save_grounding_prompt(proposition.prop_id, system_prompt, user_prompt)

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
            if history_aware:
                print(  # noqa: T201 - user-facing debug visibility for grounding prompts.
                    "[Grounding] CONVERSATION_SUMMARY_BLOCK "
                    f"for {proposition.prop_id}:\n{conversation_summary_block}\n",
                    flush=True,
                )

            response_text = await self._client.chat(system_prompt, user_prompt)
            return self._parse_response(response_text, proposition.prop_id)

        except Exception as e:
            return GroundingResult(
                match=False,
                confidence=0.0,
                unavailable=True,
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
                unavailable=True,
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
                unavailable=True,
                reasoning=f"'found'/'match' field is not a boolean: {match_val}",
                method="llm",
                prop_id=prop_id,
            )

        confidence = _parse_confidence(data.get("confidence"))
        reasoning = _parse_reasoning(data.get("reasoning"))

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


class ConversationSummaryUpdater:
    """Maintains a concise per-conversation summary with the grounding LLM.

    Fail-open: invalid JSON, API errors, or malformed outputs keep the previous
    summary unchanged.
    """

    def __init__(
        self,
        client: GroundingClientProtocol,
        system_prompt: str = "",
        user_prompt_template: str = "",
    ) -> None:
        self._client = client
        self.system_prompt = system_prompt or DEFAULT_SUMMARY_SYSTEM_PROMPT
        self.user_prompt_template = (
            user_prompt_template or DEFAULT_SUMMARY_USER_PROMPT_TEMPLATE
        )

    async def update(self, previous_summary: str, role: str, text: str) -> str:
        """Return the updated summary, or the old one if updating fails."""
        old_summary = (previous_summary or "").strip()
        summary_block = old_summary or "NONE"
        try:
            user_prompt = _replace_prompt_template_aliases(
                self.user_prompt_template,
                {
                    "conversation_summary": summary_block,
                    "CONVERSATION_SUMMARY": summary_block,
                    "role": role,
                    "ROLE": role,
                    "text": text,
                    "TEXT": text,
                },
            )
            user_prompt = _ensure_summary_update_context(
                user_prompt,
                summary_block,
                role,
                text,
            )
            response_text = await self._client.chat(self.system_prompt, user_prompt)
            data = _extract_json(response_text)
            if not isinstance(data, dict):
                return old_summary
            new_summary = data.get("summary")
            if not isinstance(new_summary, str):
                return old_summary
            return new_summary.strip()
        except Exception:
            return old_summary


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


def _ensure_grounding_summary_block(user_prompt: str, summary_block: str) -> str:
    """Guarantee the grounding prompt includes the conversation summary."""
    if GROUNDING_SUMMARY_HEADER in user_prompt:
        return user_prompt

    block = f"{GROUNDING_SUMMARY_HEADER}\n{summary_block or 'NONE'}\n\n"
    marker = "Message text:"
    if marker in user_prompt:
        return user_prompt.replace(marker, block + marker, 1)
    return user_prompt.rstrip() + "\n\n" + block.rstrip()


def _remove_grounding_summary_block(user_prompt: str) -> str:
    """Remove the summary block when rendering single-message grounding."""
    following_header = (
        r"\n\n[A-Z][^\n]*:"
        r"|\n\nRelated object"
        r"|\n\nFew-shot"
        r"|\n\nMessage role:"
        r"|\n\nMessage text:"
        r"|$"
    )
    pattern = re.compile(
        rf"\n*{re.escape(GROUNDING_SUMMARY_HEADER)}\n.*?(?={following_header})",
        re.DOTALL,
    )
    return pattern.sub("\n", user_prompt).replace("\n\n\n", "\n\n").strip()


def _ensure_summary_update_context(
    user_prompt: str,
    summary_block: str,
    role: str,
    text: str,
) -> str:
    """Guarantee the summary-update prompt includes prior summary and new message."""
    prefix_parts: list[str] = []
    if SUMMARY_PREVIOUS_HEADER not in user_prompt:
        prefix_parts.append(f"{SUMMARY_PREVIOUS_HEADER}\n{summary_block or 'NONE'}")
    if SUMMARY_NEW_MESSAGE_HEADER not in user_prompt:
        prefix_parts.append(f"{SUMMARY_NEW_MESSAGE_HEADER}\n{role}: {text}")
    if not prefix_parts:
        return user_prompt
    return "\n\n".join(prefix_parts) + "\n\n" + user_prompt


def build_grounding_prompts(
    proposition: Proposition,
    message_role: str,
    message_text: str,
    system_prompt: str,
    user_prompt_template_user: str,
    user_prompt_template_assistant: str,
    related_object_context_block: str = "NONE",
    related_object_history_block: str = "NONE",
    conversation_summary_block: str = "NONE",
    include_conversation_summary: bool = False,
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
        "CONVERSATION_SUMMARY": (
            conversation_summary_block if include_conversation_summary else "NONE"
        ),
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
        "conversation_summary_block": (
            conversation_summary_block if include_conversation_summary else "NONE"
        ),
        "predicate_block": predicate_block,
        "related_object_context": related_object_context_block,
        "related_object_history": related_object_history_block,
        "conversation_summary": (
            conversation_summary_block if include_conversation_summary else "NONE"
        ),
        "few_shot_block": few_shot_examples,
        "instance_rules": INSTANCE_RULES,
        "predicate_description": proposition.description,
        "role": message_role,
        "text": message_text,
    }

    user_prompt = _replace_prompt_template_aliases(user_template, alias_values)
    if include_conversation_summary:
        user_prompt = _ensure_grounding_summary_block(
            user_prompt,
            conversation_summary_block or "NONE",
        )
    else:
        user_prompt = _remove_grounding_summary_block(user_prompt)
    return final_system_prompt, user_prompt
