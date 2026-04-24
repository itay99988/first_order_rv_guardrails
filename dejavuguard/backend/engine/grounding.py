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
    ``object_mentions`` carries verbatim extracted argument mentions
    when the predicate has arity > 0.
    """

    match: bool
    confidence: float
    reasoning: str
    method: str  # "llm" | "cosine" | "nli" | "hybrid"
    prop_id: str = ""
    object_mentions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to a serializable dictionary."""
        return {
            "match": self.match,
            "found": self.match,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "method": self.method,
            "prop_id": self.prop_id,
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
        if isinstance(confidence_raw, (int, float)):
            confidence = float(confidence_raw)
        else:
            confidence = 1.0 if match_val else 0.0

        reasoning = data.get("reasoning", "")
        if not isinstance(reasoning, str):
            reasoning = str(reasoning)

        # Parse object_mentions when found=True
        object_mentions: list[dict] = []
        if match_val:
            raw_mentions = data.get("object_mentions", [])
            if isinstance(raw_mentions, list):
                for item in raw_mentions:
                    if isinstance(item, dict) and "object_id" in item and "mention" in item:
                        object_mentions.append({
                            "object_id": str(item["object_id"]),
                            "mention": str(item["mention"]),
                        })

        return GroundingResult(
            match=match_val,
            confidence=confidence,
            reasoning=reasoning,
            method="llm",
            prop_id=prop_id,
            object_mentions=object_mentions,
        )


def render_few_shots(proposition: Proposition, role: str) -> str:
    """Render predicate few-shot examples in the exact structure used by evaluator scripts."""
    role_label = "USER" if (role or "").strip().lower() != "assistant" else "ASSISTANT"
    positives = proposition.few_shot_positive or []
    negatives = proposition.few_shot_negative or []
    if not positives and not negatives:
        return "NONE"

    lines: list[str] = []
    i = 1
    for txt in positives:
        lines.append("Example {}:\nLabel: MATCH\n{} MESSAGE: {}".format(i, role_label, txt))
        i += 1
    for txt in negatives:
        lines.append(
            "Example {}:\nLabel: NO_MATCH\n{} MESSAGE: {}".format(i, role_label, txt)
        )
        i += 1
    return "\n\n".join(lines)


def _build_objects_section(proposition: Proposition) -> str:
    """Build the Objects section for the grounding prompt.

    When arity > 0, creates an objects list mapping o1..oN to argument
    descriptions.  When arity is 0, returns an empty string so the prompt
    uses a simplified Boolean-only format.
    """
    if proposition.arity <= 0:
        return ""

    descriptions = proposition.arg_descriptions or []
    lines: list[str] = []
    for i in range(proposition.arity):
        desc = descriptions[i] if i < len(descriptions) else f"argument {i + 1}"
        lines.append(f"  - o{i + 1}: {desc}")

    return "Objects:\n" + "\n".join(lines) + "\n"


def build_grounding_prompts(
    proposition: Proposition,
    message_role: str,
    message_text: str,
    system_prompt: str,
    user_prompt_template_user: str,
    user_prompt_template_assistant: str,
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

    user_prompt = user_template.format(
        proposition_description=proposition.description,
        proposition_role=proposition.role,
        message_role=message_role,
        message_role_upper=(message_role or "user").strip().upper(),
        message_text=message_text,
        few_shot_examples=few_shot_examples,
        objects_section=objects_section,
    )
    return final_system_prompt, user_prompt
