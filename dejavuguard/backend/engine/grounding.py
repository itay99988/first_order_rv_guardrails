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
from dataclasses import dataclass

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
    """Result of evaluating a message against a predicate."""

    match: bool
    confidence: float
    reasoning: str
    method: str  # "llm" | "cosine" | "nli" | "hybrid"
    prop_id: str = ""

    def to_dict(self) -> dict:
        """Convert to a serializable dictionary."""
        return {
            "match": self.match,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "method": self.method,
            "prop_id": self.prop_id,
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

    # Try to find a JSON object in the text
    match = re.search(r"\{[^{}]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
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

        match_val = data.get("match")
        if not isinstance(match_val, bool):
            return GroundingResult(
                match=False,
                confidence=0.0,
                reasoning=f"'match' field is not a boolean: {match_val}",
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

        return GroundingResult(
            match=match_val,
            confidence=confidence,
            reasoning=reasoning,
            method="llm",
            prop_id=prop_id,
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
        final_system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        user_template = (
            user_prompt_template_assistant or DEFAULT_USER_PROMPT_TEMPLATE_ASSISTANT
        )
    else:
        final_system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        user_template = user_prompt_template_user or DEFAULT_USER_PROMPT_TEMPLATE_USER

    few_shot_examples = render_few_shots(proposition, role)

    user_prompt = user_template.format(
        proposition_description=proposition.description,
        proposition_role=proposition.role,
        message_role=message_role,
        message_role_upper=(message_role or "user").strip().upper(),
        message_text=message_text,
        few_shot_examples=few_shot_examples,
    )
    return final_system_prompt, user_prompt
