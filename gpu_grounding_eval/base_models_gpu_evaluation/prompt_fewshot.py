"""Few-shot prompting approach for the extended grounding task.

The evaluator imports this module and calls predict(record, ...).  The prompt
uses predicate-specific few-shot examples from few_shot_examples.json when
available.
"""

from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
from typing import Any


BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "mistralai/ministral-8b-2512"
TEMPERATURE = 0.0
MAX_API_RETRIES = 3
REQUEST_TIMEOUT = 45


SYSTEM_PROMPT = """You are a strict JSON-only extraction model for extended first-order grounding.

Step 1 — decide found=true or found=false. Return found=false unless the message ACTIVELY AND EXPLICITLY performs the predicate right now. Specifically, return found=false when:
- The message uses information-request framing to ask whether the predicate holds ("Can you tell me whether X", "Please confirm if X", "I need to confirm whether X", "Can you confirm if X")
- The message queries availability or existence ("Are there flights from X to Y?", "Is X available?")
- The predicate action is purely historical, conditional, or hypothetical — applies to BOTH questions AND statements: "Last year X was with Y", "previously X held Y", "if X were to..."
- The relevant entities appear only as background context, not as the direct subject of the predicate ("I need info about the case involving X and Y" — X and Y are context only)
- The message looks for or wants to find something, rather than actually requesting or providing it ("I want the title that X held")
- Not all required objects (o1, o2, ...) are explicitly present as distinct named mentions in the message (pronouns and vague references like "my wife", "him", "her" are NOT sufficient)

Note: the grammatical form does not determine found. Direct questions, tag questions, declarative statements, and checking expressions can all be found=true as long as the predicate relationship is directly expressed between explicitly named entities. In particular, for predicates that describe "the user asks about/for X", a direct question that queries that specific relationship ("Was X on Y?", "Is X at Y?") IS the predicate (found=true), as long as it is not phrased with information-request framing.

Return found=true only when the message itself directly performs or states the predicate as a current, active fact.

Step 2 — if found=true, extract instances. Each instance is one complete predicate occurrence:
- Scan the FULL message for EVERY occurrence of the predicate. Conjunctions like "and", "and also", "as well as" often introduce additional instances — extract each one separately.
- One instance per satisfying tuple (binary) or entity (unary)
- Every required object_id must appear exactly once per instance
- Never merge two separate occurrences into one instance
- If the same entity appears in the message under different names or aliases, each distinct mention creates its own separate instance
- mention = exact substring copied from the MESSAGE TEXT — never use a span from related_object_history
- CRITICAL: Every instance must contain a non-null, non-empty mention for EVERY required object_id. If you cannot find an explicit mention for any required object, do NOT output found=true — return {"found": false} instead.
- canonical_form = normalized identity; copy exactly from related_object_history when the mention matches a history item
- canonical_source = {"type": "history", "matched_history_index": N} for history matches, {"type": "new"} otherwise

Output valid JSON only. No markdown."""


INSTANCE_RULES = """Instance rules:
- One instance per entity/pair satisfying the predicate. "A and B" → two instances, never one.
- If the same entity appears under different names in the message, each distinct mention is its own instance.
- mention = exact span from the MESSAGE TEXT, not from history.
- Canonical: match each mention to related_object_history; if found, copy canonical_form exactly and set canonical_source to {"type": "history", "matched_history_index": N}. Otherwise {"type": "new"}."""


USER_MESSAGE_PROMPT = """You are grounding a USER message.

Predicate information:
{predicate_block}

Related object history:
{related_object_history}

Few-shot examples for this predicate:
{few_shot_block}

{instance_rules}

Message role: {role}
Reminder: return {{"found": false}} unless this message actively and exactly expresses: "{predicate_description}". Closely related or similar actions do not qualify.
Message text:
{text}

Return JSON only.
If not found: {{"found": false}}
If found: {{"found": true, "instances": [...]}}"""


ASSISTANT_MESSAGE_PROMPT = """You are grounding an ASSISTANT message.

Predicate information:
{predicate_block}

Related object history:
{related_object_history}

Few-shot examples for this predicate:
{few_shot_block}

{instance_rules}

Message role: {role}
Reminder: return {{"found": false}} unless this message actively and exactly expresses: "{predicate_description}". Closely related or similar actions do not qualify.
Message text:
{text}

Return JSON only.
If not found: {{"found": false}}
If found: {{"found": true, "instances": [...]}}"""


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = _strip_json_fences(text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Expected top-level JSON object")
    return payload


def _openrouter_chat(
    api_key: str,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    request_timeout: float = REQUEST_TIMEOUT,
    max_retries: int = MAX_API_RETRIES,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(1, max_retries + 1):
        req = urllib.request.Request(
            f"{BASE_URL}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=request_timeout) as resp:
                raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            content = data["choices"][0]["message"]["content"]
            return content if isinstance(content, str) else json.dumps(content)
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            KeyError,
            IndexError,
            json.JSONDecodeError,
        ) as exc:
            if attempt == max_retries:
                raise RuntimeError(f"OpenRouter call failed after retries: {exc}") from exc
            time.sleep(1.5 * attempt)
    raise RuntimeError("unreachable")


@lru_cache(maxsize=8)
def load_few_shot_index(path: str) -> dict[str, list[dict[str, Any]]]:
    few_shot_path = Path(path)
    if not few_shot_path.exists():
        return {}
    with few_shot_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    predicates = payload.get("predicates", [])
    if not isinstance(predicates, list):
        return {}
    index: dict[str, list[dict[str, Any]]] = {}
    for item in predicates:
        if not isinstance(item, dict):
            continue
        predicate_id = item.get("predicate_id")
        examples = item.get("examples")
        if isinstance(predicate_id, str) and isinstance(examples, list):
            index[predicate_id] = [ex for ex in examples if isinstance(ex, dict)]
    return index


def _compact_output(example: dict[str, Any]) -> dict[str, Any]:
    output = {"found": bool(example.get("found"))}
    if output["found"]:
        output["instances"] = example.get("instances", [])
    return output


def _format_few_shots(examples: list[dict[str, Any]], max_examples: int = 6) -> str:
    if not examples:
        return "[]"
    def _n_instances(ex: dict[str, Any]) -> int:
        return len(ex.get("instances") or []) if ex.get("found") else 0
    examples = sorted(examples, key=_n_instances, reverse=True)
    compact = []
    for ex in examples[:max_examples]:
        compact.append(
            {
                "input": {
                    "text": ex.get("text"),
                    "role": ex.get("role"),
                },
                "output": _compact_output(ex),
            }
        )
    return json.dumps(compact, ensure_ascii=False, indent=2)


def _predicate_block(record: dict[str, Any]) -> str:
    return json.dumps(
        {
            "predicate_description": record.get("predicate_description"),
            "objects": record.get("objects", []),
        },
        ensure_ascii=False,
        indent=2,
    )


def build_messages(
    record: dict[str, Any],
    few_shot_path: str | Path,
) -> list[dict[str, str]]:
    few_shot_index = load_few_shot_index(str(few_shot_path))
    few_shots = few_shot_index.get(str(record.get("predicate_id")), [])
    template = USER_MESSAGE_PROMPT if record.get("role") == "user" else ASSISTANT_MESSAGE_PROMPT
    history = record.get("related_object_history", [])
    slim_history = [
        {"mention": item.get("mention"), "canonical_form": item.get("canonical_form")}
        for item in history
    ]
    user_prompt = template.format(
        predicate_block=_predicate_block(record),
        related_object_history=json.dumps(slim_history, ensure_ascii=False, indent=2),
        few_shot_block=_format_few_shots(few_shots),
        instance_rules=INSTANCE_RULES,
        predicate_description=record.get("predicate_description", ""),
        role=record.get("role"),
        text=record.get("text"),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def normalize_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    found = bool(prediction.get("found"))
    if not found:
        return {"found": False}

    instances = prediction.get("instances")
    if not isinstance(instances, list):
        object_mentions = prediction.get("object_mentions")
        if isinstance(object_mentions, list):
            instances = [{"instance_id": "i1", "object_mentions": object_mentions}]
        else:
            instances = []

    normalized_instances = []
    for idx, instance in enumerate(instances, start=1):
        if not isinstance(instance, dict):
            continue
        mentions = instance.get("object_mentions", [])
        if not isinstance(mentions, list):
            mentions = []
        normalized_instances.append(
            {
                "instance_id": str(instance.get("instance_id") or f"i{idx}"),
                "object_mentions": [m for m in mentions if isinstance(m, dict)],
            }
        )
    return {"found": True, "instances": normalized_instances}


def predict(
    record: dict[str, Any],
    few_shot_path: str | Path,
    model: str = MODEL_NAME,
    temperature: float = TEMPERATURE,
    request_timeout: float = REQUEST_TIMEOUT,
    max_retries: int = MAX_API_RETRIES,
) -> dict[str, Any]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    messages = build_messages(record, few_shot_path=few_shot_path)
    content = _openrouter_chat(api_key, messages, model=model, temperature=temperature,
                               request_timeout=request_timeout, max_retries=max_retries)
    return normalize_prediction(_parse_json_object(content))
