"""
Policies and predicates API router.

CRUD for predicates and policies with formula validation.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend.engine.grounding import build_grounding_prompts
from backend.models.builtins import is_builtin_proposition
from backend.models.chat import ChatMessage
from backend.models.policy import Policy, Proposition
from backend.routers.chat import invalidate_monitors
from backend.routers.settings import _load_settings
from backend.services.openrouter import OpenRouterClient, OpenRouterError
from backend.store.db import DatabaseStore

router = APIRouter(tags=["policies"])


def _get_db(request: Request) -> DatabaseStore:
    return request.app.state.db


# Request schemas


class CreatePropositionRequest(BaseModel):
    """Request body for creating a predicate.

    Attributes:
        prop_id: Predicate name (e.g., "p_fraud", "p_transfer").
        description: Canonical description for semantic grounding.
        role: Which message role this applies to ("user" or "assistant").
        arity: Number of arguments (0 = Boolean, >0 = first-order with data).
        arg_descriptions: Description for each argument (length should match arity).
    """

    prop_id: str
    description: str
    role: str  # "user" | "assistant"
    arity: int = 0
    arg_descriptions: list[str] = []


class UpdatePropositionRequest(BaseModel):
    """Request body for updating a predicate."""

    description: str | None = None
    role: str | None = None
    arity: int | None = None
    arg_descriptions: list[str] | None = None


class GroundingPromptPreview(BaseModel):
    """Rendered grounding prompt preview for a predicate."""

    prop_id: str
    role: str
    system_prompt: str
    user_prompt: str


class CreatePolicyRequest(BaseModel):
    """Request body for creating a policy."""

    name: str
    formula_str: str
    enabled: bool = True


class UpdatePolicyRequest(BaseModel):
    """Request body for updating a policy."""

    name: str | None = None
    formula_str: str | None = None
    enabled: bool | None = None


# DejaVu reserved words — these are NOT predicate IDs
_DEJAVU_KEYWORDS = frozenset({
    "true", "false", "Forall", "Exists", "forall", "exists",
    "H", "P", "S", "Z", "where", "pred", "prop",
})


def _extract_rule_definitions(formula_str: str) -> tuple[set[str], set[str]]:
    """Return (rule_names, rule_parameters) from a where clause.

    DejaVu lets a property declare local rules with:
        prop p : <body> where r(args) := <rule-body>, s(args) := <rule-body>
    Both rule names and their formal parameters are identifiers that should
    not be looked up as predicates in the database.
    """
    cleaned = re.sub(r'"[^"]*"', '', formula_str)
    cleaned = re.sub(r"'[^']*'", '', cleaned)
    where_match = re.search(r"\bwhere\b", cleaned)
    if not where_match:
        return set(), set()
    where_body = cleaned[where_match.end():]
    rule_names: set[str] = set()
    rule_params: set[str] = set()
    for match in re.finditer(
        r"([A-Za-z_]\w*)\s*(?:\(([^)]*)\))?\s*:=", where_body
    ):
        rule_names.add(match.group(1))
        params = match.group(2)
        if params:
            for param in params.split(","):
                param = param.strip()
                if param and re.fullmatch(r"[A-Za-z_]\w*", param):
                    rule_params.add(param)
    return rule_names, rule_params


def _extract_rule_names(formula_str: str) -> set[str]:
    """Backwards-compatible accessor returning only rule names."""
    return _extract_rule_definitions(formula_str)[0]


def _extract_identifiers(formula_str: str) -> set[str]:
    """Extract candidate predicate IDs from a formula string.

    Used to build pred declarations for DejaVu validation.
    DejaVu's parser performs the actual syntax validation — this only
    extracts identifiers that might be predicates. Rule names defined
    in a trailing `where` clause are excluded.
    """
    cleaned = re.sub(r'"[^"]*"', '', formula_str)
    cleaned = re.sub(r"'[^']*'", '', cleaned)
    quant_vars = set()
    for match in re.finditer(r'(?:Forall|Exists|forall|exists)\s+(\w+)', cleaned):
        quant_vars.add(match.group(1))
    all_ids = set(re.findall(r'\b([a-zA-Z_]\w*)\b', cleaned))
    rule_names, rule_params = _extract_rule_definitions(formula_str)
    return all_ids - _DEJAVU_KEYWORDS - quant_vars - rule_names - rule_params


def _split_formula_args(args_str: str) -> list[str]:
    """Split predicate call arguments while preserving quoted commas."""
    args: list[str] = []
    current: list[str] = []
    quote_char: str | None = None
    escape_next = False

    for ch in args_str:
        if escape_next:
            current.append(ch)
            escape_next = False
            continue
        if ch == "\\" and quote_char:
            current.append(ch)
            escape_next = True
            continue
        if quote_char:
            current.append(ch)
            if ch == quote_char:
                quote_char = None
            continue
        if ch in ("'", '"'):
            current.append(ch)
            quote_char = ch
            continue
        if ch == ",":
            args.append("".join(current).strip())
            current = []
            continue
        current.append(ch)

    tail = "".join(current).strip()
    if tail:
        args.append(tail)
    return args


def _extract_formula_calls(formula_str: str) -> list[tuple[str, list[str]]]:
    """Extract simple predicate calls and their raw argument strings."""
    calls: list[tuple[str, list[str]]] = []
    for match in re.finditer(r"\b([A-Za-z_]\w*)\s*\(([^()]*)\)", formula_str):
        call_name = match.group(1)
        if call_name in _DEJAVU_KEYWORDS:
            continue
        calls.append((call_name, _split_formula_args(match.group(2))))
    return calls


def _is_relation_variable(token: str) -> bool:
    """Return True for unquoted identifier arguments used as policy variables."""
    raw = (token or "").strip()
    if not raw:
        return False
    if raw[0] in ("'", '"') or raw[-1:] in ("'", '"'):
        return False
    if raw in _DEJAVU_KEYWORDS:
        return False
    return bool(re.fullmatch(r"[A-Za-z_]\w*", raw))


def _strip_formula_string_literals(formula_str: str) -> str:
    """Remove quoted literals so comparison extraction only sees variables."""
    cleaned = re.sub(r'"(?:\\.|[^"\\])*"', "", formula_str)
    return re.sub(r"'(?:\\.|[^'\\])*'", "", cleaned)


def _find_variable_comparisons(formula_str: str) -> list[tuple[str, str]]:
    """Extract variable pairs compared with equality/ordering operators."""
    cleaned = _strip_formula_string_literals(formula_str)
    comparisons: list[tuple[str, str]] = []
    comparison_re = re.compile(
        r"\b([A-Za-z_]\w*)\b\s*(?:<=|>=|!=|==|=|<|>)\s*\b([A-Za-z_]\w*)\b"
    )
    for match in comparison_re.finditer(cleaned):
        left = match.group(1)
        right = match.group(2)
        if _is_relation_variable(left) and _is_relation_variable(right):
            comparisons.append((left, right))
    return comparisons


class _VariableUnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        if item not in self._parent:
            self._parent[item] = item
        if self._parent[item] != item:
            self._parent[item] = self.find(self._parent[item])
        return self._parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self._parent[right_root] = left_root


async def _extract_related_object_relations(
    db: DatabaseStore,
    formula_str: str,
) -> list[dict]:
    """Build directed related-object edges implied by variables and comparisons."""
    positions_by_variable: dict[str, list[tuple[str, str]]] = {}
    variables = _VariableUnionFind()

    for prop_id, args in _extract_formula_calls(formula_str):
        if is_builtin_proposition(prop_id):
            continue
        prop = await db.get_proposition(prop_id)
        if not prop:
            continue

        arity = prop.get("arity", 0) or 0
        for idx, arg in enumerate(args[:arity]):
            if not _is_relation_variable(arg):
                continue
            variable = arg.strip()
            variables.find(variable)
            positions_by_variable.setdefault(variable, []).append(
                (prop_id, f"o{idx + 1}")
            )

    for left, right in _find_variable_comparisons(formula_str):
        variables.union(left, right)

    positions_by_relation_group: dict[str, list[tuple[str, str]]] = {}
    for variable, positions in positions_by_variable.items():
        positions_by_relation_group.setdefault(variables.find(variable), []).extend(
            positions
        )

    relations: list[dict] = []
    seen: set[tuple[str, str, str, str]] = set()
    for positions in positions_by_relation_group.values():
        unique_positions = list(dict.fromkeys(positions))
        if len(unique_positions) < 2:
            continue
        for prop_id, object_id in unique_positions:
            for related_prop_id, related_object_id in unique_positions:
                key = (prop_id, object_id, related_prop_id, related_object_id)
                if key in seen or (prop_id, object_id) == (
                    related_prop_id,
                    related_object_id,
                ):
                    continue
                seen.add(key)
                relations.append({
                    "prop_id": prop_id,
                    "object_id": object_id,
                    "related_prop_id": related_prop_id,
                    "related_object_id": related_object_id,
                })

    return relations


def _parse_json_list_field(raw_value) -> list[str]:
    if not raw_value:
        return []
    try:
        parsed = json.loads(raw_value)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if str(x).strip()]
    except (TypeError, json.JSONDecodeError):
        return []
    return []


def _parse_json_object_list_field(raw_value) -> list[dict[str, Any]]:
    if not raw_value:
        return []
    try:
        parsed = json.loads(raw_value)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    except (TypeError, json.JSONDecodeError):
        return []
    return []


def _row_to_proposition(row: dict) -> Proposition:
    return Proposition(
        prop_id=row["prop_id"],
        description=row["description"],
        role=row["role"],
        arity=row.get("arity", 0) or 0,
        arg_descriptions=_parse_json_list_field(row.get("arg_descriptions")),
        few_shot_positive=_parse_json_list_field(row.get("few_shot_positive")),
        few_shot_negative=_parse_json_list_field(row.get("few_shot_negative")),
        few_shot_examples=_parse_json_object_list_field(row.get("few_shot_examples")),
        few_shot_generated_at=row.get("few_shot_generated_at"),
    )


def _extract_json_object(text: str) -> dict | None:
    t = (text or "").strip()
    if not t:
        return None

    if t.startswith("```"):
        lines = t.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        t = "\n".join(lines).strip()

    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict):
        return obj

    match = re.search(r"\{[\s\S]*\}", t)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _predicate_objects(
    arity: int,
    arg_descriptions: list[str],
) -> list[dict[str, str]]:
    return [
        {
            "object_id": f"o{idx + 1}",
            "description": (
                arg_descriptions[idx]
                if idx < len(arg_descriptions) and arg_descriptions[idx].strip()
                else f"argument {idx + 1}"
            ),
        }
        for idx in range(arity)
    ]


def _validate_few_shot_example(
    example: dict[str, Any],
    role: str,
    object_ids: list[str],
) -> tuple[bool, str]:
    if not isinstance(example.get("text"), str) or not example["text"].strip():
        return False, "missing text"
    if example.get("role") != role:
        return False, "wrong role"
    if not isinstance(example.get("related_object_context", []), list):
        return False, "bad related_object_context"
    history = example.get("related_object_history", [])
    if not isinstance(history, list):
        return False, "bad related_object_history"
    if not isinstance(example.get("found"), bool):
        return False, "missing found"
    if not example["found"]:
        if "instances" in example:
            return False, "negative examples must omit instances"
        return True, "ok"

    instances = example.get("instances")
    if not isinstance(instances, list) or (object_ids and not instances):
        return False, "positive examples need instances"
    for instance in instances:
        if not isinstance(instance, dict) or not isinstance(
            instance.get("object_mentions"), list
        ):
            return False, "bad instance"
        mentions = instance["object_mentions"]
        if sorted(str(m.get("object_id")) for m in mentions if isinstance(m, dict)) != sorted(object_ids):
            return False, "instance object ids do not match predicate"
        for mention in mentions:
            if not isinstance(mention, dict):
                return False, "bad object mention"
            span = mention.get("mention")
            canonical_form = mention.get("canonical_form")
            source = mention.get("canonical_source")
            if not isinstance(span, str) or not span or span not in example["text"]:
                return False, "mention is not an exact text span"
            if not isinstance(canonical_form, str) or not canonical_form.strip():
                return False, "missing canonical form"
            if not isinstance(source, dict) or source.get("type") not in ("new", "history"):
                return False, "bad canonical source"
            if source["type"] == "history":
                history_index = source.get("matched_history_index")
                if not isinstance(history_index, int) or not (0 <= history_index < len(history)):
                    return False, "bad history index"
                history_item = history[history_index]
                if not isinstance(history_item, dict) or history_item.get("canonical_form") != canonical_form:
                    return False, "history canonical form mismatch"
    return True, "ok"


def _parse_few_shot_examples(
    raw_response: str,
    role: str,
    object_ids: list[str],
) -> list[dict[str, Any]]:
    obj = _extract_json_object(raw_response)
    if not obj:
        raise ValueError("Could not parse JSON from chat model response")

    examples = obj.get("examples", [])
    if not isinstance(examples, list):
        raise ValueError("Missing examples array")
    valid: list[dict[str, Any]] = []
    for idx, example in enumerate(examples, start=1):
        if not isinstance(example, dict):
            raise ValueError(f"Example {idx} is not an object")
        is_valid, reason = _validate_few_shot_example(example, role, object_ids)
        if not is_valid:
            raise ValueError(f"Example {idx} invalid: {reason}")
        valid.append(example)
    positives = [example for example in valid if example["found"]]
    negatives = [example for example in valid if not example["found"]]
    if len(positives) < 3 or len(negatives) < 3:
        raise ValueError("Need three valid positive and three valid negative examples")
    return positives[:3] + negatives[:3]


def _few_shot_generation_prompt(
    prop_id: str,
    prop_description: str,
    role: str,
    objects: list[dict[str, str]],
) -> str:
    return f"""Generate six structured few-shot examples for the extended grounding task.

Predicate:
{json.dumps({"predicate_id": prop_id, "predicate_description": prop_description, "predicate_role": role, "objects": objects}, indent=2)}

Generate exactly three positive and three negative examples.
- Every example must use role "{role}".
- Positive examples must directly express this predicate and include found=true plus an instances array.
- For positive examples, each instance is one complete predicate occurrence and includes every required object exactly once.
- Mentions are exact verbatim substrings of text.
- canonical_form is a stable normalized value and canonical_source is either {{"type": "new"}} or {{"type": "history", "matched_history_index": N}}.
- Include a multi-instance positive example when it is semantically plausible.
- Include at least one positive example with a plausible related_object_history reuse when objects are present: provide related_object_context and history, reuse the history canonical_form exactly, and use canonical_source type "history".
- Negative examples must be challenging near-misses using similar domain vocabulary, with found=false and no instances field.
- related_object_context and related_object_history are arrays; use [] when not needed.

Return only this JSON form:
{{
  "examples": [
    {{
      "text": "...",
      "role": "{role}",
      "related_object_context": [],
      "related_object_history": [],
      "found": true,
      "instances": [
        {{
          "instance_id": "i1",
          "object_mentions": [
            {{
              "object_id": "o1",
              "mention": "exact substring from text",
              "canonical_source": {{"type": "new"}},
              "canonical_form": "normalized value",
            }}
          ]
        }}
      ]
    }},
    {{
      "text": "...",
      "role": "{role}",
      "related_object_context": [],
      "related_object_history": [],
      "found": false
    }}
  ]
}}"""


async def _generate_few_shots_with_chat_model(
    openrouter_api_key: str,
    chat_model: str,
    proposition_id: str,
    proposition_description: str,
    role: str,
    objects: list[dict[str, str]],
    retries: int = 3,
) -> list[dict[str, Any]]:
    if not openrouter_api_key:
        raise HTTPException(
            400,
            "OpenRouter API key not configured. Configure Chat Model in Settings before adding predicates.",
        )

    client = OpenRouterClient(api_key=openrouter_api_key, model=chat_model)
    system_prompt = (
        "You generate synthetic few-shot examples for proposition matching. "
        "Return ONLY valid JSON."
    )
    user_prompt = _few_shot_generation_prompt(
        proposition_id, proposition_description, role, objects
    )
    object_ids = [obj["object_id"] for obj in objects]

    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            raw = await client.chat(
                [
                    ChatMessage(role="system", content=system_prompt),
                    ChatMessage(role="user", content=user_prompt),
                ]
            )
            return _parse_few_shot_examples(raw, role, object_ids)
        except (OpenRouterError, ValueError) as e:
            last_error = str(e)
            if attempt < retries:
                continue

    raise HTTPException(
        502,
        f"Failed to generate few-shot examples using chat model: {last_error}",
    )


async def _validate_formula(db: DatabaseStore, formula_str: str) -> tuple[list[str], str | None]:
    """Validate a DejaVu formula using the DejaVu server's parser.

    Sends the formula to DejaVu's POST /validate endpoint for server-side
    parsing. DejaVu performs full syntax and wellformedness checking using
    its own parser. No local regex-based validation fallback.

    Returns (prop_ids, error_or_none).
    """
    from backend.config import get_config
    from backend.engine.dejavu_client import DejaVuClient, DejaVuError

    formula_str = formula_str.strip()
    if not formula_str:
        return [], "Formula cannot be empty"

    # Basic sanity checks before sending to DejaVu
    if len(formula_str) > 2000:
        return [], "Formula too long (max 2000 characters)"

    # Extract identifiers to find which predicates are used in the formula
    candidate_ids = _extract_identifiers(formula_str)

    # Look up defined predicates from DB to get their arity for pred declarations.
    # This lets DejaVu validate that argument counts match.
    pred_lines = []
    for pid in sorted(candidate_ids):
        if is_builtin_proposition(pid):
            pred_lines.append(f"pred {pid}")
            continue
        prop_row = await db.get_proposition(pid)
        if prop_row:
            arity = prop_row.get("arity", 0) or 0
            if arity > 0:
                args = ", ".join(f"a{i+1}" for i in range(arity))
                pred_lines.append(f"pred {pid}({args})")
            else:
                pred_lines.append(f"pred {pid}")

    spec_parts = [*pred_lines, f"prop _validate : {formula_str}"]
    spec = "\n".join(spec_parts)

    config = get_config()
    dejavu_url = config.dejavu_url
    client = DejaVuClient(base_url=dejavu_url)
    try:
        valid, _properties, error = await client.validate_spec(spec)
        if not valid:
            return sorted(candidate_ids), error
    except DejaVuError as e:
        return [], f"DejaVu unavailable: {e}"
    except Exception as e:
        return [], f"Validation failed: {type(e).__name__}: {e}"
    finally:
        await client.close()

    # Extract prop IDs from the validated formula
    prop_ids = sorted(_extract_identifiers(formula_str))

    # Check all referenced predicates exist in the database
    missing = []
    for pid in prop_ids:
        if is_builtin_proposition(pid):
            continue
        prop = await db.get_proposition(pid)
        if prop is None:
            missing.append(pid)

    if missing:
        return prop_ids, f"Unknown predicates: {', '.join(missing)}"

    return prop_ids, None


# Predicates endpoints


@router.get("/propositions")
async def list_propositions(request: Request) -> list[Proposition]:
    """List all predicates."""
    db = _get_db(request)
    rows = await db.list_propositions()
    return [_row_to_proposition(r) for r in rows]


class CreatePropositionResponse(BaseModel):
    """Response for predicate creation, includes optional warning."""
    proposition: Proposition
    warning: str | None = None


@router.post("/propositions", status_code=201)
async def create_proposition(request: Request, body: CreatePropositionRequest) -> CreatePropositionResponse:
    """Create a new predicate."""
    db = _get_db(request)

    if not body.prop_id or not body.prop_id.strip():
        raise HTTPException(422, "Predicate ID cannot be empty.")

    if not body.description or not body.description.strip():
        raise HTTPException(422, "Predicate description cannot be empty.")

    if body.role not in ("user", "assistant"):
        raise HTTPException(422, f"Invalid role: {body.role}. Must be 'user' or 'assistant'.")

    existing = await db.get_proposition(body.prop_id)
    if existing:
        raise HTTPException(409, f"Predicate '{body.prop_id}' already exists.")

    objects = _predicate_objects(body.arity, body.arg_descriptions)
    few_shot_examples: list[dict[str, Any]] = []
    warning: str | None = None
    settings = await _load_settings(db)

    # Determine which model to use for few-shot generation
    use_grounding = (settings.few_shot_model == "grounding")

    if use_grounding:
        # Use the grounding model (local or OpenRouter)
        grounding_settings = settings.grounding
        if grounding_settings.provider == "openrouter":
            api_key = grounding_settings.api_key or settings.openrouter_api_key
            model = grounding_settings.model
            if api_key:
                try:
                    few_shot_examples = await _generate_few_shots_with_chat_model(
                        openrouter_api_key=api_key,
                        chat_model=model,
                        proposition_id=body.prop_id,
                        proposition_description=body.description,
                        role=body.role,
                        objects=objects,
                    )
                except HTTPException as e:
                    warning = (
                        f"Few-shot generation failed with grounding model ({e.detail}). "
                        "Predicate saved with zero-shot mode."
                    )
            else:
                warning = (
                    "No API key configured for grounding model. "
                    "Predicate saved with zero-shot mode."
                )
        else:
            # Local grounding model (Ollama, LM Studio, etc.) — use via grounding client
            from backend.services.grounding_client import create_grounding_client
            try:
                grounding_client = create_grounding_client(
                    provider=grounding_settings.provider,
                    base_url=grounding_settings.base_url,
                    model=grounding_settings.model,
                )
                system_prompt = (
                    "You generate synthetic few-shot examples for predicate matching. "
                    "Return ONLY valid JSON."
                )
                user_prompt = _few_shot_generation_prompt(
                    body.prop_id, body.description, body.role, objects
                )
                raw = await grounding_client.chat(system_prompt, user_prompt)
                few_shot_examples = _parse_few_shot_examples(
                    raw, body.role, [obj["object_id"] for obj in objects]
                )
            except Exception as e:
                warning = (
                    f"Few-shot generation failed with local grounding model ({e}). "
                    "Predicate saved with zero-shot mode. "
                    "Ensure your grounding LLM server is running."
                )
    else:
        # Use the chat model (OpenRouter)
        effective_chat_model = settings.openrouter_model_custom or settings.openrouter_model
        if settings.openrouter_api_key:
            try:
                few_shot_examples = await _generate_few_shots_with_chat_model(
                    openrouter_api_key=settings.openrouter_api_key,
                    chat_model=effective_chat_model,
                    proposition_id=body.prop_id,
                    proposition_description=body.description,
                    role=body.role,
                    objects=objects,
                )
            except HTTPException as e:
                warning = (
                    f"Few-shot generation failed with chat model ({e.detail}). "
                    "Predicate saved with zero-shot mode."
                )
        else:
            warning = (
                "No OpenRouter API key configured — predicate saved with zero-shot mode. "
                "To generate few-shot examples, add your API key in Settings."
            )

    generated_at = datetime.now(UTC).isoformat()
    await db.create_proposition(
        body.prop_id,
        body.description,
        body.role,
        arity=body.arity,
        arg_descriptions=body.arg_descriptions if body.arg_descriptions else None,
        few_shot_examples=few_shot_examples,
        few_shot_generated_at=generated_at,
    )
    invalidate_monitors()
    created = await db.get_proposition(body.prop_id)
    return CreatePropositionResponse(
        proposition=_row_to_proposition(created),
        warning=warning,
    )


@router.put("/propositions/{prop_id}")
async def update_proposition(
    request: Request, prop_id: str, body: UpdatePropositionRequest
) -> Proposition:
    """Update an existing predicate."""
    db = _get_db(request)
    existing = await db.get_proposition(prop_id)
    if not existing:
        raise HTTPException(404, f"Predicate '{prop_id}' not found.")

    if body.role is not None and body.role not in ("user", "assistant"):
        raise HTTPException(422, f"Invalid role: {body.role}. Must be 'user' or 'assistant'.")

    await db.update_proposition(
        prop_id,
        description=body.description,
        role=body.role,
        arg_descriptions=body.arg_descriptions,
    )
    invalidate_monitors()
    updated = await db.get_proposition(prop_id)
    return _row_to_proposition(updated)


@router.get("/propositions/{prop_id}/grounding-prompt")
async def proposition_grounding_prompt(
    request: Request,
    prop_id: str,
    message_text: str | None = None,
) -> GroundingPromptPreview:
    """Render the full grounding prompt for a predicate."""
    db = _get_db(request)
    row = await db.get_proposition(prop_id)
    if not row:
        raise HTTPException(404, f"Predicate '{prop_id}' not found.")

    proposition = _row_to_proposition(row)
    settings = await _load_settings(db)
    preview_message = (
        message_text if message_text is not None else "<MESSAGE_TEXT_GOES_HERE>"
    )
    system_prompt, user_prompt = build_grounding_prompts(
        proposition=proposition,
        message_role=proposition.role,
        message_text=preview_message,
        system_prompt=settings.grounding.system_prompt,
        user_prompt_template_user=settings.grounding.user_prompt_template_user,
        user_prompt_template_assistant=settings.grounding.user_prompt_template_assistant,
        related_object_context_block="{{RELATED_OBJECT_CONTEXT_BLOCK}}",
        related_object_history_block="{{RELATED_OBJECT_HISTORY_BLOCK}}",
    )
    return GroundingPromptPreview(
        prop_id=proposition.prop_id,
        role=proposition.role,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


@router.delete("/propositions/{prop_id}", status_code=204)
async def delete_proposition(request: Request, prop_id: str):
    """Delete a predicate. Rejects if referenced by any policy."""
    db = _get_db(request)
    existing = await db.get_proposition(prop_id)
    if not existing:
        raise HTTPException(404, f"Predicate '{prop_id}' not found.")

    # Check for referencing policies
    referencing = await db.get_policies_using_proposition(prop_id)
    if referencing:
        names = [r["name"] for r in referencing]
        raise HTTPException(
            409,
            f"Cannot delete predicate '{prop_id}': referenced by "
            f"policies: {', '.join(names)}. Remove it from those policies first.",
        )

    await db.delete_proposition(prop_id)
    invalidate_monitors()


# Policies endpoints


@router.get("/policies")
async def list_policies(request: Request) -> list[Policy]:
    """List all policies with their predicate references."""
    db = _get_db(request)
    rows = await db.list_policies()
    result = []
    for r in rows:
        try:
            props = sorted(_extract_identifiers(r["formula_str"]))
        except Exception:
            # Defensive fallback for malformed persisted formula rows.
            props = await db.get_policy_propositions(r["policy_id"])
        result.append(
            Policy(
                policy_id=r["policy_id"],
                name=r["name"],
                formula_str=r["formula_str"],
                propositions=props,
                enabled=bool(r["enabled"]),
            )
        )
    return result


MAX_FORMULA_LENGTH = 1000
MAX_POLICY_COUNT = 50


@router.post("/policies", status_code=201)
async def create_policy(request: Request, body: CreatePolicyRequest) -> Policy:
    """Create a new policy. Validates the temporal logic formula and predicate references."""
    db = _get_db(request)

    # Validate name is not empty
    if not body.name or not body.name.strip():
        raise HTTPException(422, "Policy name cannot be empty.")

    # Validate formula is not empty
    if not body.formula_str or not body.formula_str.strip():
        raise HTTPException(422, "Formula cannot be empty.")

    # Validate formula size
    if len(body.formula_str) > MAX_FORMULA_LENGTH:
        raise HTTPException(422, f"Formula too long. Maximum {MAX_FORMULA_LENGTH} characters.")

    # Validate policy count limit
    existing_policies = await db.list_policies()
    if len(existing_policies) >= MAX_POLICY_COUNT:
        raise HTTPException(422, f"Maximum of {MAX_POLICY_COUNT} policies reached.")

    prop_ids, error = await _validate_formula(db, body.formula_str)
    if error:
        raise HTTPException(422, error)

    policy_id = str(uuid.uuid4())
    await db.create_policy(policy_id, body.name, body.formula_str, body.enabled)
    await db.set_policy_propositions(
        policy_id,
        [pid for pid in prop_ids if not is_builtin_proposition(pid)],
    )
    await db.set_policy_related_objects(
        policy_id,
        await _extract_related_object_relations(db, body.formula_str),
    )
    invalidate_monitors()

    return Policy(
        policy_id=policy_id,
        name=body.name,
        formula_str=body.formula_str,
        propositions=prop_ids,
        enabled=body.enabled,
    )


@router.put("/policies/{policy_id}")
async def update_policy(request: Request, policy_id: str, body: UpdatePolicyRequest) -> Policy:
    """Update an existing policy. Re-validates formula if changed."""
    db = _get_db(request)
    existing = await db.get_policy(policy_id)
    if not existing:
        raise HTTPException(404, f"Policy '{policy_id}' not found.")

    # If formula changed, re-validate
    if body.formula_str is not None:
        if len(body.formula_str) > MAX_FORMULA_LENGTH:
            raise HTTPException(422, f"Formula too long. Maximum {MAX_FORMULA_LENGTH} characters.")
        prop_ids, error = await _validate_formula(db, body.formula_str)
        if error:
            raise HTTPException(422, error)
        await db.set_policy_propositions(
            policy_id,
            [pid for pid in prop_ids if not is_builtin_proposition(pid)],
        )
        await db.set_policy_related_objects(
            policy_id,
            await _extract_related_object_relations(db, body.formula_str),
        )

    await db.update_policy(
        policy_id,
        name=body.name,
        formula_str=body.formula_str,
        enabled=body.enabled,
    )
    invalidate_monitors()

    updated = await db.get_policy(policy_id)
    props = sorted(_extract_identifiers(updated["formula_str"]))
    return Policy(
        policy_id=updated["policy_id"],
        name=updated["name"],
        formula_str=updated["formula_str"],
        propositions=props,
        enabled=bool(updated["enabled"]),
    )


@router.delete("/policies/{policy_id}", status_code=204)
async def delete_policy(request: Request, policy_id: str):
    """Delete a policy."""
    db = _get_db(request)
    existing = await db.get_policy(policy_id)
    if not existing:
        raise HTTPException(404, f"Policy '{policy_id}' not found.")
    await db.delete_policy(policy_id)
    invalidate_monitors()


@router.post("/policies/validate")
async def validate_formula(request: Request, body: CreatePolicyRequest):
    """Validate a temporal logic formula without creating a policy."""
    db = _get_db(request)
    prop_ids, error = await _validate_formula(db, body.formula_str)
    if error:
        return {"valid": False, "error": error, "propositions": prop_ids}
    return {"valid": True, "error": None, "propositions": prop_ids}
