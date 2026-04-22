"""
Policies and predicates API router.

CRUD for predicates and policies with formula validation.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend.engine.grounding import build_grounding_prompts
from backend.models.chat import ChatMessage
from backend.models.builtins import is_builtin_proposition
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
    """

    prop_id: str
    description: str
    role: str  # "user" | "assistant"
    arity: int = 0


class UpdatePropositionRequest(BaseModel):
    """Request body for updating a predicate."""

    description: str | None = None
    role: str | None = None
    arity: int | None = None


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


def _extract_identifiers(formula_str: str) -> set[str]:
    """Extract candidate predicate IDs from a formula string.

    Used to build pred declarations for DejaVu validation.
    DejaVu's parser performs the actual syntax validation — this only
    extracts identifiers that might be predicates.
    """
    cleaned = re.sub(r'"[^"]*"', '', formula_str)
    cleaned = re.sub(r"'[^']*'", '', cleaned)
    quant_vars = set()
    for match in re.finditer(r'(?:Forall|Exists|forall|exists)\s+(\w+)', cleaned):
        quant_vars.add(match.group(1))
    all_ids = set(re.findall(r'\b([a-zA-Z_]\w*)\b', cleaned))
    return all_ids - _DEJAVU_KEYWORDS - quant_vars


def _parse_json_list_field(raw_value) -> list[str]:
    if not raw_value:
        return []
    try:
        parsed = json.loads(raw_value)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if str(x).strip()]
    except Exception:
        pass
    return []


def _row_to_proposition(row: dict) -> Proposition:
    return Proposition(
        prop_id=row["prop_id"],
        description=row["description"],
        role=row["role"],
        arity=row.get("arity", 0) or 0,
        few_shot_positive=_parse_json_list_field(row.get("few_shot_positive")),
        few_shot_negative=_parse_json_list_field(row.get("few_shot_negative")),
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
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

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


def _parse_few_shot_examples(raw_response: str) -> tuple[list[str], list[str]]:
    obj = _extract_json_object(raw_response)
    if not obj:
        raise ValueError("Could not parse JSON from chat model response")

    pos = obj.get("positive_examples", [])
    neg = obj.get("negative_examples", [])
    if not isinstance(pos, list) or not isinstance(neg, list):
        raise ValueError("Missing positive_examples/negative_examples arrays")

    positives = [str(x).strip() for x in pos if str(x).strip()]
    negatives = [str(x).strip() for x in neg if str(x).strip()]
    if len(positives) < 5 or len(negatives) < 5:
        raise ValueError("Need at least 5 positive and 5 tricky negative examples")
    return positives[:5], negatives[:5]


def _few_shot_generation_prompt(prop_description: str, role: str) -> str:
    role_norm = (role or "user").strip().lower()
    role_desc = "user messages" if role_norm == "user" else "assistant messages"
    return (
        "Create few-shot examples for proposition classification.\n\n"
        'PROPOSITION: "{}"\n'
        "ROLE: {}\n\n"
        "Generate exactly:\n"
        "1) 5 positive examples where proposition is clearly true.\n"
        "2) 5 negative examples that are tricky: same domain/terms, but proposition is false.\n\n"
        "Examples must be realistic {} and 1-2 sentences.\n\n"
        "Return JSON exactly:\n"
        "{{\n"
        '  "positive_examples": ["...", "...", "...", "...", "..."],\n'
        '  "negative_examples": ["...", "...", "...", "...", "..."]\n'
        "}}"
    ).format(prop_description, role_norm, role_desc)


async def _generate_few_shots_with_chat_model(
    openrouter_api_key: str,
    chat_model: str,
    proposition_description: str,
    role: str,
    retries: int = 3,
) -> tuple[list[str], list[str]]:
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
    user_prompt = _few_shot_generation_prompt(proposition_description, role)

    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            raw = await client.chat(
                [
                    ChatMessage(role="system", content=system_prompt),
                    ChatMessage(role="user", content=user_prompt),
                ]
            )
            return _parse_few_shot_examples(raw)
        except (OpenRouterError, ValueError) as e:
            last_error = str(e)
            if attempt < retries:
                continue

    raise HTTPException(
        502,
        "Failed to generate few-shot examples using chat model: {}".format(last_error),
    )


async def _validate_formula(db: DatabaseStore, formula_str: str) -> tuple[list[str], str | None]:
    """Validate a DejaVu formula using the DejaVu server's parser.

    Sends the formula to DejaVu's POST /validate endpoint for server-side
    parsing. DejaVu performs full syntax and wellformedness checking using
    its own parser. No local regex-based validation fallback.

    Returns (prop_ids, error_or_none).
    """
    from backend.engine.dejavu_client import DejaVuClient, DejaVuError
    from backend.config import get_config

    formula_str = formula_str.strip()
    if not formula_str:
        return [], "Formula cannot be empty"

    # Sanity checks before sending to DejaVu (prevent crashes from adversarial input)
    if len(formula_str) > 2000:
        return [], "Formula too long (max 2000 characters)"
    if formula_str.count('(') != formula_str.count(')'):
        return [], "Unbalanced parentheses"
    if formula_str.count('[') != formula_str.count(']'):
        return [], "Unbalanced brackets"

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

    spec_parts = pred_lines + [f"prop _validate : {formula_str}"]
    spec = "\n".join(spec_parts)

    config = get_config()
    dejavu_url = config.dejavu_url
    client = DejaVuClient(base_url=dejavu_url)
    try:
        valid, properties, error = await client.validate_spec(spec)
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

    few_shot_positive: list[str] = []
    few_shot_negative: list[str] = []
    warning: str | None = None
    settings = await _load_settings(db)
    effective_chat_model = settings.openrouter_model_custom or settings.openrouter_model
    if settings.openrouter_api_key:
        try:
            few_shot_positive, few_shot_negative = await _generate_few_shots_with_chat_model(
                openrouter_api_key=settings.openrouter_api_key,
                chat_model=effective_chat_model,
                proposition_description=body.description,
                role=body.role,
            )
        except HTTPException as e:
            warning = (
                f"Few-shot example generation failed ({e.detail}). "
                "Predicate saved with zero-shot mode — grounding will use the description only. "
                "To enable few-shot examples, check your OpenRouter API key and chat model in Settings."
            )
    else:
        warning = (
            "No OpenRouter API key configured — predicate saved with zero-shot mode. "
            "To generate few-shot examples, add your API key in Settings → Chat Model."
        )

    generated_at = datetime.now(timezone.utc).isoformat()
    await db.create_proposition(
        body.prop_id,
        body.description,
        body.role,
        arity=body.arity,
        few_shot_positive=few_shot_positive,
        few_shot_negative=few_shot_negative,
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

    await db.update_proposition(prop_id, description=body.description, role=body.role)
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
