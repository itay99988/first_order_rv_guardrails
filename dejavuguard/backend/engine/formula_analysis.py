"""Static analysis of DejaVu policy formulas.

Pure, dependency-free helpers shared by the policies router (which derives
related-object relations when a policy is saved) and the monitor (which needs
the same information at event-build time).

Lives under ``engine`` rather than ``routers`` so the monitor can use it
without importing router modules, which would create an import cycle.
"""

from __future__ import annotations

import re

# DejaVu reserved words -- these are not predicate IDs or policy variables.
DEJAVU_KEYWORDS = frozenset({
    "true", "false", "Forall", "Exists", "forall", "exists",
    "H", "P", "S", "Z", "where", "pred", "prop",
})

# Operators that force their operands to be numbers. Equality (`=`, `!=`)
# relates two slots without constraining their type, so it is excluded.
ORDERING_OPERATORS = ("<=", ">=", "<", ">")

_ORDERING_RE = re.compile(
    r"\b([A-Za-z_]\w*)\b\s*(<=|>=|<|>)\s*\b([A-Za-z_]\w*)\b"
)

_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(([^()]*)\)")


def strip_string_literals(formula_str: str) -> str:
    """Remove quoted literals so operators inside them are not parsed."""
    cleaned = re.sub(r'"(?:\\.|[^"\\])*"', "", formula_str)
    return re.sub(r"'(?:\\.|[^'\\])*'", "", cleaned)


def is_relation_variable(token: str) -> bool:
    """Return True for unquoted identifier arguments used as policy variables."""
    raw = (token or "").strip()
    if not raw:
        return False
    if raw[0] in ("'", '"') or raw[-1:] in ("'", '"'):
        return False
    if raw in DEJAVU_KEYWORDS:
        return False
    return bool(re.fullmatch(r"[A-Za-z_]\w*", raw))


def split_formula_args(args_str: str) -> list[str]:
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


def extract_formula_calls(formula_str: str) -> list[tuple[str, list[str]]]:
    """Extract simple predicate calls and their raw argument strings."""
    calls: list[tuple[str, list[str]]] = []
    for match in _CALL_RE.finditer(formula_str):
        call_name = match.group(1)
        if call_name in DEJAVU_KEYWORDS:
            continue
        calls.append((call_name, split_formula_args(match.group(2))))
    return calls


def find_ordering_comparisons(formula_str: str) -> list[tuple[str, str]]:
    """Return variable pairs compared with an ordering operator.

    Only `<`, `<=`, `>`, `>=` qualify. These are the comparisons DejaVu
    evaluates numerically, so both operands must carry numeric values.
    """
    cleaned = strip_string_literals(formula_str)
    pairs: list[tuple[str, str]] = []
    for match in _ORDERING_RE.finditer(cleaned):
        left, right = match.group(1), match.group(3)
        if is_relation_variable(left) and is_relation_variable(right):
            pairs.append((left, right))
    return pairs


def numeric_object_positions(
    formula_str: str,
    arities: dict[str, int],
) -> set[tuple[str, str]]:
    """Return the ``(prop_id, object_id)`` positions the formula types numeric.

    A position is numeric when the variable bound to it appears on either side
    of an ordering comparison anywhere in the formula.

    Args:
        formula_str: The DejaVu policy formula.
        arities: prop_id -> arity, used to ignore extra call arguments.

    Returns:
        Set of ``(prop_id, "oN")`` pairs that must receive numeric values.
    """
    numeric_variables: set[str] = set()
    for left, right in find_ordering_comparisons(formula_str):
        numeric_variables.add(left)
        numeric_variables.add(right)

    if not numeric_variables:
        return set()

    positions: set[tuple[str, str]] = set()
    for prop_id, args in extract_formula_calls(formula_str):
        arity = arities.get(prop_id)
        if not arity:
            continue
        for idx, arg in enumerate(args[:arity]):
            if not is_relation_variable(arg):
                continue
            if arg.strip() in numeric_variables:
                positions.add((prop_id, f"o{idx + 1}"))
    return positions
