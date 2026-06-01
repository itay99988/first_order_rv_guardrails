"""
Builds DejaVu specification strings from DejaVuGuard policies.

Users write DejaVu formulas directly, so no syntax conversion is needed.
This module assembles the full DejaVu spec (pred declarations + prop rules)
from the policy and predicate models.
"""

from __future__ import annotations

from backend.models.builtins import BUILTIN_PROPOSITIONS
from backend.models.policy import Policy, Proposition


def build_dejavu_spec(policies: list[Policy], propositions: list[Proposition]) -> str:
    """Build a complete DejaVu spec from all enabled policies.

    Each policy becomes a DejaVu property. Predicates are declared as
    predicates. The spec format is:

        pred p_fraud
        pred q_comply
        pred p_transfer(account, destination, amount)

        prop fraud_prevention : H (P p_fraud -> ! q_comply)
        prop transfer_policy : Forall acc . (p_transfer(acc, "offshore") -> q_review(acc))

    Args:
        policies: List of enabled policies.
        propositions: All predicates referenced by the policies.

    Returns:
        DejaVu specification string.
    """
    lines = []

    # Declare all predicates with correct arity
    declared: set[str] = set()
    for builtin_id in sorted(BUILTIN_PROPOSITIONS):
        lines.append(f"pred {builtin_id}")
        declared.add(builtin_id)

    for prop in propositions:
        if prop.prop_id not in declared:
            if prop.arity > 0:
                # Generate placeholder arg names: a1, a2, a3, ...
                arg_names = ", ".join(f"a{i+1}" for i in range(prop.arity))
                lines.append(f"pred {prop.prop_id}({arg_names})")
            else:
                lines.append(f"pred {prop.prop_id}")
            declared.add(prop.prop_id)

    if lines:
        lines.append("")  # blank line between preds and props

    # Add each policy as a property
    for policy in policies:
        if not policy.enabled:
            continue
        formula = policy.formula_str
        # Sanitize policy name for DejaVu (must start with letter, alphanumeric + underscore)
        safe_name = "pol_" + policy.policy_id.replace("-", "_")
        lines.append(f"prop {safe_name} : {formula}")

    return "\n".join(lines)


