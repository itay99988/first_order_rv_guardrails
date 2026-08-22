"""Deriving which predicate object positions must hold numbers.

A policy that orders two object slots with <, <=, > or >= implicitly types
those slots as numeric. Nothing else in the pipeline declares that, so it
has to be recovered from the formula.
"""

from __future__ import annotations

from backend.engine.formula_analysis import (
    find_ordering_comparisons,
    numeric_object_positions,
)

CAR_FORMULA = (
    "forall m . forall p . recommend_a(m, p) -> exists b . ( "
    "( !(exists m2 . exists b2 . ( request_u(m2, b2) & (!(m2 = m) | !(b2 = b)) )) "
    "S request_u(m, b) ) & !(b < p) )"
)

CAR_ARITIES = {"request_u": 2, "recommend_a": 2}


def test_ordering_comparison_is_detected():
    assert ("b", "p") in find_ordering_comparisons(CAR_FORMULA)


def test_equality_is_not_an_ordering_comparison():
    """`=` relates slots but does not require them to be numeric."""
    assert find_ordering_comparisons("forall a . forall c . p(a) & q(c) & (a = c)") == []


def test_ordering_marks_both_sides_numeric():
    """`b < p` types request_u.o2 and recommend_a.o2, and nothing else."""
    positions = numeric_object_positions(CAR_FORMULA, CAR_ARITIES)

    assert ("request_u", "o2") in positions
    assert ("recommend_a", "o2") in positions


def test_manufacturer_slots_are_not_numeric():
    """`m` is only ever used for identity, so o1 stays untyped."""
    positions = numeric_object_positions(CAR_FORMULA, CAR_ARITIES)

    assert ("request_u", "o1") not in positions
    assert ("recommend_a", "o1") not in positions


def test_formula_without_ordering_has_no_numeric_positions():
    formula = "H (p_fraud -> ! q_comply)"

    assert numeric_object_positions(formula, {"p_fraud": 0, "q_comply": 0}) == set()


def test_quoted_literals_do_not_create_comparisons():
    """A string literal containing '<' must not be parsed as an operator."""
    formula = 'forall a . p(a) & (a = "x < y")'

    assert numeric_object_positions(formula, {"p": 1}) == set()
