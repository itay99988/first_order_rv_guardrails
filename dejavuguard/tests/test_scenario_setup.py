"""Idempotent setup tests for scenario_runner.setup.

Uses an in-memory SQLite DatabaseStore. Mocks _validate_formula so the
DejaVu server is not required.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from backend.store.db import DatabaseStore

from scenario_runner.schema import (
    ScenarioObject,
    ScenarioPolicy,
    ScenarioPredicate,
    ScenarioRelatedObjects,
)
from scenario_runner.setup import (
    SetupConflict,
    ensure_policy,
    ensure_predicate,
    ensure_related_objects,
    _expand_pairs_to_relations,
)


@pytest.fixture
async def db():
    store = DatabaseStore(":memory:")
    await store.initialize()
    yield store
    await store.close()


def _pred(prop_id: str = "p_x", description: str = "user says X") -> ScenarioPredicate:
    return ScenarioPredicate(
        prop_id=prop_id,
        description=description,
        role="user",
        objects=[
            ScenarioObject(object_id="o1", description="thing", entity_type="Object")
        ],
        few_shot_examples=[{"text": "hello", "instances": []}],
    )


@pytest.mark.asyncio
async def test_predicate_created_when_absent(db: DatabaseStore):
    status = await ensure_predicate(db, _pred(), overwrite=False,
                                     scenario_few_shot_model=None)
    assert status == "created"
    row = await db.get_proposition("p_x")
    assert row is not None
    assert row["role"] == "user"
    assert row["arity"] == 1


@pytest.mark.asyncio
async def test_predicate_reused_when_shape_matches(db: DatabaseStore):
    await ensure_predicate(db, _pred(), overwrite=False,
                            scenario_few_shot_model=None)
    status = await ensure_predicate(db, _pred(), overwrite=False,
                                     scenario_few_shot_model=None)
    assert status == "reused"


@pytest.mark.asyncio
async def test_predicate_shape_conflict_aborts(db: DatabaseStore):
    await ensure_predicate(db, _pred(), overwrite=False,
                            scenario_few_shot_model=None)
    bad = _pred(description="different description")
    with pytest.raises(SetupConflict, match="different shape"):
        await ensure_predicate(db, bad, overwrite=False,
                                scenario_few_shot_model=None)


@pytest.mark.asyncio
async def test_predicate_overwrite_updates(db: DatabaseStore):
    await ensure_predicate(db, _pred(), overwrite=False,
                            scenario_few_shot_model=None)
    updated = _pred(description="updated description")
    status = await ensure_predicate(db, updated, overwrite=True,
                                     scenario_few_shot_model=None)
    assert status == "updated"
    row = await db.get_proposition("p_x")
    assert row["description"] == "updated description"


@pytest.mark.asyncio
async def test_predicate_overwrite_refreshes_few_shots_on_shape_match(
    db: DatabaseStore,
):
    """When --overwrite is set and shape is identical, new few-shots
    supplied by the scenario should still replace the stored ones."""
    import json
    await ensure_predicate(db, _pred(), overwrite=False,
                            scenario_few_shot_model=None)
    new_examples = [
        {"text": "added example", "instances": []},
        {"text": "second new example", "instances": []},
    ]
    refreshed = ScenarioPredicate(
        prop_id="p_x", description="user says X", role="user",
        objects=[ScenarioObject(object_id="o1", description="thing",
                                  entity_type="Object")],
        few_shot_examples=new_examples,
    )
    status = await ensure_predicate(db, refreshed, overwrite=True,
                                     scenario_few_shot_model=None)
    assert status == "updated"
    row = await db.get_proposition("p_x")
    stored_examples = json.loads(row["few_shot_examples"])
    assert stored_examples == new_examples


@pytest.mark.asyncio
async def test_predicate_no_overwrite_keeps_stored_few_shots(
    db: DatabaseStore,
):
    """Without --overwrite, identical-shape predicates are reused as-is
    (few-shots are not touched)."""
    import json
    await ensure_predicate(db, _pred(), overwrite=False,
                            scenario_few_shot_model=None)
    new_examples = [{"text": "should NOT appear", "instances": []}]
    refreshed = ScenarioPredicate(
        prop_id="p_x", description="user says X", role="user",
        objects=[ScenarioObject(object_id="o1", description="thing",
                                  entity_type="Object")],
        few_shot_examples=new_examples,
    )
    status = await ensure_predicate(db, refreshed, overwrite=False,
                                     scenario_few_shot_model=None)
    assert status == "reused"
    row = await db.get_proposition("p_x")
    stored_examples = json.loads(row["few_shot_examples"])
    assert stored_examples != new_examples


@pytest.mark.asyncio
async def test_predicate_with_no_few_shots_and_no_model_aborts(db: DatabaseStore):
    pred = ScenarioPredicate(
        prop_id="p_y", description="d", role="user",
        objects=[ScenarioObject(object_id="o1", description="x", entity_type="X")],
    )
    with pytest.raises(SetupConflict, match="no chat model configured"):
        await ensure_predicate(db, pred, overwrite=False,
                                scenario_few_shot_model=None)


def _policy(formula: str = "H p_x") -> ScenarioPolicy:
    return ScenarioPolicy(
        policy_id="pol1", name="must X", formula_str=formula
    )


@pytest.mark.asyncio
async def test_policy_created_when_absent(db: DatabaseStore):
    with patch(
        "scenario_runner.setup._validate_formula",
        AsyncMock(return_value=([], None)),
    ):
        status = await ensure_policy(db, _policy(), overwrite=False)
    assert status == "created"
    row = await db.get_policy("pol1")
    assert row is not None
    assert row["formula_str"] == "H p_x"


@pytest.mark.asyncio
async def test_policy_reused_when_match(db: DatabaseStore):
    with patch(
        "scenario_runner.setup._validate_formula",
        AsyncMock(return_value=([], None)),
    ):
        await ensure_policy(db, _policy(), overwrite=False)
        status = await ensure_policy(db, _policy(), overwrite=False)
    assert status == "reused"


@pytest.mark.asyncio
async def test_policy_conflict_aborts(db: DatabaseStore):
    with patch(
        "scenario_runner.setup._validate_formula",
        AsyncMock(return_value=([], None)),
    ):
        await ensure_policy(db, _policy(), overwrite=False)
        with pytest.raises(SetupConflict, match="different fields"):
            await ensure_policy(db, _policy(formula="P p_x"), overwrite=False)


@pytest.mark.asyncio
async def test_policy_overwrite_updates(db: DatabaseStore):
    with patch(
        "scenario_runner.setup._validate_formula",
        AsyncMock(return_value=([], None)),
    ):
        await ensure_policy(db, _policy(), overwrite=False)
        status = await ensure_policy(
            db, _policy(formula="P p_x"), overwrite=True
        )
    assert status == "updated"
    row = await db.get_policy("pol1")
    assert row["formula_str"] == "P p_x"


@pytest.mark.asyncio
async def test_policy_validation_failure_aborts(db: DatabaseStore):
    with patch(
        "scenario_runner.setup._validate_formula",
        AsyncMock(return_value=([], "bad syntax")),
    ):
        with pytest.raises(SetupConflict, match="failed validation"):
            await ensure_policy(db, _policy(formula="garbage"), overwrite=False)


def test_expand_pairs_to_relations_bidirectional():
    entry = ScenarioRelatedObjects(
        policy_id="pol1",
        pairs=[["user_car.o1", "assistant_car.o1"]],
    )
    rels = _expand_pairs_to_relations(entry)
    assert len(rels) == 2
    assert {(r["prop_id"], r["object_id"], r["related_prop_id"], r["related_object_id"])
            for r in rels} == {
        ("user_car", "o1", "assistant_car", "o1"),
        ("assistant_car", "o1", "user_car", "o1"),
    }


@pytest.mark.asyncio
async def test_related_objects_created_when_absent(db: DatabaseStore):
    # Must seed predicates + policy first because related_objects FK
    # points at both.
    await ensure_predicate(db, _pred(prop_id="a"), overwrite=False,
                            scenario_few_shot_model=None)
    await ensure_predicate(db, _pred(prop_id="b"), overwrite=False,
                            scenario_few_shot_model=None)
    with patch(
        "scenario_runner.setup._validate_formula",
        AsyncMock(return_value=([], None)),
    ):
        await ensure_policy(db, _policy(), overwrite=False)
    entry = ScenarioRelatedObjects(
        policy_id="pol1",
        pairs=[["a.o1", "b.o1"]],
    )
    status = await ensure_related_objects(db, entry, overwrite=False)
    assert status == "created"
    rows = await db.list_related_objects()
    # Bidirectional: one pair -> two rows
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_related_objects_reused_on_match(db: DatabaseStore):
    await ensure_predicate(db, _pred(prop_id="a"), overwrite=False,
                            scenario_few_shot_model=None)
    await ensure_predicate(db, _pred(prop_id="b"), overwrite=False,
                            scenario_few_shot_model=None)
    with patch(
        "scenario_runner.setup._validate_formula",
        AsyncMock(return_value=([], None)),
    ):
        await ensure_policy(db, _policy(), overwrite=False)
    entry = ScenarioRelatedObjects(
        policy_id="pol1",
        pairs=[["a.o1", "b.o1"]],
    )
    await ensure_related_objects(db, entry, overwrite=False)
    status = await ensure_related_objects(db, entry, overwrite=False)
    assert status == "reused"


@pytest.mark.asyncio
async def test_related_objects_conflict_aborts(db: DatabaseStore):
    await ensure_predicate(db, _pred(prop_id="a"), overwrite=False,
                            scenario_few_shot_model=None)
    await ensure_predicate(db, _pred(prop_id="b"), overwrite=False,
                            scenario_few_shot_model=None)
    with patch(
        "scenario_runner.setup._validate_formula",
        AsyncMock(return_value=([], None)),
    ):
        await ensure_policy(db, _policy(), overwrite=False)
    await ensure_related_objects(
        db,
        ScenarioRelatedObjects(policy_id="pol1", pairs=[["a.o1", "b.o1"]]),
        overwrite=False,
    )
    with pytest.raises(SetupConflict, match="related_objects"):
        await ensure_related_objects(
            db,
            ScenarioRelatedObjects(policy_id="pol1", pairs=[["a.o2", "b.o2"]]),
            overwrite=False,
        )


@pytest.mark.asyncio
async def test_related_objects_overwrite_replaces(db: DatabaseStore):
    await ensure_predicate(db, _pred(prop_id="a"), overwrite=False,
                            scenario_few_shot_model=None)
    await ensure_predicate(db, _pred(prop_id="b"), overwrite=False,
                            scenario_few_shot_model=None)
    with patch(
        "scenario_runner.setup._validate_formula",
        AsyncMock(return_value=([], None)),
    ):
        await ensure_policy(db, _policy(), overwrite=False)
    await ensure_related_objects(
        db,
        ScenarioRelatedObjects(policy_id="pol1", pairs=[["a.o1", "b.o1"]]),
        overwrite=False,
    )
    status = await ensure_related_objects(
        db,
        ScenarioRelatedObjects(policy_id="pol1", pairs=[["a.o2", "b.o2"]]),
        overwrite=True,
    )
    assert status == "updated"
    rows = await db.list_related_objects()
    assert len(rows) == 2
    keys = {(r["prop_id"], r["object_id"]) for r in rows}
    assert ("a", "o2") in keys and ("b", "o2") in keys
