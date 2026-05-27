"""Schema validation tests for scenario_runner.schema."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from scenario_runner.schema import (
    Scenario,
    ScenarioObject,
    ScenarioPredicate,
    load_scenario,
)


def _minimal() -> dict:
    return {
        "scenario_id": "demo",
        "description": "demo",
        "model": {
            "grounding_provider": "ollama",
            "grounding_model": "llama3:8b",
        },
        "predicates": [
            {
                "prop_id": "p_x",
                "description": "user says X",
                "role": "user",
                "objects": [
                    {"object_id": "o1", "description": "thing", "entity_type": "Object"}
                ],
            }
        ],
        "policies": [
            {
                "policy_id": "pol1",
                "name": "must X",
                "formula_str": "H p_x",
            }
        ],
        "messages": [
            {"role": "user", "text": "hi", "expected_verdict": {"pol1": True}},
            {"role": "assistant", "text": "hello"},
        ],
    }


def test_happy_path_validates():
    s = Scenario.model_validate(_minimal())
    assert s.scenario_id == "demo"
    assert s.predicates[0].arity == 1
    assert s.predicates[0].arg_descriptions == ["thing"]


def test_predicate_arity_inferred_from_objects():
    data = _minimal()
    data["predicates"][0].pop("arity", None)
    s = Scenario.model_validate(data)
    assert s.predicates[0].arity == 1


def test_predicate_arity_mismatch_with_objects_rejected():
    data = _minimal()
    data["predicates"][0]["arity"] = 5
    with pytest.raises(ValidationError, match="disagrees"):
        Scenario.model_validate(data)


def test_predicate_arg_descriptions_inferred_from_objects():
    data = _minimal()
    data["predicates"][0].pop("arg_descriptions", None)
    s = Scenario.model_validate(data)
    assert s.predicates[0].arg_descriptions == ["thing"]


def test_bad_role_on_predicate_rejected():
    data = _minimal()
    data["predicates"][0]["role"] = "bot"
    with pytest.raises(ValidationError, match="role"):
        Scenario.model_validate(data)


def test_bad_role_on_message_rejected():
    data = _minimal()
    data["messages"][0]["role"] = "system"
    with pytest.raises(ValidationError, match="role"):
        Scenario.model_validate(data)


def test_expected_verdict_with_unknown_policy_rejected():
    data = _minimal()
    data["messages"][0]["expected_verdict"] = {"unknown_policy": True}
    with pytest.raises(ValidationError, match="unknown"):
        Scenario.model_validate(data)


def test_duplicate_predicate_ids_rejected():
    data = _minimal()
    data["predicates"].append(data["predicates"][0])
    with pytest.raises(ValidationError, match="prop_ids must be unique"):
        Scenario.model_validate(data)


def test_duplicate_policy_ids_rejected():
    data = _minimal()
    data["policies"].append(data["policies"][0])
    with pytest.raises(ValidationError, match="policy_ids must be unique"):
        Scenario.model_validate(data)


def test_extra_fields_rejected():
    data = _minimal()
    data["unexpected_field"] = "bad"
    with pytest.raises(ValidationError, match="unexpected"):
        Scenario.model_validate(data)


def test_load_scenario_from_disk():
    data = _minimal()
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(data, f)
        path = f.name
    try:
        s = load_scenario(path)
        assert s.scenario_id == "demo"
    finally:
        Path(path).unlink()


def test_load_scenario_bad_json_raises():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        f.write("{ not json")
        path = f.name
    try:
        with pytest.raises(json.JSONDecodeError):
            load_scenario(path)
    finally:
        Path(path).unlink()


def test_zero_arity_predicate_allowed():
    pred = ScenarioPredicate(
        prop_id="p",
        description="boolean predicate",
        role="user",
    )
    assert pred.arity == 0
    assert pred.arg_descriptions == []


def test_related_objects_validate_pair_shape():
    data = _minimal()
    data["predicates"].append({
        "prop_id": "p_y", "description": "d", "role": "user",
        "objects": [{"object_id": "o1", "description": "x", "entity_type": "X"}]
    })
    data["related_objects"] = [
        {"policy_id": "pol1", "pairs": [["p_x.o1", "p_y.o1"]]}
    ]
    s = Scenario.model_validate(data)
    assert len(s.related_objects) == 1
    assert s.related_objects[0].pairs == [["p_x.o1", "p_y.o1"]]


def test_related_objects_pair_must_have_two_elements():
    data = _minimal()
    data["related_objects"] = [
        {"policy_id": "pol1", "pairs": [["p_x.o1"]]}
    ]
    with pytest.raises(ValidationError, match="exactly two elements"):
        Scenario.model_validate(data)


def test_related_objects_endpoint_must_use_dot_format():
    data = _minimal()
    data["related_objects"] = [
        {"policy_id": "pol1", "pairs": [["p_x_o1", "p_x_o2"]]}
    ]
    with pytest.raises(ValidationError, match="must be 'prop_id.object_id'"):
        Scenario.model_validate(data)


def test_related_objects_unknown_policy_rejected():
    data = _minimal()
    data["related_objects"] = [
        {"policy_id": "unknown_pol", "pairs": [["p_x.o1", "p_x.o1"]]}
    ]
    with pytest.raises(ValidationError, match="unknown policy_id"):
        Scenario.model_validate(data)


def test_related_objects_unknown_predicate_rejected():
    data = _minimal()
    data["related_objects"] = [
        {"policy_id": "pol1", "pairs": [["unknown_pred.o1", "p_x.o1"]]}
    ]
    with pytest.raises(ValidationError, match="unknown predicate"):
        Scenario.model_validate(data)


def test_objects_only_arity_inferred():
    pred = ScenarioPredicate(
        prop_id="p",
        description="d",
        role="user",
        objects=[
            ScenarioObject(object_id="o1", description="a", entity_type="X"),
            ScenarioObject(object_id="o2", description="b", entity_type="Y"),
        ],
    )
    assert pred.arity == 2
    assert pred.arg_descriptions == ["a", "b"]
