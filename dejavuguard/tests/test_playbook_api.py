"""Playbook CRUD, and the membership report that makes consequences visible."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "api.db"))
    with TestClient(create_app()) as c:
        yield c


def _policy(client: TestClient, prop_id: str, policy_id_name: str) -> str:
    client.post("/api/propositions", json={
        "prop_id": prop_id, "description": prop_id, "role": "user"})
    resp = client.post("/api/policies", json={
        "name": policy_id_name, "formula_str": prop_id})
    return resp.json()["policy_id"]


def test_create_and_list_playbooks(client):
    created = client.post("/api/playbooks", json={"name": "Budget"})
    assert created.status_code == 201

    listed = client.get("/api/playbooks").json()
    assert [p["name"] for p in listed] == ["Budget"]


def test_a_new_playbook_reports_one_state_and_one_behaviour(client):
    """No members means the empty vector: exactly one state."""
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()

    listed = client.get("/api/playbooks").json()[0]
    assert listed["state_count"] == 1
    assert listed["behaviour_count"] == 1
    assert listed["playbook_id"] == pb["playbook_id"]


def test_setting_members_reports_state_growth(client):
    policy_id = _policy(client, "p_a", "A")
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]

    resp = client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": policy_id, "position": 0, "fires_on": False, "guidance": "R."}]})

    assert resp.status_code == 200
    assert resp.json()["state_count"] == 2


def test_states_endpoint_groups_by_behaviour(client):
    policy_id = _policy(client, "p_a", "A")
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]
    client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": policy_id, "position": 0, "fires_on": False, "guidance": "R."}]})

    body = client.get(f"/api/playbooks/{pb}/states").json()
    names = {b["name"] for b in body["behaviours"]}
    assert "(no guidance)" in names
    assert body["state_count"] == 2


def test_overriding_a_state_then_reverting(client):
    policy_id = _policy(client, "p_a", "A")
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]
    client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": policy_id, "position": 0, "fires_on": False, "guidance": "R."}]})
    key = f"{policy_id}=F"

    client.put(f"/api/playbooks/{pb}/states/{key}",
               json={"rule_refs": [], "flagged": True, "label": "Stop"})
    states = client.get(f"/api/playbooks/{pb}/states").json()
    flagged = [b for b in states["behaviours"] if b["flagged"]]
    assert len(flagged) == 1

    client.put(f"/api/playbooks/{pb}/states/{key}",
               json={"rule_refs": None, "flagged": False, "label": None})
    states = client.get(f"/api/playbooks/{pb}/states").json()
    assert not any(b["flagged"] for b in states["behaviours"])


def test_adding_a_member_expands_existing_overrides(client):
    a = _policy(client, "p_a", "A")
    b = _policy(client, "p_b", "B")
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]
    client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": a, "position": 0, "fires_on": False, "guidance": "R."}]})
    client.put(f"/api/playbooks/{pb}/states/{a}=F",
               json={"rule_refs": [], "flagged": True, "label": "Stop"})

    resp = client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": a, "position": 0, "fires_on": False, "guidance": "R."},
        {"policy_id": b, "position": 1, "fires_on": False, "guidance": "S."}]})

    assert resp.json()["overrides_expanded"] == 2


def test_enforcement_warning_when_no_flagged_state_can_block(client):
    """R1: a playbook with no flagged state silently blocks nothing."""
    a = _policy(client, "p_a", "A")
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]
    client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": a, "position": 0, "fires_on": False, "guidance": "R."}]})

    warnings = client.get(f"/api/playbooks/{pb}/states").json()["warnings"]
    assert any("can no longer block" in w for w in warnings)


def test_deleting_a_playbook(client):
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]
    assert client.delete(f"/api/playbooks/{pb}").status_code == 204
    assert client.get("/api/playbooks").json() == []
