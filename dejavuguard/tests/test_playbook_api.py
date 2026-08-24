"""Playbook CRUD, and the membership report that makes consequences visible."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.routers.playbooks import _is_irrevocable
from backend.store.db import DatabaseStore


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


def test_setting_globals_then_reading_them_back(client):
    """PUT replaces the whole set; GET must let a client see what survived
    before writing again, or it would silently wipe existing globals."""
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]

    put_resp = client.put(f"/api/playbooks/{pb}/globals", json={"globals": [
        {"name": "Escalate", "guidance": "Call it out.", "position": 0,
         "apply_to_all": True},
    ]})
    assert put_resp.status_code == 200

    got = client.get(f"/api/playbooks/{pb}/globals").json()
    assert len(got) == 1
    assert got[0]["name"] == "Escalate"
    assert got[0]["guidance"] == "Call it out."
    assert got[0]["position"] == 0
    assert bool(got[0]["apply_to_all"]) is True


def test_deleting_a_playbook(client):
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]
    assert client.delete(f"/api/playbooks/{pb}").status_code == 204
    assert client.get("/api/playbooks").json() == []


def test_trace_returns_nodes_and_observed_edges(client):
    """Edges come from the messages a session actually produced."""
    a = _policy(client, "p_a", "A")
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]
    client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": a, "position": 0, "fires_on": False, "guidance": "R."}]})

    body = client.get(f"/api/playbooks/{pb}/trace?session_id=none").json()

    assert {n["name"] for n in body["nodes"]} == {"(no guidance)", "R."}
    assert body["edges"] == []
    assert body["current"] is None


async def _seed_cycle_session(db_path: str) -> None:
    """A two-member playbook, labelled states, and a session that visits
    'Clear', then 'Over budget', then back to 'Clear' -- a cycle that leaves
    every visited node with an incoming edge (R-18)."""
    db = DatabaseStore(db_path)
    await db.initialize()
    await db.create_proposition("p_a", "a", "user")
    await db.create_proposition("p_b", "b", "user")
    await db.create_policy("pol-a", "A", "p_a", True)
    await db.create_policy("pol-b", "B", "p_b", True)
    await db.set_policy_propositions("pol-a", ["p_a"])
    await db.set_policy_propositions("pol-b", ["p_b"])
    await db.create_playbook("pb1", "Budget")
    await db.set_playbook_members("pb1", [
        {"policy_id": "pol-a", "position": 0, "fires_on": True, "guidance": "Over."},
        {"policy_id": "pol-b", "position": 1, "fires_on": True, "guidance": "Other."}])
    # Each state needs DIFFERENT guidance to be a different behaviour. Merging
    # keys on (rules, flagged), not on the label -- four states sharing empty
    # guidance would collapse into one node however they are named.
    await db.set_playbook_override("pb1", "pol-a=F;pol-b=F", [], False, "Clear")
    await db.set_playbook_override(
        "pb1", "pol-a=T;pol-b=F",
        [{"type": "member", "policy_id": "pol-a"}], False, "Over budget",
    )
    await db.set_playbook_override(
        "pb1", "pol-a=F;pol-b=T",
        [{"type": "member", "policy_id": "pol-b"}], False, "Other flag",
    )
    await db.create_session("s1")
    await db.add_message("s1", 0, "user", "hi",
                          monitor_state={"pol-a": False, "pol-b": False})
    await db.add_message("s1", 1, "user", "hi",
                          monitor_state={"pol-a": True, "pol-b": False})
    await db.add_message("s1", 2, "user", "hi",
                          monitor_state={"pol-a": False, "pol-b": False})
    await db.close()


def test_trace_reports_first_visit_in_chronological_order_on_a_cycle(
    tmp_path, monkeypatch,
):
    """R-18: the server, not the client, knows the true visit order. A
    session that returns to its starting node ('Clear') still gets a
    correct chronological first_visit, and nodes the session never reached
    get null."""
    db_path = str(tmp_path / "trace.db")
    monkeypatch.setenv("DATABASE_PATH", db_path)
    asyncio.run(_seed_cycle_session(db_path))

    with TestClient(create_app()) as client:
        body = client.get("/api/playbooks/pb1/trace?session_id=s1").json()

    by_name = {n["name"]: n for n in body["nodes"]}
    # Clear is visited first and again last; its first_visit stays 0.
    assert by_name["Clear"]["first_visit"] == 0
    assert by_name["Over budget"]["first_visit"] == 1
    # Every node the session never reached carries null, whatever it is named.
    assert all(
        node["first_visit"] is None
        for node in body["nodes"]
        if not node["visited"]
    )


# --- Reachability heuristic (R-17) ---


def test_is_irrevocable_accepts_leading_h_with_or_without_a_space():
    assert _is_irrevocable("H (p_fraud -> !q_comply)")
    assert _is_irrevocable("H(p_fraud -> !q_comply)")


def test_is_irrevocable_rejects_an_identifier_that_merely_starts_with_h():
    assert not _is_irrevocable("Hello")


def test_is_irrevocable_rejects_a_non_h_formula():
    assert not _is_irrevocable("p_fraud -> !q_comply")


async def _seed_irrevocable_session(db_path: str) -> None:
    """A playbook with one irrevocable member, and a session that saw it go False."""
    db = DatabaseStore(db_path)
    await db.initialize()
    await db.create_proposition("p_a", "a", "user")
    await db.create_policy("pol-a", "A", "H(p_a)", True)
    await db.set_policy_propositions("pol-a", ["p_a"])
    await db.create_playbook("pb1", "Budget")
    await db.set_playbook_members("pb1", [
        {"policy_id": "pol-a", "position": 0, "fires_on": True,
         "guidance": "Blocked."}])
    await db.create_session("s1")
    await db.add_message("s1", 0, "user", "hi", monitor_state={"pol-a": False})
    await db.close()


def test_trace_marks_states_requiring_a_false_irrevocable_member_unreachable(
    tmp_path, monkeypatch,
):
    """R-17: once an irrevocable member is False, states requiring it True
    are permanently unreachable -- shown, not hidden, and labelled a
    heuristic rather than a proof."""
    db_path = str(tmp_path / "trace.db")
    monkeypatch.setenv("DATABASE_PATH", db_path)
    asyncio.run(_seed_irrevocable_session(db_path))

    with TestClient(create_app()) as client:
        body = client.get("/api/playbooks/pb1/trace?session_id=s1").json()

    assert body["current"] == "(no guidance)"
    by_name = {n["name"]: n for n in body["nodes"]}
    assert by_name["(no guidance)"]["reachable"] is True
    assert by_name["Blocked."]["reachable"] is False

    member = body["members"][0]
    assert member["policy_id"] == "pol-a"
    assert member["irrevocable"] is True


def test_trace_treats_everything_reachable_when_there_is_no_current_state(client):
    a = _policy(client, "p_a", "A")
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]
    client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": a, "position": 0, "fires_on": False, "guidance": "R."}]})

    body = client.get(f"/api/playbooks/{pb}/trace?session_id=none").json()

    assert body["current"] is None
    assert all(n["reachable"] for n in body["nodes"])


def test_states_endpoint_reports_irrevocable_per_member(client):
    a = _policy(client, "p_a", "A")
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]
    client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": a, "position": 0, "fires_on": False, "guidance": "R."}]})

    body = client.get(f"/api/playbooks/{pb}/states").json()

    assert body["members"][0]["irrevocable"] is False
