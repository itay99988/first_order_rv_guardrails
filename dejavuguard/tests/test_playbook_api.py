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


async def _seed_traced_member(db_path: str) -> None:
    """One member, attached to a rule, and a session that never ran.

    Seeded through the store rather than over HTTP so the row is exactly
    what a migrated database holds: linked to a rule, with its own guidance
    column left empty, so the node below can only be named from the rule.
    """
    db = DatabaseStore(db_path)
    await db.initialize()
    await db.create_proposition("p_a", "a", "user")
    await db.create_policy("pol-a", "A", "p_a", True)
    await db.set_policy_propositions("pol-a", ["p_a"])
    await db.create_playbook("pb1", "Budget")
    await db.set_playbook_members("pb1", [
        {"policy_id": "pol-a", "position": 0, "fires_on": False, "guidance": ""}])
    await db.create_rule("r1", "R", "R.")
    await db._db.execute(
        "UPDATE playbook_members SET rule_id = 'r1' WHERE playbook_id = 'pb1'"
    )
    await db._db.commit()
    await db.close()


def test_trace_returns_nodes_and_observed_edges(tmp_path, monkeypatch):
    """Edges come from the messages a session actually produced."""
    db_path = str(tmp_path / "trace.db")
    monkeypatch.setenv("DATABASE_PATH", db_path)
    asyncio.run(_seed_traced_member(db_path))

    with TestClient(create_app()) as client:
        body = client.get("/api/playbooks/pb1/trace?session_id=none").json()

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


def _override_keys(client: TestClient, pb: str) -> dict[str, dict]:
    """state_key -> the stored override, read back through the states view."""
    body = client.get(f"/api/playbooks/{pb}/states").json()
    return {
        s["state_key"]: {"flagged": b["flagged"], "label": s["label"],
                         "rules": b["rules"]}
        for b in body["behaviours"] for s in b["states"]
    }


def _two_member_playbook(client: TestClient) -> tuple[str, str, str]:
    """A playbook over two policies, returned sorted by policy id.

    State keys sort by policy id, so the caller needs them in that order to
    name a state at all.
    """
    a = _policy(client, "p_a", "A")
    b = _policy(client, "p_b", "B")
    first, second = sorted((a, b))
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]
    client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": first, "position": 0, "fires_on": False, "guidance": "R."},
        {"policy_id": second, "position": 1, "fires_on": False, "guidance": "S."}]})
    return pb, first, second


def test_deleting_a_member_policy_rekeys_surviving_overrides(client):
    """Overrides must follow the shrinking state space, not be orphaned.

    The FK cascade drops the member row; without a migration the override
    rows keep their old two-policy keys and match no state at all, so every
    flag silently stops firing.
    """
    pb, first, second = _two_member_playbook(client)
    for other in ("T", "F"):
        client.put(f"/api/playbooks/{pb}/states/{first}=F;{second}={other}",
                   json={"rule_refs": [], "flagged": True, "label": "Stop"})

    assert client.delete(f"/api/policies/{second}").status_code == 204

    states = _override_keys(client, pb)
    assert set(states) == {f"{first}=T", f"{first}=F"}
    assert states[f"{first}=F"]["flagged"] is True
    assert states[f"{first}=F"]["label"] == "Stop"
    assert states[f"{first}=F"]["rules"] == []
    assert states[f"{first}=T"]["flagged"] is False


def test_deleting_a_member_policy_keeps_the_playbook_able_to_block(client):
    """The user-visible property: a blocking playbook still blocks."""
    pb, first, second = _two_member_playbook(client)
    for other in ("T", "F"):
        client.put(f"/api/playbooks/{pb}/states/{first}=F;{second}={other}",
                   json={"rule_refs": [], "flagged": True, "label": "Stop"})

    client.delete(f"/api/policies/{second}")

    body = client.get(f"/api/playbooks/{pb}/states").json()
    assert any(b["flagged"] for b in body["behaviours"])
    assert not any("can no longer block" in w for w in body["warnings"])


def test_deleting_a_member_policy_resolves_a_conflict_to_the_not_firing_branch(
    client,
):
    """Branches that disagree cannot pause a delete for the user.

    The collapse's own preference -- the not-firing (False) branch -- is
    taken, matching what set_members proposes for the same conflict.
    """
    pb, first, second = _two_member_playbook(client)
    client.put(f"/api/playbooks/{pb}/states/{first}=F;{second}=F",
               json={"rule_refs": [], "flagged": True, "label": "Stop"})
    client.put(f"/api/playbooks/{pb}/states/{first}=F;{second}=T",
               json={"rule_refs": [], "flagged": False, "label": "Go"})

    client.delete(f"/api/policies/{second}")

    states = _override_keys(client, pb)
    assert states[f"{first}=F"]["flagged"] is True
    assert states[f"{first}=F"]["label"] == "Stop"


def test_deleting_a_policy_no_playbook_uses_still_works(client):
    policy_id = _policy(client, "p_a", "A")

    assert client.delete(f"/api/policies/{policy_id}").status_code == 204
    assert client.get("/api/policies").json() == []


def _state_rows(client: TestClient, pb: str) -> dict[str, dict]:
    """state_key -> its row, as the states view returns it."""
    body = client.get(f"/api/playbooks/{pb}/states").json()
    return {s["state_key"]: s for b in body["behaviours"] for s in b["states"]}


def test_states_endpoint_returns_the_stored_rule_refs(client):
    """A client cannot infer a pin from the resolved guidance.

    Pinning exactly the rules a state already derives -- the obvious thing to
    do when the tick boxes start pre-ticked -- resolves identically to no pin
    at all. Only the stored refs tell them apart, and they stop being the same
    thing the moment a member is added.
    """
    a = _policy(client, "p_a", "A")
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]
    client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": a, "position": 0, "fires_on": False, "guidance": "R."}]})
    key = f"{a}=F"

    client.put(f"/api/playbooks/{pb}/states/{key}",
               json={"rule_refs": [{"type": "member", "policy_id": a}],
                     "flagged": False, "label": None})

    row = _state_rows(client, pb)[key]
    assert row["rule_refs"] == [{"type": "member", "policy_id": a}]
    assert row["customised"] is True


def test_states_endpoint_keeps_null_and_empty_rule_refs_apart(client):
    """SQL NULL must arrive as null and [] must arrive as [], both ways."""
    a = _policy(client, "p_a", "A")
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]
    client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": a, "position": 0, "fires_on": False, "guidance": "R."}]})
    fires, quiet = f"{a}=F", f"{a}=T"

    # Deliberately no guidance, kept customised by the flag alone elsewhere.
    client.put(f"/api/playbooks/{pb}/states/{fires}",
               json={"rule_refs": [], "flagged": False, "label": None})
    rows = _state_rows(client, pb)
    assert rows[fires]["rule_refs"] == []
    assert rows[fires]["rule_refs"] is not None
    # Never edited: derive.
    assert rows[quiet]["rule_refs"] is None

    # Flag-only: still deriving, so still null rather than an empty list.
    client.put(f"/api/playbooks/{pb}/states/{fires}",
               json={"rule_refs": None, "flagged": True, "label": None})
    rows = _state_rows(client, pb)
    assert rows[fires]["rule_refs"] is None
    assert rows[fires]["customised"] is True


def test_two_states_in_one_behaviour_report_their_own_rule_refs(client):
    """rule_refs is per state, never per behaviour.

    Behaviours group on (resolved rules, flagged), so a state pinned to
    exactly the rules another state derives lands in the *same* behaviour as
    that state -- which is correct, they do behave identically today. But
    they are not the same instruction: add a member and the derived one picks
    it up while the pinned one does not. Reporting rule_refs on the behaviour
    would hand both states one value and collapse exactly the distinction
    this endpoint is meant to carry.
    """
    pb, first, second = _two_member_playbook(client)
    # Both members fire on F, so first=F;second=F derives ("R.", "S.") and
    # first=T;second=T derives nothing. Pinning the latter to both members
    # makes it resolve identically to the former.
    derived_key, pinned_key = f"{first}=F;{second}=F", f"{first}=T;{second}=T"
    client.put(f"/api/playbooks/{pb}/states/{pinned_key}",
               json={"rule_refs": [{"type": "member", "policy_id": first},
                                   {"type": "member", "policy_id": second}],
                     "flagged": False, "label": None})

    body = client.get(f"/api/playbooks/{pb}/states").json()
    shared = [
        b for b in body["behaviours"]
        if {s["state_key"] for s in b["states"]} >= {derived_key, pinned_key}
    ]

    # One behaviour, because they resolve alike: refs must not fragment the
    # grouping into nodes that behave identically.
    assert len(shared) == 1
    rows = {s["state_key"]: s for s in shared[0]["states"]}
    assert rows[derived_key]["rule_refs"] is None
    assert rows[pinned_key]["rule_refs"] == [
        {"type": "member", "policy_id": first},
        {"type": "member", "policy_id": second},
    ]


def test_states_reports_the_members_resolved_rule_guidance(tmp_path, monkeypatch):
    """The truth table names behaviours from the rule library.

    /states is assembled from the same loader as /trace, but through a
    different payload, so it is worth pinning separately -- a member whose
    only guidance is its rule must still read as guidance here.
    """
    db_path = str(tmp_path / "states.db")
    monkeypatch.setenv("DATABASE_PATH", db_path)
    asyncio.run(_seed_traced_member(db_path))

    with TestClient(create_app()) as client:
        body = client.get("/api/playbooks/pb1/states").json()

    assert [m["guidance"] for m in body["members"]] == ["R."]
    assert {b["name"] for b in body["behaviours"]} == {"(no guidance)", "R."}


def _rule(client: TestClient, name: str, guidance: str) -> str:
    resp = client.post("/api/rules", json={"name": name, "guidance": guidance})
    assert resp.status_code == 201
    return resp.json()["rule_id"]


def test_members_can_name_an_existing_rule_by_id(client):
    """A member attaches to a rule it names, without re-sending the text.

    The library is the one copy of the text, so a caller that already has a
    rule should be able to point at it; re-sending the words would mint a
    second rule saying the same thing.
    """
    policy_id = _policy(client, "p_a", "A")
    rule_id = _rule(client, "Budget", "Stay within budget.")
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]

    resp = client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": policy_id, "position": 0, "fires_on": False,
         "rule_id": rule_id}]})
    assert resp.status_code == 200

    member = client.get(f"/api/playbooks/{pb}/states").json()["members"][0]
    assert member["rule_id"] == rule_id
    assert member["guidance"] == "Stay within budget."


def test_a_member_naming_an_unknown_rule_is_rejected(client):
    """422, not a silent drop: the member would contribute nothing at all."""
    policy_id = _policy(client, "p_a", "A")
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]

    resp = client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": policy_id, "position": 0, "fires_on": False,
         "rule_id": "no-such-rule"}]})

    assert resp.status_code == 422
    assert "no-such-rule" in resp.json()["detail"]


def test_a_member_naming_an_unknown_policy_is_rejected(client):
    """422, not a 500 (R-27).

    The row is written against a foreign key, so an unknown policy reached
    the client as `sqlite3.IntegrityError: FOREIGN KEY constraint failed`
    wrapped in a 500 -- while the globals path three functions away already
    answered the same class of mistake with a clean 422. `POST /policies`
    ignores a client-supplied `policy_id` and generates its own, which is
    how a caller ends up naming one that does not exist.
    """
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]

    resp = client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": "no-such-policy", "position": 0, "fires_on": False,
         "guidance": "Stay within budget."}]})

    assert resp.status_code == 422
    assert "no-such-policy" in resp.json()["detail"]


def test_a_refused_member_save_leaves_no_rule_behind(client):
    """The refusal has to come before the first rule is minted (R-28).

    Guidance with no `rule_id` is resolved onto a library rule on the way
    in, and each mint commits. A save that validated afterwards therefore
    left one orphan rule per failed request -- invisible, permanent, and
    growing every time the 500 above fired.
    """
    good = _policy(client, "p_a", "A")
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]
    before = {r["rule_id"] for r in client.get("/api/rules").json()}

    resp = client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": good, "position": 0, "fires_on": False,
         "guidance": "This text would mint a rule."},
        {"policy_id": "no-such-policy", "position": 1, "fires_on": False,
         "guidance": "So would this one."}]})

    assert resp.status_code == 422
    assert {r["rule_id"] for r in client.get("/api/rules").json()} == before
    assert client.get(f"/api/playbooks/{pb}/states").json()["members"] == []


def test_naming_one_policy_twice_is_rejected(client):
    """The member table is keyed on (playbook, policy), so this was a 500.

    Same shape as the unknown policy above and the same cost: the request
    reaches the INSERT, trips PRIMARY KEY, and leaves behind the rules its
    two guidance strings had already minted. The current UI greys out a
    policy the playbook already holds, so this arrives over the API rather
    than through the editor -- which is precisely why nothing caught it.
    """
    policy_id = _policy(client, "p_a", "A")
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]
    before = {r["rule_id"] for r in client.get("/api/rules").json()}

    resp = client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": policy_id, "position": 0, "fires_on": False, "guidance": "One."},
        {"policy_id": policy_id, "position": 1, "fires_on": True, "guidance": "Two."}]})

    assert resp.status_code == 422
    assert policy_id in resp.json()["detail"]
    assert {r["rule_id"] for r in client.get("/api/rules").json()} == before


def test_naming_one_global_row_id_twice_is_rejected(client):
    """`rule_id` is that table's primary key, so a repeat was a 500.

    The playbook-wide twin of the member case above, minting the same way
    before it failed. A row's id is what a `{type: "global"}` pin names, so
    two rows claiming one id are not merely a duplicate -- they are two
    different rules answering to one pin.
    """
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]
    before = {r["rule_id"] for r in client.get("/api/rules").json()}

    resp = client.put(f"/api/playbooks/{pb}/globals", json={"globals": [
        {"rule_id": "same-row", "name": "A", "guidance": "One.", "position": 0},
        {"rule_id": "same-row", "name": "B", "guidance": "Two.", "position": 1}]})

    assert resp.status_code == 422
    assert "same-row" in resp.json()["detail"]
    assert {r["rule_id"] for r in client.get("/api/rules").json()} == before
    assert client.get(f"/api/playbooks/{pb}/globals").json() == []


def test_a_member_rule_id_wins_over_inline_guidance(client):
    """The explicit link is the statement of intent; the text is legacy."""
    policy_id = _policy(client, "p_a", "A")
    rule_id = _rule(client, "Budget", "Stay within budget.")
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]

    client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": policy_id, "position": 0, "fires_on": False,
         "rule_id": rule_id, "guidance": "Something else entirely."}]})

    member = client.get(f"/api/playbooks/{pb}/states").json()["members"][0]
    assert member["rule_id"] == rule_id
    assert member["guidance"] == "Stay within budget."


def test_an_empty_rule_ref_id_falls_through_to_the_text(client):
    """The playbook-wide twin of the member case below.

    `rule_ref_id: ""` is falsy, so it skips both the 422 validation and the
    linked branch and lands on the text alias -- the same path as omitting
    the field. Task 10 made it behave that way deliberately, matching the
    member side, but only the member side was pinned; a refactor to
    `is not None` would turn one into a 422 and leave the other alone.
    """
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]

    response = client.put(f"/api/playbooks/{pb}/globals", json={"globals": [
        {"name": "House style", "guidance": "Be brief.", "position": 0,
         "apply_to_all": True, "rule_ref_id": ""}]})

    assert response.status_code == 200
    row = client.get(f"/api/playbooks/{pb}/globals").json()[0]
    assert row["guidance"] == "Be brief."
    assert row["rule_ref_id"]  # resolved through the text alias, not left unlinked


def test_an_empty_rule_id_falls_through_to_the_text(client):
    """`rule_id: ""` means absent, not "link to nothing".

    Empty string is falsy, so it skips both the 422 validation and the
    linked branch and lands on the text alias -- the same path as omitting
    the field. Defensible, but nothing pinned it, so a refactor could turn
    it into a 422 or a hard error with no test noticing.
    """
    policy_id = _policy(client, "p_a", "A")
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]

    response = client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": policy_id, "position": 0, "fires_on": False,
         "rule_id": "", "guidance": "Stay within budget."}]})

    assert response.status_code == 200
    member = client.get(f"/api/playbooks/{pb}/states").json()["members"][0]
    assert member["guidance"] == "Stay within budget."
    assert member["rule_id"]  # resolved through the text alias, not left unlinked


def test_editing_a_rule_changes_what_the_globals_endpoint_returns(client):
    """The inline guidance column is a stale display copy (R-17).

    A rule edited through the rules API reaches the assistant, because the
    loader resolves through the link. If the editor kept showing the old
    text, a user would reasonably conclude the edit had failed.
    """
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]
    client.put(f"/api/playbooks/{pb}/globals", json={"globals": [
        {"name": "Escalate", "guidance": "Call it out.", "position": 0,
         "apply_to_all": True}]})

    rule_id = client.get(f"/api/playbooks/{pb}/globals").json()[0]["rule_ref_id"]
    assert client.put(f"/api/rules/{rule_id}",
                      json={"guidance": "Escalate to a human."}).status_code == 200

    got = client.get(f"/api/playbooks/{pb}/globals").json()
    assert got[0]["guidance"] == "Escalate to a human."


def test_setting_globals_returns_the_resolved_text(client):
    """The PUT's own reply resolves the same way its GET does."""
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]
    rule_id = _rule(client, "Escalate", "Escalate to a human.")

    body = client.put(f"/api/playbooks/{pb}/globals", json={"globals": [
        {"name": "Escalate", "guidance": "Escalate to a human.", "position": 0,
         "apply_to_all": True}]}).json()

    assert body[0]["guidance"] == "Escalate to a human."
    assert body[0]["rule_ref_id"] == rule_id


def test_states_reports_rule_names_alongside_the_guidance(client):
    """Nodes are labelled by rule name, which only the API can supply."""
    first = _policy(client, "p_a", "A")
    second = _policy(client, "p_b", "B")
    rule_a = _rule(client, "Rule_A", "Stay within budget.")
    rule_b = _rule(client, "Rule_B", "Avoid the allergen.")
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]
    client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": first, "position": 0, "fires_on": False, "rule_id": rule_a},
        {"policy_id": second, "position": 1, "fires_on": False, "rule_id": rule_b}]})

    behaviours = client.get(f"/api/playbooks/{pb}/states").json()["behaviours"]
    by_rules = {tuple(b["rules"]): b["rule_names"] for b in behaviours}

    assert by_rules[("Stay within budget.", "Avoid the allergen.")] == [
        "Rule_A", "Rule_B"]
    assert by_rules[("Stay within budget.",)] == ["Rule_A"]
    assert by_rules[()] == []


def test_trace_nodes_report_rule_names(tmp_path, monkeypatch):
    """The trace graph labels the same nodes, so it needs the same names."""
    db_path = str(tmp_path / "trace_names.db")
    monkeypatch.setenv("DATABASE_PATH", db_path)
    asyncio.run(_seed_traced_member(db_path))

    with TestClient(create_app()) as client:
        nodes = client.get("/api/playbooks/pb1/trace?session_id=none").json()["nodes"]

    by_name = {n["name"]: n for n in nodes}
    assert by_name["R."]["rule_names"] == ["R"]
    assert by_name["(no guidance)"]["rule_names"] == []


def test_saving_the_globals_editor_after_a_library_edit_keeps_the_edit(client):
    """The editor round-trip must not revert the edit it was meant to show.

    The editor loads `guidance` from GET /globals and PUTs the whole set
    back on Save. If the GET hands back the pre-edit text, Save writes that
    text again: the library edit is silently reverted for this playbook,
    and text no rule carries any more mints a duplicate -- leaving the rule
    the user actually edited at usage zero, where the delete guard lets it
    be removed. Asserting the resolved text alone would pass on a fix to
    the GET that still re-mints, so the rule count is pinned too.
    """
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]
    client.put(f"/api/playbooks/{pb}/globals", json={"globals": [
        {"name": "House style", "guidance": "Be brief.", "position": 0,
         "apply_to_all": True}]})
    rule_id = client.get(f"/api/playbooks/{pb}/globals").json()[0]["rule_ref_id"]
    client.put(f"/api/rules/{rule_id}", json={"guidance": "Be concise and warm."})
    rule_count = len(client.get("/api/rules").json())

    reopened = client.get(f"/api/playbooks/{pb}/globals").json()
    assert client.put(f"/api/playbooks/{pb}/globals",
                      json={"globals": reopened}).status_code == 200

    states = client.get(f"/api/playbooks/{pb}/states").json()
    assert states["behaviours"][0]["rules"] == ["Be concise and warm."]

    rules = client.get("/api/rules").json()
    assert len(rules) == rule_count
    assert [r["usage_count"] for r in rules if r["rule_id"] == rule_id] == [1]


def test_a_playbook_wide_rule_can_name_a_library_rule(client):
    """Playbook-wide rules draw from the library, by id rather than by text.

    Matching a rule on its resolved text works only while the two agree: a
    rule edited between the editor's load and its save no longer matches the
    text the editor holds, and the save mints a duplicate instead of keeping
    the link. Naming the rule outright turns that coincidence into a
    guarantee -- and it is the only way to say which rule a row uses once
    the inline column is gone.
    """
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]
    rule_id = _rule(client, "Escalate", "Escalate to a human.")
    rule_count = len(client.get("/api/rules").json())

    body = client.put(f"/api/playbooks/{pb}/globals", json={"globals": [
        {"name": "Escalate", "guidance": "", "position": 0,
         "apply_to_all": True, "rule_ref_id": rule_id}]}).json()

    assert body[0]["rule_ref_id"] == rule_id
    assert body[0]["guidance"] == "Escalate to a human."
    # Named, not re-derived: nothing new was minted from the empty text.
    assert len(client.get("/api/rules").json()) == rule_count

    assert client.put(f"/api/rules/{rule_id}",
                      json={"guidance": "Escalate now."}).status_code == 200
    got = client.get(f"/api/playbooks/{pb}/globals").json()
    assert got[0]["guidance"] == "Escalate now."


def test_a_global_rule_ref_id_wins_over_inline_guidance(client):
    """The explicit link is the statement of intent; the text is legacy.

    Same rule as a member's `rule_id`: text sent beside a named rule is
    ignored rather than rewriting the rule or minting a second one.
    """
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]
    rule_id = _rule(client, "Escalate", "Escalate to a human.")
    rule_count = len(client.get("/api/rules").json())

    body = client.put(f"/api/playbooks/{pb}/globals", json={"globals": [
        {"name": "Escalate", "guidance": "Something else entirely.", "position": 0,
         "apply_to_all": True, "rule_ref_id": rule_id}]}).json()

    assert body[0]["rule_ref_id"] == rule_id
    assert body[0]["guidance"] == "Escalate to a human."
    assert len(client.get("/api/rules").json()) == rule_count
    assert client.get(f"/api/rules/{rule_id}").json()["guidance"] == (
        "Escalate to a human.")


def test_naming_a_missing_rule_in_the_globals_is_refused(client):
    """An unlinked playbook-wide rule contributes nothing, silently.

    Saving it unlinked would look like success and remove the guidance, so
    the id is checked instead -- the same refusal a member's rule_id gets.
    """
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]

    resp = client.put(f"/api/playbooks/{pb}/globals", json={"globals": [
        {"name": "Escalate", "guidance": "Escalate to a human.", "position": 0,
         "apply_to_all": True, "rule_ref_id": "no-such-rule"}]})

    assert resp.status_code == 422
    assert client.get(f"/api/playbooks/{pb}/globals").json() == []


def test_resaving_the_globals_keeps_a_state_pinned_to_a_playbook_wide_rule(client):
    """A globals save must not orphan the pins that name its rows (R-18).

    `playbook_global_rules.rule_id` is what a `{type: "global"}` ref points
    at, and the PUT replaces the whole set -- so a save that omits the id
    mints a fresh one and every pin naming the old id resolves to nothing.
    An unrelated edit in the globals pane would quietly drop guidance the
    user pinned to one specific state, with nothing reporting it.
    """
    policy_id = _policy(client, "p_a", "A")
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]
    client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": policy_id, "position": 0, "fires_on": False, "guidance": "R."}]})
    client.put(f"/api/playbooks/{pb}/globals", json={"globals": [
        {"name": "Escalate", "guidance": "Escalate to a human.", "position": 0,
         "apply_to_all": False}]})

    reopened = client.get(f"/api/playbooks/{pb}/globals").json()
    pinned_id = reopened[0]["rule_id"]
    client.put(f"/api/playbooks/{pb}/states/{policy_id}=F", json={
        "rule_refs": [{"type": "global", "rule_id": pinned_id}],
        "flagged": True, "label": "Stop"})

    # The editor reopens the pane, changes something unrelated, and saves.
    reopened[0]["name"] = "Escalate to a human"
    assert client.put(f"/api/playbooks/{pb}/globals",
                      json={"globals": reopened}).status_code == 200

    assert client.get(f"/api/playbooks/{pb}/globals").json()[0]["rule_id"] == pinned_id
    states = client.get(f"/api/playbooks/{pb}/states").json()
    pinned = [
        s for b in states["behaviours"] for s in b["states"]
        if s["state_key"] == f"{policy_id}=F"
    ]
    assert [b["rules"] for b in states["behaviours"]
            if any(s["state_key"] == f"{policy_id}=F" for s in b["states"])] == [
        ["Escalate to a human."]]
    assert pinned[0]["rule_refs"] == [{"type": "global", "rule_id": pinned_id}]
