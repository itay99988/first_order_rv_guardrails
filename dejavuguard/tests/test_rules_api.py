"""The rules library API.

Rules are shared, so the two numbers that matter are the usage count -- how
many playbooks an edit or a delete would reach -- and the refusal built on
it. A delete that trusted a stale count would orphan a member while
reporting that it had protected it, so the in-use tests below go through the
real write path rather than seeding the link by hand.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.routers import chat


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "rules-api.db"))
    with TestClient(create_app()) as c:
        yield c


def _create(client: TestClient, name: str, guidance: str):
    return client.post("/api/rules", json={"name": name, "guidance": guidance})


def _rule_in_use(client: TestClient, guidance: str = "Stay within budget.") -> str:
    """Attach a rule to a playbook member the way the UI does, and return it."""
    client.post("/api/propositions", json={
        "prop_id": "p_a", "description": "a", "role": "user"})
    policy_id = client.post("/api/policies", json={
        "name": "A", "formula_str": "p_a"}).json()["policy_id"]
    pb = client.post("/api/playbooks", json={"name": "Budget"}).json()["playbook_id"]
    client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": policy_id, "position": 0, "fires_on": True,
         "guidance": guidance}]})

    attached = [r for r in client.get("/api/rules").json() if r["guidance"] == guidance]
    assert len(attached) == 1, "the member save should have minted exactly one rule"
    return attached[0]["rule_id"]


# --- Round trip ---


def test_create_then_list_a_rule(client):
    created = _create(client, "Rule_Budget", "Stay within budget.")
    assert created.status_code == 201
    assert created.json()["name"] == "Rule_Budget"

    listed = client.get("/api/rules").json()
    assert [(r["name"], r["guidance"]) for r in listed] == [
        ("Rule_Budget", "Stay within budget.")]


def test_rules_are_listed_with_their_usage_count(client):
    """The number the edit warning and the delete refusal both read."""
    unused = _create(client, "Rule_Unused", "Never attached.").json()["rule_id"]
    used = _rule_in_use(client)

    by_id = {r["rule_id"]: r for r in client.get("/api/rules").json()}
    assert by_id[used]["usage_count"] == 1
    assert by_id[unused]["usage_count"] == 0


def test_read_one_rule(client):
    rule_id = _create(client, "Rule_Budget", "Stay within budget.").json()["rule_id"]

    got = client.get(f"/api/rules/{rule_id}")
    assert got.status_code == 200
    assert got.json()["guidance"] == "Stay within budget."


def test_reading_a_rule_that_does_not_exist_is_a_404(client):
    assert client.get("/api/rules/ghost").status_code == 404


def test_update_a_rule(client):
    rule_id = _create(client, "Rule_Budget", "Stay within budget.").json()["rule_id"]

    updated = client.put(f"/api/rules/{rule_id}",
                         json={"name": "Rule_Spend", "guidance": "Ask first."})

    assert updated.status_code == 200
    assert updated.json()["name"] == "Rule_Spend"
    assert client.get(f"/api/rules/{rule_id}").json()["guidance"] == "Ask first."


def test_updating_a_rule_that_does_not_exist_is_a_404(client):
    assert client.put("/api/rules/ghost", json={"guidance": "x"}).status_code == 404


def test_an_edit_reaches_every_playbook_holding_the_rule(client):
    """What sharing is for -- and what the usage count is warning about."""
    rule_id = _rule_in_use(client)
    pb = client.get("/api/playbooks").json()[0]["playbook_id"]

    client.put(f"/api/rules/{rule_id}", json={"guidance": "Ask before overspending."})

    states = client.get(f"/api/playbooks/{pb}/states").json()
    assert [m["guidance"] for m in states["members"]] == ["Ask before overspending."]


# --- Names are unique, and the API says so rather than raising ---


def test_creating_a_rule_with_a_taken_name_is_a_409(client):
    _create(client, "Rule_Budget", "Stay within budget.")

    clash = _create(client, "Rule_Budget", "Different text.")

    assert clash.status_code == 409
    assert "Rule_Budget" in clash.json()["detail"]


def test_renaming_a_rule_onto_a_taken_name_is_a_409(client):
    _create(client, "Rule_Budget", "A")
    other = _create(client, "Rule_Tone", "B").json()["rule_id"]

    clash = client.put(f"/api/rules/{other}", json={"name": "Rule_Budget"})

    assert clash.status_code == 409
    assert client.get(f"/api/rules/{other}").json()["name"] == "Rule_Tone"


def test_a_rule_needs_a_name(client):
    assert _create(client, "   ", "Stay within budget.").status_code == 422


# --- Delete ---


def test_deleting_a_rule_in_use_is_refused_and_says_how_many(client):
    """Never silently orphan a member: the refusal carries the count."""
    rule_id = _rule_in_use(client)

    refused = client.delete(f"/api/rules/{rule_id}")

    assert refused.status_code == 409
    assert "1" in refused.json()["detail"]
    assert client.get(f"/api/rules/{rule_id}").status_code == 200


def test_a_refused_delete_leaves_the_member_resolving_as_before(client):
    """The guard has to have actually protected what it says it protected."""
    rule_id = _rule_in_use(client)
    pb = client.get("/api/playbooks").json()[0]["playbook_id"]

    client.delete(f"/api/rules/{rule_id}")

    states = client.get(f"/api/playbooks/{pb}/states").json()
    assert [m["guidance"] for m in states["members"]] == ["Stay within budget."]


def test_a_rule_nothing_uses_can_be_deleted(client):
    """Every guidance edit mints a rule the old text no longer holds.

    Without this the library fills with orphans and offers no way to clear
    them, so a zero usage count has to mean deletable.
    """
    rule_id = _rule_in_use(client, "First wording.")
    pb = client.get("/api/playbooks").json()[0]["playbook_id"]
    policy_id = client.get("/api/policies").json()[0]["policy_id"]
    client.put(f"/api/playbooks/{pb}/members", json={"members": [
        {"policy_id": policy_id, "position": 0, "fires_on": True,
         "guidance": "Second wording."}]})

    by_id = {r["rule_id"]: r for r in client.get("/api/rules").json()}
    assert by_id[rule_id]["usage_count"] == 0

    assert client.delete(f"/api/rules/{rule_id}").status_code == 204
    assert client.get(f"/api/rules/{rule_id}").status_code == 404


def test_deleting_a_rule_that_does_not_exist_is_a_404(client):
    assert client.delete("/api/rules/ghost").status_code == 404


# --- Live sessions ---


@pytest.mark.parametrize("edit", ["update", "delete"])
def test_changing_a_rule_evicts_cached_monitors(client, edit):
    """A cached monitor holds a resolved snapshot of its playbook.

    Left in place it would keep injecting the old text for the rest of the
    session -- an edit that reports success and reaches nobody.
    """
    rule_id = _create(client, "Rule_Budget", "Stay within budget.").json()["rule_id"]
    chat._monitors["s1"] = object()

    if edit == "update":
        client.put(f"/api/rules/{rule_id}", json={"guidance": "Ask first."})
    else:
        client.delete(f"/api/rules/{rule_id}")

    assert "s1" not in chat._monitors
