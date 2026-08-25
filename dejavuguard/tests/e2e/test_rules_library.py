"""
E2E tests: the shared rule library.

Guidance used to be an inline string, retyped into every playbook that wanted
it. It now lives once in a shared library, and members and playbook-wide rows
name the rule they draw from. Three claims carry that change end to end, and
this module pins each of them:

* a policy can be added to a playbook through the guided ``+ Add policy``
  flow, and the playbook it lands in stops offering it a second time;
* one rule attached to *two* playbooks and edited *once* changes the resolved
  guidance of both -- the entire reason a library exists rather than a copy
  per playbook, and the only claim no single-playbook test can make;
* a state-graph node names the rules that apply in it, rather than reprinting
  their text.

Everything here is asserted through the UI. Where a claim is about what the
server now holds -- an added member, an edited rule reaching a playbook that
was never opened -- it is asserted against the API as well, because a screen
can agree with itself while nothing was written.

Shared development database
---------------------------
Nothing below assumes the database is empty, and nothing deletes a row this
module did not create. Every name it writes carries :data:`PREFIX`, the
fixture sweeps exactly those names before and after the run, and the sweep
runs in a ``finally`` so a failing test still leaves the database as it found
it. Rules are swept only after the playbooks holding them are gone: the API
refuses to delete a rule a playbook still names, which is itself the
behaviour that makes the sweep safe.
"""

from __future__ import annotations

import re

import httpx
import pytest
from playwright.sync_api import Page, expect

API = "http://localhost:8000/api"

#: Everything this module creates is named with this prefix so a crashed run
#: can be swept clean on the next one -- the backend uses a shared dev DB.
PREFIX = "e2e-ruleslib"

PREDICATE_ID = "e2e_ruleslib_u"

#: The policy count is capped server-side, and a developer's database already
#: holds policies. Two is what this module needs; the fixture refuses to run
#: rather than fail somewhere less legible if that would not fit.
POLICIES_NEEDED = 2
MAX_POLICY_COUNT = 50

#: Rule names, all of them short enough that the graph draws them whole --
#: past 28 characters a node elides the middle, and an elided name is not the
#: name the graph test is looking for.
ADDED_RULE = f"{PREFIX} added"
SHARED_RULE = f"{PREFIX} shared"
GRAPH_RULE = f"{PREFIX} graph"

ADDED_GUIDANCE = "Answer the alpha question first."

#: The shared rule before and after the single edit. Neither is a substring of
#: the other, so "changed" cannot be satisfied by text that merely contains
#: the old string.
SHARED_BEFORE = "Ask for the alpha ticket number."
SHARED_AFTER = "Escalate to a duty manager instead."

#: Under 40 characters, so the behaviour it names is not truncated and the
#: node's test id is the guidance verbatim. It shares no words with
#: :data:`GRAPH_RULE`, which is what lets the graph test tell a node showing
#: the rule's *name* from one reprinting its *text*.
GRAPH_GUIDANCE = "Confirm the seat map before booking."


def _api() -> httpx.Client:
    return httpx.Client(base_url=API, timeout=60.0)


def _sweep(client: httpx.Client) -> None:
    """Delete anything this module created, here or in an earlier run.

    Playbooks first: a rule a playbook still names cannot be deleted, and the
    rules created through the UI are exactly the ones those playbooks hold.
    """
    for playbook in client.get("/playbooks").json():
        if playbook["name"].startswith(PREFIX):
            client.delete(f"/playbooks/{playbook['playbook_id']}")
    for rule in client.get("/rules").json():
        if rule["name"].startswith(PREFIX):
            client.delete(f"/rules/{rule['rule_id']}")
    for policy in client.get("/policies").json():
        if policy["name"].startswith(PREFIX):
            client.delete(f"/policies/{policy['policy_id']}")
    client.delete(f"/propositions/{PREDICATE_ID}")


def _rule_named(client: httpx.Client, name: str) -> dict | None:
    """The library's rule of that name, or None -- with its usage count."""
    for rule in client.get("/rules").json():
        if rule["name"] == name:
            return rule
    return None


def _member_guidance(client: httpx.Client, playbook_id: str, policy_id: str) -> str:
    """The guidance the server resolves for one member, right now.

    Read from ``/states`` rather than from the rules table: this is the text
    the playbook would actually inject, which is the claim a shared edit has
    to satisfy. It arrives resolved through the member's rule id, so a member
    that quietly detached onto a copy shows its own stale text here.
    """
    states = client.get(f"/playbooks/{playbook_id}/states").json()
    for member in states["members"]:
        if member["policy_id"] == policy_id:
            return member["guidance"]
    raise AssertionError(f"playbook {playbook_id} has no member {policy_id}")


@pytest.fixture(scope="module")
def library_env():
    """One predicate, two policies and four playbooks, over the API.

    Each test gets a playbook of its own so none of them has to run after
    another, and the two that the shared-rule test needs are created empty:
    that test's whole subject is the rule it attaches to both of them
    through the UI.
    """
    with _api() as client:
        _sweep(client)

        existing = len(client.get("/policies").json())
        assert existing + POLICIES_NEEDED <= MAX_POLICY_COUNT, (
            f"{existing} policies already exist and the server caps them at "
            f"{MAX_POLICY_COUNT}; this module needs {POLICIES_NEEDED} more. "
            "Remove some policies before running the e2e suite."
        )
        for name in (ADDED_RULE, SHARED_RULE, GRAPH_RULE):
            assert _rule_named(client, name) is None, (
                f"a rule named {name!r} survived the sweep -- a playbook "
                "outside this module holds it, and this module needs the name"
            )

        created = client.post(
            "/propositions",
            json={
                "prop_id": PREDICATE_ID,
                "description": "the user mentions the alpha keyword",
                "role": "user",
                "arity": 0,
                "arg_descriptions": [],
            },
        )
        assert created.status_code == 201, created.text

        policies: dict[str, str] = {}
        for keyword in ("alpha", "beta"):
            response = client.post(
                "/policies",
                json={
                    "name": f"{PREFIX} {keyword}",
                    "formula_str": f"! {PREDICATE_ID}",
                },
            )
            assert response.status_code == 201, response.text
            policies[keyword] = response.json()["policy_id"]

        names: dict[str, str] = {}

        def make(suffix: str) -> str:
            name = f"{PREFIX} {suffix}"
            response = client.post("/playbooks", json={"name": name})
            assert response.status_code == 201, response.text
            playbook_id = response.json()["playbook_id"]
            names[playbook_id] = name
            return playbook_id

        add_target = make("add")
        first = make("first")
        second = make("second")
        graph = make("graph")

        # The graph playbook is the one case set up entirely over the API:
        # its subject is what a node *displays*, so building it by clicking
        # would leave a failure ambiguous between the two.
        rule = client.post(
            "/rules", json={"name": GRAPH_RULE, "guidance": GRAPH_GUIDANCE}
        )
        assert rule.status_code == 201, rule.text
        graph_rule_id = rule.json()["rule_id"]
        saved = client.put(
            f"/playbooks/{graph}/members",
            json={
                "members": [
                    {
                        "policy_id": policies["alpha"],
                        "position": 0,
                        "fires_on": False,
                        "guidance": "",
                        "rule_id": graph_rule_id,
                    }
                ]
            },
        )
        assert saved.status_code == 200, saved.text

        env = {
            "policies": policies,
            "names": names,
            "add_target": add_target,
            "first": first,
            "second": second,
            "graph": graph,
            "graph_rule_id": graph_rule_id,
        }
        try:
            yield env
        finally:
            _sweep(client)


def _open_playbooks(page: Page) -> None:
    """The playbook list, from wherever in the section we already are.

    The nav link is a route, and the editor and the library are states inside
    the ``/playbooks`` route -- so clicking "Playbooks" while one of them is
    open navigates nowhere and changes nothing. Each test here opens more
    than one playbook, so the way back out is the screen's own Back control.
    """
    page.click('[data-testid="nav-playbooks"]')
    for back in ("playbook-editor-back", "rule-library-back"):
        control = page.get_by_test_id(back)
        if control.count():
            control.click()
    expect(page.get_by_test_id("playbooks-view")).to_be_visible()


def _open_editor(page: Page, env: dict, playbook_id: str) -> None:
    _open_playbooks(page)
    page.get_by_test_id(f"playbook-card-{playbook_id}").get_by_role(
        "button", name=env["names"][playbook_id], exact=True
    ).click()
    expect(page.get_by_test_id("playbook-editor")).to_be_visible()


def _open_library(page: Page) -> None:
    _open_playbooks(page)
    page.get_by_test_id("open-rule-library").click()
    expect(page.get_by_test_id("rule-library")).to_be_visible()


def _pick_policy(page: Page, policy_id: str) -> None:
    """Steps 1 and 2 of the add flow: this policy, when it is violated."""
    page.get_by_test_id("add-policy").click()
    expect(page.get_by_test_id("add-policy-modal")).to_be_visible()
    expect(page.get_by_test_id("add-policy-step")).to_contain_text("Step 1 of 3")

    page.get_by_test_id(f"policy-option-{policy_id}").click()
    expect(page.get_by_test_id("fires-on-step")).to_be_visible()
    page.get_by_test_id("fires-on-violated").click()
    page.get_by_test_id("fires-on-next").click()
    expect(page.get_by_test_id("rule-step")).to_be_visible()


def _save_members(page: Page) -> None:
    page.get_by_test_id("save-members").click()
    expect(page.get_by_test_id("members-save-report")).to_be_visible()
    expect(page.get_by_test_id("members-save-error")).to_have_count(0)


class TestAddPolicyFlow:
    """`+ Add policy` puts a member on the playbook, once."""

    def test_added_policy_is_saved_and_then_offered_no_second_time(
        self, app_page: Page, library_env
    ):
        """Add a policy through the modal, save it, reopen the modal.

        Both halves matter and neither implies the other. A flow that added
        the row to the screen without writing it would pass a reopen check
        driven by unsaved component state, and a flow that wrote the member
        but kept offering the policy would let a user add it twice -- the
        state space doubles and the second copy silently wins.

        So: the member is read back from the server, and the reopened modal
        is checked against a playbook whose membership came back from it.
        """
        policy_id = library_env["policies"]["alpha"]
        _open_editor(app_page, library_env, library_env["add_target"])
        expect(app_page.get_by_test_id("no-members")).to_be_visible()

        _pick_policy(app_page, policy_id)
        app_page.get_by_test_id("rule-mode-create").check()
        app_page.get_by_test_id("new-rule-name").fill(ADDED_RULE)
        app_page.get_by_test_id("new-rule-guidance").fill(ADDED_GUIDANCE)
        app_page.get_by_test_id("add-policy-confirm").click()

        expect(app_page.get_by_test_id("add-policy-modal")).to_have_count(0)
        row = app_page.get_by_test_id(f"member-row-{policy_id}")
        expect(row).to_be_visible()
        expect(app_page.get_by_test_id(f"member-rule-{policy_id}")).to_have_text(
            ADDED_RULE
        )

        _save_members(app_page)

        with _api() as client:
            states = client.get(
                f"/playbooks/{library_env['add_target']}/states"
            ).json()
            assert [m["policy_id"] for m in states["members"]] == [policy_id], (
                "the added member did not reach the server"
            )
            member = states["members"][0]
            assert member["fires_on"] is False
            assert member["guidance"] == ADDED_GUIDANCE

            rule = _rule_named(client, ADDED_RULE)
            assert rule is not None, "the modal did not create the rule"
            assert member["rule_id"] == rule["rule_id"], (
                "the member holds guidance text of its own instead of naming "
                "the rule the modal created"
            )

        # Reopen: the policy is still listed -- "what do I already have?" is
        # otherwise only answerable by closing the dialog -- but inert.
        app_page.get_by_test_id("add-policy").click()
        expect(app_page.get_by_test_id("add-policy-modal")).to_be_visible()
        option = app_page.get_by_test_id(f"policy-option-{policy_id}")
        expect(option).to_be_visible()
        expect(option).to_have_attribute("aria-disabled", "true")
        expect(option).to_contain_text("already in this playbook")

        # Forced past the actionability check on purpose: Playwright already
        # refuses to click an `aria-disabled` option, and a test that stopped
        # there would prove the attribute is set, not that the flow honours
        # it. This dispatches the click the attribute is supposed to make
        # meaningless.
        option.click(force=True)
        expect(app_page.get_by_test_id("add-policy-step")).to_contain_text(
            "Step 1 of 3"
        )
        expect(app_page.get_by_test_id("fires-on-step")).to_have_count(0)

        # The other policy is unaffected: greying is per policy, not a dead
        # list.
        expect(
            app_page.get_by_test_id(f"policy-option-{library_env['policies']['beta']}")
        ).to_have_attribute("aria-disabled", "false")


class TestSharedRuleReachesEveryPlaybook:
    """The claim the library exists for."""

    def test_editing_one_rule_changes_both_playbooks_that_use_it(
        self, app_page: Page, library_env
    ):
        """One rule, two playbooks, one edit, both playbooks changed.

        This is the whole value proposition, and nothing smaller proves it.
        A single playbook cannot tell a shared rule from a private copy: both
        show the edit. Only a second playbook that was never opened during
        the edit can, and only if its guidance is read back from the server
        rather than from a screen that might be echoing the edit it just made.

        The rule is created in the first playbook and *reused* in the second
        through the modal's own list, so the linkage under test is the one a
        user would create.
        """
        policy_id = library_env["policies"]["alpha"]
        first, second = library_env["first"], library_env["second"]

        # -- first playbook: create the rule -----------------------------
        _open_editor(app_page, library_env, first)
        _pick_policy(app_page, policy_id)
        app_page.get_by_test_id("rule-mode-create").check()
        app_page.get_by_test_id("new-rule-name").fill(SHARED_RULE)
        app_page.get_by_test_id("new-rule-guidance").fill(SHARED_BEFORE)
        app_page.get_by_test_id("add-policy-confirm").click()
        expect(app_page.get_by_test_id(f"member-row-{policy_id}")).to_be_visible()
        _save_members(app_page)

        with _api() as client:
            rule = _rule_named(client, SHARED_RULE)
            assert rule is not None, "the modal did not create the shared rule"
            rule_id = rule["rule_id"]

        # -- second playbook: reuse the same rule ------------------------
        _open_editor(app_page, library_env, second)
        _pick_policy(app_page, policy_id)
        app_page.get_by_test_id("rule-mode-reuse").check()
        app_page.get_by_test_id("rule-search").fill(SHARED_RULE)
        option = app_page.get_by_test_id(f"rule-option-{rule_id}")
        expect(option).to_be_visible()
        option.click()
        app_page.get_by_test_id("add-policy-confirm").click()
        expect(app_page.get_by_test_id(f"member-rule-{policy_id}")).to_have_text(
            SHARED_RULE
        )
        _save_members(app_page)

        with _api() as client:
            for playbook_id in (first, second):
                assert (
                    _member_guidance(client, playbook_id, policy_id) == SHARED_BEFORE
                ), "the two playbooks did not start from the same text"
            assert _rule_named(client, SHARED_RULE)["usage_count"] == 2, (
                "the second playbook took a copy instead of the shared rule"
            )

        # -- the library: edit it once -----------------------------------
        _open_library(app_page)
        app_page.get_by_test_id(f"rule-edit-{rule_id}").click()
        expect(app_page.get_by_test_id("rule-editor")).to_be_visible()
        # The blast radius is stated before the edit is made, and states it
        # correctly: two playbooks, not one and not "all of them".
        expect(app_page.get_by_test_id("rule-shared-warning")).to_contain_text(
            "used by 2 playbooks"
        )
        app_page.get_by_test_id("rule-editor-guidance").fill(SHARED_AFTER)
        expect(app_page.get_by_test_id("rule-editor-save")).to_have_text(
            "Save for 2 playbooks"
        )
        app_page.get_by_test_id("rule-editor-save").click()
        expect(app_page.get_by_test_id("rule-editor")).to_have_count(0)
        expect(app_page.get_by_test_id(f"rule-row-{rule_id}")).to_contain_text(
            SHARED_AFTER
        )

        # -- both playbooks now resolve to the new text ------------------
        with _api() as client:
            for playbook_id, label in ((first, "first"), (second, "second")):
                resolved = _member_guidance(client, playbook_id, policy_id)
                assert resolved == SHARED_AFTER, (
                    f"the {label} playbook still resolves to {resolved!r}: the "
                    "edit did not reach it, so its guidance is a copy rather "
                    "than the shared rule"
                )

        for playbook_id in (first, second):
            _open_editor(app_page, library_env, playbook_id)
            expect(
                app_page.get_by_test_id(f"member-guidance-{policy_id}")
            ).to_have_value(SHARED_AFTER)
            expect(
                app_page.get_by_test_id(f"member-rule-{policy_id}")
            ).to_have_text(SHARED_RULE)
            # Still attached: a row whose text no longer matches its rule
            # warns that saving would detach it, and neither of these was
            # edited in place.
            expect(
                app_page.get_by_test_id(f"member-detached-{policy_id}")
            ).to_have_count(0)


class TestGraphNamesItsRules:
    """A node says which rules apply, by name."""

    def test_a_node_lists_the_names_of_the_rules_that_apply_in_it(
        self, app_page: Page, library_env
    ):
        """The node draws the rule's name, not the rule's text.

        A behaviour reaches the client as resolved guidance -- the engine
        never learns that rules exist -- and the node is *identified* by that
        text. Drawing it again would be the pre-library rendering and would
        still look plausible, so the discriminating assertion is the negative
        one: the text that names the node must not appear inside it.
        """
        _open_editor(app_page, library_env, library_env["graph"])
        app_page.get_by_test_id("states-view-graph").click()
        expect(app_page.get_by_test_id("playbook-graph")).to_be_visible()

        node = app_page.get_by_test_id(f"node-{GRAPH_GUIDANCE}")
        expect(node).to_be_visible()
        expect(node).to_contain_text("1 rule")

        # Both halves are asserted against the drawn `<text>` captions rather
        # than the node's subtree, and that is not a detail. The node's
        # `<title>` -- the hover text -- carries the behaviour name *and* the
        # rule names, so a subtree-level "contains the name" passes even when
        # the drawn caption is the guidance text, and a subtree-level "does
        # not contain the text" fails on a node that is drawing exactly what
        # it should. Only the captions distinguish the two.
        captions = node.locator("text").all_text_contents()
        assert f"· {GRAPH_RULE}" in captions, (
            f"the node does not name the rule that applies in it: {captions}"
        )
        assert not any(GRAPH_GUIDANCE in caption for caption in captions), (
            f"the node reprints its rule's text instead of naming it: {captions}"
        )

        # The hover text keeps every name in full, which is what the drawn
        # caption's 28-character clip is allowed to rely on.
        expect(node.locator("title")).to_contain_text(f"· {GRAPH_RULE}")

        # The rules a node applies are named for a screen reader too, since
        # the drawn list is clipped and an SVG caption is not a label.
        expect(node).to_have_attribute(
            "aria-label", re.compile(rf"^Rules applied: {re.escape(GRAPH_RULE)}\.")
        )

        # The control: the sibling behaviour applies no rule and says so,
        # which is what stops "contains the name" being satisfiable by a node
        # that prints every rule in the playbook.
        silent = app_page.get_by_test_id("node-(no guidance)")
        expect(silent).to_be_visible()
        expect(silent).to_contain_text("no rules")
        expect(silent).not_to_contain_text(GRAPH_RULE)
