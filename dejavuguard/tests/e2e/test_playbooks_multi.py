"""
E2E tests: playbooks with two, three and four member policies.

``test_playbooks.py`` covers a playbook with exactly one member, and a
one-member playbook is a policy with extra steps: its state space is {T, F}
and its "combination" of verdicts is a single verdict. Everything that
justifies the feature -- that a *combination* means something no single
verdict does, that states sharing a behaviour merge, that guidance composes
in member order -- only appears from two members up. This module covers that.

The load-bearing test is ``test_only_the_combination_of_both_verdicts_blocks``:
with members A and B and *only* ``A=F;B=F`` flagged, A=F alone must not block,
B=F alone must not block, and the two together must. If playbook mode ever
degraded into per-policy blocking, the first two turns would block; if the
flag stopped selecting on the whole combination, the third would not.

Independent verdicts
--------------------
A combination can only be exercised if the members can disagree. Each member
here is a Boolean predicate over its own marker word -- ``ALPHA-ON``,
``BRAVO-ON``, ``CHARLIE-ON``, ``DELTA-ON`` -- recognised by the deterministic
stub grounder on :9099 from rules added to
``scenario_runner/support/playbook_grounding.json``. One message can turn any
subset of them on, so every one of the 2^n states is reachable from a single
turn, and the fixtures below assert that they really are before anything
depends on it.

Each policy is ``! <predicate>``: the predicate grounding true means the
policy is violated, so a marker present makes that member's verdict False.
"""

from __future__ import annotations

import httpx
import pytest
from playwright.sync_api import Page, expect

from tests.e2e.test_playbooks import (
    CHAT_TIMEOUT_MS,
    _open_playbooks,
    _open_session,
    _send,
)

API = "http://localhost:8000/api"

#: Everything this module creates is named with this prefix so a crashed run
#: can be swept clean on the next one -- the backend uses a shared dev DB.
PREFIX = "e2e-pbmulti"

KEYWORDS = ("alpha", "bravo", "charlie", "delta")


def _predicate_id(keyword: str) -> str:
    return f"e2e_pbmulti_{keyword}_u"


def _drive(*on: str) -> str:
    """A user message that turns exactly ``on`` markers on and the rest off.

    The markers a message omits are what makes the states independent: the
    stub grounds a predicate true only when its own marker is in the text.
    """
    markers = ", ".join(f"{k.upper()}-ON" for k in on) if on else "nothing"
    return f"Systems check for this run: {markers}."


def _api() -> httpx.Client:
    return httpx.Client(base_url=API, timeout=60.0)


def _sweep(client: httpx.Client) -> None:
    """Delete anything a previous run of this module left behind."""
    for playbook in client.get("/playbooks").json():
        if playbook["name"].startswith(PREFIX):
            client.delete(f"/playbooks/{playbook['playbook_id']}")
    for policy in client.get("/policies").json():
        if policy["name"].startswith(PREFIX):
            client.delete(f"/policies/{policy['policy_id']}")
    for keyword in KEYWORDS:
        client.delete(f"/propositions/{_predicate_id(keyword)}")


def _state_key(verdicts: dict[str, bool]) -> str:
    """The backend's canonical state key: sorted by policy id, never position."""
    return ";".join(
        f"{pid}={'T' if verdicts[pid] else 'F'}" for pid in sorted(verdicts)
    )


def _reverse_position_members(
    policy_ids: list[str], guidance: list[str]
) -> list[dict]:
    """Members whose positions are the exact reverse of policy-id order.

    Policy ids are uuids, so declaring positions in id order would let a
    result that sorted by id pass by accident. Reversing them means guidance
    assembled by id comes out backwards and the position tests fail.
    """
    ordered = sorted(policy_ids)
    last = len(ordered) - 1
    return [
        {
            "policy_id": policy_id,
            "position": last - index,
            "fires_on": False,
            "guidance": guidance[last - index],
        }
        for index, policy_id in enumerate(ordered)
    ]


@pytest.fixture(scope="module")
def multi_env():
    """Four independent predicates, four policies, and six playbooks.

    Each playbook isolates one claim, so no test has to leave another test's
    fixture in a different shape than it found it -- except the collapse
    playbook, which exists to be edited.
    """
    with _api() as client:
        original_settings = client.get("/settings").json()
        _sweep(client)

        settings = client.get("/settings").json()
        settings["grounding"]["provider"] = "custom"
        settings["grounding"]["base_url"] = "http://localhost:9099/v1"
        settings["grounding"]["model"] = "stub-grounder"
        assert client.put("/settings", json=settings).status_code == 200

        policies: dict[str, str] = {}
        for keyword in KEYWORDS:
            prop_id = _predicate_id(keyword)
            created = client.post(
                "/propositions",
                json={
                    "prop_id": prop_id,
                    "description": f"the user mentions the {keyword} keyword",
                    "role": "user",
                    "arity": 0,
                    "arg_descriptions": [],
                },
            )
            assert created.status_code == 201, created.text
            policy = client.post(
                "/policies",
                json={
                    "name": f"{PREFIX} {keyword}",
                    "formula_str": f"! {prop_id}",
                },
            )
            assert policy.status_code == 201, policy.text
            policies[keyword] = policy.json()["policy_id"]

        names: dict[str, str] = {}

        def make(suffix: str, members: list[dict]) -> str:
            name = f"{PREFIX} {suffix}"
            response = client.post("/playbooks", json={"name": name})
            assert response.status_code == 201, response.text
            playbook_id = response.json()["playbook_id"]
            names[playbook_id] = name
            saved = client.put(
                f"/playbooks/{playbook_id}/members", json={"members": members}
            )
            assert saved.status_code == 200, saved.text
            return playbook_id

        def flag(playbook_id: str, key: str, label: str | None) -> None:
            response = client.put(
                f"/playbooks/{playbook_id}/states/{key}",
                json={"rule_refs": None, "flagged": True, "label": label},
            )
            assert response.status_code == 200, response.text

        # -- two members, one flagged combination -------------------------
        combo_ids = [policies["alpha"], policies["bravo"]]
        combo = make(
            "combo",
            [
                {"policy_id": policies["alpha"], "position": 0,
                 "fires_on": False, "guidance": "Alpha guidance."},
                {"policy_id": policies["bravo"], "position": 1,
                 "fires_on": False, "guidance": "Bravo guidance."},
            ],
        )
        combo_both_false = _state_key({pid: False for pid in combo_ids})
        flag(combo, combo_both_false, "Both broken")

        # -- three members, positions reversed against policy-id order ----
        three_ids = [policies[k] for k in ("alpha", "bravo", "charlie")]
        three_members = _reverse_position_members(
            three_ids, ["P0 rule.", "P1 rule.", "P2 rule."]
        )
        three = make("three", three_members)
        three_all_false = _state_key({pid: False for pid in three_ids})
        # Two states flagged, both unlabelled. Flagged, because only a blocked
        # turn reports the guidance it resolved -- a turn that passes the
        # monitor goes on to a chat model this environment has no key for.
        # Unlabelled, so each behaviour keeps its guidance-derived name.
        flag(three, three_all_false, None)
        three_partial = {pid: False for pid in three_ids}
        three_partial[policies["charlie"]] = True
        three_partial_key = _state_key(three_partial)
        flag(three, three_partial_key, None)
        three_partial_rules = [
            m["guidance"]
            for m in sorted(three_members, key=lambda m: m["position"])
            if not three_partial[m["policy_id"]]
        ]

        # -- four members, all distinct: nothing merges -------------------
        wide_ids = [policies[k] for k in KEYWORDS]
        wide = make(
            "wide",
            _reverse_position_members(
                wide_ids, ["A-rule", "B-rule", "C-rule", "D-rule"]
            ),
        )

        # -- four members, two of them silent: everything merges ----------
        merged_members = _reverse_position_members(
            wide_ids, ["A-rule", "B-rule", "", ""]
        )
        merged = make("merged", merged_members)
        split = make("split", merged_members)
        # Guiding members False, silent members True: one of the four states
        # of the "A-rule + B-rule" behaviour, identical to its three siblings
        # in everything but the flag.
        split_key = _state_key(
            {m["policy_id"]: not m["guidance"] for m in merged_members}
        )
        flag(split, split_key, "Escalate")

        # -- three members, an override pair straddling the one removed ---
        collapse_ids = [policies[k] for k in ("bravo", "charlie", "delta")]
        # The member to be removed carries no guidance, so the two states its
        # verdict distinguishes share a behaviour before the removal as well
        # as after it -- the override pair is then visible as one node whose
        # state count drops from two to one.
        collapse_members = _reverse_position_members(
            collapse_ids, ["C0.", "C1.", ""]
        )
        collapse = make("collapse", collapse_members)
        removed_id = sorted(collapse_ids)[0]
        assert not next(
            m["guidance"] for m in collapse_members if m["policy_id"] == removed_id
        )
        kept_false = {pid: False for pid in sorted(collapse_ids)[1:]}
        collapsed_key = _state_key(kept_false)
        for branch in (True, False):
            flag(collapse, _state_key(kept_false | {removed_id: branch}), "Pair")

        env = {
            "policies": policies,
            "names": names,
            "combo": combo,
            "combo_both_false": combo_both_false,
            "combo_alpha_false": _state_key(
                {policies["alpha"]: False, policies["bravo"]: True}
            ),
            "combo_bravo_false": _state_key(
                {policies["alpha"]: True, policies["bravo"]: False}
            ),
            "three": three,
            "three_all_false": three_all_false,
            "three_partial_key": three_partial_key,
            "three_partial_rules": three_partial_rules,
            "wide": wide,
            "merged": merged,
            "split": split,
            "split_key": split_key,
            "collapse": collapse,
            "collapse_removed": removed_id,
            "collapse_collapsed_key": collapsed_key,
        }
        try:
            yield env
        finally:
            _sweep(client)
            client.put("/settings", json=original_settings)


@pytest.fixture()
def session_factory():
    """Chat sessions created over the API and removed afterwards."""
    created: list[str] = []

    def make(playbook_id: str) -> str:
        with _api() as client:
            session_id = client.post("/chat/sessions").json()["session_id"]
            created.append(session_id)
            response = client.patch(
                f"/chat/sessions/{session_id}/monitoring",
                json={"mode": "playbook", "playbook_id": playbook_id},
            )
            assert response.status_code == 200, response.text
        return session_id

    yield make

    with _api() as client:
        for session_id in created:
            client.delete(f"/chat/sessions/{session_id}")


def _open_editor(page: Page, env: dict, playbook_id: str) -> None:
    _open_playbooks(page)
    page.get_by_test_id(f"playbook-card-{playbook_id}").get_by_role(
        "button", name=env["names"][playbook_id], exact=True
    ).click()
    expect(page.locator('[data-testid="playbook-editor"]')).to_be_visible()
    expect(page.locator('[data-testid="playbook-states"]')).to_be_visible()


def _turn(page: Page, session_factory, playbook_id: str, message: str):
    """One user turn in a fresh session on ``playbook_id``.

    Returns the ``/api/chat`` response and the session id. Both are needed: a
    turn the monitor blocks reports the state it landed in on the response,
    while a turn that *passes* goes on to a chat model this environment has no
    key for, so its verdicts have to be read back from the stored message.
    """
    session_id = session_factory(playbook_id)
    _open_session(page, session_id)
    return _send(page, message), session_id


def _stored_verdicts(session_id: str) -> dict[str, bool]:
    """The per-policy verdicts the monitor recorded for the first user turn."""
    with _api() as client:
        messages = client.get(f"/chat/sessions/{session_id}").json()["messages"]
    assert messages, "the turn was never stored"
    return messages[0]["monitor_state"]


class TestVerdictCombination:
    """The premise of the whole feature: a combination means something."""

    def test_only_the_combination_of_both_verdicts_blocks(
        self, app_page: Page, multi_env, session_factory
    ):
        """A=F alone passes, B=F alone passes, A=F and B=F together block.

        The three turns run against the *same* playbook, in which exactly one
        state -- ``A=F;B=F`` -- is flagged. Move that flag onto either
        single-member state and the pass/block pattern inverts: whichever
        member was flagged blocks on its own, which is per-policy blocking
        wearing a playbook's name.

        The state key reported for each turn is asserted too, so a failure
        distinguishes "the verdicts were wrong" from "the flag was read
        wrong" without a rerun.
        """
        alpha, bravo = (
            multi_env["policies"]["alpha"],
            multi_env["policies"]["bravo"],
        )

        alpha_only, alpha_session = _turn(
            app_page, session_factory, multi_env["combo"], _drive("alpha")
        )
        assert alpha_only.json().get("blocked") is not True, (
            "one member's verdict being False blocked on its own -- playbook "
            "mode has fallen back to per-policy blocking"
        )
        assert _stored_verdicts(alpha_session) == {alpha: False, bravo: True}
        expect(app_page.get_by_test_id("violation-alert")).to_have_count(0)

        bravo_only, bravo_session = _turn(
            app_page, session_factory, multi_env["combo"], _drive("bravo")
        )
        assert bravo_only.json().get("blocked") is not True, (
            "the other member's verdict being False blocked on its own"
        )
        assert _stored_verdicts(bravo_session) == {alpha: True, bravo: False}
        expect(app_page.get_by_test_id("violation-alert")).to_have_count(0)

        both, _ = _turn(
            app_page, session_factory, multi_env["combo"],
            _drive("alpha", "bravo"),
        )
        assert both.status == 200, both.text()
        assert both.json()["blocked"] is True, (
            "the flagged combination did not block, though each member "
            "reached the same verdict that passed on its own"
        )
        assert (
            both.json()["playbook_state"]["state_key"]
            == multi_env["combo_both_false"]
        )

        alert = app_page.get_by_test_id("violation-alert")
        expect(alert).to_be_visible(timeout=CHAT_TIMEOUT_MS)
        expect(alert).to_contain_text("Message blocked by playbook")
        expect(alert.get_by_test_id("violation-playbook-state")).to_contain_text(
            "Both broken"
        )
        expect(alert.get_by_test_id("violation-playbook-state")).to_contain_text(
            multi_env["combo_both_false"]
        )
        expect(app_page.get_by_test_id("playbook-state-badge")).to_have_text(
            "Both broken"
        )

    def test_every_state_of_a_two_member_playbook_is_reachable(
        self, app_page: Page, multi_env, session_factory
    ):
        """The fourth combination, and the guarantee the test above rests on.

        Three of the four states are visited by the test above; this one adds
        the state where neither member fires. A fixture that could not reach
        every state would let a combination test pass while proving nothing,
        so reachability is asserted rather than assumed.
        """
        neither, session_id = _turn(
            app_page, session_factory, multi_env["combo"], _drive()
        )
        assert neither.json().get("blocked") is not True
        assert _stored_verdicts(session_id) == {
            multi_env["policies"]["alpha"]: True,
            multi_env["policies"]["bravo"]: True,
        }


class TestStateSpaceSize:
    """2^n states, counted in the product rather than in the engine."""

    @pytest.mark.parametrize(
        ("playbook", "members", "states"),
        [("combo", 2, 4), ("three", 3, 8), ("wide", 4, 16)],
    )
    def test_members_expand_the_state_space_by_powers_of_two(
        self, app_page: Page, multi_env, playbook: str, members: int, states: int
    ):
        """The card and the states pane both count the real state space.

        Every member in these three playbooks carries distinct guidance, so
        no two states share a behaviour and the behaviour count matches the
        state count exactly.
        """
        playbook_id = multi_env[playbook]
        _open_playbooks(app_page)
        card = app_page.get_by_test_id(f"playbook-card-{playbook_id}")
        expect(card).to_contain_text(f"{members} policies")
        expect(card).to_contain_text(f"{states} states → {states} behaviours")

        _open_editor(app_page, multi_env, playbook_id)
        expect(app_page.get_by_test_id("playbook-states")).to_contain_text(
            f"{states} behaviours · {states} states"
        )


class TestBehaviourMerging:
    """States that behave identically collapse; states that differ do not."""

    def test_sixteen_states_merge_into_four_behaviours(
        self, app_page: Page, multi_env
    ):
        """Two of four members carry no guidance, so their bits change nothing.

        Sixteen states, four behaviours, four states each: a table that
        listed sixteen nodes would be listing the same behaviour four times
        over, which is the readability problem behaviours exist to solve.
        """
        _open_editor(app_page, multi_env, multi_env["merged"])
        expect(app_page.get_by_test_id("playbook-states")).to_contain_text(
            "4 behaviours · 16 states"
        )
        for name in ("(no guidance)", "A-rule", "B-rule", "A-rule + B-rule"):
            expect(app_page.get_by_test_id(f"behaviour-{name}")).to_contain_text(
                "4 states"
            )

    def test_a_flag_splits_a_behaviour_from_its_identical_twin(
        self, app_page: Page, multi_env
    ):
        """Identical guidance, different flag: two nodes, not one.

        ``split`` is ``merged`` with a single state flagged. That state leaves
        its group of four and becomes a node of its own, while the remaining
        three stay merged -- and the two nodes still carry byte-identical
        guidance, so a grouping key that ignored the flag would put them back
        together and the playbook would stop blocking.
        """
        _open_editor(app_page, multi_env, multi_env["split"])
        expect(app_page.get_by_test_id("playbook-states")).to_contain_text(
            "5 behaviours · 16 states"
        )

        escalate = app_page.get_by_test_id("behaviour-Escalate")
        twin = app_page.get_by_test_id("behaviour-A-rule + B-rule")
        expect(app_page.get_by_test_id("behaviour-flag-Escalate")).to_be_visible()
        expect(
            app_page.get_by_test_id("behaviour-flag-A-rule + B-rule")
        ).to_have_count(0)
        expect(escalate).to_contain_text("1 states")
        expect(twin).to_contain_text("3 states")

        # The split is the flag and nothing else: same rules, same order.
        expect(escalate.get_by_role("listitem")).to_have_text(
            ["A-rule", "B-rule"]
        )
        expect(twin.get_by_role("listitem")).to_have_text(["A-rule", "B-rule"])
        expect(
            escalate.get_by_test_id(f"state-row-{multi_env['split_key']}")
        ).to_be_visible()


class TestGuidanceComposition:
    """Three members firing at once, composed in the order the user declared."""

    def test_three_firing_members_compose_in_position_order(
        self, app_page: Page, multi_env, session_factory
    ):
        """All three rules, in member position order, not policy-id order.

        The fixture declares positions in the exact reverse of policy-id
        order, so guidance assembled by id arrives backwards. Both ends are
        checked: what the editor shows, and what the monitor actually hands
        the assistant for the turn.
        """
        _open_editor(app_page, multi_env, multi_env["three"])
        behaviour = app_page.get_by_test_id(
            "behaviour-P0 rule. + P1 rule. + P2 rule."
        )
        expect(behaviour).to_be_visible()
        expect(behaviour.get_by_role("listitem")).to_have_text(
            ["P0 rule.", "P1 rule.", "P2 rule."]
        )
        expect(
            behaviour.get_by_test_id(f"state-row-{multi_env['three_all_false']}")
        ).to_be_visible()

        response, _ = _turn(
            app_page, session_factory, multi_env["three"],
            _drive("alpha", "bravo", "charlie"),
        )
        assert response.json()["blocked"] is True, (
            "the all-members-firing state is flagged and should have blocked"
        )
        state = response.json()["playbook_state"]
        assert state["state_key"] == multi_env["three_all_false"]
        assert state["rules"] == ["P0 rule.", "P1 rule.", "P2 rule."], (
            "guidance reached the turn out of member position order"
        )

    def test_a_member_that_does_not_fire_contributes_nothing(
        self, app_page: Page, multi_env, session_factory
    ):
        """The control: drop one marker and exactly that rule disappears.

        Without it, guidance that carried every member's rule regardless of
        verdict would satisfy the test above. The surviving pair is still in
        position order, so this also pins the composition on a state where
        the absent member is not the last one in policy-id order.
        """
        response, _ = _turn(
            app_page, session_factory, multi_env["three"], _drive("alpha", "bravo")
        )
        assert response.json()["blocked"] is True, (
            "the state with charlie satisfied is flagged and should have blocked"
        )
        state = response.json()["playbook_state"]
        assert state["state_key"] == multi_env["three_partial_key"]
        assert len(multi_env["three_partial_rules"]) == 2
        assert state["rules"] == multi_env["three_partial_rules"], (
            "the non-firing member's rule survived, or the surviving two are "
            "out of member position order"
        )


class TestGraphAtScale:
    """The state-machine view with sixteen behaviours."""

    def test_the_graph_renders_every_one_of_sixteen_behaviours(
        self, app_page: Page, multi_env
    ):
        """Four members, sixteen distinct behaviours, sixteen nodes.

        Names are how the graph identifies a node -- as its React key, its
        test id, and the key the trace marks visited -- so the count is
        asserted on distinct node elements, which also pins the names apart.
        """
        _open_editor(app_page, multi_env, multi_env["wide"])
        app_page.get_by_test_id("states-view-graph").click()
        expect(app_page.get_by_test_id("playbook-graph")).to_be_visible()

        expect(app_page.locator('[data-testid^="node-"]')).to_have_count(16)
        for name in (
            "(no guidance)",
            "A-rule",
            "D-rule",
            "A-rule + B-rule",
            "B-rule + C-rule + D-rule",
            "A-rule + B-rule + C-rule + D-rule",
        ):
            node = app_page.get_by_test_id(f"node-{name}")
            expect(node).to_be_visible()
            expect(node).to_have_attribute("data-visited", "false")


class TestMemberRemoval:
    """Overrides re-key onto the smaller state space when a member leaves."""

    def test_removing_a_member_collapses_its_override_pair(
        self, app_page: Page, multi_env
    ):
        """Both branches of the removed member agree, so they collapse silently.

        The two flagged states differ only in the verdict of the member being
        removed, so after the removal there is one flagged state, keyed on the
        two survivors, still flagged and still labelled. Dropping the
        overrides instead would leave a playbook that quietly stops blocking;
        keeping both keys would leave two overrides pointing at states that no
        longer exist.
        """
        _open_editor(app_page, multi_env, multi_env["collapse"])
        expect(app_page.get_by_test_id("playbook-states")).to_contain_text(
            "4 behaviours · 8 states"
        )
        expect(app_page.get_by_test_id("behaviour-Pair")).to_contain_text(
            "2 states"
        )

        app_page.get_by_test_id(
            f"member-included-{multi_env['collapse_removed']}"
        ).uncheck()
        app_page.get_by_test_id("save-members").click()

        expect(app_page.get_by_test_id("members-save-report")).to_be_visible()
        expect(app_page.get_by_test_id("members-conflicts")).to_have_count(0)

        expect(app_page.get_by_test_id("playbook-states")).to_contain_text(
            "4 behaviours · 4 states"
        )
        pair = app_page.get_by_test_id("behaviour-Pair")
        expect(pair).to_contain_text("1 states")
        expect(app_page.get_by_test_id("behaviour-flag-Pair")).to_be_visible()
        expect(
            pair.get_by_test_id(f"state-row-{multi_env['collapse_collapsed_key']}")
        ).to_be_visible()

        with _api() as client:
            states = client.get(
                f"/playbooks/{multi_env['collapse']}/states"
            ).json()
        keys = [
            state["state_key"]
            for behaviour in states["behaviours"]
            for state in behaviour["states"]
        ]
        assert all(
            multi_env["collapse_removed"] not in key for key in keys
        ), "a state key still names the removed member"
        flagged = [b for b in states["behaviours"] if b["flagged"]]
        assert [b["name"] for b in flagged] == ["Pair"]
        assert [s["state_key"] for s in flagged[0]["states"]] == [
            multi_env["collapse_collapsed_key"]
        ]
