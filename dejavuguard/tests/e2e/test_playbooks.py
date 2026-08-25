"""
E2E tests: Playbooks — the states table, session monitoring mode, and the
one behaviour the whole feature exists for: a flagged state blocks a turn.

There is no chat-LLM API key in this environment, so an assistant reply
cannot be generated. That does not stop the enforcement path being tested:
a blocked user turn never reaches the LLM, because the monitor decides
before the OpenRouter call. So the flagged/not-flagged pair below is a real
end-to-end discriminator --

  * ``test_flagged_state_blocks_the_user_turn`` sends the violating message
    with the F state flagged and asserts the turn is blocked, naming the
    playbook and the state;
  * ``test_unflagged_state_does_not_block_the_same_turn`` sends the *same*
    message against an *identical* playbook whose F state is not flagged,
    and asserts the turn is not blocked -- it gets all the way to the chat
    model, which is the only reason it fails there.

Only the flag differs, so together they pin the block to ``flagged`` rather
than to the member's verdict. If playbook mode ever fell back to per-policy
blocking, the second test fails; if the flag stopped blocking, the first does.

Data is set up over the API and asserted through the UI: a fixture that
clicks its own preconditions makes a failure ambiguous.
"""

from __future__ import annotations

import httpx
import pytest
from playwright.sync_api import Page, expect

API = "http://localhost:8000/api"

#: Everything this module creates is named with this prefix so a crashed run
#: can be swept clean on the next one -- the backend uses a shared dev DB.
PREFIX = "e2e-playbooks"

PREDICATE_ID = "e2e_playbooks_budget_u"

#: Matched by the deterministic stub grounder on
#: scenario_runner/support/playbook_grounding.json: the predicate description
#: supplies "states a maximum budget" and this text supplies "ceiling is
#: $12,000", so the predicate grounds true without any model.
OVER_BUDGET_MESSAGE = (
    "I'm planning a project and my absolute ceiling is $12,000 "
    "- that's everything I've budgeted."
)

MEMBER_GUIDANCE = "Stay within the stated budget."
FLAGGED_LABEL = "Over budget"

# The monitor talks to the real DejaVu server, so a chat turn is slower than
# a plain render.
CHAT_TIMEOUT_MS = 60_000


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
    client.delete(f"/propositions/{PREDICATE_ID}")


@pytest.fixture(scope="module")
def playbook_env():
    """Predicates, a policy and four playbooks, over the API.

    The playbooks are identical but for their flags, because the discriminating
    pair needs the flag to be the *only* difference:

    * ``flagged``  -- the F state is flagged and labelled; this one blocks.
    * ``open``     -- same member, nothing flagged; this one cannot block.
    * ``ui``       -- same again, for flagging through the editor, so the
      control above is never left flagged by a test that failed mid-edit.
    * ``blank``    -- no members, for editing membership through the UI.
    """
    with _api() as client:
        original_settings = client.get("/settings").json()
        _sweep(client)

        # Point grounding at the deterministic stub so verdicts are fixed.
        settings = client.get("/settings").json()
        settings["grounding"]["provider"] = "custom"
        settings["grounding"]["base_url"] = "http://localhost:9099/v1"
        settings["grounding"]["model"] = "stub-grounder"
        assert client.put("/settings", json=settings).status_code == 200

        created = client.post(
            "/propositions",
            json={
                "prop_id": PREDICATE_ID,
                "description": (
                    "the user states a maximum budget they are willing to spend"
                ),
                "role": "user",
                "arity": 1,
                "arg_descriptions": [
                    "the maximum budget amount, in US dollars, the user is "
                    "willing to spend"
                ],
            },
        )
        assert created.status_code == 201, created.text

        policy = client.post(
            "/policies",
            json={
                "name": f"{PREFIX} budget guard",
                "formula_str": f"forall b . ! {PREDICATE_ID}(b)",
            },
        )
        assert policy.status_code == 201, policy.text
        policy_id = policy.json()["policy_id"]

        names: dict[str, str] = {}

        def make(suffix: str, with_member: bool) -> str:
            name = f"{PREFIX} {suffix}"
            response = client.post("/playbooks", json={"name": name})
            assert response.status_code == 201, response.text
            playbook_id = response.json()["playbook_id"]
            names[playbook_id] = name
            if with_member:
                members = client.put(
                    f"/playbooks/{playbook_id}/members",
                    json={
                        "members": [
                            {
                                "policy_id": policy_id,
                                "position": 0,
                                "fires_on": False,
                                "guidance": MEMBER_GUIDANCE,
                            }
                        ]
                    },
                )
                assert members.status_code == 200, members.text
            return playbook_id

        flagged_id = make("flagged", with_member=True)
        open_id = make("open", with_member=True)
        ui_id = make("ui", with_member=True)
        blank_id = make("blank", with_member=False)

        false_key = f"{policy_id}=F"
        override = client.put(
            f"/playbooks/{flagged_id}/states/{false_key}",
            json={"rule_refs": None, "flagged": True, "label": FLAGGED_LABEL},
        )
        assert override.status_code == 200, override.text

        env = {
            "policy_id": policy_id,
            "flagged_id": flagged_id,
            "open_id": open_id,
            "ui_id": ui_id,
            "blank_id": blank_id,
            "names": names,
            "false_key": false_key,
            "true_key": f"{policy_id}=T",
        }
        try:
            yield env
        finally:
            _sweep(client)
            client.put("/settings", json=original_settings)


@pytest.fixture()
def session_factory():
    """Create chat sessions over the API and remove them afterwards.

    Sessions are created here rather than through the "New session" button so
    the id is known, and so nothing leaks into the chat suite -- which asserts
    the empty-session state.
    """
    created: list[str] = []

    def make(monitoring: dict | None = None) -> str:
        with _api() as client:
            session_id = client.post("/chat/sessions").json()["session_id"]
            created.append(session_id)
            if monitoring:
                response = client.patch(
                    f"/chat/sessions/{session_id}/monitoring", json=monitoring
                )
                assert response.status_code == 200, response.text
        return session_id

    yield make

    with _api() as client:
        for session_id in created:
            client.delete(f"/chat/sessions/{session_id}")


@pytest.fixture()
def only_this_modules_policies_enabled():
    """Suspend every policy this module did not create, then restore them.

    Policy mode monitors *every* enabled policy, so a policy that happens to
    live in the developer's database can reach a verdict on the same message
    and block the turn before this module's does. That is correct product
    behaviour and a test that never said which policies it was testing; the
    fixture says it. The previous enabled state is put back in a ``finally``,
    because a fixture that leaves the database mutated after a failing test is
    its own defect.
    """
    with _api() as client:
        suspended = [
            policy["policy_id"]
            for policy in client.get("/policies").json()
            if policy["enabled"] and not policy["name"].startswith(PREFIX)
        ]
        for policy_id in suspended:
            response = client.put(f"/policies/{policy_id}", json={"enabled": False})
            assert response.status_code == 200, response.text
        try:
            yield
        finally:
            for policy_id in suspended:
                client.put(f"/policies/{policy_id}", json={"enabled": True})


def _dismiss_intro(page: Page) -> None:
    """The intro overlay covers the app on a fresh page load."""
    overlay = page.locator('[data-testid="intro-overlay"]')
    if overlay.count() and overlay.is_visible():
        overlay.click()
    page.wait_for_selector('[data-testid="app-layout"]', timeout=10_000)


def _open_playbooks(page: Page) -> None:
    page.click('[data-testid="nav-playbooks"]')
    expect(page.locator('[data-testid="playbooks-view"]')).to_be_visible()


def _open_editor(page: Page, env: dict, playbook_id: str) -> None:
    _open_playbooks(page)
    page.get_by_test_id(f"playbook-card-{playbook_id}").get_by_role(
        "button", name=env["names"][playbook_id], exact=True
    ).click()
    expect(page.locator('[data-testid="playbook-editor"]')).to_be_visible()
    expect(page.locator('[data-testid="playbook-states"]')).to_be_visible()


def _open_session(page: Page, session_id: str) -> None:
    """Reload, then open a session from the sidebar.

    The reload is needed because the fixtures create the session over the API
    after the page has already loaded its session list -- and it is also what
    the mode-persistence test is asserting, so everything the UI shows below
    was read back from the server rather than kept in component state.
    """
    page.reload()
    _dismiss_intro(page)
    page.click('[data-testid="nav-chat"]')
    page.get_by_test_id(f"session-{session_id}").click()
    expect(page.locator('[data-testid="message-input-form"]')).to_be_visible()


def _send(page: Page, message: str):
    """Type a message, send it, and return the /api/chat response."""
    page.get_by_test_id("message-input").fill(message)
    with page.expect_response(
        lambda r: r.request.method == "POST" and r.url.endswith("/api/chat"),
        timeout=CHAT_TIMEOUT_MS,
    ) as info:
        page.get_by_test_id("send-button").click()
    return info.value


class TestPlaybookList:
    """The playbooks list and the enforcement warning on each card."""

    def test_playbook_card_shows_derived_state_counts(
        self, app_page: Page, playbook_env
    ):
        """A one-member playbook reads as 1 policy, 2 states, 2 behaviours."""
        _open_playbooks(app_page)
        card = app_page.get_by_test_id(
            f"playbook-card-{playbook_env['flagged_id']}"
        )
        expect(card).to_be_visible()
        expect(card).to_contain_text("1 policies")
        expect(card).to_contain_text("2 states → 2 behaviours")

    def test_playbook_with_no_flagged_state_warns_it_cannot_block(
        self, app_page: Page, playbook_env
    ):
        """The R1 mitigation: a playbook that flags nothing says so.

        The paired negative matters as much as the warning: a card that always
        warned would be no more informative than one that never did.
        """
        _open_playbooks(app_page)
        open_card = app_page.get_by_test_id(f"playbook-card-{playbook_env['open_id']}")
        expect(
            open_card.get_by_test_id("playbook-no-block-warning")
        ).to_be_visible()

        flagged_card = app_page.get_by_test_id(
            f"playbook-card-{playbook_env['flagged_id']}"
        )
        expect(
            flagged_card.get_by_test_id("playbook-no-block-warning")
        ).to_have_count(0)

    def test_create_and_delete_a_playbook_through_the_ui(
        self, app_page: Page, playbook_env
    ):
        """Create through the form, confirm it persists, then delete it."""
        name = f"{PREFIX} created-in-ui"
        _open_playbooks(app_page)
        app_page.get_by_test_id("add-playbook").click()
        app_page.get_by_test_id("new-playbook-name-input").fill(name)
        app_page.get_by_test_id("new-playbook-save").click()

        card = app_page.locator('[data-testid^="playbook-card-"]').filter(
            has_text=name
        )
        expect(card).to_be_visible()

        with _api() as client:
            stored = [p["name"] for p in client.get("/playbooks").json()]
        assert name in stored, "the new playbook was never persisted"

        card.get_by_role("button", name=f"Delete {name}").click()
        expect(card).to_have_count(0)

        with _api() as client:
            stored = [p["name"] for p in client.get("/playbooks").json()]
        assert name not in stored, "the deleted playbook is still stored"


class TestStatesTable:
    """The truth table, behaviour grouping, flagging, filters and revert."""

    def test_truth_table_lists_every_verdict_combination(
        self, app_page: Page, playbook_env
    ):
        """One member yields exactly two states, T and F, each with its badge."""
        _open_editor(app_page, playbook_env, playbook_env["flagged_id"])
        expect(app_page.get_by_test_id("playbook-states")).to_contain_text(
            "2 behaviours · 2 states"
        )

        true_row = app_page.get_by_test_id(f"state-row-{playbook_env['true_key']}")
        false_row = app_page.get_by_test_id(f"state-row-{playbook_env['false_key']}")
        expect(true_row).to_be_visible()
        expect(false_row).to_be_visible()
        expect(true_row).to_contain_text(f"{playbook_env['policy_id']}=T")
        expect(false_row).to_contain_text(f"{playbook_env['policy_id']}=F")

    def test_flagged_state_is_its_own_labelled_behaviour(
        self, app_page: Page, playbook_env
    ):
        """The flagged state groups under its label and carries the flag."""
        _open_editor(app_page, playbook_env, playbook_env["flagged_id"])
        expect(
            app_page.get_by_test_id(f"behaviour-{FLAGGED_LABEL}")
        ).to_be_visible()
        expect(
            app_page.get_by_test_id(f"behaviour-flag-{FLAGGED_LABEL}")
        ).to_be_visible()

        # The unflagged half is a separate node, and is not flagged.
        expect(app_page.get_by_test_id("behaviour-(no guidance)")).to_be_visible()
        expect(
            app_page.get_by_test_id("behaviour-flag-(no guidance)")
        ).to_have_count(0)

    def test_flagging_a_state_through_the_ui_makes_the_playbook_enforce(
        self, app_page: Page, playbook_env
    ):
        """Flag the F state from the editor, then revert it.

        This is the control the product shipped without: without it, a
        playbook built through the UI can never block anything. The warning
        disappearing and coming back is what proves the flag reached the
        backend rather than only the local draft.
        """
        _open_editor(app_page, playbook_env, playbook_env["ui_id"])
        expect(app_page.get_by_test_id("playbook-warnings")).to_contain_text(
            "it can no longer block anything"
        )

        false_key = playbook_env["false_key"]
        app_page.get_by_test_id(f"edit-{false_key}").click()
        editor = app_page.get_by_test_id(f"state-override-{false_key}")
        expect(editor).to_be_visible()
        editor.get_by_test_id("override-flagged").check()
        editor.get_by_test_id("override-label").fill("Blocks now")
        editor.get_by_test_id("override-save").click()

        try:
            expect(app_page.get_by_test_id("behaviour-flag-Blocks now")).to_be_visible()
            expect(app_page.get_by_test_id("playbook-warnings")).to_have_count(0)

            with _api() as client:
                states = client.get(
                    f"/playbooks/{playbook_env['ui_id']}/states"
                ).json()
            flagged = [b for b in states["behaviours"] if b["flagged"]]
            assert [b["name"] for b in flagged] == ["Blocks now"], (
                "flagging through the UI did not reach the backend"
            )
        finally:
            app_page.get_by_test_id(f"revert-{false_key}").click()

        expect(app_page.get_by_test_id("behaviour-flag-Blocks now")).to_have_count(0)
        expect(app_page.get_by_test_id("playbook-warnings")).to_contain_text(
            "it can no longer block anything"
        )

    def test_only_flagged_filter_hides_the_unflagged_behaviour(
        self, app_page: Page, playbook_env
    ):
        """The filter narrows the table to the states that can block."""
        _open_editor(app_page, playbook_env, playbook_env["flagged_id"])
        expect(app_page.get_by_test_id("behaviour-(no guidance)")).to_be_visible()

        app_page.get_by_test_id("filter-only-flagged").check()

        expect(app_page.get_by_test_id(f"behaviour-{FLAGGED_LABEL}")).to_be_visible()
        expect(app_page.get_by_test_id("behaviour-(no guidance)")).to_have_count(0)

    def test_only_customised_filter_hides_the_default_state(
        self, app_page: Page, playbook_env
    ):
        """The filter narrows the table to the states someone has edited."""
        _open_editor(app_page, playbook_env, playbook_env["flagged_id"])
        expect(
            app_page.get_by_test_id(f"state-row-{playbook_env['true_key']}")
        ).to_be_visible()

        app_page.get_by_test_id("filter-only-customised").check()

        expect(
            app_page.get_by_test_id(f"state-row-{playbook_env['false_key']}")
        ).to_be_visible()
        expect(
            app_page.get_by_test_id(f"state-row-{playbook_env['true_key']}")
        ).to_have_count(0)

    def test_editing_membership_expands_the_state_space(
        self, app_page: Page, playbook_env
    ):
        """Adding a member through the UI turns 1 state into 2, and warns."""
        _open_editor(app_page, playbook_env, playbook_env["blank_id"])
        expect(app_page.get_by_test_id("playbook-states")).to_contain_text(
            "1 behaviours · 1 states"
        )

        policy_id = playbook_env["policy_id"]
        app_page.get_by_test_id(f"member-included-{policy_id}").check()
        app_page.get_by_test_id(f"member-fires-on-{policy_id}").select_option("false")
        app_page.get_by_test_id(f"member-guidance-{policy_id}").fill(MEMBER_GUIDANCE)
        app_page.get_by_test_id("save-members").click()

        expect(app_page.get_by_test_id("members-warnings")).to_contain_text(
            "it can no longer block anything"
        )
        expect(app_page.get_by_test_id("playbook-states")).to_contain_text(
            "2 behaviours · 2 states"
        )
        expect(
            app_page.get_by_test_id(f"state-row-{playbook_env['false_key']}")
        ).to_be_visible()

    def test_graph_view_renders_every_behaviour_as_a_node(
        self, app_page: Page, playbook_env
    ):
        """The editor's graph is the playbook's map: all nodes, none visited."""
        _open_editor(app_page, playbook_env, playbook_env["flagged_id"])
        app_page.get_by_test_id("states-view-graph").click()

        expect(app_page.get_by_test_id("playbook-graph")).to_be_visible()
        node = app_page.get_by_test_id(f"node-{FLAGGED_LABEL}")
        expect(node).to_be_visible()
        expect(node).to_have_attribute("data-visited", "false")
        expect(app_page.get_by_test_id("node-(no guidance)")).to_be_visible()


class TestSessionMonitoringMode:
    """Switching a session between policy and playbook monitoring."""

    def test_switching_to_playbook_mode_persists_across_a_reload(
        self, app_page: Page, playbook_env, session_factory
    ):
        """The mode is read from the session, not from the last turn."""
        session_id = session_factory()
        _open_session(app_page, session_id)

        selector = app_page.get_by_test_id("monitoring-selector")
        expect(selector.get_by_role("radio", name="Policies")).to_be_checked()

        selector.get_by_role("radio", name="Playbook").check()
        with app_page.expect_response(
            lambda r: r.request.method == "PATCH" and "/monitoring" in r.url
        ):
            app_page.get_by_test_id("playbook-select").select_option(
                playbook_env["flagged_id"]
            )

        # _open_session reloads, so nothing below survives in component state.
        _open_session(app_page, session_id)

        selector = app_page.get_by_test_id("monitoring-selector")
        expect(selector.get_by_role("radio", name="Playbook")).to_be_checked()
        expect(app_page.get_by_test_id("playbook-select")).to_have_value(
            playbook_env["flagged_id"]
        )


class TestFlaggedStateBlocks:
    """The behaviour the feature exists for, and its control."""

    def test_flagged_state_blocks_the_user_turn(
        self, app_page: Page, playbook_env, session_factory
    ):
        """A user turn landing in a flagged state is blocked, and says why.

        The block names the playbook and the state rather than a policy and a
        formula: in playbook mode there is no single policy to blame.
        """
        session_id = session_factory(
            {"mode": "playbook", "playbook_id": playbook_env["flagged_id"]}
        )
        _open_session(app_page, session_id)

        response = _send(app_page, OVER_BUDGET_MESSAGE)
        assert response.status == 200, response.text()
        assert response.json()["blocked"] is True, (
            "the flagged state did not block the turn"
        )

        alert = app_page.get_by_test_id("violation-alert")
        expect(alert).to_be_visible(timeout=CHAT_TIMEOUT_MS)
        expect(alert).to_contain_text("Message blocked by playbook")
        expect(alert).to_contain_text(f"{PREFIX} flagged")
        expect(alert.get_by_test_id("violation-playbook-state")).to_contain_text(
            FLAGGED_LABEL
        )
        expect(alert.get_by_test_id("violation-playbook-state")).to_contain_text(
            playbook_env["false_key"]
        )
        # A playbook block has no formula to show.
        expect(alert.get_by_test_id("violation-formula")).to_have_count(0)

        # The turn itself is struck out as blocked, and the header badge names
        # the state the session is now in.
        expect(app_page.get_by_test_id("message-blocked")).to_be_visible()
        expect(app_page.get_by_test_id("playbook-state-badge")).to_have_text(
            FLAGGED_LABEL
        )

    def test_unflagged_state_does_not_block_the_same_turn(
        self, app_page: Page, playbook_env, session_factory
    ):
        """The control: same message, same member verdict, no flag, no block.

        The member's verdict here is False -- exactly the verdict that blocks
        in policy mode -- so if playbook mode ever blocked on a member verdict
        rather than on the state flag, this turn would be blocked and the test
        would fail. Instead the turn passes the monitor and reaches the chat
        model, which is the only thing that stops it in this environment.
        """
        session_id = session_factory(
            {"mode": "playbook", "playbook_id": playbook_env["open_id"]}
        )
        _open_session(app_page, session_id)

        response = _send(app_page, OVER_BUDGET_MESSAGE)
        assert response.json().get("blocked") is not True, (
            "an unflagged state blocked the turn"
        )

        expect(app_page.get_by_test_id("violation-alert")).to_have_count(0)

        # The monitor let the turn through and stored it unblocked; reopening
        # reloads, so this is read back from the server rather than from the
        # optimistic copy the failed send removed.
        _open_session(app_page, session_id)
        expect(app_page.get_by_test_id("message-content")).to_contain_text(
            "Passed"
        )
        expect(app_page.get_by_test_id("message-blocked")).to_have_count(0)

    def test_policy_mode_still_blocks_on_the_policy_verdict(
        self,
        app_page: Page,
        playbook_env,
        only_this_modules_policies_enabled,
        session_factory,
    ):
        """The same message in policy mode blocks by policy, naming the formula.

        Together with the two above this pins each mode to its own rule: the
        playbook's flag decides in playbook mode, the policy verdict decides in
        policy mode, and the session's mode chooses between them.

        Policy mode monitors every enabled policy, so which policy blocks is
        only a fact about this one while no other policy is enabled -- hence
        the fixture.
        """
        session_id = session_factory({"mode": "policies"})
        _open_session(app_page, session_id)

        response = _send(app_page, OVER_BUDGET_MESSAGE)
        assert response.status == 200, response.text()
        assert response.json()["blocked"] is True

        alert = app_page.get_by_test_id("violation-alert")
        expect(alert).to_be_visible(timeout=CHAT_TIMEOUT_MS)
        expect(alert).to_contain_text("Message blocked by policy")
        expect(alert).to_contain_text(f"{PREFIX} budget guard")
        expect(alert.get_by_test_id("violation-formula")).to_contain_text(
            PREDICATE_ID
        )
        expect(alert.get_by_test_id("violation-playbook-state")).to_have_count(0)
        expect(app_page.get_by_test_id("playbook-state-badge")).to_have_count(0)


class TestStateGraphFromChat:
    """The graph opened from the chat header badge."""

    def test_badge_opens_the_graph_on_the_state_the_session_is_in(
        self, app_page: Page, playbook_env, session_factory
    ):
        """The blocked turn is a visit: its node is current and visited.

        The graph's edges and current node are reconstructed from the stored
        per-turn verdicts, so this fails if a blocked turn stopped recording
        the state it landed in.
        """
        session_id = session_factory(
            {"mode": "playbook", "playbook_id": playbook_env["flagged_id"]}
        )
        _open_session(app_page, session_id)
        _send(app_page, OVER_BUDGET_MESSAGE)

        badge = app_page.get_by_test_id("playbook-state-badge")
        expect(badge).to_be_visible(timeout=CHAT_TIMEOUT_MS)
        badge.click()

        expect(app_page.get_by_test_id("playbook-graph")).to_be_visible()
        node = app_page.get_by_test_id(f"node-{FLAGGED_LABEL}")
        expect(node).to_have_attribute("data-visited", "true")
        expect(node).to_have_attribute("data-current", "true")

        # The state the session never reached stays unvisited.
        expect(app_page.get_by_test_id("node-(no guidance)")).to_have_attribute(
            "data-visited", "false"
        )
