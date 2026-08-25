"""E2E tests: can a human actually finish the task on this screen?

Every other suite in this repository asserts that the right things are
*rendered*. None of them, until this one, asserted that the things rendered
can be *reached*. That gap shipped a modal whose confirm button was off the
bottom of the screen with nothing to scroll: 436 frontend tests and 166 e2e
tests passed, because jsdom computes no layout and Playwright's
``to_be_visible()`` means "in the DOM and not hidden", not "a person can click
this". It was found by a human looking at a screenshot.

So this module drives the playbook feature the way a person does, and asserts
operability rather than presence at every step. See ``reachability.py`` for
what "operable" means and why it is two checks rather than one.

Three things here are deliberate and load-bearing:

**The playbooks are built through the UI.** Every other multi-policy test
constructs its fixture over HTTP, so the guided ``+ Add policy`` flow had
never been driven repeatedly to assemble a real playbook -- which is exactly
the flow the bug was in. :class:`TestBuildingAPlaybookThroughTheUI` clicks
through it four times over, once per member.

**The rule library is populated first.** The confirm button only became
unreachable once the rule list had content: the reuse step's list grows the
panel. A fixture against an empty library would have passed while the product
was broken, so the fixture seeds rules before anything opens the rule step.

**Four viewports, not three.** The bug appeared at a laptop height and not on
a large monitor, which is why it survived a session of manual inspection, so
the guided flow is driven at 1280x720, 1440x900 and 1920x1080. Measuring the
panel that shipped, though, says none of those three reproduces it: a
Playwright viewport is *content* height, and the tallest add-policy panel is
646px centred with no cap, so its confirm button only leaves a 598px-tall
screen -- less room than any of the three. :class:`TestAShortBrowserWindow`
adds the size that does reproduce it, and is the one that fails when the fix
is reverted.
"""

from __future__ import annotations

import httpx
import pytest
from playwright.sync_api import Page, expect

from tests.e2e.reachability import (
    UnreachableError,
    expect_all_reachable,
    expect_reachable,
)
from tests.e2e.test_playbooks import _dismiss_intro, _open_playbooks, _rule_prefix

API = "http://localhost:8000/api"

#: Everything this module creates is named with this prefix, so a crashed run
#: can be swept on the next one -- the backend uses a shared dev database that
#: also holds playbooks a person is using. Nothing outside the prefix is ever
#: touched.
PREFIX = "e2e-reach"

RULE_PREFIX = _rule_prefix(PREFIX)

#: The four members a built playbook ends up with, in the order they are added.
KEYWORDS = ("alpha", "bravo", "charlie", "delta")

#: A policy whose name is long enough that the rule name derived from it --
#: `Rule_<POLICY_NAME>`, slugged -- overflows its input. The create branch
#: fills that name in by default, so this is the tallest the step ever gets:
#: radios, a name box, a guidance box, a collision warning and the confirm row.
LONG_POLICY_NAME = (
    f"{PREFIX} the assistant must never restate a budget the user gave it "
    "earlier in the conversation without first asking whether it still holds"
)

#: The three sizes required. 1280x720 first, because it is the one that fails.
VIEWPORTS = [(1280, 720), (1440, 900), (1920, 1080)]

VIEWPORT_IDS = [f"{w}x{h}" for w, h in VIEWPORTS]

#: The smallest of the three: where deliverable 4's modal sweep runs, and
#: where the states table and graph are checked at sixteen states.
LAPTOP = VIEWPORTS[0]

#: A short window, and the only size at which the bug that started all this
#: actually reproduces.
#:
#: Measured, with the shared ``Modal`` reverted to what shipped: the tallest
#: add-policy panel is 646px, centred in a viewport with no cap, so its
#: confirm button leaves the screen below a content height of 598px, and the
#: reuse branch's shorter panel leaves it below 512px. A Playwright viewport
#: is *content* height, so the 1280x720 the three sizes above start from is
#: already more room than a maximised browser has on a 1280x720 screen -- and
#: none of the three reproduces it. This one does, on both branches, with
#: room to spare.
#:
#: 460px of content is not a contrivance: it is a browser with the devtools
#: drawer docked at the bottom, or simply a window that is not maximised. The
#: fix caps the panel at 90vh and scrolls its body, so it holds at *any*
#: height -- which is what makes this a fair thing to require.
SHORT_WINDOW = (1280, 460)


def _api() -> httpx.Client:
    return httpx.Client(base_url=API, timeout=60.0)


def _sweep(client: httpx.Client) -> None:
    """Delete anything a previous run of this module left behind.

    Playbooks before the rules they hold: the API refuses to delete a rule a
    playbook still names, and that refusal is what stops this loop reaching a
    rule someone else is using. Nothing here is deleted by force, and nothing
    outside ``PREFIX`` is looked at.
    """
    for playbook in client.get("/playbooks").json():
        if playbook["name"].startswith(PREFIX):
            client.delete(f"/playbooks/{playbook['playbook_id']}")
    for rule in client.get("/rules").json():
        if rule["name"].startswith(RULE_PREFIX):
            client.delete(f"/rules/{rule['rule_id']}")
    for policy in client.get("/policies").json():
        if policy["name"].startswith(PREFIX):
            client.delete(f"/policies/{policy['policy_id']}")
    for keyword in (*KEYWORDS, "long"):
        client.delete(f"/propositions/e2e_reach_{keyword}_u")


@pytest.fixture(scope="module")
def reach_env():
    """Five policies, a seeded rule library, and a four-member playbook.

    The library is seeded on purpose. The reachability bug this module exists
    for only appeared once the reuse step had rules to list, so a fixture that
    left the library empty would assert a shorter panel than the one users
    actually see. Two rules of this module's own are added on top of whatever
    the developer's database already holds, so the list is never empty even on
    a fresh database.
    """
    with _api() as client:
        _sweep(client)

        policies: dict[str, str] = {}
        names: dict[str, str] = {}
        for keyword in (*KEYWORDS, "long"):
            prop_id = f"e2e_reach_{keyword}_u"
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
            name = LONG_POLICY_NAME if keyword == "long" else f"{PREFIX} {keyword}"
            policy = client.post(
                "/policies", json={"name": name, "formula_str": f"! {prop_id}"}
            )
            assert policy.status_code == 201, policy.text
            policies[keyword] = policy.json()["policy_id"]
            names[keyword] = name

        seeded: dict[str, str] = {}
        for suffix, guidance in (
            ("shared", "Reuse me: shared guidance from the seeded library."),
            ("second", "A second seeded rule, so the list is never one row."),
        ):
            rule_name = f"{RULE_PREFIX}_seed_{suffix}"
            rule = client.post(
                "/rules", json={"name": rule_name, "guidance": guidance}
            )
            assert rule.status_code == 201, rule.text
            seeded[suffix] = rule_name

        # A rule already holding the name the create branch derives from the
        # long policy. That makes the branch show its "the library already
        # holds ..." warning -- two long names' worth of wrapped text, and the
        # tallest the add-policy panel ever gets. Without it the panel is some
        # 60px shorter, and the short-window test stops discriminating.
        owned = _rule_prefix(LONG_POLICY_NAME)
        held = client.post("/rules", json={"name": owned, "guidance": "Held already."})
        assert held.status_code == 201, held.text

        # A sixteen-state playbook built over the API. The guided flow is
        # tested by building one through the UI; this one exists so the table
        # and graph can be measured at scale without paying for that build
        # again, and its four members carry distinct guidance so no two states
        # merge and the table really does list sixteen behaviours.
        wide = client.post("/playbooks", json={"name": f"{PREFIX} wide"})
        assert wide.status_code == 201, wide.text
        wide_id = wide.json()["playbook_id"]
        members = client.put(
            f"/playbooks/{wide_id}/members",
            json={
                "members": [
                    {
                        "policy_id": policies[keyword],
                        "position": index,
                        "fires_on": False,
                        "guidance": f"{keyword.title()} guidance for the wide playbook.",
                    }
                    for index, keyword in enumerate(KEYWORDS)
                ]
            },
        )
        assert members.status_code == 200, members.text

        env = {
            "policies": policies,
            "names": names,
            "seeded": seeded,
            "wide": wide_id,
            "wide_name": f"{PREFIX} wide",
        }
        try:
            yield env
        finally:
            _sweep(client)


def _at(page: Page, size: tuple[int, int]) -> None:
    """Size the window, then land on the playbooks list."""
    width, height = size
    page.set_viewport_size({"width": width, "height": height})
    page.reload()
    _dismiss_intro(page)
    _open_playbooks(page)


def _open_editor(page: Page, name: str) -> None:
    card = page.locator('[data-testid^="playbook-card-"]').filter(has_text=name)
    expect(card).to_be_visible()
    opener = card.get_by_role("button", name=name, exact=True)
    expect_reachable(opener, f'the "{name}" card on the playbooks list')
    opener.click()
    expect(page.get_by_test_id("playbook-editor")).to_be_visible()


# -- the guided + Add policy flow, one member at a time --------------------
#
# Each step asserts the controls it offers are operable before it uses them,
# so a failure names the step and the control rather than reporting a stale
# click somewhere further on.


def _step_one_pick_policy(page: Page, policy_id: str, policy_name: str) -> None:
    add = page.get_by_test_id("add-policy")
    expect_reachable(add, "the + Add policy button")
    add.click()
    expect(page.get_by_test_id("add-policy-modal")).to_be_visible()

    option = page.get_by_test_id(f"policy-option-{policy_id}")
    expect_all_reachable(
        {
            "the modal's close button on step 1": page.get_by_test_id("modal-close"),
            "Cancel on step 1": page.get_by_test_id("add-policy-cancel"),
            f'the policy option for "{policy_name}" on step 1': option,
        }
    )
    option.click()
    expect(page.get_by_test_id("fires-on-step")).to_be_visible()


def _step_two_fires_on(page: Page, *, satisfied: bool) -> None:
    expect_all_reachable(
        {
            '"When violated" on step 2': page.get_by_test_id("fires-on-violated"),
            '"When satisfied" on step 2': page.get_by_test_id("fires-on-satisfied"),
            "Back on step 2": page.get_by_test_id("add-policy-back"),
            "Next on step 2": page.get_by_test_id("fires-on-next"),
        }
    )
    page.get_by_test_id(
        "fires-on-satisfied" if satisfied else "fires-on-violated"
    ).click()
    page.get_by_test_id("fires-on-next").click()
    expect(page.get_by_test_id("rule-step")).to_be_visible()


def _step_three_rule(
    page: Page,
    mode: str,
    *,
    search: str | None = None,
    reuse_named: str | None = None,
    guidance: str | None = None,
) -> str | None:
    """Choose the rule for this member, and confirm the add.

    The confirm button is the control this whole module was written for: it is
    the one that shipped off screen, and it is only ever below a rule list
    that has content, so it is measured *after* the chosen mode has expanded.

    Returns the name of the rule created, when the mode was ``create``. The
    name is read off the box rather than derived here because the modal
    suffixes it past whatever the library already holds -- the second run of
    this module against an unswept database gets ``..._2``, and a test that
    assumed otherwise would go looking for a rule that is not there.
    """
    expect_all_reachable(
        {
            "the reuse radio on step 3": page.get_by_test_id("rule-mode-reuse"),
            "the create radio on step 3": page.get_by_test_id("rule-mode-create"),
            "the no-guidance radio on step 3": page.get_by_test_id("rule-mode-none"),
            "Back on step 3": page.get_by_test_id("add-policy-back"),
        }
    )
    page.get_by_test_id(f"rule-mode-{mode}").check()
    created: str | None = None

    if mode == "reuse":
        box = page.get_by_test_id("rule-search")
        expect_reachable(box, "the rule search box on step 3")
        if search is not None:
            box.fill(search)
        option = (
            page.get_by_test_id("rule-list")
            .locator('[data-testid^="rule-option-"]')
            .filter(has_text=reuse_named)
            .first
        )
        expect(option).to_be_visible()
        expect_reachable(option, f'the rule option for "{reuse_named}" on step 3')
        option.click()
    elif mode == "create":
        name_box = page.get_by_test_id("new-rule-name")
        guidance_box = page.get_by_test_id("new-rule-guidance")
        expect_all_reachable(
            {
                "the new rule's name box on step 3": name_box,
                "the new rule's guidance box on step 3": guidance_box,
            }
        )
        created = name_box.input_value()
        assert created.startswith(RULE_PREFIX), (
            f"the default rule name {created!r} is not derived from the "
            "policy, so the modal is naming rules from something else"
        )
        if guidance is not None:
            guidance_box.fill(guidance)
    else:
        expect_reachable(
            page.get_by_test_id("rule-none-hint"), "the no-guidance hint on step 3"
        )

    confirm = page.get_by_test_id("add-policy-confirm")
    expect_reachable(confirm, '"Add to playbook" on step 3')
    expect(confirm).to_be_enabled()
    confirm.click()
    expect(page.get_by_test_id("add-policy-modal")).to_have_count(0)
    return created


def _save_members(page: Page, expected_members: int, expected_states: int) -> None:
    save = page.get_by_test_id("save-members")
    expect_reachable(save, f'"Save members" with {expected_members} members staged')
    save.click()
    expect(page.get_by_test_id("members-save-report")).to_be_visible()
    expect(page.get_by_test_id("playbook-states")).to_contain_text(
        f"· {expected_states} states"
    )
    rows = page.locator('[data-testid^="member-row-"]')
    expect(rows).to_have_count(expected_members)


def _member_rows_are_operable(page: Page, policy_ids: list[str]) -> None:
    """Every saved member row, and every control on it."""
    for policy_id in policy_ids:
        row = page.get_by_test_id(f"member-row-{policy_id}")
        expect(row).to_be_visible()
        expect_all_reachable(
            {
                f"the include checkbox on member {policy_id}": page.get_by_test_id(
                    f"member-included-{policy_id}"
                ),
                f"the applies-when select on member {policy_id}": page.get_by_test_id(
                    f"member-fires-on-{policy_id}"
                ),
                f"the guidance box on member {policy_id}": page.get_by_test_id(
                    f"member-guidance-{policy_id}"
                ),
            }
        )


class TestTheHelperItself:
    """Break what the helper guards, and watch it fail.

    A reachability assertion that cannot fail is worth exactly as much as the
    ``to_be_visible()`` it replaces, and nothing in the tests below would
    notice if it silently stopped checking. These three cases pin both halves
    of it, against a real control on a real screen: the same button passes
    untouched, fails when pushed past the fold, and fails when covered.
    """

    def test_a_reachable_control_passes(self, app_page: Page, reach_env):
        _at(app_page, LAPTOP)
        expect_reachable(
            app_page.get_by_test_id("add-playbook"), "the New playbook button"
        )

    def test_a_control_pushed_past_the_fold_is_rejected(
        self, app_page: Page, reach_env
    ):
        """Off screen with nothing to scroll -- the modal bug, in miniature.

        ``position: fixed`` is what makes this the real thing rather than a
        contrivance: the page can scroll all it likes and the button does not
        move, which is precisely why the shipped modal could not be reached.
        Playwright still calls it visible, which is the point.
        """
        _at(app_page, LAPTOP)
        button = app_page.get_by_test_id("add-playbook")
        button.evaluate(
            "(el) => { el.style.position = 'fixed';"
            " el.style.top = (window.innerHeight + 200) + 'px';"
            " el.style.left = '20px'; }"
        )

        expect(button).to_be_visible()
        with pytest.raises(UnreachableError, match="outside the viewport"):
            expect_reachable(button, "the New playbook button")

    def test_a_control_under_an_overlay_is_rejected(self, app_page: Page, reach_env):
        """On screen, unscrolled, and still not clickable.

        This is the half ``to_be_visible()`` cannot see at all and the
        viewport check cannot see either: the button is exactly where it
        should be, and a click lands on the sheet over it.
        """
        _at(app_page, LAPTOP)
        button = app_page.get_by_test_id("add-playbook")
        app_page.evaluate(
            "() => { const sheet = document.createElement('div');"
            " sheet.setAttribute('data-testid', 'test-overlay');"
            " sheet.style.cssText = 'position:fixed;inset:0;z-index:2147483647;"
            "background:rgba(0,0,0,0.5)';"
            " document.body.appendChild(sheet); }"
        )

        expect(button).to_be_visible()
        with pytest.raises(UnreachableError, match="on top of it"):
            expect_reachable(button, "the New playbook button")


class TestBuildingAPlaybookThroughTheUI:
    """A playbook assembled the way a person assembles one, at three sizes.

    One playbook per viewport, grown from empty to four members through the
    guided flow, saved at two, three and four members. Saving at each size is
    what makes this a 2-, a 3- and a 4-member playbook rather than only the
    last of the three: the states pane is asserted to have doubled each time,
    so a member that never reached the server would be caught there.

    The four members exercise every branch of the rule step -- one rule
    created, one seeded rule reused, one member with no guidance at all, and
    finally the rule created in step one reused by a later member, which is
    the only thing that proves a rule minted mid-flow is immediately
    available to the next policy.
    """

    @pytest.mark.parametrize("size", VIEWPORTS, ids=VIEWPORT_IDS)
    def test_four_members_added_through_the_guided_flow_stay_operable(
        self, app_page: Page, reach_env, size: tuple[int, int]
    ):
        width, height = size
        name = f"{PREFIX} built {width}x{height}"
        _at(app_page, size)

        # The playbook itself is created through the UI too -- "entirely
        # through the UI" starts at New playbook, not at the first member.
        add = app_page.get_by_test_id("add-playbook")
        expect_reachable(add, "the New playbook button")
        add.click()
        expect_all_reachable(
            {
                "the new playbook name box": app_page.get_by_test_id(
                    "new-playbook-name-input"
                ),
                "the new playbook description box": app_page.get_by_test_id(
                    "new-playbook-description-input"
                ),
                "Cancel on the new playbook form": app_page.get_by_test_id(
                    "new-playbook-cancel"
                ),
                "Create on the new playbook form": app_page.get_by_test_id(
                    "new-playbook-save"
                ),
            }
        )
        app_page.get_by_test_id("new-playbook-name-input").fill(name)
        app_page.get_by_test_id("new-playbook-save").click()

        _open_editor(app_page, name)
        expect(app_page.get_by_test_id("no-members")).to_be_visible()

        policies = reach_env["policies"]

        # Member 1 -- create a new rule, named after the policy.
        _step_one_pick_policy(
            app_page, policies["alpha"], reach_env["names"]["alpha"]
        )
        _step_two_fires_on(app_page, satisfied=False)
        created_rule = _step_three_rule(
            app_page, "create", guidance="Guidance minted during the guided flow."
        )
        assert created_rule, "the create branch did not report the rule it named"

        # Member 2 -- reuse a rule that was already in the library.
        _step_one_pick_policy(
            app_page, policies["bravo"], reach_env["names"]["bravo"]
        )
        _step_two_fires_on(app_page, satisfied=True)
        _step_three_rule(
            app_page,
            "reuse",
            search="seed_shared",
            reuse_named=reach_env["seeded"]["shared"],
        )

        _save_members(app_page, expected_members=2, expected_states=4)
        _member_rows_are_operable(
            app_page, [policies["alpha"], policies["bravo"]]
        )

        # Member 3 -- no guidance at all.
        _step_one_pick_policy(
            app_page, policies["charlie"], reach_env["names"]["charlie"]
        )
        _step_two_fires_on(app_page, satisfied=False)
        _step_three_rule(app_page, "none")

        _save_members(app_page, expected_members=3, expected_states=8)
        _member_rows_are_operable(
            app_page, [policies["alpha"], policies["bravo"], policies["charlie"]]
        )

        # Member 4 -- reuse the rule member 1 created a moment ago.
        _step_one_pick_policy(
            app_page, policies["delta"], reach_env["names"]["delta"]
        )
        _step_two_fires_on(app_page, satisfied=False)
        _step_three_rule(
            app_page, "reuse", search=created_rule, reuse_named=created_rule
        )

        _save_members(app_page, expected_members=4, expected_states=16)
        _member_rows_are_operable(
            app_page, [policies[keyword] for keyword in KEYWORDS]
        )

        # The rule created in the flow is a real library rule, and the member
        # that reused it points at the same one rather than at a copy.
        expect(
            app_page.get_by_test_id(f"member-rule-{policies['alpha']}")
        ).to_contain_text(created_rule)
        expect(
            app_page.get_by_test_id(f"member-rule-{policies['delta']}")
        ).to_contain_text(created_rule)

        # Read back from the server, not from the screen: everything above
        # asserted what the editor drew, and a playbook that only exists in a
        # component's state is exactly the failure that would look identical.
        with _api() as client:
            listed = next(
                p for p in client.get("/playbooks").json() if p["name"] == name
            )
            states = client.get(
                f"/playbooks/{listed['playbook_id']}/states"
            ).json()
        assert listed["member_count"] == 4, (
            "four members were added through the UI but the server holds "
            f"{listed['member_count']}"
        )
        assert listed["state_count"] == 16
        keys = [
            state["state_key"]
            for behaviour in states["behaviours"]
            for state in behaviour["states"]
        ]
        assert len(keys) == 16
        for keyword in KEYWORDS:
            assert all(policies[keyword] in key for key in keys), (
                f"the {keyword} member never reached the stored state space"
            )


class TestStatesAtScale:
    """Sixteen states on a laptop screen -- the table and the graph.

    Both panes sit below a members list that is itself four rows tall, and the
    graph gained a ``max-h-[65vh] overflow-auto`` box that is the only thing
    keeping sixteen nodes on screen. The smallest of the three viewports is
    where either of those runs out of room.
    """

    def test_the_states_table_is_operable_at_sixteen_states(
        self, app_page: Page, reach_env
    ):
        _at(app_page, LAPTOP)
        _open_editor(app_page, reach_env["wide_name"])
        expect(app_page.get_by_test_id("playbook-states")).to_contain_text(
            "16 behaviours · 16 states"
        )

        expect_all_reachable(
            {
                "the only-customised filter": app_page.get_by_test_id(
                    "filter-only-customised"
                ),
                "the only-flagged filter": app_page.get_by_test_id(
                    "filter-only-flagged"
                ),
                "the Table/Graph toggle": app_page.get_by_test_id(
                    "states-view-table"
                ),
            }
        )

        # Every behaviour's header and every state row's Edit button, all
        # sixteen of them -- the last is the one at the bottom of a long page.
        toggles = app_page.locator('[data-testid^="behaviour-toggle-"]')
        expect(toggles).to_have_count(16)
        for index in range(16):
            expect_reachable(
                toggles.nth(index), f"behaviour header {index + 1} of 16"
            )

        edits = app_page.locator('[data-testid^="edit-"]')
        expect(edits).to_have_count(16)
        for index in range(16):
            expect_reachable(
                edits.nth(index), f"the Edit button on state row {index + 1} of 16"
            )

    def test_the_graph_is_operable_at_sixteen_behaviours(
        self, app_page: Page, reach_env
    ):
        _at(app_page, LAPTOP)
        _open_editor(app_page, reach_env["wide_name"])

        graph_toggle = app_page.get_by_test_id("states-view-graph")
        expect_reachable(graph_toggle, "the Graph view toggle")
        graph_toggle.click()
        expect(app_page.get_by_test_id("playbook-graph")).to_be_visible()

        expect_reachable(
            app_page.get_by_test_id("graph-member-legend"), "the graph's member legend"
        )

        nodes = app_page.locator('[data-testid^="node-"]')
        expect(nodes).to_have_count(16)
        for index in range(16):
            expect_reachable(nodes.nth(index), f"graph node {index + 1} of 16")


class TestEveryModalOnALaptopScreen:
    """Deliverable 4: each dialog and inline pane in the feature, at 1280x720.

    The fix that started this went into the shared ``Modal``, so it "should"
    hold everywhere -- and "should" is what failed last time. Each of these
    opens the thing and measures its actions rather than assuming the shared
    component covers it.
    """

    def test_the_add_policy_create_branch_with_a_long_default_name(
        self, app_page: Page, reach_env
    ):
        """The tallest the add-policy modal ever gets.

        A policy with a very long name fills the create branch's name box with
        a very long derived name, and puts the "already held" warning under it
        -- radios, two boxes, a warning and the confirm row, in one panel.
        """
        _at(app_page, LAPTOP)
        _open_editor(app_page, reach_env["wide_name"])

        _step_one_pick_policy(
            app_page, reach_env["policies"]["long"], reach_env["names"]["long"]
        )
        _step_two_fires_on(app_page, satisfied=False)

        page = app_page
        expect_all_reachable(
            {
                "the reuse radio on step 3": page.get_by_test_id("rule-mode-reuse"),
                "the create radio on step 3": page.get_by_test_id("rule-mode-create"),
                "the no-guidance radio on step 3": page.get_by_test_id(
                    "rule-mode-none"
                ),
            }
        )
        page.get_by_test_id("rule-mode-create").check()

        name_box = page.get_by_test_id("new-rule-name")
        expect_reachable(name_box, "the long derived rule name box")
        derived = name_box.input_value()
        assert len(derived) > 60, (
            f"the derived name is only {len(derived)} characters, so this is "
            "not testing the long-name branch"
        )
        page.get_by_test_id("new-rule-guidance").fill("x" * 400)

        expect_all_reachable(
            {
                "the new rule's guidance box": page.get_by_test_id(
                    "new-rule-guidance"
                ),
                "Back on the long-name step 3": page.get_by_test_id(
                    "add-policy-back"
                ),
                '"Add to playbook" on the long-name step 3': page.get_by_test_id(
                    "add-policy-confirm"
                ),
                "the modal's close button": page.get_by_test_id("modal-close"),
            }
        )

        # Nothing is added: this playbook is a fixture the other tests read.
        page.get_by_test_id("modal-close").click()
        expect(page.get_by_test_id("add-policy-modal")).to_have_count(0)

    def test_the_rule_library_editor(self, app_page: Page, reach_env):
        _at(app_page, LAPTOP)
        library = app_page.get_by_test_id("open-rule-library")
        expect_reachable(library, "the Rule library button")
        library.click()
        expect(app_page.get_by_test_id("rule-library")).to_be_visible()

        search = app_page.get_by_test_id("rule-search")
        expect_reachable(search, "the rule library's search box")
        search.fill(reach_env["seeded"]["shared"])

        row = app_page.locator('[data-testid^="rule-row-"]').first
        expect(row).to_be_visible()
        edit = row.locator('[data-testid^="rule-edit-"]')
        expect_reachable(edit, "the Edit button on a library rule")
        edit.click()

        expect(app_page.get_by_test_id("rule-editor")).to_be_visible()
        expect_all_reachable(
            {
                "the rule editor's name box": app_page.get_by_test_id(
                    "rule-editor-name"
                ),
                "the rule editor's guidance box": app_page.get_by_test_id(
                    "rule-editor-guidance"
                ),
                "Cancel in the rule editor": app_page.get_by_test_id(
                    "rule-editor-cancel"
                ),
                "Save in the rule editor": app_page.get_by_test_id(
                    "rule-editor-save"
                ),
                "the rule editor's close button": app_page.get_by_test_id(
                    "modal-close"
                ),
            }
        )
        app_page.get_by_test_id("rule-editor-cancel").click()
        expect(app_page.get_by_test_id("rule-editor")).to_have_count(0)

    def test_the_rule_delete_confirmation(self, app_page: Page, reach_env):
        """Opened on this module's own seeded rule, and never confirmed."""
        _at(app_page, LAPTOP)
        app_page.get_by_test_id("open-rule-library").click()
        app_page.get_by_test_id("rule-search").fill(reach_env["seeded"]["second"])

        row = app_page.locator('[data-testid^="rule-row-"]').first
        expect(row).to_be_visible()
        delete = row.locator('[data-testid^="rule-delete-"]')
        expect_reachable(delete, "the Delete button on a library rule")
        delete.click()

        prompt = app_page.locator('[data-testid^="rule-delete-prompt-"]')
        expect(prompt).to_be_visible()
        expect_all_reachable(
            {
                "the delete confirmation prompt": prompt,
                "Cancel in the delete confirmation": prompt.locator(
                    '[data-testid^="rule-delete-cancel-"]'
                ),
                "Delete in the delete confirmation": prompt.locator(
                    '[data-testid^="rule-delete-confirm-"]'
                ),
            }
        )
        prompt.locator('[data-testid^="rule-delete-cancel-"]').click()
        expect(prompt).to_have_count(0)

    def test_the_state_override_editor_on_the_last_of_sixteen_states(
        self, app_page: Page, reach_env
    ):
        """The pane furthest down the longest page in the feature.

        Opened on the sixteenth state rather than the first: the editor
        expands *below* an already tall table, so the first row would prove
        nothing about the row that is actually at risk.
        """
        _at(app_page, LAPTOP)
        _open_editor(app_page, reach_env["wide_name"])

        last_edit = app_page.locator('[data-testid^="edit-"]').last
        expect_reachable(last_edit, "the Edit button on the sixteenth state row")
        last_edit.click()

        editor = app_page.locator('[data-testid^="state-override-"]')
        expect(editor).to_be_visible()
        expect_all_reachable(
            {
                "the flag checkbox in the state override editor": editor.get_by_test_id(
                    "override-flagged"
                ),
                "the label box in the state override editor": editor.get_by_test_id(
                    "override-label"
                ),
                "the derived-guidance radio": editor.get_by_test_id(
                    "override-source-derived"
                ),
                "the no-guidance radio": editor.get_by_test_id("override-source-none"),
                "the pinned-guidance radio": editor.get_by_test_id(
                    "override-source-pinned"
                ),
            }
        )

        # Pinning expands the panel by one checkbox per member rule, which is
        # the tallest this editor gets.
        editor.get_by_test_id("override-source-pinned").check()
        refs = editor.get_by_test_id("override-refs")
        expect(refs).to_be_visible()
        checkboxes = refs.locator('[data-testid^="override-ref-"]')
        for index in range(checkboxes.count()):
            expect_reachable(
                checkboxes.nth(index), f"pinnable rule {index + 1} in the override editor"
            )

        expect_all_reachable(
            {
                '"Save state" in the state override editor': editor.get_by_test_id(
                    "override-save"
                ),
                "Cancel in the state override editor": editor.get_by_test_id(
                    "override-cancel"
                ),
            }
        )

        # Cancelled, not saved: this playbook is a fixture the tests above read.
        editor.get_by_test_id("override-cancel").click()
        expect(app_page.locator('[data-testid^="state-override-"]')).to_have_count(0)

    def test_the_new_playbook_form_and_the_card_delete_button(
        self, app_page: Page, reach_env
    ):
        """The list view's two actions, with the list already populated.

        The form pushes every card down the page, so the delete button being
        reachable is only interesting once the form above it is open.
        """
        _at(app_page, LAPTOP)
        add = app_page.get_by_test_id("add-playbook")
        expect_reachable(add, "the New playbook button")
        add.click()

        expect_all_reachable(
            {
                "the new playbook name box": app_page.get_by_test_id(
                    "new-playbook-name-input"
                ),
                "the new playbook description box": app_page.get_by_test_id(
                    "new-playbook-description-input"
                ),
                "Cancel on the new playbook form": app_page.get_by_test_id(
                    "new-playbook-cancel"
                ),
                "Create on the new playbook form": app_page.get_by_test_id(
                    "new-playbook-save"
                ),
            }
        )

        card = app_page.locator('[data-testid^="playbook-card-"]').filter(
            has_text=reach_env["wide_name"]
        )
        expect_reachable(
            card.get_by_role("button", name=f"Delete {reach_env['wide_name']}"),
            "the Delete button on a playbook card",
        )

        app_page.get_by_test_id("new-playbook-cancel").click()
        expect(app_page.get_by_test_id("new-playbook-form")).to_have_count(0)


class TestAShortBrowserWindow:
    """The reported bug, at the only size that actually reproduces it.

    Deliverable 3's three viewports are all *larger* than the window the bug
    was seen in -- a Playwright viewport is content height, so 1280x720 there
    is more room than a maximised browser gets on a 1280x720 screen. Measured
    against the panel that shipped, the tallest add-policy step is 646px and
    is centred with no cap, so its confirm button only leaves the screen below
    598px of content, and the reuse branch's shorter panel below 512px.

    So the three required sizes assert that the flow is operable; this one
    asserts the thing that broke. All three rule branches are driven, because
    each grows the panel by a different amount and the one that was reported
    -- reuse, with a populated list -- is the *shortest* of the three.

    Nothing is confirmed here: the fixture playbook these open belongs to the
    tests above, and adding a member to it would change what they count.
    """

    @pytest.mark.parametrize("mode", ["reuse", "create", "none"])
    def test_the_add_policy_actions_stay_on_screen(
        self, app_page: Page, reach_env, mode: str
    ):
        _at(app_page, SHORT_WINDOW)
        _open_editor(app_page, reach_env["wide_name"])
        _step_one_pick_policy(
            app_page, reach_env["policies"]["long"], reach_env["names"]["long"]
        )
        _step_two_fires_on(app_page, satisfied=False)

        page = app_page
        expect_all_reachable(
            {
                f"the {mode} radio in a short window": page.get_by_test_id(
                    f"rule-mode-{mode}"
                ),
                "Back on step 3 in a short window": page.get_by_test_id(
                    "add-policy-back"
                ),
            }
        )
        page.get_by_test_id(f"rule-mode-{mode}").check()

        if mode == "reuse":
            expect_reachable(
                page.get_by_test_id("rule-search"),
                "the rule search box in a short window",
            )
            options = page.get_by_test_id("rule-list").locator(
                '[data-testid^="rule-option-"]'
            )
            # The list having content is the whole point: it is what grows the
            # panel, and a run against an empty library would pass here while
            # the product was broken.
            assert options.count() > 0, (
                "the rule library is empty, so this is not exercising the case "
                "that shipped broken"
            )
            expect_reachable(
                options.first, "the first rule option in a short window"
            )
            options.first.click()
        elif mode == "create":
            # The collision warning is seeded, so this is the tallest the
            # panel gets rather than merely a tall one.
            expect(page.get_by_test_id("rule-name-taken")).to_be_visible()
            expect_reachable(
                page.get_by_test_id("new-rule-guidance"),
                "the new rule's guidance box in a short window",
            )

        confirm = page.get_by_test_id("add-policy-confirm")
        expect_reachable(
            confirm, f'"Add to playbook" on the {mode} branch in a short window'
        )
        expect(confirm).to_be_enabled()

        close = page.get_by_test_id("modal-close")
        expect_reachable(close, "the modal's close button in a short window")
        close.click()
        expect(page.get_by_test_id("add-policy-modal")).to_have_count(0)
