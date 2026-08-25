"""
E2E tests: Rules screen — propositions and policies CRUD.
"""

from __future__ import annotations

import json
import re

from playwright.sync_api import Page, expect

#: A predicate and a policy as the API serves them, for the halves of the
#: empty-state contract that need a non-empty list.
LISTED_PREDICATE = {
    "prop_id": "e2e_rules_listed_u",
    "description": "a predicate the view under test is given to list",
    "role": "user",
    "grounding_scope": "single_message",
    "arity": 0,
    "arg_descriptions": [],
}

LISTED_POLICY = {
    "policy_id": "e2e-rules-listed",
    "name": "e2e-rules listed policy",
    "formula_str": f"! {LISTED_PREDICATE['prop_id']}",
    "propositions": [LISTED_PREDICATE["prop_id"]],
    "enabled": True,
}


def _serve(page: Page, collection: str, items: list[dict]) -> None:
    """Serve a fixed list for one API collection, for this page only.

    The contract is that the empty-state message shows *iff* the list is
    empty, and its empty half cannot be asserted against a shared development
    database without deleting whatever that database already holds. Handing
    the view its list here pins both halves on any machine: the emptiness
    under test is the view's input rather than the developer's data. The
    latest registered handler wins, so calling this again switches the answer.
    """
    page.route(
        f"**/api/{collection}",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(items),
        ),
    )


def _reopen_rules(page: Page) -> None:
    """Leave the Rules screen and come back, so its lists are fetched again."""
    page.click('[data-testid="nav-chat"]')
    page.click('[data-testid="nav-rules"]')


class TestRulesPageLoad:
    """Verify the Rules page renders correctly."""

    def test_rules_page_renders(self, app_page: Page):
        """Rules page loads with heading."""
        app_page.click('[data-testid="nav-rules"]')
        expect(app_page.locator('[data-testid="rules-view"]')).to_be_visible()

    def test_propositions_heading(self, app_page: Page):
        """Predicates section heading is visible."""
        app_page.click('[data-testid="nav-rules"]')
        expect(
            app_page.get_by_role("heading", name="Predicates", exact=True)
        ).to_be_visible()

    def test_policies_heading(self, app_page: Page):
        """Policies section heading is visible."""
        app_page.click('[data-testid="nav-rules"]')
        expect(
            app_page.get_by_role("heading", name="Policies", exact=True)
        ).to_be_visible()

    def test_add_proposition_button(self, app_page: Page):
        """Add proposition button is visible."""
        app_page.click('[data-testid="nav-rules"]')
        expect(app_page.locator('[data-testid="add-proposition"]')).to_be_visible()

    def test_add_policy_button(self, app_page: Page):
        """Add policy button is visible."""
        app_page.click('[data-testid="nav-rules"]')
        expect(app_page.locator('[data-testid="add-policy"]')).to_be_visible()

    def test_propositions_empty_message_shows_iff_the_list_is_empty(
        self, app_page: Page
    ):
        """No predicates listed, the message; one listed, no message.

        Asserting only the empty half would make the test a claim about the
        developer's database rather than about the view, and it would fail on
        every machine that has predicates in it.
        """
        message = app_page.locator('[data-testid="no-propositions"]')
        cards = app_page.locator('[data-testid^="proposition-card-"]')

        _serve(app_page, "propositions", [])
        app_page.click('[data-testid="nav-rules"]')
        expect(cards).to_have_count(0)
        expect(message).to_be_visible()

        _serve(app_page, "propositions", [LISTED_PREDICATE])
        _reopen_rules(app_page)
        expect(cards).to_have_count(1)
        expect(message).to_have_count(0)

    def test_policies_empty_message_shows_iff_the_list_is_empty(
        self, app_page: Page
    ):
        """No policies listed, the message; one listed, no message."""
        message = app_page.locator('[data-testid="no-policies"]')
        cards = app_page.locator('[data-testid^="policy-card-"]')

        _serve(app_page, "policies", [])
        app_page.click('[data-testid="nav-rules"]')
        expect(cards).to_have_count(0)
        expect(message).to_be_visible()

        _serve(app_page, "policies", [LISTED_POLICY])
        _reopen_rules(app_page)
        expect(cards).to_have_count(1)
        expect(message).to_have_count(0)


class TestPropositionEditor:
    """Verify proposition creation modal."""

    def test_clicking_add_opens_modal(self, app_page: Page):
        """Clicking Add opens the proposition editor modal."""
        app_page.click('[data-testid="nav-rules"]')
        app_page.click('[data-testid="add-proposition"]')
        expect(app_page.locator('[data-testid="modal"]')).to_be_visible()
        expect(
            app_page.get_by_role("heading", name="New Predicate", exact=True)
        ).to_be_visible()

    def test_modal_has_prop_id_input(self, app_page: Page):
        """Modal contains proposition ID input."""
        app_page.click('[data-testid="nav-rules"]')
        app_page.click('[data-testid="add-proposition"]')
        expect(app_page.locator('[data-testid="prop-id-input"]')).to_be_visible()

    def test_modal_has_role_select(self, app_page: Page):
        """Modal contains role radio buttons."""
        app_page.click('[data-testid="nav-rules"]')
        app_page.click('[data-testid="add-proposition"]')
        expect(app_page.locator('[data-testid="prop-role-user"]')).to_be_visible()
        expect(app_page.locator('[data-testid="prop-role-assistant"]')).to_be_visible()

    def test_modal_has_description_input(self, app_page: Page):
        """Modal contains description textarea."""
        app_page.click('[data-testid="nav-rules"]')
        app_page.click('[data-testid="add-proposition"]')
        expect(app_page.locator('[data-testid="prop-description-input"]')).to_be_visible()

    def test_save_disabled_when_empty(self, app_page: Page):
        """Save button is disabled when fields are empty."""
        app_page.click('[data-testid="nav-rules"]')
        app_page.click('[data-testid="add-proposition"]')
        expect(app_page.locator('[data-testid="prop-save"]')).to_be_disabled()

    def test_save_enabled_when_filled(self, app_page: Page):
        """Save button enables when all fields are filled."""
        app_page.click('[data-testid="nav-rules"]')
        app_page.click('[data-testid="add-proposition"]')
        app_page.locator('[data-testid="prop-id-input"]').fill("p_test")
        app_page.locator('[data-testid="prop-description-input"]').fill("Test description")
        expect(app_page.locator('[data-testid="prop-save"]')).to_be_enabled()

    def test_cancel_closes_modal(self, app_page: Page):
        """Cancel button closes the modal."""
        app_page.click('[data-testid="nav-rules"]')
        app_page.click('[data-testid="add-proposition"]')
        app_page.click('[data-testid="prop-cancel"]')
        expect(app_page.locator('[data-testid="modal"]')).not_to_be_visible()

    def test_close_button_closes_modal(self, app_page: Page):
        """X button closes the modal."""
        app_page.click('[data-testid="nav-rules"]')
        app_page.click('[data-testid="add-proposition"]')
        app_page.click('[data-testid="modal-close"]')
        expect(app_page.locator('[data-testid="modal"]')).not_to_be_visible()

    def test_user_role_selected_by_default(self, app_page: Page):
        """User role is selected by default."""
        app_page.click('[data-testid="nav-rules"]')
        app_page.click('[data-testid="add-proposition"]')
        expect(app_page.locator('[data-testid="prop-role-user"]')).to_be_checked()

    def test_can_select_assistant_role(self, app_page: Page):
        """Can switch to assistant role."""
        app_page.click('[data-testid="nav-rules"]')
        app_page.click('[data-testid="add-proposition"]')
        app_page.locator('[data-testid="prop-role-assistant"]').click()
        expect(app_page.locator('[data-testid="prop-role-assistant"]')).to_be_checked()
        expect(app_page.locator('[data-testid="prop-role-user"]')).not_to_be_checked()

    def test_prop_id_placeholder(self, app_page: Page):
        """Proposition ID input has a placeholder."""
        app_page.click('[data-testid="nav-rules"]')
        app_page.click('[data-testid="add-proposition"]')
        expect(app_page.locator('[data-testid="prop-id-input"]')).to_have_attribute(
            "placeholder", "p_fraud"
        )


class TestFormulaBuilder:
    """Verify formula builder modal."""

    def test_clicking_add_policy_opens_modal(self, app_page: Page):
        """Clicking Add policy opens the formula builder modal."""
        app_page.click('[data-testid="nav-rules"]')
        app_page.locator('[data-testid="add-policy"]').click()
        expect(app_page.locator('[data-testid="modal"]')).to_be_visible()
        expect(
            app_page.get_by_role("heading", name="New Policy", exact=True)
        ).to_be_visible()

    def test_policy_name_input_present(self, app_page: Page):
        """Policy name input exists in formula builder."""
        app_page.click('[data-testid="nav-rules"]')
        app_page.locator('[data-testid="add-policy"]').click()
        expect(app_page.locator('[data-testid="policy-name-input"]')).to_be_visible()

    def test_formula_input_present(self, app_page: Page):
        """Formula input field exists."""
        app_page.click('[data-testid="nav-rules"]')
        app_page.locator('[data-testid="add-policy"]').click()
        expect(app_page.locator('[data-testid="formula-input"]')).to_be_visible()

    def test_operator_buttons_present(self, app_page: Page):
        """Operator buttons section is visible."""
        app_page.click('[data-testid="nav-rules"]')
        app_page.locator('[data-testid="add-policy"]').click()
        expect(app_page.locator('[data-testid="operator-buttons"]')).to_be_visible()

    def test_temporal_reference_panel(self, app_page: Page):
        """Temporal operators reference panel is visible."""
        app_page.click('[data-testid="nav-rules"]')
        app_page.locator('[data-testid="add-policy"]').click()
        expect(app_page.locator("text=DejaVu Operators Reference")).to_be_visible()

    def test_save_disabled_initially(self, app_page: Page):
        """Save button is disabled when formula is empty."""
        app_page.click('[data-testid="nav-rules"]')
        app_page.locator('[data-testid="add-policy"]').click()
        expect(app_page.locator('[data-testid="policy-save"]')).to_be_disabled()

    def test_cancel_closes_modal(self, app_page: Page):
        """Cancel button closes formula builder."""
        app_page.click('[data-testid="nav-rules"]')
        app_page.locator('[data-testid="add-policy"]').click()
        app_page.click('[data-testid="policy-cancel"]')
        expect(app_page.locator('[data-testid="modal"]')).not_to_be_visible()

    def test_formula_input_monospace(self, app_page: Page):
        """Formula input uses monospace font."""
        app_page.click('[data-testid="nav-rules"]')
        app_page.locator('[data-testid="add-policy"]').click()
        formula_input = app_page.locator('[data-testid="formula-input"]')
        expect(formula_input).to_have_class(re.compile(r"font-mono"))

    def test_add_policy_enabled_without_propositions(self, app_page: Page):
        """Add policy stays enabled with no predicates: user_turn is built in."""
        app_page.click('[data-testid="nav-rules"]')
        add_btn = app_page.locator('[data-testid="add-policy"]')
        expect(add_btn).to_be_enabled()
