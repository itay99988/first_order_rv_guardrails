"""Assert that a control is *operable*, not merely present in the DOM.

Why this exists
---------------
A modal shipped whose "Add to playbook" button sat below the bottom of the
screen with nothing to scroll. It passed 436 frontend tests and 166 e2e tests,
because neither layer can see it:

* **jsdom computes no layout.** Every box is 0x0 at (0, 0). A component that
  overflows the viewport renders and asserts exactly like one that does not,
  so no unit test can distinguish them.
* **Playwright's ``to_be_visible()`` is not reachability.** It asserts the
  element is in the DOM, has a non-empty bounding box and is not hidden by
  ``display:none``/``visibility:hidden``. An element pushed past the fold, or
  sitting under an overlay that eats its clicks, satisfies all three.

So the suite could describe a screen in detail and still not know whether a
human could finish the task on it. :func:`expect_reachable` is the missing
assertion, and it makes exactly two checks:

1. **On screen.** After scrolling every scrollable ancestor as far as it will
   go, the element's rectangle lies wholly inside the viewport. This is the
   one that catches the modal bug: the panel was ``position: fixed``, so no
   amount of page scrolling moved it, and its own body had no scroller.
2. **Not covered.** ``document.elementFromPoint`` at the element's centre
   returns that element or a descendant of it. This is the one that catches an
   overlay, a sticky header or a toast intercepting the click -- the element is
   on screen, and a click still does not reach it.

Neither check on its own is enough. The first passes for a button under a
modal backdrop; the second passes for a button 400px below the fold, because
``elementFromPoint`` outside the viewport returns nothing to compare against
and would have to be treated as "nothing in the way".

The scroll is deliberate and is not a weakening: a control further down a
scrollable page *is* reachable, because the user scrolls to it. What the modal
bug proved is that "the user can scroll to it" has to be demonstrated rather
than assumed, and ``scroll_into_view_if_needed`` demonstrates it -- when there
is no scroller that can bring the element in, the checks below still fail.

Both self-checks are exercised by ``TestTheHelperItself`` in
``test_reachability.py``: one control is pushed off screen and one is covered,
and the helper is asserted to reject each. A helper that could not fail is
worth no more than the ``to_be_visible()`` it was written to replace.
"""

from __future__ import annotations

import time
from typing import Any

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Locator

#: How long to keep re-measuring before giving up. React re-renders and CSS
#: transitions can move a box for a frame or two after the element is
#: actionable, and a single measurement taken inside that window is a flake,
#: not a finding. A failure still reports the *last* measurement, so nothing
#: is hidden by retrying -- only settled.
SETTLE_MS = 2_000


class UnreachableError(AssertionError):
    """A control is present but a human could not operate it.

    An ``AssertionError`` so pytest reports it as a failed assertion rather
    than an error, and a distinct type so the helper's own tests can assert
    that it rejected something for *this* reason and not by crashing.
    """


#: Measured in the page, in one round trip, so the rectangle and the hit test
#: describe the same instant. Two separate evaluates could straddle a scroll.
#:
#: ``documentElement.clientWidth``/``clientHeight`` rather than
#: ``window.innerWidth``/``innerHeight``: the latter includes a classic
#: scrollbar, and the strip under the scrollbar is not a place a click can
#: land. ``elementFromPoint`` agrees -- it answers in client coordinates -- so
#: this keeps the two checks measuring the same rectangle.
_MEASURE_JS = """
(el) => {
  const rect = el.getBoundingClientRect();
  const vw = document.documentElement.clientWidth;
  const vh = document.documentElement.clientHeight;
  const cx = rect.left + rect.width / 2;
  const cy = rect.top + rect.height / 2;
  const inside =
    rect.width > 0 &&
    rect.height > 0 &&
    rect.top >= 0 &&
    rect.left >= 0 &&
    rect.bottom <= vh &&
    rect.right <= vw;
  const describe = (n) => {
    if (!n) return "nothing";
    const parts = [n.tagName ? n.tagName.toLowerCase() : String(n.nodeName)];
    const testid = n.getAttribute && n.getAttribute("data-testid");
    if (testid) parts.push(`[data-testid="${testid}"]`);
    if (n.id) parts.push(`#${n.id}`);
    const cls = typeof n.className === "string" ? n.className.trim() : "";
    if (cls) parts.push("." + cls.split(/\\s+/).slice(0, 4).join("."));
    return parts.join("");
  };
  // Only meaningful for a point that is on screen; off screen it answers
  // null, which must not be read as "nothing is in the way".
  const onScreen = cx >= 0 && cy >= 0 && cx <= vw && cy <= vh;
  const hit = onScreen ? document.elementFromPoint(cx, cy) : null;
  // Blink does not hit-test disabled form controls: `elementFromPoint` over
  // one answers with whatever is *behind* it, which is its own ancestor. That
  // is not something covering the control, and reporting it as one would send
  // a reader hunting for an overlay that does not exist. Only an ancestor is
  // ever forgiven, and only for a disabled control -- an overlay is never an
  // ancestor of the thing it covers, so this cannot excuse a real one.
  const disabled =
    !!el.disabled || !!(el.closest && el.closest(":disabled,[aria-disabled=true]"));
  return {
    rect: {
      top: rect.top, left: rect.left, bottom: rect.bottom,
      right: rect.right, width: rect.width, height: rect.height,
    },
    viewport: { width: vw, height: vh },
    point: { x: cx, y: cy },
    inside_viewport: inside,
    disabled: disabled,
    hit: describe(hit),
    hit_is_self_or_descendant: !!hit && (hit === el || el.contains(hit)),
    hit_is_ancestor: !!hit && hit !== el && hit.contains(el),
  };
}
"""


def measure(locator: Locator) -> dict[str, Any]:
    """Where this element is, and what a click at its centre would land on."""
    return locator.evaluate(_MEASURE_JS)


def _why(report: dict[str, Any]) -> str | None:
    """The reason this element is unreachable, or ``None`` if it is not."""
    rect, view = report["rect"], report["viewport"]
    if rect["width"] <= 0 or rect["height"] <= 0:
        return f"it has no area: {rect['width']}x{rect['height']}"
    if not report["inside_viewport"]:
        return (
            "it lies outside the viewport -- "
            f"x {rect['left']:.0f}..{rect['right']:.0f} of "
            f"0..{view['width']}, y {rect['top']:.0f}..{rect['bottom']:.0f} of "
            f"0..{view['height']}. Nothing on the page can scroll it into "
            "view, so a user cannot reach it"
        )
    if not report["hit_is_self_or_descendant"]:
        if report["disabled"] and report["hit_is_ancestor"]:
            # Disabled, on screen, with only its own ancestor behind it: the
            # control is where the user can see it. Whether it *should* be
            # disabled is a question about the product, which the caller
            # asserts with `to_be_enabled()`; it is not a layout defect.
            return None
        return (
            "something else is on top of it: a click at its centre "
            f"({report['point']['x']:.0f}, {report['point']['y']:.0f}) lands "
            f"on {report['hit']}"
        )
    return None


def expect_reachable(locator: Locator, what: str) -> dict[str, Any]:
    """Assert a human can operate ``locator``; return where it ended up.

    ``what`` names the control in the failure message -- these assertions fire
    from loops over a dozen controls, and "an element" is not a bug report.

    Raises :class:`UnreachableError` with the measured rectangle, the viewport and,
    when something is covering the control, what that something is.
    """
    try:
        locator.scroll_into_view_if_needed(timeout=SETTLE_MS)
    except PlaywrightError as err:
        # Playwright refuses to scroll to an element it cannot make
        # actionable. That refusal is the finding, not an error in the test.
        raise UnreachableError(
            f"{what} could not be scrolled into view, so a user cannot reach "
            f"it: {err.message.splitlines()[0]}"
        ) from err

    deadline = time.monotonic() + SETTLE_MS / 1000
    while True:
        report = measure(locator)
        reason = _why(report)
        if reason is None:
            return report
        if time.monotonic() >= deadline:
            raise UnreachableError(
                f"{what} is present and 'visible', but {reason}. "
                f"Viewport {report['viewport']['width']}x"
                f"{report['viewport']['height']}."
            )
        locator.page.wait_for_timeout(100)


def expect_all_reachable(controls: dict[str, Locator]) -> None:
    """Assert every control in ``controls`` is reachable, naming each.

    Fails on the first unreachable one, because a screen with two unreachable
    controls is not twice as broken -- it is the same broken screen, and the
    first name is enough to find it.
    """
    for what, locator in controls.items():
        expect_reachable(locator, what)
