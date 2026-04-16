"""
Root test configuration.

Reorders tests so sync E2E tests (Playwright) run after async unit tests.
Without this, Playwright's sync API closes the event loop during teardown,
corrupting pytest-asyncio's runner for subsequent async tests.
"""

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Reorder tests: run sync E2E tests AFTER async unit tests.

    This prevents Playwright's event loop teardown from corrupting
    pytest-asyncio's runner for subsequent async tests.
    """
    e2e_tests = []
    other_tests = []

    for item in items:
        if "/e2e/" in str(item.fspath):
            e2e_tests.append(item)
        else:
            other_tests.append(item)

    items[:] = other_tests + e2e_tests
