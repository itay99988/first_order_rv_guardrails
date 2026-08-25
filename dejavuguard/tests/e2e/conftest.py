"""
Playwright E2E test fixtures.

These tests require both the backend and frontend dev servers running:
  - Backend: uvicorn backend.main:app --port 8000
  - Frontend: cd frontend && npm run dev (Vite on port 5173)

The frontend Vite dev server proxies /api → http://localhost:8000.
"""

from __future__ import annotations

import pytest


def pytest_configure(config):
    """Coverage is meaningless for this suite, so do not gate on it.

    These tests drive a separate uvicorn process; nothing of `backend` executes
    inside the pytest process, so --cov reports 0% and --cov-fail-under=80 fails
    a run in which every test passed. Exiting 1 on success is worse than not
    measuring: it trains people to ignore the exit code.
    """
    cov = config.pluginmanager.get_plugin("_cov")
    if cov is not None and getattr(cov, "options", None) is not None:
        cov.options.cov_fail_under = 0
    config.option.cov_fail_under = 0


@pytest.fixture(scope="session")
def base_url() -> str:
    """Base URL for the frontend dev server."""
    return "http://localhost:5173"


@pytest.fixture()
def app_page(page, base_url):
    """Navigate to the app and wait for it to load."""
    page.goto(base_url)
    page.wait_for_selector('[data-testid="app-layout"]', timeout=10_000)
    return page
