"""
Playwright E2E test fixtures.

These tests require both the backend and frontend dev servers running:
  - Backend: uvicorn backend.main:app --port 8000
  - Frontend: cd frontend && npm run dev (Vite on port 5173)

The frontend Vite dev server proxies /api → http://localhost:8000.
"""

from __future__ import annotations

import pytest


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
