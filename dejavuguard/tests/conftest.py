"""
Root test configuration.

Starts one bundled DejaVu server for the whole test session, so that
policy validation and monitoring tests exercise the real engine instead
of failing on an unreachable server. Also reorders tests so sync E2E
tests (Playwright) run after async unit tests. Without the reordering,
Playwright's sync API closes the event loop during teardown, corrupting
pytest-asyncio's runner for subsequent async tests.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

_DEJAVU_JAR = Path(__file__).resolve().parent.parent / "backend" / "libs" / "dejavu.jar"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_health(url: str, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=1):  # noqa: S310 — fixed local http URL
                return True
        except OSError:
            time.sleep(0.3)
    return False


@pytest.fixture(scope="session", autouse=True)
def dejavu_server(tmp_path_factory: pytest.TempPathFactory):
    """Run the bundled DejaVu server for the test session.

    Policy formula validation and conversation monitoring talk to the
    DejaVu HTTP server configured through DEJAVU_URL. Starting the
    bundled jar on a free port makes the suite self-contained; it only
    requires Java 17 or later, which is already a documented development
    prerequisite.
    """
    java = shutil.which("java")
    if java is None or not _DEJAVU_JAR.is_file():
        pytest.exit(
            "Java 17+ and backend/libs/dejavu.jar are required to run the "
            "backend tests (the suite starts the bundled DejaVu server).",
            returncode=2,
        )

    port = _free_port()
    storage = tmp_path_factory.mktemp("dejavu-sessions")
    proc = subprocess.Popen(  # noqa: S603 — fixed, repo-local command
        [java, "-jar", str(_DEJAVU_JAR), "--server", "--port", str(port),
         "--storage", str(storage)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    if not _wait_for_health(url):
        proc.terminate()
        pytest.exit(f"Bundled DejaVu server failed to start on {url}.", returncode=2)

    previous = os.environ.get("DEJAVU_URL")
    os.environ["DEJAVU_URL"] = url
    try:
        yield url
    finally:
        if previous is None:
            os.environ.pop("DEJAVU_URL", None)
        else:
            os.environ["DEJAVU_URL"] = previous
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


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
