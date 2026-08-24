"""Guidance reaches the model as an ephemeral system message.

Two properties matter: the model must see guidance as instruction rather than
as something the user said, and the stored conversation must stay verbatim so
guidance never accumulates in history.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.routers.chat import render_guidance
from backend.store.db import DatabaseStore


def test_render_guidance_is_a_labelled_bullet_list():
    rendered = render_guidance(["Stay within budget.", "Avoid the allergen."])

    assert rendered == (
        "Active guidance:\n- Stay within budget.\n- Avoid the allergen."
    )


def test_render_guidance_of_one_rule():
    assert render_guidance(["Only this."]) == "Active guidance:\n- Only this."


class _StubGrounding:
    """Grounds p_a False (fires) and p_b True (does not)."""

    async def evaluate(self, message, proposition, **kwargs):
        from backend.engine.grounding import GroundingResult
        return GroundingResult(
            match=proposition.prop_id == "p_a", confidence=1.0,
            reasoning="stub", method="stub", prop_id=proposition.prop_id,
        )


async def _seed(db_path: str) -> None:
    db = DatabaseStore(db_path)
    await db.initialize()
    await db.create_proposition("p_a", "a", "user")
    await db.create_policy("pol-a", "A", "p_a", True)
    await db.set_policy_propositions("pol-a", ["p_a"])
    await db.create_playbook("pb1", "Budget")
    await db.set_playbook_members("pb1", [
        {"policy_id": "pol-a", "position": 0, "fires_on": True,
         "guidance": "Stay within budget."}])
    await db.create_session("s1")
    await db.set_session_monitoring("s1", "playbook", "pb1")
    await db.set_setting("openrouter_api_key", "simulated")
    await db.set_setting("openrouter_model", "simulated/model")
    await db.close()


@pytest.mark.parametrize("expect_guidance", [True])
def test_guidance_is_sent_as_system_before_the_user_turn(tmp_path, monkeypatch,
                                                         expect_guidance):
    db_path = str(tmp_path / "chat.db")
    monkeypatch.setenv("DATABASE_PATH", db_path)
    asyncio.run(_seed(db_path))

    captured: dict = {}

    async def _fake_chat(messages, model=None):
        captured["messages"] = messages
        return "sure"

    import backend.routers.chat as chat_mod
    chat_mod.invalidate_monitors()

    with patch.object(chat_mod, "LLMGrounding", lambda **kw: _StubGrounding()), \
         patch("backend.routers.chat.OpenRouterClient") as mock_or:
        mock_or.return_value.chat = AsyncMock(side_effect=_fake_chat)
        with TestClient(create_app()) as client:
            client.post("/api/chat", json={"message": "hi", "session_id": "s1"})

    roles = [m.role for m in captured["messages"]]
    assert roles[-2:] == ["system", "user"]
    assert "Stay within budget." in captured["messages"][-2].content


def test_the_stored_user_message_stays_verbatim(tmp_path, monkeypatch):
    db_path = str(tmp_path / "chat2.db")
    monkeypatch.setenv("DATABASE_PATH", db_path)
    asyncio.run(_seed(db_path))

    import backend.routers.chat as chat_mod
    chat_mod.invalidate_monitors()

    with patch.object(chat_mod, "LLMGrounding", lambda **kw: _StubGrounding()), \
         patch("backend.routers.chat.OpenRouterClient") as mock_or:
        mock_or.return_value.chat = AsyncMock(return_value="sure")
        with TestClient(create_app()) as client:
            client.post("/api/chat", json={"message": "hi", "session_id": "s1"})
            body = client.get("/api/chat/sessions/s1").json()

    user_messages = [m for m in body["messages"] if m["role"] == "user"]
    assert user_messages[0]["content"] == "hi"


def test_switching_monitoring_mode(tmp_path, monkeypatch):
    db_path = str(tmp_path / "chat3.db")
    monkeypatch.setenv("DATABASE_PATH", db_path)
    asyncio.run(_seed(db_path))

    with TestClient(create_app()) as client:
        resp = client.patch("/api/chat/sessions/s1/monitoring",
                            json={"mode": "policies"})
        assert resp.status_code == 200
        assert resp.json()["monitoring_mode"] == "policies"
        assert resp.json()["playbook_id"] is None


def test_monitoring_mode_round_trips_through_get_and_list(tmp_path, monkeypatch):
    """R-20: the mode set via PATCH .../monitoring must be readable back --
    otherwise a client has no way to know which specification a session is
    actually running, and would show a mode that doesn't match reality."""
    db_path = str(tmp_path / "chat4.db")
    monkeypatch.setenv("DATABASE_PATH", db_path)
    asyncio.run(_seed(db_path))

    with TestClient(create_app()) as client:
        # s1 was seeded straight into playbook mode -- both read paths must
        # agree with that, not just the write path that set it.
        get_body = client.get("/api/chat/sessions/s1").json()
        assert get_body["monitoring_mode"] == "playbook"
        assert get_body["playbook_id"] == "pb1"

        list_body = client.get("/api/chat/sessions").json()
        s1 = next(s for s in list_body if s["session_id"] == "s1")
        assert s1["monitoring_mode"] == "playbook"
        assert s1["playbook_id"] == "pb1"

        # Switch back to policies through the API and confirm both read
        # paths follow.
        resp = client.patch("/api/chat/sessions/s1/monitoring",
                            json={"mode": "policies"})
        assert resp.status_code == 200

        get_body = client.get("/api/chat/sessions/s1").json()
        assert get_body["monitoring_mode"] == "policies"
        assert get_body["playbook_id"] is None

        list_body = client.get("/api/chat/sessions").json()
        s1 = next(s for s in list_body if s["session_id"] == "s1")
        assert s1["monitoring_mode"] == "policies"
        assert s1["playbook_id"] is None


def test_never_switched_session_reads_as_policies(tmp_path, monkeypatch):
    """A session row predating the monitoring_mode column (or simply never
    PATCHed) must read as policy mode with no playbook, not None/missing."""
    db_path = str(tmp_path / "chat5.db")
    monkeypatch.setenv("DATABASE_PATH", db_path)
    asyncio.run(_seed(db_path))

    with TestClient(create_app()) as client:
        session_id = client.post("/api/chat/sessions").json()["session_id"]

        get_body = client.get(f"/api/chat/sessions/{session_id}").json()
        assert get_body["monitoring_mode"] == "policies"
        assert get_body["playbook_id"] is None

        list_body = client.get("/api/chat/sessions").json()
        row = next(s for s in list_body if s["session_id"] == session_id)
        assert row["monitoring_mode"] == "policies"
        assert row["playbook_id"] is None


class _FakeRequest:
    """Just enough of fastapi.Request for chat._get_db(request)."""

    def __init__(self, db: DatabaseStore) -> None:
        self.app = type("_App", (), {"state": type("_State", (), {"db": db})()})()


async def _open_seeded_db(tmp_path, name: str) -> DatabaseStore:
    db_path = str(tmp_path / name)
    await _seed(db_path)
    db = DatabaseStore(db_path)
    await db.initialize()
    await db.set_session_monitoring("s1", "policies", None)
    return db


async def test_a_mode_switch_mid_construction_does_not_resurrect_the_old_monitor(
    tmp_path,
):
    """R-26: a PATCH landing between the mode read and the cache store.

    _get_or_create_monitor reads monitoring_mode, then awaits several DB
    round-trips before storing the monitor. A mode switch inside that window
    pops nothing -- there is nothing cached yet -- and the store afterwards
    would put a monitor built from the OLD mode back into the cache, where it
    survives every later turn. The session would then keep enforcing the old
    specification while the UI shows the new one.
    """
    import backend.routers.chat as chat_mod

    db = await _open_seeded_db(tmp_path, "race.db")
    chat_mod.invalidate_monitors()
    chat_mod._monitors.clear()
    chat_mod._monitor_generation.clear()

    started = asyncio.Event()
    release = asyncio.Event()
    real_get_policy_propositions = db.get_policy_propositions

    async def _barrier(policy_id: str):
        # Awaited strictly after the mode read and strictly before the store.
        started.set()
        await release.wait()
        return await real_get_policy_propositions(policy_id)

    db.get_policy_propositions = _barrier  # type: ignore[method-assign]

    async def _switch_mid_flight():
        await started.wait()
        # The race precondition: nothing cached, so the PATCH's pop is a no-op.
        assert "s1" not in chat_mod._monitors
        body = chat_mod.MonitoringRequest(mode="playbook", playbook_id="pb1")
        await chat_mod.set_session_monitoring(_FakeRequest(db), "s1", body)
        release.set()

    stale, _ = await asyncio.gather(
        chat_mod._get_or_create_monitor(db, "s1"), _switch_mid_flight()
    )

    # The in-flight turn finishes on the mode it started with, but that
    # staleness must not outlive it.
    assert stale._playbook is None
    assert "s1" not in chat_mod._monitors

    db.get_policy_propositions = real_get_policy_propositions  # type: ignore[method-assign]
    rebuilt = await chat_mod._get_or_create_monitor(db, "s1")
    assert rebuilt is not stale
    assert rebuilt._playbook is not None
    await db.close()


async def test_the_uncontended_path_still_caches_the_monitor(tmp_path):
    """The guard must not degrade into never caching.

    Without this, a fix that simply stopped storing monitors would pass the
    race test above while rebuilding the monitor -- and losing its DejaVu
    session -- on every single turn.
    """
    import backend.routers.chat as chat_mod

    db = await _open_seeded_db(tmp_path, "uncontended.db")
    chat_mod.invalidate_monitors()
    chat_mod._monitors.clear()
    chat_mod._monitor_generation.clear()

    first = await chat_mod._get_or_create_monitor(db, "s1")

    assert chat_mod._monitors.get("s1") is first
    assert await chat_mod._get_or_create_monitor(db, "s1") is first
    await db.close()


async def test_deleting_a_session_mid_construction_does_not_cache_its_monitor(
    tmp_path,
):
    """delete_session has the same resurrection hole as the mode switch.

    It pops _monitors, which evicts nothing while the monitor is still being
    built, and the store afterwards would cache a monitor for a session that
    no longer exists -- keyed by an id nothing will ever evict again.
    """
    import backend.routers.chat as chat_mod

    db = await _open_seeded_db(tmp_path, "delete-race.db")
    chat_mod.invalidate_monitors()
    chat_mod._monitors.clear()
    chat_mod._monitor_generation.clear()

    started = asyncio.Event()
    release = asyncio.Event()
    real_get_policy_propositions = db.get_policy_propositions

    async def _barrier(policy_id: str):
        started.set()
        await release.wait()
        return await real_get_policy_propositions(policy_id)

    db.get_policy_propositions = _barrier  # type: ignore[method-assign]

    async def _delete_mid_flight():
        await started.wait()
        assert "s1" not in chat_mod._monitors
        await chat_mod.delete_session(_FakeRequest(db), "s1")
        release.set()

    await asyncio.gather(
        chat_mod._get_or_create_monitor(db, "s1"), _delete_mid_flight()
    )

    assert "s1" not in chat_mod._monitors
    assert await db.get_session("s1") is None
    await db.close()
