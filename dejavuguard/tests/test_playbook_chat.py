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
