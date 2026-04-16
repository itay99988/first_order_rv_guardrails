"""
DejaVu HTTP client for runtime verification.

Manages DejaVu sessions via the REST API. Each conversation gets its own
session. Events are sent as composites (simultaneous) per message step.
"""

from __future__ import annotations

import httpx
from dataclasses import dataclass, field


@dataclass
class DejaVuVerdict:
    """Result from evaluating one step in a DejaVu session."""
    event_number: int
    verdicts: dict[str, bool]    # property_name -> True/False
    violations: list[str]        # property names that are violated


class DejaVuClient:
    """Async HTTP client for DejaVu runtime verification server.

    Each conversation maps to one DejaVu session. Events are sent as
    composite events (simultaneous) per message step -- all true predicates
    from grounding are sent together.

    Future grounding may extract predicate data (args), e.g.:
      p_transfer("1234", "offshore", "50000")
    These args are passed through to DejaVu for first-order quantification.
    """

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=30.0)

    async def create_session(self, spec: str, bits: int = 20) -> tuple[str, list[str]]:
        """Create a DejaVu monitoring session.

        Args:
            spec: DejaVu specification string (first-order past-time LTL).
            bits: BDD bits per variable.

        Returns:
            Tuple of (session_id, list of property names).

        Raises:
            DejaVuError: If spec is invalid or server unreachable.
        """
        try:
            resp = await self._client.post(
                f"{self.base_url}/sessions",
                json={"spec": spec, "bits": bits}
            )
            if resp.status_code == 201:
                data = resp.json()
                return data["session_id"], data["properties"]
            elif resp.status_code == 400:
                raise DejaVuError(f"Invalid specification: {resp.json().get('error', '')}")
            else:
                raise DejaVuError(f"DejaVu server error: {resp.status_code}")
        except httpx.ConnectError:
            raise DejaVuError(f"Cannot connect to DejaVu server at {self.base_url}")

    async def send_events(
        self, session_id: str, events: list[dict[str, list[str] | str]]
    ) -> DejaVuVerdict:
        """Send composite events (simultaneous) to a session.

        All events in the list are treated as happening at the same logical
        time step. This is used to send all true predicates from grounding
        as one atomic step.

        Args:
            session_id: DejaVu session identifier.
            events: List of {"name": str, "args": list[str]} dicts.
                    args can carry extracted entity data for first-order specs.

        Returns:
            DejaVuVerdict with per-property verdicts and violations.
        """
        try:
            resp = await self._client.post(
                f"{self.base_url}/sessions/{session_id}/events",
                json=events
            )
            if resp.status_code == 200:
                data = resp.json()
                return DejaVuVerdict(
                    event_number=data["event_number"],
                    verdicts=data["verdicts"],
                    violations=data["violations"]
                )
            elif resp.status_code == 404:
                raise DejaVuSessionNotFound(session_id)
            else:
                raise DejaVuError(f"DejaVu error: {resp.status_code} {resp.text}")
        except httpx.ConnectError:
            raise DejaVuError(f"Cannot connect to DejaVu server at {self.base_url}")

    async def send_event(
        self, session_id: str, name: str, args: list[str] | None = None
    ) -> DejaVuVerdict:
        """Send a single event to a session.

        For sending one event at a time (no composite).
        """
        try:
            resp = await self._client.post(
                f"{self.base_url}/sessions/{session_id}/event",
                json={"name": name, "args": args or []}
            )
            if resp.status_code == 200:
                data = resp.json()
                return DejaVuVerdict(
                    event_number=data["event_number"],
                    verdicts=data["verdicts"],
                    violations=data["violations"]
                )
            elif resp.status_code == 404:
                raise DejaVuSessionNotFound(session_id)
            else:
                raise DejaVuError(f"DejaVu error: {resp.status_code} {resp.text}")
        except httpx.ConnectError:
            raise DejaVuError(f"Cannot connect to DejaVu server at {self.base_url}")

    async def delete_session(self, session_id: str) -> bool:
        """Delete a DejaVu session."""
        try:
            resp = await self._client.delete(
                f"{self.base_url}/sessions/{session_id}"
            )
            return resp.status_code == 200
        except httpx.ConnectError:
            return False

    async def get_session(self, session_id: str) -> dict | None:
        """Get session status. Returns None if not found."""
        try:
            resp = await self._client.get(
                f"{self.base_url}/sessions/{session_id}"
            )
            if resp.status_code == 200:
                return resp.json()
            return None
        except httpx.ConnectError:
            return None

    async def validate_spec(self, spec: str) -> tuple[bool, list[str], str | None]:
        """Validate a DejaVu specification without creating a session.

        Parses the spec server-side using DejaVu's own parser. No compilation,
        no session creation — just syntax and wellformedness checking.

        Args:
            spec: DejaVu specification string.

        Returns:
            Tuple of (valid, property_names, error_message_or_none).
        """
        try:
            resp = await self._client.post(
                f"{self.base_url}/validate",
                json={"spec": spec}
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("valid"):
                    return True, data.get("properties", []), None
                else:
                    return False, [], data.get("error", "Invalid specification")
            return False, [], f"Validation failed: {resp.status_code}"
        except httpx.ConnectError:
            raise DejaVuError(f"Cannot connect to DejaVu server at {self.base_url}")

    async def health_check(self) -> bool:
        """Check if DejaVu server is reachable."""
        try:
            resp = await self._client.get(f"{self.base_url}/health")
            return resp.status_code == 200
        except httpx.ConnectError:
            return False

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()


class DejaVuError(Exception):
    """Base error for DejaVu client."""
    pass


class DejaVuSessionNotFound(DejaVuError):
    """Session not found (expired or deleted)."""
    def __init__(self, session_id: str):
        super().__init__(f"DejaVu session not found: {session_id}")
        self.session_id = session_id
