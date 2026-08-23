"""Deterministic OpenAI-compatible grounding server for offline scenario runs.

Returns a fixed grounding verdict per (predicate, phrase) rule so scenarios
are reproducible without an API key or a model. Start it with:

    uv run python -m scenario_runner.support.stub_grounding --port 9099 \
        --rules scenario_runner/support/playbook_grounding.json
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar


class _Handler(BaseHTTPRequestHandler):
    rules: ClassVar[list[dict]] = []

    def log_message(self, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        self._send({"data": [{"id": "stub-grounder"}]})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = "\n".join(m.get("content", "") for m in body.get("messages", []))
        self._send({
            "choices": [
                {"message": {"role": "assistant", "content": json.dumps(self._decide(prompt))}}
            ]
        })

    def _decide(self, prompt: str) -> dict:
        for rule in self.rules:
            if all(marker in prompt for marker in rule["when"]):
                return rule["respond"]
        return {"found": False}

    def _send(self, obj: dict) -> None:
        raw = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main() -> None:
    parser = argparse.ArgumentParser(prog="stub_grounding")
    parser.add_argument("--port", type=int, default=9099)
    parser.add_argument("--rules", required=True)
    args = parser.parse_args()
    with open(args.rules, encoding="utf-8") as handle:
        _Handler.rules = json.load(handle)
    HTTPServer(("127.0.0.1", args.port), _Handler).serve_forever()


if __name__ == "__main__":
    main()
