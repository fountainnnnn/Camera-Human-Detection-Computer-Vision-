from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Callable


class HealthHandler(BaseHTTPRequestHandler):
    provider: Callable[[], dict]

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return
        payload = self.provider()
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


class HealthServer:
    def __init__(self, host: str, port: int, provider: Callable[[], dict]) -> None:
        handler = type("ConfiguredHealthHandler", (HealthHandler,), {})
        handler.provider = staticmethod(provider)
        self._server = ThreadingHTTPServer((host, port), handler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
