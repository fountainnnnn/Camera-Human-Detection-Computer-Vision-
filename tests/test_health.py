import json
import time
import urllib.error
import urllib.request

from src.health import HealthServer


def test_health_server_serves_json_payload() -> None:
    server = HealthServer("127.0.0.1", 0, lambda: {"status": "ok", "fps": 4.2})
    server.start()

    try:
        port = server._server.server_port
        body = None
        for _ in range(10):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                    assert response.status == 200
                    body = json.loads(response.read().decode("utf-8"))
                    break
            except OSError:
                time.sleep(0.05)
        assert body == {"status": "ok", "fps": 4.2}
    finally:
        server.stop()


def test_health_server_returns_404_for_other_paths() -> None:
    server = HealthServer("127.0.0.1", 0, lambda: {"status": "ok"})
    server.start()

    try:
        port = server._server.server_port
        for _ in range(10):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/missing", timeout=1)
            except urllib.error.HTTPError as exc:
                assert exc.code == 404
                return
            except OSError:
                time.sleep(0.05)
        raise AssertionError("health server did not return 404 in time")
    finally:
        server.stop()
