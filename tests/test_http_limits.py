from http.server import BaseHTTPRequestHandler
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest

from mill.http_server import BoundedHTTPServer


def test_idle_unauthenticated_connections_release_request_slots():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
        def log_message(self, *args):
            pass
    server = BoundedHTTPServer(("127.0.0.1", 0), Handler, max_requests=2, read_timeout=0.3)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    sockets = [socket.create_connection(server.server_address, timeout=1) for _ in range(2)]
    url = f"http://127.0.0.1:{server.server_port}/health"
    try:
        until = time.monotonic() + 0.2
        while server._slots._value:
            assert time.monotonic() < until
            time.sleep(0.005)
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url, timeout=1)
        assert error.value.code == 503
        time.sleep(0.35)
        with urllib.request.urlopen(url, timeout=1) as response:
            assert response.status == 200
    finally:
        for connection in sockets:
            connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
