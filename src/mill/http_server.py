from http.server import ThreadingHTTPServer
import threading


class BoundedHTTPServer(ThreadingHTTPServer):
    """Bound live request threads; overload gets an explicit retryable response."""
    def __init__(self, address, handler, max_requests=16, read_timeout=10):
        super().__init__(address, handler)
        self._slots = threading.BoundedSemaphore(max_requests)
        self._read_timeout = read_timeout

    def get_request(self):
        request, address = super().get_request()
        request.settimeout(self._read_timeout)
        return request, address

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            try:
                request.settimeout(1)
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\n"
                                b"Content-Length: 0\r\nRetry-After: 1\r\nConnection: close\r\n\r\n")
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()
