"""Directory-scoped HTTP server registration and discovery."""
import fcntl
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import requests

from .server_common import ServerInfo, ServerRegistry


class HTTPServerRegistry(ServerRegistry):
    """Use a separate registry from the LanceDB gRPC service."""

    def get_server_info(self):
        # Readers must never unlink the lock file, including on corrupt JSON.
        try:
            return ServerInfo.from_dict(json.loads(Path(self.registry_file).read_text()))
        except (OSError, ValueError, TypeError):
            return None

    def is_server_alive(self, info):
        if info is None:
            return False
        if info.machine_id == self._get_machine_id():
            try:
                os.kill(info.pid, 0)
            except ProcessLookupError:
                return False
            except PermissionError:
                pass
        try:
            response = requests.get(server_url(info) + "/health", timeout=2)
            return response.status_code == 200 and response.json().get("healthy") is True
        except (requests.RequestException, ValueError):
            return False


def server_url(info):
    host = info.host
    if ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{info.port}"


def serve(db, host, port, app_factory, registry_dir=None):
    """Run one HTTP server per database directory, or return its existing info.

    Keep an OS lock for the server's lifetime. Contenders can discover it once
    startup completes; crashes release the lock automatically. The lock file is
    never deleted, so all processes continue locking the same inode.
    """
    db = str(Path(db).expanduser().resolve())
    registry = HTTPServerRegistry(
        db, registry_dir or str(Path(db).parent / ".wiki_retriever_servers")
    )
    fd = os.open(registry.lock_file, os.O_CREAT | os.O_RDWR, 0o600)
    owns_lock = False
    registered = False
    try:
        deadline = time.monotonic() + 120
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                owns_lock = True
            except BlockingIOError:
                pass
            info = registry.get_server_info()
            if registry.is_server_alive(info):
                print(f"Using existing server at {server_url(info)}", flush=True)
                return info
            if owns_lock:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Timed out waiting for server startup for {db}")
            time.sleep(0.1)

        # Only the lock owner may remove a stale registration. Keep the lock file.
        Path(registry.registry_file).unlink(missing_ok=True)
        app = app_factory()
        original_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app):
            nonlocal registered
            async with original_lifespan(app):
                advertised_host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
                info = registry.register_server(os.getpid(), host=advertised_host, port=port)
                registered = True
                print(f"Server registered at {server_url(info)}", flush=True)
                yield

        app.router.lifespan_context = lifespan
        import uvicorn
        uvicorn.run(app, host=host, port=port)
        return registry.get_server_info()
    finally:
        if owns_lock:
            if registered:
                Path(registry.registry_file).unlink(missing_ok=True)
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
