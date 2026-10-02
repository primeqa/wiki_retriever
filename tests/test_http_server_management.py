import asyncio
import fcntl
import os
from pathlib import Path

from fastapi import FastAPI

from wiki_retriever.http_server_management import HTTPServerRegistry, serve


def test_registration_reuse_and_shutdown(tmp_path, monkeypatch, capsys):
    db = tmp_path / 'db'
    db.mkdir()
    registry_dir = tmp_path / 'registry'
    registry = HTTPServerRegistry(str(db), str(registry_dir))

    def run(app, **kwargs):
        async def exercise():
            async with app.router.lifespan_context(app):
                info = registry.get_server_info()
                assert info.db_path == str(db)
                assert info.port == 8000
                monkeypatch.setattr(HTTPServerRegistry, 'is_server_alive', lambda self, info: info is not None)
                def unexpected():
                    raise AssertionError('second caller loaded the model')
                reused = serve(str(db / '..' / 'db'), 'localhost', 9000, unexpected, str(registry_dir))
                assert reused == info
        asyncio.run(exercise())

    monkeypatch.setattr('uvicorn.run', run)
    serve(str(db), '0.0.0.0', 8000, FastAPI, str(registry_dir))
    assert registry.get_server_info() is None
    assert Path(registry.lock_file).exists()
    assert 'Using existing server at http://127.0.0.1:8000' in capsys.readouterr().out


def test_stale_registration_and_startup_failure(tmp_path, monkeypatch):
    registry = HTTPServerRegistry(str(tmp_path / 'db'), str(tmp_path / 'registry'))
    registry.register_server(os.getpid(), host='localhost', port=8000)
    monkeypatch.setattr(HTTPServerRegistry, 'is_server_alive', lambda *args: False)
    import pytest
    def fail():
        raise ValueError('model failed')
    with pytest.raises(ValueError, match='model failed'):
        serve(registry.db_path, 'localhost', 8000, fail, registry.registry_dir)
    assert registry.get_server_info() is None
    fd = os.open(registry.lock_file, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)


def test_corrupt_registration_keeps_lock(tmp_path):
    registry = HTTPServerRegistry(str(tmp_path / 'db'), str(tmp_path / 'registry'))
    Path(registry.lock_file).touch()
    Path(registry.registry_file).write_text('{')
    assert registry.get_server_info() is None
    assert Path(registry.lock_file).exists()
