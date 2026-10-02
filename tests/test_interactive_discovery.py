import sys
from types import SimpleNamespace

import pytest

from wiki_retriever import interactive_search
from wiki_retriever.http_server_management import HTTPServerRegistry


@pytest.mark.parametrize('alive,force_local', [(True, False), (False, False), (True, True)])
def test_db_uri_discovers_http_server(tmp_path, monkeypatch, alive, force_local):
    db = tmp_path / 'wiki.lancedb'
    db.mkdir()
    monkeypatch.setattr(HTTPServerRegistry, 'get_server_info', lambda self: SimpleNamespace(host='localhost', port=8123))
    checks = []
    def is_alive(self, info):
        checks.append(self.db_path)
        return alive
    monkeypatch.setattr(HTTPServerRegistry, 'is_server_alive', is_alive)
    argv = ['wiki-search', '--db_uri', str(db), '--collection_name', 'wiki']
    if force_local:
        argv.append('--dont_use_api')
    monkeypatch.setattr(sys, 'argv', argv)
    args = interactive_search.parse_args()
    assert args.server_url == ('http://localhost:8123' if alive and not force_local else None)
    assert args.use_api is (False if force_local else None)
    assert checks == ([] if force_local else [str(db)])


def test_retriever_discovery_never_starts_server(tmp_path, monkeypatch):
    from wiki_retriever import dense_lancedb
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(use_api=False)
    monkeypatch.setattr(dense_lancedb, 'create_lancedb_server', create)
    retriever = dense_lancedb.DenseLanceDB.__new__(dense_lancedb.DenseLanceDB)
    retriever.db_uri = str(tmp_path)
    retriever._use_api = None
    retriever._start_server = False
    retriever._init_backend()
    assert calls[0]['use_api'] is None
    assert calls[0]['start_server'] is False
    assert retriever.engine_type == 'lancedb'


def test_discovered_http_skips_local_model(tmp_path, monkeypatch):
    db = tmp_path / 'wiki.lancedb'
    db.mkdir()
    monkeypatch.setattr(HTTPServerRegistry, 'get_server_info', lambda self: SimpleNamespace(host='localhost', port=8123))
    monkeypatch.setattr(HTTPServerRegistry, 'is_server_alive', lambda *args: True)
    monkeypatch.setattr(sys, 'argv', ['wiki-search', '--db_uri', str(db), '--embedding_config', '/missing'])
    monkeypatch.setattr(interactive_search, '_import_backend', lambda *args: pytest.fail('local backend initialized'))
    monkeypatch.setattr('builtins.input', lambda prompt: 'exit')
    closed = []
    monkeypatch.setattr(interactive_search.HTTPSearchClient, 'close', lambda self: closed.append(self.url))
    interactive_search.main()
    assert closed == ['http://localhost:8123/search']


@pytest.mark.parametrize('extension', ['yaml', 'yml', 'json', 'jsonl'])
@pytest.mark.parametrize('custom_registry', [False, True])
def test_config_registry_beside_resolved_database(tmp_path, monkeypatch, extension, custom_registry):
    import json
    import yaml
    db = tmp_path / 'data' / 'wiki'
    db.mkdir(parents=True)
    configs = tmp_path / 'configs'
    configs.mkdir()
    config = configs / f'wiki.{extension}'
    values = {'db_uri': '../data/wiki', 'collection_name': 'wiki', 'backend': 'lancedb'}
    config.write_text(yaml.safe_dump(values) if extension in ('yaml', 'yml') else json.dumps(values) + '\n')
    monkeypatch.setattr(HTTPServerRegistry, 'is_server_alive', lambda *args: False)
    argv = ['wiki-search', '--db_uri', str(config)]
    expected = tmp_path / 'custom' if custom_registry else db.parent / '.wiki_retriever_servers'
    if custom_registry:
        argv.extend(['--registry-dir', str(expected)])
    monkeypatch.setattr(sys, 'argv', argv)
    args = interactive_search.parse_args()
    assert args.db_uri == str(db)
    assert args.registry_dir == str(expected)
    assert expected.is_dir()
