import bz2
import json
import pytest
from wiki_retriever.prepare import extract, split


@pytest.mark.parametrize('extension', ['yaml', 'yml', 'json', 'jsonl'])
@pytest.mark.parametrize('command', ['serve', 'index'])
def test_cli_database_config(tmp_path, monkeypatch, extension, command):
    import sys
    import yaml
    import uvicorn
    from wiki_retriever import cli, service, index

    db = tmp_path / 'wiki.lancedb'
    db.mkdir()
    (db / 'wiki-en.wiki-retriever.json').write_text(json.dumps({'model': 'recorded-model'}))
    docs = tmp_path / 'docs.json'
    docs.write_text('{}')
    config = tmp_path / f'wiki.{extension}'
    values = {'db_uri': 'wiki.lancedb', 'collection_name': 'wiki-en',
              'doc_store_path': 'docs.json', 'backend': 'lancedb'}
    config.write_text(yaml.safe_dump(values) if extension in ('yaml', 'yml') else json.dumps(values) + '\n')
    captured = {}
    def capture(**kwargs):
        captured.update(kwargs)
        return {}
    monkeypatch.setattr(service, 'create_app', capture)
    monkeypatch.setattr(index, 'build_index', capture)
    monkeypatch.setattr(uvicorn, 'run', lambda *a, **kw: None)
    argv = ['wiki-retriever', command, '--db', str(config)]
    if command == 'index':
        argv += ['--input', 'articles.jsonl']
    monkeypatch.setattr(sys, 'argv', argv)
    cli.main()
    assert captured['db'] == str(db)
    assert captured['table'] == 'wiki-en'
    if command == 'serve':
        assert captured['model'] == 'recorded-model'
        assert captured['doc_store_path'] == str(docs)
    monkeypatch.setattr(sys, 'argv', argv + ['--table', 'override'])
    cli.main()
    assert captured['table'] == 'override'


@pytest.mark.parametrize('command', ['serve', 'index'])
@pytest.mark.parametrize('use_config', [False, True])
def test_cli_requires_table(tmp_path, monkeypatch, capsys, command, use_config):
    import sys
    from wiki_retriever.cli import main
    db = tmp_path / 'wiki.lancedb'
    db.mkdir()
    if use_config:
        config = tmp_path / 'wiki.json'
        config.write_text(json.dumps({'db_uri': str(db)}))
        db = config
    argv = ['wiki-retriever', command, '--db', str(db)]
    if command == 'index':
        argv += ['--input', 'articles.jsonl']
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert '--table is required' in capsys.readouterr().err


def test_list_tables_cli(tmp_path, monkeypatch, capsys):
    import sys
    import lancedb
    from wiki_retriever.cli import main

    db_path = tmp_path / 'wiki.lancedb'
    db = lancedb.connect(str(db_path))
    monkeypatch.setattr(sys, 'argv', ['wiki-retriever', 'list-tables', '--db', str(db_path)])
    main()
    assert capsys.readouterr().out == ''

    names = [f'table_{i:03d}' for i in range(101)]
    for name in names:
        db.create_table(name, [{'id': 1}])
    main()
    assert capsys.readouterr().out.splitlines() == names

    missing = tmp_path / 'missing'
    monkeypatch.setattr(sys, 'argv', ['wiki-retriever', 'list-tables', '--db', str(missing)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert 'LanceDB directory does not exist' in capsys.readouterr().err
    assert not missing.exists()


def test_extract_and_shard(tmp_path):
    source = tmp_path / 'wiki.xml.bz2'
    xml = '''<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
    <page><title>Albert Einstein</title><ns>0</ns><id>1</id><revision><id>900</id><text>''' + "'''Scientist''' and [[physicist]].\n\nSecond paragraph." + '''</text></revision></page>
    <page><title>Redirect</title><ns>0</ns><id>2</id><redirect title="Albert Einstein"/><revision><text>#REDIRECT</text></revision></page>
    <page><title>Talk</title><ns>1</ns><id>3</id><revision><text>Ignore</text></revision></page>
    <page><title>Another</title><ns>0</ns><id>4</id><revision><text>Article</text></revision></page>
    </mediawiki>'''
    source.write_bytes(bz2.compress(xml.encode()))
    output = tmp_path / 'articles.jsonl.bz2'
    assert extract(source, output, workers=2, chunk=1) == 2
    with bz2.open(output, 'rt') as stream:
        articles = [json.loads(line) for line in stream]
    assert articles[0]['id'] == 1
    assert articles[0]['text'] == 'Scientist and physicist.\n\nSecond paragraph.'
    assert articles[0]['url'].endswith('/Albert_Einstein')
    assert split(output, tmp_path / 'shards', records=1) == 2
    shards = sorted((tmp_path / 'shards').glob('*.bz2'))
    assert len(shards) == 2
    assert json.loads(bz2.decompress(shards[1].read_bytes())) == articles[1]
    with pytest.raises(FileExistsError):
        split(output, tmp_path / 'shards', records=1)


def test_http_search(tmp_path):
    from fastapi.testclient import TestClient
    from wiki_retriever.service import create_app
    class Client:
        def has_collection(self, name):
            return name == 'wiki'
        def close(self):
            self.closed = True
    class Retriever:
        client = Client()
        def search(self, query, top_k):
            assert query == 'Einstein' and top_k == 2
            return [{'id': '1-0-10', 'text': 'Physicist', 'score': -1.0}]
    retriever = Retriever()
    with TestClient(create_app(str(tmp_path), 'wiki', 'test', retriever=retriever)) as client:
        assert client.get('/health').json()['healthy']
        assert client.post('/search', json={'query': 'Einstein', 'top_k': 2}).json()['results'][0]['id'] == '1-0-10'
        assert client.post('/search', json={'query': '', 'top_k': 0}).status_code == 422
    assert retriever.client.closed


def test_grpc_roundtrip(tmp_path):
    from wiki_retriever.lancedb_client_adapter import LanceDBClientAdapter
    from wiki_retriever.milvus_grpc_server import serve_grpc
    from wiki_retriever.milvus_grpc_client import MilvusGRPCClient
    import grpc
    from wiki_retriever import milvus_service_pb2 as pb
    from wiki_retriever import milvus_service_pb2_grpc as rpc
    adapter = LanceDBClientAdapter(str(tmp_path))
    adapter.db.create_table('wiki', [{'id': '1', 'text': 'Einstein', 'title': 'Albert', 'url': 'url', 'vector': [1., 0.]}, {'id': '2', 'text': 'Other', 'title': 'Other', 'url': 'url2', 'vector': [0., 1.]}])
    socket = str(tmp_path / 'test.sock')
    server = serve_grpc(adapter, socket_path=socket)
    try:
        with grpc.insecure_channel('unix://' + socket) as channel:
            stub = rpc.MilvusServiceStub(channel)
            assert stub.HealthCheck(pb.HealthCheckRequest(), timeout=5).healthy
            response = stub.Search(pb.SearchRequest(collection_name='wiki', query_vector=[1., 0.], limit=1), timeout=5)
            assert response.results[0].id == '1'
            assert response.results[0].entity['text'] == 'Einstein'
    finally:
        server.stop(0).wait()
        adapter.close()


def test_streaming_index_and_model_metadata(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import numpy as np
    import lancedb
    from wiki_retriever.index import build_index
    class Tokenizer:
        def num_special_tokens_to_add(self, pair=False):
            return 0
        def encode(self, text, add_special_tokens=False):
            return list(range(len(text.split())))
        def decode(self, tokens):
            return ' '.join(str(t) for t in tokens)
    class Encoder:
        max_seq_length = 10
        tokenizer = Tokenizer()
        def encode(self, texts, show_progress_bar=False):
            assert len(texts) <= 2
            return np.array([[1., 0.] for text in texts], dtype=np.float32)
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(SentenceTransformer=lambda *a, **kw: Encoder()))
    source = tmp_path / 'articles.jsonl'
    source.write_text(json.dumps({'id': 1, 'title': 'Article', 'text': 'a b c d e f g', 'url': 'url'}) + '\n')
    db = str(tmp_path / 'wiki.lancedb')
    assert build_index(str(source), db, 'wiki-en', model='test', window=4, overlap=1, batch_size=2) == {'documents': 1, 'passages': 2}
    table = lancedb.connect(db).open_table('wiki-en')
    assert table.count_rows() == 2
    assert {r['id'] for r in table.to_arrow().to_pylist()} == {'1-0-4', '1-3-7'}
    assert json.loads((tmp_path / 'wiki.lancedb/wiki-en.wiki-retriever.json').read_text())['model'] == 'test'
    with pytest.raises(FileExistsError):
        build_index(str(source), db, 'wiki-en', model='test')
