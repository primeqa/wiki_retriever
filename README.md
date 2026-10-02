# wiki_retriever

An independently installable Python package for Wikipedia preparation and search
during RL sessions. LanceDB stores passage vectors; an HTTP server owns one
embedding model and accepts text queries. The existing vector gRPC protocol and
interactive RITS client are also included. No DocUVerse installation is required.

## Install

```bash
pip install "wiki-retriever[prepare,local]"
```

To install from a source checkout instead, run `pip install -e ".[prepare,local]"`
in the repository root. Choose a CUDA-compatible PyTorch installation when using a GPU.
`prepare` installs the wiki parser; `local` installs SentenceTransformers.
Downloads require `wget`. Full dumps and vector indexes require substantial disk
space. Extraction and ingestion stream bounded batches, although each individual
article is tokenized in memory. A model download requires Hugging Face access.

## Prepare and index Wikipedia

```bash
wiki-retriever download -o data/enwiki-latest-pages-articles.xml.bz2
wiki-retriever extract data/enwiki-latest-pages-articles.xml.bz2 \
  -o data/wikipedia.en.jsonl.bz2 -w 24
wiki-retriever split data/wikipedia.en.jsonl.bz2 --output-dir data/shards
wiki-retriever index --input 'data/shards/wikipedia.en_*.jsonl.bz2' \
  --db data/wiki.lancedb --table wiki-en --device cuda \
  --window 1024 --overlap 100 --batch-size 128 --ann-partitions 256
```

Use `--date YYYYMMDD` for a dated dump. Keep its checksum for reproducibility.
Extracted JSONL contains `id`, `title`, `text`, and `url`; only namespace-zero,
non-redirect pages with title and text are retained. Extraction uses
`mwparserfromhell.strip_code()` and preserves paragraph breaks. `--chunk` bounds
the queued article batch. Sharding is optional: `index --input` also accepts a
single plain or compressed JSONL file. Use either corpus or shards, never both.

For a smoke build use `--max-documents 1000` and a separate table name. Existing
tables are refused to prevent accidental duplication or replacement; a failed
build may leave a partial table, so retry with a fresh name. Indexing uses token
windows and local passage embeddings from
`ibm-granite/granite-embedding-english-r2` by default. Model identity is recorded
beside the table and reused by `serve`. Search scores are cosine dissimilarities
(`1 - cosine_similarity`; lower is better). Without `--ann-partitions`, search is
exhaustive. Use the optional IVF_FLAT
index for full-dump latency; omit it for tiny smoke builds that cannot train
256 partitions. ANN construction may need additional memory.

List the tables in an existing LanceDB directory:

```bash
wiki-retriever list-tables --db data/wiki.lancedb
```

This prints one table name per line, including all pages of results. An empty
database produces no output; a missing directory produces an error.

## Start a server and search from an RL worker

`--db` also accepts a `.yaml`, `.yml`, `.json`, or `.jsonl` configuration file.
For example, save the following as `wiki.yaml` beside your `data/` directory:

```yaml
db_uri: data/wiki.lancedb
collection_name: wiki-en
backend: lancedb
```

Then run `wiki-retriever serve --db wiki.yaml --device cuda`. For `serve` and
`index`, `--table` can be omitted when the config supplies `collection_name`;
an explicit `--table` overrides that value. Directory paths still require
`--table`. Relative `db_uri` and optional `doc_store_path` values are resolved
relative to the config file, using `DenseRetriever`'s config reader. JSONL configs
use their first line. `list-tables` and `serve-grpc` also accept these configs.
The server reads embedding model metadata from the resolved database and table;
use the existing CLI flags for model, endpoint, device, and server options.

Search interactively from another machine using the HTTP server:

```bash
wiki-search --server-url http://retrieval-host:8000 --top_k 5
# Or, from a source checkout, run the script directly:
python wiki_retriever/interactive_search.py --server-url http://retrieval-host:8000
```

With `wiki-search --db_uri data/wiki.lancedb`, search discovers an existing
registered HTTP server first, then an existing LanceDB gRPC server. If neither
is running, it opens the database locally without starting a server. Config files
also work; their default registry is `.wiki_retriever_servers` beside the resolved
database directory, matching `serve`. Use `--registry-dir` to match a custom HTTP registry, or
`--dont_use_api` to force local access. Discovered HTTP servers use their own
configured table and model. Full-document searches and `--top_k` above 100 use
the gRPC/local path because the HTTP API supports passages with up to 100 results.

The server computes the embeddings and selects its configured table. HTTP mode
needs no local index, embedding config, or RITS credentials. Use `--timeout 120`
to adjust the request timeout; type `exit` to quit. HTTP results contain passages.
Existing ANN indexes built with dot distance must be rebuilt with `metric="cosine"`
to accelerate cosine searches; new `index --ann-partitions` builds use cosine.

```bash
wiki-retriever serve --db data/wiki.lancedb --table wiki-en --device cuda \
  --host 0.0.0.0 --port 8000
curl http://localhost:8000/health
curl http://localhost:8000/search -H 'Content-Type: application/json' \
  -d '{"query":"Who developed general relativity?","top_k":5}'
```

```python
import requests

response = requests.post(
    "http://retrieval-host:8000/search",
    json={"query": "Who developed general relativity?", "top_k": 5},
    timeout=60,
)
response.raise_for_status()
passages = response.json()["results"]
```

Results include passage `id`, `title`, `text`, `url`, and `score` (cosine dissimilarity).
`/docs` provides the HTTP schema. Requests share one model; search runs serially
within each server process. `serve` registers one HTTP server per resolved database
directory in `.wiki_retriever_servers` beside the database (override with
`--registry-dir`). Starting it again for that database prints the existing URL
and exits, reusing the first server's table, model, host, and port. Concurrent
starts wait for the first server to become ready. Stale registrations are replaced
after the owning process exits. Use the same registry directory for all callers.
The default bind
address is localhost; binding to all interfaces exposes an unauthenticated API,
so use the intended private training network.

### Search from Python with DenseRetriever

Use `DenseLanceDB`, the LanceDB subclass of `DenseRetriever`, to search an
existing index directly from a Python script. It computes query embeddings
through a RITS service. Set `RITS_API_KEY` and `RITS_ENDPOINT` in your environment;
the endpoint is the service base URL without `/v1`. Use the same embedding model
that built the index (the example below uses the indexing default).

```python
import os
from wiki_retriever.dense_lancedb import DenseLanceDB

retriever = DenseLanceDB(
    model_name="ibm-granite/granite-embedding-english-r2",
    endpoint=os.environ["RITS_ENDPOINT"],
    db_uri="data/wiki.lancedb",
    collection_name="wiki-en",
    use_api=False,
)
try:
    results = retriever.search("Who developed general relativity?", top_k=5)
    for passage in results:
        print(passage["title"], passage["score"], passage["url"])
        print(passage["text"])
finally:
    retriever.client.close()
    retriever.openai.close()
```

`use_api=False` opens the local LanceDB directory directly, so this script needs
access to the index files and does not require the HTTP or gRPC search server.
Reuse the retriever for subsequent queries. Scores are cosine dissimilarities, with lower
values indicating better matches. `DenseRetriever` itself is a base class; use
the backend subclass to initialize its database client.

From a source checkout, a container recipe is included for CPU deployment (GPU deployment needs a
CUDA-compatible base/runtime):

```bash
docker build -t wiki-retriever .
docker run --rm -p 8000:8000 -v "$PWD/data:/data" wiki-retriever \
  --db /data/wiki.lancedb --table wiki-en
```

Mount a Hugging Face cache when running without model-download access.

To use a RITS embedding service instead of a local model, set `RITS_API_KEY` and
pass `--endpoint URL` without `/v1` and the same `--model` used during ingestion.
For an existing index without metadata, explicitly specify its embedding model.
The legacy retriever also supports optional full-document SQLite/JSON/pickle
stores; only load trusted pickle files.

Vector-only gRPC serving (embedding computation belongs to the caller):

```bash
wiki-retriever serve-grpc --db data/wiki.lancedb --host 0.0.0.0 --port 8766
```

Use `wiki_retriever.milvus_grpc_client.MilvusGRPCClient` to connect. The retained
Milvus service names describe the wire protocol; the included backend is
LanceDB. Interactive RITS search is available through `wiki-search --backend
lancedb --db_uri data/wiki.lancedb --collection_name wiki-en --dont_use_api`.
The model config is bundled, and `--embedding_config` overrides it.

## Tests

```bash
pip install -e ".[prepare,dev]"
pytest tests
```
