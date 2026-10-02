# wiki_retriever

An independently installable Python package for Wikipedia preparation and search
during RL sessions. LanceDB stores passage vectors; an HTTP server owns one
embedding model and accepts text queries. The existing vector gRPC protocol and
interactive RITS client are also included. No DocUVerse installation is required.

## Install

```bash
python -m pip install -e './for_pavan[prepare,local]'
```

For another machine, copy `for_pavan/` and run `pip install '.[prepare,local]'`
inside it. Choose a CUDA-compatible PyTorch installation when using a GPU.
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
beside the table and reused by `serve`. Searches use dot distance (lower is
better). Without `--ann-partitions`, search is exhaustive. Use the optional IVF_FLAT
index for full-dump latency; omit it for tiny smoke builds that cannot train
256 partitions. ANN construction may need additional memory.

## Start a server and search from an RL worker

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

Results include passage `id`, `title`, `text`, `url`, and `score` (dot distance).
`/docs` provides the HTTP schema. Requests share one model; search runs serially
within each server process. Run one process per assigned GPU. The default bind
address is localhost; binding to all interfaces exposes an unauthenticated API,
so use the intended private training network.

A container recipe is included for CPU deployment (GPU deployment needs a
CUDA-compatible base/runtime):

```bash
docker build -t wiki-retriever for_pavan
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

## Euler and the original DocUVerse workflow

The original preparation scripts were copied from
`euler:/local2/raduf/wikipedia/` into `tools/original/`. The package commands adapt
that workflow with bounded queues and configurable language URLs. Existing
corpus files on Euler can be reused without downloading/extracting again:

```bash
scp euler:/local2/raduf/wikipedia/wikipedia.en.jsonl.bz2 data/
```

Use your configured `euler` SSH alias. The original DocUVerse ingestion workflow
and historical paths remain documented in [`../README_wiki.md`](../README_wiki.md).
The new package replaces loose module imports from `for_pavan`; import modules
under `wiki_retriever` or use the installed commands.

## Tests

```bash
pip install -e './for_pavan[prepare,dev]'
pytest for_pavan/tests
```
