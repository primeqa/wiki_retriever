#!/usr/bin/env python
"""
Interactive search tool for testing the retrieval setup.

Initializes a DenseRetriever backend and drops into an interactive loop
where you type queries and see the top-k results.  Type "exit" to quit.

Usage:
    wiki-search --backend lancedb --db_uri /path/to/wiki.lancedb --collection_name wiki-en --dont_use_api
    wiki-search --server-url http://retrieval-host:8000
"""

import argparse
import os
import readline  # noqa: F401 — enables line editing & history for input()
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse
import yaml

# Direct file execution needs the package root for relative imports.
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "wiki_retriever"

from .backends import BACKENDS, _import_backend, infer_backend
from .dense_retriever import DenseRetriever
from .timer import timer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive dense retrieval search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--backend", type=str, default=None, choices=list(BACKENDS),
        help="Vector database backend (default: auto-detect from --db_uri)",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--db_uri", type=str, help="Local database directory or config file")
    source.add_argument("--server-url", help="HTTP search server base URL, e.g. http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=60, help="HTTP request timeout in seconds (default: 60)")
    parser.add_argument("--registry-dir", help="HTTP server registry directory; config inputs default to .wiki_retriever_servers beside the resolved database")
    parser.add_argument(
        "--collection_name", type=str,
        default="nq_train_short_milvus_dense_granite125m_512_100_20250623",
        help="Collection / table name",
    )
    parser.add_argument(
        "--embedding_config", type=str,
        default=os.path.join(os.path.dirname(__file__), "configs", "granite-embedding-149m-english_config.yaml"),
        help="Embedding model config path",
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of results per query (default: 5)",
    )
    parser.add_argument(
        "--truncate", type=int, default=150,
        help="Truncate result text to this many characters (default: 150, 0=no truncation)",
    )
    parser.add_argument(
        "--return_docs", action="store_true",
        help="Return full documents from the doc store instead of passages. "
             "Requires --doc_store_path.",
    )
    parser.add_argument(
        "--doc_store_path", type=str, default=None,
        help="Path to serialized document store (.pkl.gz, .pkl, .json). "
             "Required when --return_docs is set.",
    )
    parser.add_argument(
        "--dont_use_api", action="store_false", dest="use_api", default=None,
        help="Force local file access instead of discovering an existing server",
    )
    parser.add_argument(
        "--use_glow", action="store_true",
        help="Render result text as Markdown using 'glow' (must be installed)",
    )

    args = parser.parse_args()

    if args.top_k < 1 or args.timeout <= 0:
        parser.error("--top_k and --timeout must be positive")
    if args.use_glow and not shutil.which("glow"):
        print("Warning: 'glow' not found in PATH — falling back to plain output")
        args.use_glow = False
    if args.server_url:
        url = urlparse(args.server_url)
        if url.scheme not in ("http", "https") or not url.netloc:
            parser.error("--server-url must be an HTTP or HTTPS base URL")
        if args.return_docs or args.doc_store_path:
            parser.error("HTTP search returns passages; document stores require --db_uri")
        if args.top_k > 100:
            parser.error("HTTP search supports --top_k up to 100")
        return args

    # If db_uri is a config file, read backend (and other fields) from it
    config_input = Path(args.db_uri).suffix.lower() in ('.yaml', '.yml', '.json', '.jsonl')
    collection_name, db_uri, doc_store_path, config_backend = \
        DenseRetriever.read_db_config(args.db_uri, args.collection_name, args.doc_store_path)
    args.db_uri = db_uri
    args.db_uri = str(Path(args.db_uri).expanduser().resolve())
    if config_input and args.registry_dir is None:
        args.registry_dir = str(Path(args.db_uri).parent / ".wiki_retriever_servers")
    args.collection_name = collection_name
    if doc_store_path is not None:
        args.doc_store_path = doc_store_path

    # Resolve backend: CLI flag > config file > auto-detect from db_uri
    if args.backend is None:
        if config_backend is not None:
            args.backend = config_backend
            print(f"  Backend from config: {args.backend}")
        else:
            args.backend = infer_backend(args.db_uri)
            print(f"  Auto-detected backend: {args.backend}")

    if args.return_docs and args.doc_store_path is None:
        ext = os.path.splitext(args.db_uri)[1].lower()
        if ext not in ('.yaml', '.yml', '.json', '.jsonl'):
            parser.error(
                "--return_docs requires --doc_store_path (unless --db_uri "
                "points to a .yaml/.json/.jsonl config that includes doc_store_path)"
            )

    if args.use_api is not False and not args.return_docs and args.top_k <= 100:
        from .http_server_management import HTTPServerRegistry, server_url
        registry_dir = args.registry_dir or str(Path(args.db_uri).parent / ".wiki_retriever_servers")
        registry = HTTPServerRegistry(args.db_uri, registry_dir)
        info = registry.get_server_info()
        if registry.is_server_alive(info):
            args.server_url = server_url(info)

    return args


class HTTPSearchClient:
    """Text-query client for the wiki-retriever HTTP server."""

    def __init__(self, server_url, timeout=60):
        import requests
        self.url = server_url.rstrip("/") + "/search"
        self.timeout = timeout
        self.session = requests.Session()

    def search(self, query, top_k, return_docs=False):
        response = self.session.post(
            self.url, json={"query": query, "top_k": top_k}, timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["results"]

    def close(self):
        self.session.close()


def render_glow(text):
    """Pipe *text* through 'glow' for Markdown rendering."""
    try:
        result = subprocess.run(
            ["glow", "-"],
            input=text, capture_output=True, text=True, timeout=5,
        )
        return result.stdout if result.returncode == 0 else text
    except Exception:
        return text


def truncate(text, maxlen):
    """Truncate text to maxlen chars, adding ellipsis if shortened."""
    if text is None:
        return "<None>"
    if maxlen <= 0 or len(text) <= maxlen:
        return text
    return text[:maxlen] + "..."


def main():
    tm = timer("interactive_search")

    args = parse_args()

    ANSI_COLORS = {
        'black': '30', 'red': '31', 'green': '32', 'yellow': '33',
        'blue': '34', 'magenta': '35', 'cyan': '36', 'white': '37',
        'bright_black': '90', 'bright_red': '91', 'bright_green': '92',
        'bright_yellow': '93', 'bright_blue': '94', 'bright_magenta': '95',
        'bright_cyan': '96', 'bright_white': '97',
        'bold': '1', 'dim': '2', 'italic': '3', 'underline': '4',
    }

    def color(text, c):
        if args.use_glow:
            # Markdown bold/italic as a rough equivalent
            if c in ('bold',):
                return f"**{text}**"
            if c in ('italic',):
                return f"*{text}*"
            # Glow doesn't support arbitrary colors; return as-is
            return text
        code = ANSI_COLORS.get(c)
        if code is None:
            return text
        return f"\033[{code}m{text}\033[0m"

    if args.server_url:
        client = HTTPSearchClient(args.server_url, args.timeout)
        print(f"HTTP search server: {args.server_url}")
    else:
        # Load embedding config
        tm.mark()
        print(f"Loading embedding config from {args.embedding_config}...")
        with open(args.embedding_config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        tm.add_timing("load_config")

        # Initialize retriever
        RetrieverClass = _import_backend(args.backend)

        tm.mark()
        client = RetrieverClass(
            model_name=config['model']['engine'],
            endpoint=config['model']['args']['endpoint'],
            db_uri=args.db_uri,
            collection_name=args.collection_name,
            use_api=args.use_api,
            start_server=False,
            doc_store_path=args.doc_store_path,
        )
        tm.add_timing("init_retriever")
        print(f"\nInitializing {args.backend} retriever...")
        print(f"  DB URI:     {client.db_uri}")
        print(f"  Collection: {client.collection_name}")


        # Verify collection exists
        try:
            if not client.client.has_collection(client.collection_name):
                available = client.client.list_collections()
                print(f"\nError: collection '{client.collection_name}' not found.")
                if available:
                    print("  Available collections:")
                    for name in available:
                        print(f"    - {name}")
                else:
                    print("  (no collections found in database)")
                return
        except Exception:
            pass  # some backends may not support has_collection; let search fail later

        if args.return_docs:
            if client.doc_store is None:
                print("Warning: doc store failed to load — 'doc' field will be empty")
            else:
                print(f"  Doc store:  {len(client.doc_store):,} documents loaded")

    print(f"\n  Init time: {tm.time_since_beginning()}")

    try:
        # Interactive loop
        print(f"\nReady.  top_k={args.top_k}, truncate={args.truncate}")
        print('Type a query and press Enter.  Type "exit" to quit.\n')

        num_queries = 0

        while True:
            try:
                query = input("query> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not query:
                continue
            if query.lower() == "exit":
                print("Bye.")
                break

            tm.mark()
            try:
                results = client.search(query, args.top_k, return_docs=args.return_docs)
            except Exception as e:
                print(f"  Error: {e}\n")
                continue
            query_ms = tm.mark_and_return_milliseconds()
            num_queries += 1
            tm.add_timing("queries", query_ms)

            if not results:
                print(f"  (no results)  [{query_ms} ms]\n")
                continue

            output_lines = []
            for i, doc in enumerate(results, 1):
                score = doc.get('score', doc.get('distance', '?'))
                passage_id = doc.get('id', '?')
                doc_id = DenseRetriever._get_orig_docid(passage_id) if passage_id != '?' else '?'
                title = doc.get('title', '')
                text = doc.get('doc') if args.return_docs else doc.get('text', '')

                label = "doc" if args.return_docs else "text"
                output_lines.append(color(f" [{i}] score={score:.4f}  doc_id={doc_id}  passage_id={passage_id}",
                                          'yellow' if not args.use_glow else 'bold', ))
                output_lines.append(color(f"   Title: {truncate(title, 80)}", 'cyan'))
                output_lines.append(f"         {label}:\n {truncate(text, args.truncate)}\n")

            output_lines.append(f"  ({query_ms} ms)\n")

            output = "\n".join(output_lines)
            if args.use_glow:
                print(render_glow(output))
            else:
                # Apply yellow to text lines in plain mode
                print(output)

        # Final timing summary
        if num_queries > 0:
            print()
            tm.display_timing(tm.milliseconds_since_beginning(),
                              keys={"queries": num_queries})

    finally:
        if args.server_url:
            client.close()
        else:
            client.client.close()
            client.openai.close()


if __name__ == "__main__":
    main()
