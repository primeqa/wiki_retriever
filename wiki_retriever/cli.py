"""Command line entry point; heavy dependencies are imported per command."""
import argparse
import json
from pathlib import Path
from .index import DEFAULT_MODEL


def main():
    parser = argparse.ArgumentParser(prog="wiki-retriever")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("download", help="Download/resume a Wikipedia XML dump (requires wget)")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--language", default="en")
    p.add_argument("--date", default="latest")
    p = commands.add_parser("extract", help="Extract plain article text to JSONL[.bz2]")
    p.add_argument("source")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("-w", "--workers", type=int, default=1)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--language", default="en")
    p = commands.add_parser("split", help="Split JSONL[.bz2] into compressed shards")
    p.add_argument("source")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--records", type=int, default=200000)
    p.add_argument("--prefix", default="wikipedia.en")
    p = commands.add_parser("index", help="Stream articles into a new LanceDB table")
    p.add_argument("--input", dest="inputs", required=True)
    p.add_argument("--db", required=True)
    p.add_argument("--table", help="Table name; defaults to collection_name in a --db config")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--window", type=int, default=1024)
    p.add_argument("--overlap", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device")
    p.add_argument("--max-documents", type=int)
    p.add_argument("--ann-partitions", type=int, help="Build an IVF_FLAT cosine-distance index after ingestion")
    p = commands.add_parser("list-tables", help="List tables in an existing LanceDB directory")
    p.add_argument("--db", required=True)
    for name in ("serve", "serve-grpc"):
        p = commands.add_parser(name)
        p.add_argument("--db", required=True)
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--port", type=int, default=8000 if name == "serve" else 8766)
        if name == "serve":
            p.add_argument("--table", help="Table name; defaults to collection_name in a --db config")
            p.add_argument("--model", default=None)
            p.add_argument("--endpoint", help="OpenAI-compatible embedding endpoint without /v1; uses RITS_API_KEY")
            p.add_argument("--device", default="cpu")
    args = vars(parser.parse_args())
    command = args.pop("command")
    if "db" in args and Path(args["db"]).suffix.lower() in (".yaml", ".yml", ".json", ".jsonl"):
        from .dense_retriever import DenseRetriever
        try:
            table, db, doc_store_path, backend = DenseRetriever.read_db_config(
                args["db"], args.get("table")
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(f"Invalid database config: {error}")
        if backend not in (None, "lancedb"):
            parser.error("wiki-retriever supports only the lancedb backend")
        args["db"] = db
        if "table" in args:
            args["table"] = args["table"] or table
        if command == "serve":
            args["doc_store_path"] = doc_store_path
    if command in ("index", "serve") and not args["table"]:
        parser.error("--table is required unless --db is a config containing collection_name")
    if command in ("download", "extract", "split"):
        from . import prepare
        result = getattr(prepare, command)(**args)
        if result is not None:
            print(f"Processed {result} articles")
    elif command == "index":
        from .index import build_index
        print(json.dumps(build_index(**args)))
    elif command == "list-tables":
        import lancedb
        db_path = Path(args["db"]).expanduser()
        if not db_path.is_dir():
            parser.error(f"LanceDB directory does not exist: {db_path}")
        db = lancedb.connect(str(db_path))
        page_token = None
        while True:
            if hasattr(db, "list_tables"):
                page = db.list_tables(page_token=page_token, limit=100)
                names, next_token = page.tables, page.page_token
            else:
                # Compatibility with older supported LanceDB versions.
                names = list(db.table_names(page_token=page_token, limit=100))
                next_token = names[-1] if len(names) == 100 else None
            for name in names:
                print(name)
            if not next_token:
                break
            page_token = next_token
    elif command == "serve":
        import uvicorn
        from .service import create_app
        host, port = args.pop("host"), args.pop("port")
        metadata = Path(args["db"]) / f"{args['table']}.wiki-retriever.json"
        recorded = json.loads(metadata.read_text())["model"] if metadata.exists() else None
        if args["model"] and recorded and args["model"] != recorded:
            parser.error("--model differs from the model used to build this table")
        args["model"] = args["model"] or recorded or DEFAULT_MODEL
        uvicorn.run(create_app(**args), host=host, port=port)
    else:
        from .lancedb_client_adapter import LanceDBClientAdapter
        from .milvus_grpc_server import serve_grpc
        client = LanceDBClientAdapter(args.pop("db"))
        server = serve_grpc(client, **args)
        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            pass
        finally:
            server.stop(5).wait()
            client.close()
