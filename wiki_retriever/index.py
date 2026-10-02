"""Streaming, token-windowed LanceDB ingestion using a local embedding model."""
import glob
import json
from .prepare import open_text

DEFAULT_MODEL = "ibm-granite/granite-embedding-english-r2"


def build_index(inputs, db, table, model=DEFAULT_MODEL, window=1024, overlap=100,
                batch_size=128, device=None, max_documents=None, ann_partitions=None):
    import lancedb
    from sentence_transformers import SentenceTransformer
    if window < 1 or not 0 <= overlap < window or batch_size < 1:
        raise ValueError("Require window > overlap >= 0 and batch_size > 0")
    if max_documents is not None and max_documents < 1:
        raise ValueError("max_documents must be positive")
    if ann_partitions is not None and ann_partitions < 1:
        raise ValueError("ann_partitions must be positive")
    files = sorted(glob.glob(inputs))
    if not files:
        raise FileNotFoundError(inputs)
    connection = lancedb.connect(db)
    if table in connection.table_names():
        raise FileExistsError(f"Table {table!r} already exists; choose a new name")
    encoder = SentenceTransformer(model, device=device)
    if window > encoder.max_seq_length:
        raise ValueError(f"window exceeds model max_seq_length={encoder.max_seq_length}")
    tokenizer = encoder.tokenizer
    budget = window - tokenizer.num_special_tokens_to_add(pair=False)
    if overlap >= budget:
        raise ValueError("overlap must leave space for content and special tokens")
    pending = []
    target = None
    passages = documents = 0
    def flush():
        nonlocal target, passages
        vectors = encoder.encode([r["text"] for r in pending], show_progress_bar=False)
        for row, vector in zip(pending, vectors):
            row["vector"] = vector.tolist()
        if target is None:
            target = connection.create_table(table, pending)
        else:
            target.add(pending)
        passages += len(pending)
        pending.clear()
    for filename in files:
        with open_text(filename) as source:
            for line in source:
                article = json.loads(line)
                tokens = tokenizer.encode(article["text"], add_special_tokens=False)
                for start in range(0, len(tokens), budget - overlap):
                    end = min(start + budget, len(tokens))
                    pending.append({"id": f"{article['id']}-{start}-{end}",
                                    "title": article.get("title", ""),
                                    "text": tokenizer.decode(tokens[start:end]),
                                    "url": article.get("url", "")})
                    if len(pending) >= batch_size:
                        flush()
                    if end == len(tokens):
                        break
                documents += 1
                if max_documents and documents >= max_documents:
                    break
        if max_documents and documents >= max_documents:
            break
    if pending:
        flush()
    if target is None:
        raise ValueError("Corpus contains no nonempty articles")
    if ann_partitions is not None:
        if ann_partitions < 1:
            raise ValueError("ann_partitions must be positive")
        target.create_index(metric="cosine", index_type="IVF_FLAT", num_partitions=ann_partitions)
    # Record embedding identity beside the index for serving with the same model.
    from pathlib import Path
    (Path(db) / f"{table}.wiki-retriever.json").write_text(json.dumps({"model": model, "window": window, "overlap": overlap}))
    return {"documents": documents, "passages": passages}
