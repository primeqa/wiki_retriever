"""Memory-bounded Wikipedia download, extraction and JSONL sharding."""
import bz2
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import re
import subprocess
from urllib.parse import quote
import xml.etree.ElementTree as ET


def open_text(path, mode="rt"):
    return (bz2.open if str(path).endswith(".bz2") else open)(path, mode, encoding="utf-8")


def download(output, language="en", date="latest"):
    if not re.fullmatch(r"[a-z][a-z0-9-]*", language) or not re.fullmatch(r"latest|\d{8}", date):
        raise ValueError("Invalid language or dump date")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    url = f"https://dumps.wikimedia.org/{language}wiki/{date}/{language}wiki-{date}-pages-articles.xml.bz2"
    subprocess.run(["wget", "-c", "-O", str(output), url], check=True)


def pages(path):
    opener = bz2.open if str(path).endswith(".bz2") else open
    with opener(path, "rb") as source:
        events = ET.iterparse(source, events=("start", "end"))
        _, root = next(events)
        for event, element in events:
            if event != "end" or element.tag.rsplit("}", 1)[-1] != "page":
                continue
            ns = element.tag[:-4]
            title = element.findtext(ns + "title")
            text = element.findtext(ns + "revision/" + ns + "text")
            if element.findtext(ns + "ns") == "0" and element.find(ns + "redirect") is None and title and text:
                yield int(element.findtext(ns + "id")), title, text
            root.clear()


def clean_article(item):
    import mwparserfromhell
    page_id, title, markup, language = item
    text = mwparserfromhell.parse(markup).strip_code()
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text).strip()
    return {"id": page_id, "title": title, "text": text,
            "url": f"https://{language}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"}


def extract(source, output, workers=1, chunk=256, language="en"):
    if workers < 1 or chunk < 1:
        raise ValueError("workers and chunk must be positive")
    if Path(source).resolve() == Path(output).resolve():
        raise ValueError("Input and output must differ")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    # Submit bounded batches: Executor.map over the full dump can queue every page.
    with open_text(output, "wt") as dest, ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        batch = []
        def write_batch(items):
            for article in pool.map(clean_article, items, chunksize=max(1, chunk // workers)):
                dest.write(json.dumps(article, ensure_ascii=False) + "\n")
        for page in pages(source):
            batch.append((*page, language))
            count += 1
            if len(batch) >= chunk:
                write_batch(batch)
                batch.clear()
        if batch:
            write_batch(batch)
    return count


def split(source, output_dir, records=200000, prefix="wikipedia.en"):
    if records < 1 or Path(prefix).name != prefix:
        raise ValueError("Positive record count and a filename prefix are required")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if list(output_dir.glob(f"{prefix}_*.jsonl.bz2")):
        raise FileExistsError("Shard prefix already exists; use a fresh directory or prefix")
    dest = None
    count = 0
    try:
        with open_text(source) as src:
            for line in src:
                if count % records == 0:
                    if dest:
                        dest.close()
                    dest = bz2.open(output_dir / f"{prefix}_{count // records + 1:03d}.jsonl.bz2", "wt", encoding="utf-8")
                dest.write(line)
                count += 1
    finally:
        if dest:
            dest.close()
    return count
