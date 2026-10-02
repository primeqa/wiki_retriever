#!/usr/bin/env python3
"""Extract plain text from a Wikipedia XML dump in parallel using mwparserfromhell.

Outputs JSONL with fields: id, title, text, url

Usage:
    python3 extract_wiki.py enwiki-latest-pages-articles.xml.bz2 -o output.jsonl -w 24
"""

import argparse
import bz2
import json
import multiprocessing as mp
import os
import sys
import xml.etree.ElementTree as ET
from urllib.parse import quote


def parse_wikitext(args):
    """Worker function: convert wikitext to plain text via mwparserfromhell."""
    page_id, title, wikitext = args
    import mwparserfromhell

    try:
        parsed = mwparserfromhell.parse(wikitext)
        text = parsed.strip_code()
        # Collapse excessive whitespace while preserving paragraph breaks
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            lines.append(stripped)
        text = "\n".join(lines)
        # Remove runs of 3+ newlines
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")
        text = text.strip()
    except Exception:
        text = ""

    url = "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"), safe="/:()")

    return json.dumps({"id": page_id, "title": title, "text": text, "url": url}, ensure_ascii=False)


def iter_pages(dump_path):
    """Stream pages from a (possibly bz2-compressed) MediaWiki XML dump.

    Yields (page_id, title, wikitext) for article-namespace, non-redirect pages.
    """
    if dump_path.endswith(".bz2"):
        source = bz2.open(dump_path, "rb")
    else:
        source = open(dump_path, "rb")

    # We need to handle the MediaWiki XML namespace.
    # Detect it from the root element.
    ns = ""
    context = ET.iterparse(source, events=("start", "end", "start-ns"))

    for event, elem in context:
        if event == "start-ns":
            prefix, uri = elem
            if prefix == "" or "mediawiki" in uri.lower():
                ns = "{" + uri + "}"
                break

    # Re-open since iterparse consumed some of the stream
    source.close()
    if dump_path.endswith(".bz2"):
        source = bz2.open(dump_path, "rb")
    else:
        source = open(dump_path, "rb")

    tag_page = ns + "page"
    tag_title = ns + "title"
    tag_ns = ns + "ns"
    tag_id = ns + "id"
    tag_redirect = ns + "redirect"
    tag_text = ns + "text"
    tag_revision = ns + "revision"

    context = ET.iterparse(source, events=("end",))
    count = 0

    for event, elem in context:
        if elem.tag != tag_page:
            continue

        # Extract fields from the page element
        title_el = elem.find(tag_title)
        ns_el = elem.find(tag_ns)
        redirect_el = elem.find(tag_redirect)
        rev_el = elem.find(tag_revision)

        # Only article namespace (ns=0), skip redirects
        if ns_el is not None and ns_el.text != "0":
            elem.clear()
            continue
        if redirect_el is not None:
            elem.clear()
            continue

        title = title_el.text if title_el is not None else ""

        # The first <id> directly under <page> is the page id
        page_id = None
        for child in elem:
            if child.tag == tag_id:
                page_id = int(child.text)
                break

        text = ""
        if rev_el is not None:
            text_el = rev_el.find(tag_text)
            if text_el is not None and text_el.text:
                text = text_el.text

        # Free memory
        elem.clear()

        if not text or not title:
            continue

        count += 1
        yield (page_id, title, text)

        if count % 100000 == 0:
            print(f"  ... streamed {count:,} articles so far", file=sys.stderr, flush=True)

    source.close()
    print(f"  Total articles streamed: {count:,}", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Extract text from Wikipedia XML dump to JSONL")
    parser.add_argument("dump", help="Path to the XML dump (optionally bz2-compressed)")
    parser.add_argument("-o", "--output", default="wiki_articles.jsonl", help="Output JSONL path (default: wiki_articles.jsonl)")
    parser.add_argument("-w", "--workers", type=int, default=max(1, os.cpu_count() - 2),
                        help="Number of parallel workers (default: ncpus - 2)")
    parser.add_argument("--chunk", type=int, default=256,
                        help="Chunk size for imap_unordered (default: 256)")
    args = parser.parse_args()

    print(f"Input:   {args.dump}", file=sys.stderr)
    print(f"Output:  {args.output}", file=sys.stderr)
    print(f"Workers: {args.workers}", file=sys.stderr)
    print(file=sys.stderr)

    if args.output.endswith(".bz2"):
        open_func = lambda p: bz2.open(p, "wt", encoding="utf-8")
    else:
        open_func = lambda p: open(p, "w", encoding="utf-8")

    written = 0
    with mp.Pool(args.workers) as pool, open_func(args.output) as out:
        results = pool.imap_unordered(parse_wikitext, iter_pages(args.dump), chunksize=args.chunk)
        for line in results:
            out.write(line)
            out.write("\n")
            written += 1
            if written % 100000 == 0:
                print(f"  ... wrote {written:,} articles", file=sys.stderr, flush=True)

    print(f"\nDone. Wrote {written:,} articles to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
