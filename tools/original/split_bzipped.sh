#!/bin/bash
set -euo pipefail

INPUT="${1:-wikipedia.en.jsonl.bz2}"
CHUNK_SIZE="${2:-200000}"
PREFIX="${INPUT%.jsonl.bz2}"

if [[ ! -f "$INPUT" ]]; then
    echo "Error: $INPUT not found" >&2
    exit 1
fi

echo "Splitting $INPUT into chunks of $CHUNK_SIZE lines..."

bzcat "$INPUT" | split \
    --lines="$CHUNK_SIZE" \
    --numeric-suffixes=1 \
    --suffix-length=3 \
    --filter='bzip2 > $FILE.jsonl.bz2' \
    - "${PREFIX}_"

echo "Done. Output files:"
ls -lh "${PREFIX}_"*.jsonl.bz2
