"""
LanceDB client adapter that provides the same interface as pymilvus.MilvusClient.

This adapter allows the existing gRPC server (MilvusServicer) to work with LanceDB
without modification, since both MilvusClient and LanceDBClientAdapter expose the
same search/list_collections/has_collection interface.
"""

import logging
from typing import List, Dict, Any, Optional

import lancedb


class LanceDBClientAdapter:
    """Adapter that wraps a LanceDB connection to match the pymilvus.MilvusClient interface.

    This allows the existing gRPC server to serve LanceDB data using the same
    proto and servicer code used for Milvus.
    """

    def __init__(self, uri: str, metric: str = "cosine"):
        """
        Initialize the LanceDB client adapter.

        Args:
            uri: Path to the LanceDB database directory.
        """
        self.metric = metric
        self.uri = uri
        self.db = lancedb.connect(uri)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"LanceDB client adapter initialized with database at: {uri}")

    def list_collections(self) -> List[str]:
        """List all table names (equivalent to Milvus collections)."""
        return self.db.table_names()

    def has_collection(self, collection_name: str) -> bool:
        """Check if a table (collection) exists."""
        return collection_name in self.db.table_names()

    def search(self, collection_name: str, data: List[List[float]],
               limit: int = 10, output_fields: Optional[List[str]] = None,
               search_params: Optional[Dict[str, Any]] = None) -> List[List[Dict[str, Any]]]:
        """
        Perform vector search matching pymilvus.MilvusClient.search() interface.

        Args:
            collection_name: Name of the LanceDB table to search.
            data: List of query vectors. Each element is a vector (list of floats).
                  Typically a single-element list: [[0.1, 0.2, ...]].
            limit: Maximum number of results per query.
            output_fields: Fields to include in results. If ["*"] or None, returns all fields.
            search_params: Additional search parameters (unused for LanceDB compatibility).

        Returns:
            List of result lists (one per query vector), matching Milvus format:
            [[{"id": ..., "distance": ..., "entity": {"field": "value", ...}}, ...], ...]
        """
        table = self.db.open_table(collection_name)
        all_results = []

        for query_vector in data:
            results = (
                table.search(query_vector).distance_type(self.metric)
                .limit(limit)
                .to_list()
            )

            formatted = []
            for row in results:
                hit = {
                    "id": row.get("id", ""),
                    "distance": row.get("_distance", 0.0),
                    "entity": {},
                }

                # Determine which fields to include
                if output_fields is None or output_fields == ["*"]:
                    include_fields = set(row.keys()) - {"id", "_distance", "vector"}
                else:
                    include_fields = set(output_fields) - {"id", "_distance", "vector"}

                for field in include_fields:
                    if field in row:
                        hit["entity"][field] = row[field]

                formatted.append(hit)

            all_results.append(formatted)

        return all_results

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get table statistics."""
        table = self.db.open_table(collection_name)
        return {
            "row_count": table.count_rows(),
        }

    def close(self):
        """Close the connection (no-op for LanceDB file-based access)."""
        self.db = None
