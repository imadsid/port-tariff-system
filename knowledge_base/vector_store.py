"""
knowledge_base/vector_store.py
ChromaDB vector store — stores two types of chunks:
  - semantic_rule : LLM-extracted rule/condition descriptions (rich metadata)
  - raw           : sliding-window text from PDF pages (broad context)
"""
from pathlib import Path
from typing import Optional

from config.settings import settings
from monitoring import get_logger

log = get_logger(__name__)


class VectorStore:

    def __init__(self) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embed_model = None
        log.info("ChromaDB ready", chunks=self.collection.count())

    @property
    def embed_model(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer(settings.embedding_model)
        return self._embed_model

    def _embed(self, text: str) -> list[float]:
        return self.embed_model.encode(text, convert_to_numpy=True).tolist()

    def upsert_chunks(self, chunks: list) -> int:
        if not chunks:
            return 0

        ids, documents, metadatas, embeddings = [], [], [], []
        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            # Store rich metadata for filtering
            meta = {
                "page_num":    chunk.page_num,
                "section":     chunk.section or "",
                "subsection":  chunk.subsection or "",
                "due_types":   ",".join(chunk.due_types),
                "chunk_type":  chunk.chunk_type,   # "semantic_rule" | "raw"
            }
            # Merge any extra metadata from semantic rules
            if chunk.metadata:
                for k, v in chunk.metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        meta[k] = str(v)
            metadatas.append(meta)
            if chunk.embedding:
                embeddings.append(chunk.embedding)

        if len(embeddings) == len(ids):
            self.collection.upsert(
                ids=ids, documents=documents,
                metadatas=metadatas, embeddings=embeddings,
            )
        else:
            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

        log.info("Chunks upserted to ChromaDB", count=len(ids))
        return len(ids)

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        due_type_filter: Optional[str] = None,
        chunk_type_filter: Optional[str] = None,   # "semantic_rule" | "raw"
        query_embedding: Optional[list[float]] = None,
    ) -> list[dict]:
        """
        Semantic search with optional filters.
        Prefer semantic_rule chunks for explanation (richer metadata).
        """
        total = self.collection.count()
        if total == 0:
            return []

        n_results = min(n_results, total)
        if not query_embedding:
            query_embedding = self._embed(query_text)

        # Build where clause
        filters = []
        if due_type_filter:
            filters.append({"due_types": {"$contains": due_type_filter}})
        if chunk_type_filter:
            filters.append({"chunk_type": {"$eq": chunk_type_filter}})

        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results":        n_results,
        }
        if len(filters) == 1:
            kwargs["where"] = filters[0]
        elif len(filters) > 1:
            kwargs["where"] = {"$and": filters}

        try:
            results = self.collection.query(**kwargs)
        except Exception:
            # Retry without filters if ChromaDB raises
            kwargs.pop("where", None)
            results = self.collection.query(**kwargs)

        hits: list[dict] = []
        if results and results.get("documents"):
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results.get("distances", [[]])[0],
            ):
                hits.append({
                    "text":       doc,
                    "metadata":   meta,
                    "similarity": round(1.0 - dist, 4),
                    "chunk_type": meta.get("chunk_type", "raw"),
                })
        return hits

    def query_semantic_rules(self, query_text: str, due_type: str, n: int = 3) -> list[dict]:
        """Convenience: query only semantic rule chunks for a specific due type."""
        return self.query(
            query_text=query_text,
            n_results=n,
            due_type_filter=due_type,
            chunk_type_filter="semantic_rule",
        )

    def count(self) -> int:
        return self.collection.count()

    def stats(self) -> dict:
        """Return breakdown of chunk types stored."""
        total = self.collection.count()
        if total == 0:
            return {"total": 0, "semantic_rule": 0, "raw": 0}
        try:
            sem = self.collection.get(where={"chunk_type": {"$eq": "semantic_rule"}})
            raw = self.collection.get(where={"chunk_type": {"$eq": "raw"}})
            return {
                "total":         total,
                "semantic_rule": len(sem["ids"]),
                "raw":           len(raw["ids"]),
            }
        except Exception:
            return {"total": total, "semantic_rule": 0, "raw": 0}
