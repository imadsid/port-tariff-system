"""
ingestion/chunker.py
Hybrid chunker — creates ChromaDB chunks from two sources:

  1. Semantic rules extracted by the LLM (structured descriptions)
  2. Raw text sliding-window chunks (fallback for context retrieval)

Uses sentence-transformers for local embeddings (no API key needed).
"""
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config.settings import settings
from monitoring import get_logger

log = get_logger(__name__)


@dataclass
class TextChunk:
    chunk_id:    str
    text:        str
    page_num:    int
    section:     str
    subsection:  str
    chunk_type:  str = "raw"        # "semantic_rule" | "raw"
    due_types:   list[str] = field(default_factory=list)
    metadata:    dict = field(default_factory=dict)
    embedding:   Optional[list[float]] = None


DUE_TYPE_KEYWORDS: dict[str, list[str]] = {
    "light_dues":         ["light dues", "lighthouse", "1.1"],
    "vts_dues":           ["vts", "vessel traffic", "2.1"],
    "towage_dues":        ["towage", "tug", "3.6"],
    "pilotage_dues":      ["pilotage", "pilot", "3.3"],
    "running_lines_dues": ["running lines", "berthing", "mooring", "3.8"],
    "port_dues":          ["port dues", "4.1"],
}


class Chunker:
    """
    Produces TextChunks for ChromaDB from:
      - LLM-extracted semantic rules (rich metadata, clean descriptions)
      - Raw text pages (sliding window, for broad context retrieval)
    """

    def __init__(self) -> None:
        self.chunk_size = settings.chunk_size
        self.overlap    = settings.chunk_overlap
        self._model     = None  # lazy

    @property
    def embed_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            log.info("Loading embedding model", model=settings.embedding_model)
            self._model = SentenceTransformer(settings.embedding_model)
        return self._model

    # ── Public ────────────────────────────────────────────────────────────────

    def chunks_from_semantic_rules(self, rules: list[dict]) -> list[TextChunk]:
        """
        Convert LLM-extracted semantic rules into TextChunks.
        Each rule becomes one chunk — the text is its full description,
        enriched with title and applies_to for better retrieval.
        """
        chunks: list[TextChunk] = []
        for i, rule in enumerate(rules):
            due_type    = rule.get("due_type", "general")
            title       = rule.get("title", "")
            description = rule.get("description", "")
            rule_type   = rule.get("rule_type", "policy")
            applies_to  = rule.get("applies_to", "ALL")
            section     = rule.get("section", "") or rule.get("source_section", "")

            if not description.strip():
                continue

            # Compose rich text: title + description + context
            text = f"[{due_type.replace('_', ' ').upper()} — {rule_type.upper()}]\n"
            if title:
                text += f"Title: {title}\n"
            if applies_to and applies_to != "ALL":
                text += f"Applies to: {applies_to}\n"
            text += f"\n{description}"

            chunk_id = self._make_id(text, section, i, prefix="sem")
            chunks.append(TextChunk(
                chunk_id   =chunk_id,
                text       =text,
                page_num   =0,
                section    =section,
                subsection =rule_type,
                chunk_type ="semantic_rule",
                due_types  =[due_type] if due_type != "general" else [],
                metadata   ={
                    "rule_type":  rule_type,
                    "applies_to": applies_to,
                    "title":      title,
                    "due_type":   due_type,
                    "section":    section,
                },
            ))

        log.info("Semantic rule chunks created", count=len(chunks))
        return chunks

    def chunks_from_pages(self, pages: list) -> list[TextChunk]:
        """
        Create sliding-window raw text chunks from extracted PDF pages.
        Used as fallback context for explanation generation.
        """
        chunks: list[TextChunk] = []
        global_idx = 0
        for page in pages:
            for raw in self._sliding_window(page.raw_text):
                if len(raw.strip()) < 50:
                    continue
                chunk_id = self._make_id(raw, page.section or "", global_idx, prefix="raw")
                chunks.append(TextChunk(
                    chunk_id   =chunk_id,
                    text       =raw,
                    page_num   =page.page_num,
                    section    =page.section or "Unknown",
                    subsection =page.subsection or "",
                    chunk_type ="raw",
                    due_types  =self._detect_due_types(raw),
                ))
                global_idx += 1

        log.info("Raw page chunks created", count=len(chunks))
        return chunks

    def generate_embeddings(self, chunks: list[TextChunk]) -> list[TextChunk]:
        """Generate local embeddings for all chunks using sentence-transformers."""
        if not chunks:
            return chunks

        log.info("Generating embeddings", chunks=len(chunks))
        texts = [c.text for c in chunks]
        embeddings = self.embed_model.encode(
            texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb.tolist()

        log.info("Embeddings generated", total=len(chunks))
        return chunks

    # ── Private ───────────────────────────────────────────────────────────────

    def _sliding_window(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return []
        step = max(1, self.chunk_size - self.overlap)
        return [
            " ".join(words[i: i + self.chunk_size])
            for i in range(0, len(words), step)
            if words[i: i + self.chunk_size]
        ]

    @staticmethod
    def _make_id(text: str, section: str, idx: int, prefix: str = "chunk") -> str:
        digest = hashlib.md5(f"{section}:{idx}:{text[:80]}".encode()).hexdigest()[:12]
        return f"{prefix}_{idx}_{digest}"

    @staticmethod
    def _detect_due_types(text: str) -> list[str]:
        text_lower = text.lower()
        return [
            dt for dt, keywords in DUE_TYPE_KEYWORDS.items()
            if any(kw in text_lower for kw in keywords)
        ]
