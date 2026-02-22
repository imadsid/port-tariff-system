"""
ingestion/pipeline.py
Hybrid Ingestion Pipeline
"""
import hashlib
import re
from pathlib import Path

from ingestion.chunker import Chunker
from ingestion.pdf_extractor import PDFTextExtractor
from ingestion.rule_extractor import RuleExtractor
from knowledge_base.sqlite_store import SQLiteStore
from knowledge_base.vector_store import VectorStore
from monitoring import get_logger

log = get_logger(__name__)

SECTION_HINTS: list[dict] = [
    {"title": "1.1 Light Dues",              "keywords": ["LIGHT DUES", "1.1"]},
    {"title": "2.1 VTS Charges",             "keywords": ["VESSEL TRAFFIC SERVICES", "VTS CHARGES", "2.1"]},
    {"title": "3.3 Pilotage Services",       "keywords": ["PILOTAGE SERVICES", "3.3"]},
    {"title": "3.6 Tug/Vessel Assistance",   "keywords": ["TUGS/VESSEL ASSISTANCE", "3.6", "TUG ASSISTANCE"]},
    {"title": "3.8 Berthing Services",       "keywords": ["BERTHING SERVICES", "3.8"]},
    {"title": "3.9 Running of Vessel Lines", "keywords": ["RUNNING OF VESSEL LINES", "3.9"]},
    {"title": "4.1.1 Port Dues",             "keywords": ["PORT DUES", "4.1.1", "PORT FEES ON VESSELS"]},
    {"title": "4.1.2 Berth Dues",            "keywords": ["BERTH DUES", "4.1.2"]},
    {"title": "7 Cargo Dues",                "keywords": ["CARGO DUES", "SECTION 7"]},
]

SECTION_RE = re.compile(r"(?=SECTION\s+\d+|^\d+\.\d+\s+[A-Z])", re.MULTILINE)


class IngestionPipeline:

    def __init__(self) -> None:
        self.pdf_extractor = PDFTextExtractor()
        self.rule_extractor = RuleExtractor()
        self.chunker        = Chunker()
        self.sqlite_store   = SQLiteStore()
        self.vector_store   = VectorStore()

    def run(self, pdf_path: str | Path, force_reingest: bool = False) -> dict:
        """
        Run the full hybrid ingestion pipeline.
        Returns a summary dict with counts for each store.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Check if already ingested
        pdf_hash = self._hash_file(pdf_path)
        if not force_reingest and self.sqlite_store.count() > 0:
            log.info("Knowledge base already populated — skipping ingestion. Use force_reingest=True to re-run.")
            return {"status": "skipped", "reason": "already_ingested"}

        log.info("Starting hybrid ingestion", pdf=str(pdf_path), force=force_reingest)

        if force_reingest:
            self.sqlite_store.clear_all()
            log.info("SQLite store cleared for re-ingestion")

        # ── Step 1: Extract text ──────────────────────────────────────────────
        doc = self.pdf_extractor.extract(pdf_path)
        log.info("PDF extracted", pages=doc.total_pages, chars=len(doc.full_text))

        # ── Step 2: Split into sections ───────────────────────────────────────
        sections = self._split_into_sections(doc.full_text)
        log.info("Sections identified", count=len(sections))

        # ── Step 3: Dual LLM extraction ───────────────────────────────────────
        log.info("Running hybrid LLM extraction (structured + semantic)...")
        extracted = self.rule_extractor.extract_all_sections(sections)

        structured = extracted["structured"]   # rates, tiers, surcharges, reductions, minimums
        semantic   = extracted["semantic"]     # rule/condition descriptions

        # ── Step 4a: Persist structured data to SQLite ────────────────────────
        sqlite_rows = 0
        sqlite_rows += self._safe_insert(self.sqlite_store.insert_rates,      structured["rates"])
        sqlite_rows += self._safe_insert(self.sqlite_store.insert_tiers,      structured["tiers"])
        sqlite_rows += self._safe_insert(self.sqlite_store.insert_surcharges, structured["surcharges"])
        sqlite_rows += self._safe_insert(self.sqlite_store.insert_reductions, structured["reductions"])
        sqlite_rows += self._safe_insert(self.sqlite_store.insert_minimums,   structured["minimums"])

        log.info("SQLite populated", rows=sqlite_rows, breakdown=self.sqlite_store.stats())

        # ── Step 4b: Build ChromaDB chunks ────────────────────────────────────
        # Source 1: semantic rule chunks (rich LLM descriptions)
        sem_chunks = self.chunker.chunks_from_semantic_rules(semantic)

        # Source 2: raw page text chunks (broad context fallback)
        raw_chunks = self.chunker.chunks_from_pages(doc.pages)

        all_chunks = sem_chunks + raw_chunks
        log.info("Chunks created", semantic=len(sem_chunks), raw=len(raw_chunks), total=len(all_chunks))

        # ── Step 4c: Embed and upsert to ChromaDB ────────────────────────────
        all_chunks = self.chunker.generate_embeddings(all_chunks)
        chroma_count = self.vector_store.upsert_chunks(all_chunks)

        # ── Step 5: Log ingestion ─────────────────────────────────────────────
        self.sqlite_store.log_ingestion(str(pdf_path), pdf_hash, sqlite_rows)

        sqlite_stats = self.sqlite_store.stats()
        chroma_stats = self.vector_store.stats()

        summary = {
            "status": "success",
            "pdf":    str(pdf_path),
            "pages":  doc.total_pages,
            "sections_processed": len(sections),
            "sqlite": {
                "total_rows": sqlite_rows,
                "rates":      sqlite_stats.get("rates", 0),
                "tiers":      sqlite_stats.get("tiers", 0),
                "surcharges": sqlite_stats.get("surcharges", 0),
                "reductions": sqlite_stats.get("reductions", 0),
                "minimums":   sqlite_stats.get("minimums", 0),
            },
            "chromadb": {
                "total_chunks":    chroma_count,
                "semantic_rules":  chroma_stats.get("semantic_rule", 0),
                "raw_text_chunks": chroma_stats.get("raw", 0),
            },
        }
        log.info("Hybrid ingestion complete", **{k: v for k, v in summary.items() if k != "pdf"})
        return summary

    # ── Private ───────────────────────────────────────────────────────────────

    def _split_into_sections(self, full_text: str) -> list[dict[str, str]]:
        """
        Split document into named sections using known tariff headings.

        Uses the LAST occurrence of each keyword to skip the Table of Contents
        (which lists section titles early in the document) and land on the
        actual content body where the tariff rates and rules live.
        """
        import re as _re

        found: list[dict] = []
        for hint in SECTION_HINTS:
            best_pos  = -1
            best_match = None
            for kw in hint["keywords"]:
                # Find ALL occurrences — use the last one (body, not TOC)
                pattern = _re.compile(_re.escape(kw), _re.IGNORECASE)
                matches = list(pattern.finditer(full_text))
                if matches:
                    # Last match is deepest in the document = actual section body
                    last = matches[-1]
                    if last.start() > best_pos:
                        best_pos   = last.start()
                        best_match = {"pos": last.start(), "title": hint["title"]}
            if best_match:
                found.append(best_match)

        # Sort by position
        found.sort(key=lambda x: x["pos"])

        # Deduplicate by title
        seen: set[str] = set()
        unique: list[dict] = []
        for item in found:
            if item["title"] not in seen:
                seen.add(item["title"])
                unique.append(item)

        if len(unique) < 2:
            log.warning("Section detection found < 2 sections — using fixed windows")
            window, step = 4000, 3500
            return [
                {"title": f"window_{i}", "text": full_text[i: i + window]}
                for i in range(0, len(full_text), step)
            ]

        # Slice text between section positions
        sections: list[dict[str, str]] = []
        for i, item in enumerate(unique):
            start = item["pos"]
            end   = unique[i + 1]["pos"] if i + 1 < len(unique) else len(full_text)
            text  = full_text[start:end].strip()
            if len(text) > 200:  # skip anything still too short
                sections.append({"title": item["title"], "text": text})

        log.info(
            "Sections split",
            count=len(sections),
            sizes={s["title"]: len(s["text"]) for s in sections},
        )
        return sections

    @staticmethod
    def _safe_insert(fn, rows: list) -> int:
        if not rows:
            return 0
        # Normalise: fill missing keys with defaults
        defaults = {
            "port": "ALL", "vessel_type": "ALL", "notes": None,
            "condition": None, "applies_to": "ALL", "section": "",
            "gt_max": None, "rate_per_unit": 0,
        }
        normalised = [{**defaults, **r} for r in rows if isinstance(r, dict)]
        try:
            return fn(normalised)
        except Exception as exc:
            log.warning("Insert failed", fn=fn.__name__, error=str(exc))
            return 0

    @staticmethod
    def _hash_file(path: Path) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
