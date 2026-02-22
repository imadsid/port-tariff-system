"""
explanation/generator.py
Explanation Generator using Groq.
Retrieves relevant tariff context from ChromaDB and uses Groq to produce
a human-readable, cited explanation of the calculated dues.
"""
import json

from config.settings import settings
from monitoring import get_logger

log = get_logger(__name__)

_PROMPT = """You are a senior port tariff consultant explaining a dues calculation to a shipping agent.

Vessel: {vessel_name} — Port of {port}
Gross Tonnage: {gt:,} GT | Days in Port: {days:.2f}

Calculation Results:
{results_json}

Relevant Tariff Sections:
{context}

Write a clear, professional explanation that:
1. States the grand total payable (incl. 15% VAT).
2. Explains each due type step-by-step with the formula used.
3. Notes any reductions, surcharges, or exemptions that apply.
4. Cites the relevant tariff section numbers.

Use South African Rand (R) notation. Be precise with all numbers.
"""


class ExplanationGenerator:

    def __init__(self) -> None:
        self._client      = None  # lazy
        self._vector_store = None  # lazy

    @property
    def client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=settings.groq_api_key)
        return self._client

    @property
    def vector_store(self):
        if self._vector_store is None:
            from knowledge_base.vector_store import VectorStore
            self._vector_store = VectorStore()
        return self._vector_store

    def generate(self, query, result) -> dict:
        # Retrieve relevant context for each due type
        context_chunks: list[dict] = []
        for due_type in result.dues:
            try:
                hits = self.vector_store.query(
                    query_text=f"{due_type.replace('_', ' ')} {query.port}",
                    n_results=2,
                    due_type_filter=due_type,
                )
                context_chunks.extend(hits)
            except Exception:
                pass

        # Deduplicate
        seen: set[str] = set()
        unique_chunks: list[str] = []
        for hit in context_chunks:
            text = hit["text"]
            if text not in seen:
                seen.add(text)
                section = hit["metadata"].get("section", "")
                unique_chunks.append(f"[{section}]: {text[:400]}")

        context_text = "\n\n".join(unique_chunks[:6]) or "No context retrieved."

        # Serialise results
        results_summary: dict = {}
        for dt, dr in result.dues.items():
            results_summary[dt] = {
                "net_amount_excl_vat": dr.net_amount,
                "vat_amount":          dr.vat_amount,
                "total_incl_vat":      dr.total_with_vat,
                "formula":             dr.formula_applied,
                "exempted":            dr.exempted,
            }
        results_summary["grand_total"] = {
            "excl_vat": result.grand_total_excl_vat,
            "vat":      result.grand_total_vat,
            "incl_vat": result.grand_total_incl_vat,
        }

        prompt = _PROMPT.format(
            vessel_name  =result.vessel_name,
            port         =result.port,
            gt           =int(result.gross_tonnage),
            days         =result.days_in_port,
            results_json =json.dumps(results_summary, indent=2),
            context      =context_text,
        )

        response = self.client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2048,
        )
        explanation = response.choices[0].message.content
        citations   = self._build_citations(context_chunks)

        log.info("Explanation generated", chars=len(explanation), citations=len(citations))
        return {"explanation": explanation, "citations": citations, "context_chunks_used": len(unique_chunks)}

    @staticmethod
    def _build_citations(hits: list[dict]) -> list[dict]:
        seen: set[str] = set()
        citations: list[dict] = []
        for hit in hits:
            meta    = hit.get("metadata", {})
            section = meta.get("section", "Unknown")
            if section not in seen:
                seen.add(section)
                citations.append({
                    "section":         section,
                    "page":            meta.get("page_num"),
                    "relevance_score": hit.get("similarity", 0),
                })
        return citations
