"""
explanation/generator.py
Explanation Generator using LangChain RAG chain.

LangChain pipeline:
  1. ChromaDB retrieval — fetch relevant tariff policy chunks per due type
  2. PromptTemplate — build a grounded prompt with retrieved context
  3. ChatGroq — generate a professional cited explanation
  4. StrOutputParser — extract clean string from AIMessage
"""
import json
from typing import Any

from config.settings import settings
from monitoring import get_logger

log = get_logger(__name__)

_TEMPLATE = """You are a senior port tariff consultant explaining a dues calculation to a shipping agent.

Vessel: {vessel_name} — Port of {port}
Gross Tonnage: {gt} GT | Days in Port: {days}

Calculation Results:
{results_json}

Relevant Tariff Sections (retrieved from the TNPA Tariff Book):
{context}

Write a clear, professional explanation that:
1. States the grand total payable (incl. 15% VAT).
2. Explains each due type step-by-step with the formula used.
3. Notes any reductions, surcharges, or exemptions that apply.
4. Cites the relevant tariff section numbers from the context above.

Use South African Rand (R) notation. Be precise with all numbers.
"""


class ExplanationGenerator:
    """
    RAG-powered explanation generator using LangChain.

    Chain:
      PromptTemplate | ChatGroq | StrOutputParser
    """

    def __init__(self) -> None:
        self._llm          = None   # lazy ChatGroq
        self._chain        = None   # lazy LCEL chain
        self._vector_store = None   # lazy ChromaDB

    def _get_llm(self):
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=0.2,
            max_tokens=2048,
        )

    def _get_chain(self):
        """
        Build LCEL chain:
          PromptTemplate | ChatGroq | StrOutputParser

        StrOutputParser extracts the plain string from AIMessage.content
        so the caller gets a clean string rather than a message object.
        """
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        if self._llm is None:
            self._llm = self._get_llm()

        prompt = PromptTemplate(
            template=_TEMPLATE,
            input_variables=["vessel_name", "port", "gt", "days", "results_json", "context"],
        )
        return prompt | self._llm | StrOutputParser()

    @property
    def chain(self):
        if self._chain is None:
            self._chain = self._get_chain()
        return self._chain

    @property
    def vector_store(self):
        if self._vector_store is None:
            from knowledge_base.vector_store import VectorStore
            self._vector_store = VectorStore()
        return self._vector_store

    def generate(self, query: Any, result: Any) -> dict:
        """
        RAG pipeline:
          1. Retrieve relevant chunks from ChromaDB for each due type
          2. Build grounded prompt with retrieved context
          3. Invoke LangChain chain → professional explanation
        """
        # ── Step 1: Retrieve context from ChromaDB ────────────────────────────
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

        # Deduplicate chunks
        seen: set[str] = set()
        unique_chunks: list[str] = []
        for hit in context_chunks:
            text = hit["text"]
            if text not in seen:
                seen.add(text)
                section = hit["metadata"].get("section", "")
                unique_chunks.append(f"[{section}]: {text[:400]}")

        context_text = "\n\n".join(unique_chunks[:6]) or "No context retrieved."

        # ── Step 2: Build results summary ─────────────────────────────────────
        results_summary: dict = {}
        for dt, dr in result.dues.items():
            results_summary[dt] = {
                "net_amount_excl_vat": dr.net_amount,
                "vat_amount":          dr.vat_amount,
                "total_incl_vat":      dr.total_with_vat,
                "formula":             dr.formula_applied,
                "rate_source":         getattr(dr, "rate_source", "hardcoded"),
                "exempted":            dr.exempted,
            }
        results_summary["grand_total"] = {
            "excl_vat": result.grand_total_excl_vat,
            "vat":      result.grand_total_vat,
            "incl_vat": result.grand_total_incl_vat,
        }

        # ── Step 3: Invoke LangChain RAG chain ────────────────────────────────
        explanation = self.chain.invoke({
            "vessel_name":  result.vessel_name,
            "port":         result.port,
            "gt":           f"{int(result.gross_tonnage):,}",
            "days":         f"{result.days_in_port:.2f}",
            "results_json": json.dumps(results_summary, indent=2),
            "context":      context_text,
        })

        citations = self._build_citations(context_chunks)
        log.info(
            "Explanation generated via LangChain RAG",
            chars=len(explanation),
            citations=len(citations),
            context_chunks=len(unique_chunks),
        )
        return {
            "explanation":          explanation,
            "citations":            citations,
            "context_chunks_used":  len(unique_chunks),
        }

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
