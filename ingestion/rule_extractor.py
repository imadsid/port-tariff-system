"""
ingestion/rule_extractor.py
Hybrid extraction — two LangChain LLM calls per section:
  1. STRUCTURED extraction -> numbers/tables -> SQLite
  2. SEMANTIC extraction   -> rules/conditions -> ChromaDB

Uses LangChain's ChatGroq wrapper with automatic retry on rate limits.
"""
import json
import time
from typing import Any

from config.settings import settings
from monitoring import get_logger

log = get_logger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────
# String concatenation only — tariff text may contain { } % characters
# that would break .format() or % templating.

_STRUCTURED_PROMPT = """You are a port tariff data extraction specialist.
Extract ALL numeric rates, fees, tiers, surcharges, and reductions from the tariff text below.

Return ONLY a valid JSON object. No markdown code fences, no explanation.

Schema:
{
  "rates": [{"due_type": "light_dues|port_dues|vts_dues|towage_dues|pilotage_dues|running_lines_dues", "port": "Durban|Cape Town|Richards Bay|ALL", "vessel_type": "ALL", "rate": 0.0, "unit": "per_100gt|per_gt|per_call|flat", "section": "", "notes": ""}],
  "tiers": [{"due_type": "towage_dues|pilotage_dues|running_lines_dues", "port": "ALL", "gt_min": 0, "gt_max": null, "base_fee": 0.0, "rate_per_unit": 0.0, "section": "", "notes": null}],
  "surcharges": [{"due_type": "", "name": "outside_working_hours|weekend|holiday", "pct": 25.0, "applies_to": "ALL", "condition": "", "section": ""}],
  "reductions": [{"due_type": "", "name": "coaster|double_hull_tanker|short_stay|government_vessel", "pct": 35.0, "applies_to": "ALL", "condition": "", "section": ""}],
  "minimums": [{"due_type": "", "port": "ALL", "amount": 0.0, "condition": null, "section": ""}]
}

TARIFF TEXT:
"""

_SEMANTIC_PROMPT = """You are a port tariff policy expert.
Extract RULES, CONDITIONS, EXEMPTIONS and POLICY TEXT from the tariff section below.
Focus on: who is exempt, when surcharges apply, special conditions, definitions.
Do NOT extract raw numbers — those are captured separately.

Return ONLY a JSON array. No markdown code fences, no explanation.

Schema:
[{"due_type": "light_dues|port_dues|vts_dues|towage_dues|pilotage_dues|running_lines_dues|general", "rule_type": "exemption|condition|definition|policy|surcharge_condition|reduction_condition", "title": "short title", "description": "full plain English description", "applies_to": "ALL", "section": ""}]

TARIFF TEXT:
"""


def _strip_fences(raw: str) -> str:
    """Remove markdown code fences from LLM response."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        elif lines[0].strip().startswith("```"):
            lines = lines[1:]
        raw = "\n".join(lines)
    return raw.strip()


class RuleExtractor:
    """
    Uses LangChain's ChatGroq for structured and semantic extraction.

    LangChain benefits here:
    - Standardised LLM interface — swap provider by changing one import
    - Built-in message formatting (HumanMessage)
    - Easy to add output parsers or chains in future
    """

    def __init__(self) -> None:
        self._llm_fast = None   # llama-3.1-8b-instant — for structured JSON
        self._llm_smart = None  # llama-3.3-70b-versatile — for semantic rules

    def _get_llm(self, model: str):
        """Lazy-init LangChain ChatGroq instance."""
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=settings.groq_api_key,
            model=model,
            temperature=0,
            max_tokens=3000,
        )

    @property
    def llm_fast(self):
        if self._llm_fast is None:
            self._llm_fast = self._get_llm("llama-3.1-8b-instant")
        return self._llm_fast

    @property
    def llm_smart(self):
        if self._llm_smart is None:
            self._llm_smart = self._get_llm("llama-3.1-8b-instant")
        return self._llm_smart

    def _invoke(self, llm, prompt: str, max_retries: int = 3) -> str:
        """
        Invoke a LangChain LLM with retry logic for rate limits.
        LangChain's .invoke() returns an AIMessage — we extract .content.
        """
        from langchain_core.messages import HumanMessage
        last_exc = None
        for attempt in range(max_retries):
            try:
                message  = HumanMessage(content=prompt)
                response = llm.invoke([message])
                return _strip_fences(response.content)
            except Exception as exc:
                last_exc = exc
                err = str(exc)
                if "tokens per day" in err or "TPD" in err:
                    log.error("Daily token limit reached", error=err)
                    raise RuntimeError(
                        "Groq daily token limit reached. "
                        "Wait until tomorrow or use a new API key."
                    ) from exc
                elif "rate_limit" in err.lower() or "tokens per minute" in err:
                    wait = 30 * (attempt + 1)
                    log.warning("Rate limit — retrying", wait_secs=wait, attempt=attempt + 1)
                    time.sleep(wait)
                else:
                    raise
        raise last_exc

    def extract_structured(self, text: str, section: str = "") -> dict[str, list]:
        """
        Structured extraction using LangChain ChatGroq (fast 8b model).
        Returns rates, tiers, surcharges, reductions, minimums for SQLite.
        """
        empty = {"rates": [], "tiers": [], "surcharges": [], "reductions": [], "minimums": []}
        if len(text.strip()) < 50:
            return empty
        try:
            raw  = self._invoke(self.llm_fast, _STRUCTURED_PROMPT + text)
            data = json.loads(raw)
            result = {k: data.get(k) or [] for k in empty}
            total  = sum(len(v) for v in result.values())
            log.info("Structured extraction done", section=section, total_rows=total)
            return result
        except Exception as exc:
            log.warning("Structured extraction failed", section=section, error=str(exc))
            return empty

    def extract_semantic(self, text: str, section: str = "") -> list[dict[str, Any]]:
        """
        Semantic extraction using LangChain ChatGroq.
        Returns rules, conditions, exemptions for ChromaDB.
        """
        if len(text.strip()) < 50:
            return []
        try:
            raw   = self._invoke(self.llm_smart, _SEMANTIC_PROMPT + text, max_retries=3)
            rules = json.loads(raw)
            if not isinstance(rules, list):
                rules = rules.get("rules", []) if isinstance(rules, dict) else []
            for r in rules:
                r["source_section"] = section
            log.info("Semantic extraction done", section=section, rules=len(rules))
            return rules
        except Exception as exc:
            log.warning("Semantic extraction failed", section=section, error=str(exc))
            return []

    def extract_all_sections(self, sections: list[dict]) -> dict:
        all_structured = {"rates": [], "tiers": [], "surcharges": [], "reductions": [], "minimums": []}
        all_semantic: list[dict] = []

        for i, section in enumerate(sections):
            text  = section.get("text", "")
            title = section.get("title", "Unknown")
            log.info("Extracting section", title=title, num=f"{i+1}/{len(sections)}", chars=len(text))

            structured = self.extract_structured(text, title)
            for key in all_structured:
                all_structured[key].extend(structured.get(key, []))

            time.sleep(1)

            semantic = self.extract_semantic(text, title)
            all_semantic.extend(semantic)

            if (i + 1) % 3 == 0:
                log.info("Rate limit pause", after_section=i + 1)
                time.sleep(3)

        log.info(
            "All sections extracted",
            rates      =len(all_structured["rates"]),
            tiers      =len(all_structured["tiers"]),
            surcharges =len(all_structured["surcharges"]),
            reductions =len(all_structured["reductions"]),
            minimums   =len(all_structured["minimums"]),
            semantic   =len(all_semantic),
        )
        return {"structured": all_structured, "semantic": all_semantic}
