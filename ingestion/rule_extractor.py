"""
ingestion/rule_extractor.py
Hybrid extraction — two separate LLM calls per section:
  1. STRUCTURED extraction -> numbers/tables -> SQLite
  2. SEMANTIC extraction   -> rules/conditions -> ChromaDB
"""
import json
import re
import time
from typing import Any

from config.settings import settings
from monitoring import get_logger

log = get_logger(__name__)


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


def _clean_json(raw: str) -> str:
    """Strip markdown code fences and whitespace from LLM response."""
    raw = raw.strip()
    # Remove opening fence
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    # Remove closing fence
    raw = re.sub(r"\s*```\s*$", "", raw)
    return raw.strip()


class RuleExtractor:

    def __init__(self) -> None:
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=settings.groq_api_key)
        return self._client

    def _call_llm(self, prompt: str, max_tokens: int = 3000, model: str = None) -> str:
        # Use fast 8b model for structured extraction to save daily token quota
        # Fall back to settings model for semantic extraction
        use_model = model or settings.groq_model
        response = self.client.chat.completions.create(
            model=use_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences manually — no regex escaping issues
        if raw.startswith("```"):
            lines = raw.split("\n")
            # Remove first line (```json or ```) and last line (```)
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            elif lines[0].strip().startswith("```"):
                lines = lines[1:]
            raw = "\n".join(lines)
        return raw.strip()

    def extract_structured(self, text: str, section: str = "") -> dict[str, list]:
        empty = {"rates": [], "tiers": [], "surcharges": [], "reductions": [], "minimums": []}
        if len(text.strip()) < 50:
            return empty
        try:
            # Append text after prompt — no string formatting needed
            prompt = _STRUCTURED_PROMPT + text
            # Use fast 8b model — structured JSON needs less reasoning power
            raw = self._call_llm(prompt, model="llama-3.1-8b-instant")
            data = json.loads(raw)
            result = {k: data.get(k) or [] for k in empty}
            total = sum(len(v) for v in result.values())
            log.info("Structured extraction done", section=section, total_rows=total)
            return result
        except Exception as exc:
            log.warning("Structured extraction failed", section=section, error=str(exc))
            return empty

    def extract_semantic(self, text: str, section: str = "") -> list[dict[str, Any]]:
        if len(text.strip()) < 50:
            return []
        try:
            prompt = _SEMANTIC_PROMPT + text
            raw = self._call_llm(prompt, max_tokens=2000)
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

            time.sleep(1)  # brief pause between the two calls per section

            semantic = self.extract_semantic(text, title)
            all_semantic.extend(semantic)

            # Pause every 3 sections to respect rate limits
            if (i + 1) % 3 == 0:
                log.info("Rate limit pause", after_section=i+1)
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
