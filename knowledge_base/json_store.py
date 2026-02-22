"""
knowledge_base/json_store.py
Structured JSON Store
"""
import json
from pathlib import Path
from typing import Any, Optional

from config.settings import settings
from monitoring import get_logger

log = get_logger(__name__)


class JSONStore:
    """
    Read-optimised interface to the tariff JSON knowledge base.
    Lazy-loads on first access and caches in memory.
    """

    def __init__(self) -> None:
        self._store: Optional[dict[str, Any]] = None

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def store(self) -> dict[str, Any]:
        if self._store is None:
            self._store = self._load()
        return self._store

    def reload(self) -> None:
        """Force reload from disk (call after re-ingestion)."""
        self._store = None
        log.info("JSON store cache cleared — will reload on next access")

    def get_rules_by_type(self, due_type: str) -> list[dict[str, Any]]:
        return self.store.get("rules_by_type", {}).get(due_type, [])

    def get_all_rules(self) -> list[dict[str, Any]]:
        return self.store.get("all_rules", [])

    def get_vat_rate(self) -> float:
        return float(self.store.get("vat_rate", settings.vat_rate))

    def get_rates_for_port(self, due_type: str, port: str) -> list[dict[str, Any]]:
        """Return all rate entries matching a specific port name (or 'ALL')."""
        port_lower = port.lower()
        matched: list[dict[str, Any]] = []
        for rule in self.get_rules_by_type(due_type):
            for rate in rule.get("rates", []):
                rate_port = (rate.get("port") or "ALL").lower()
                if rate_port == "all" or port_lower in rate_port or rate_port in port_lower:
                    matched.append({**rate, "_rule_section": rule.get("section_id", "")})
        return matched

    def get_surcharges(self, due_type: str) -> list[dict[str, Any]]:
        return [sc for rule in self.get_rules_by_type(due_type) for sc in rule.get("surcharges", [])]

    def get_reductions(self, due_type: str) -> list[dict[str, Any]]:
        return [rd for rule in self.get_rules_by_type(due_type) for rd in rule.get("reductions", [])]

    def get_exemptions(self, due_type: str) -> list[str]:
        return [ex for rule in self.get_rules_by_type(due_type) for ex in rule.get("exemptions", [])]

    # ── Private ───────────────────────────────────────────────────────────────

    def _load(self) -> dict[str, Any]:
        path = Path(settings.json_store_path)
        if not path.exists():
            log.warning(
                "JSON store not found — using empty store. Run ingestion first.",
                path=str(path),
            )
            return {
                "rules_by_type": {},
                "all_rules": [],
                "vat_rate": settings.vat_rate,
            }
        with open(path) as f:
            data: dict[str, Any] = json.load(f)
        log.info("JSON store loaded", rules=len(data.get("all_rules", [])))
        return data

    def count(self) -> int:
        """Return total number of rules in the store."""
        return len(self.get_all_rules())

