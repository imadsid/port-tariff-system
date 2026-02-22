"""
ingestion/json_builder.py
JSON Structure Builder
"""
import json
from pathlib import Path
from typing import Any

from config.settings import settings
from monitoring import get_logger

log = get_logger(__name__)


class JSONStructureBuilder:
    """
    Assembles and persists the tariff knowledge base as a structured JSON file.
    """

    def __init__(self) -> None:
        self.store_path = Path(settings.json_store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

    def build(self, raw_rules: list[dict[str, Any]]) -> dict[str, Any]:
        """Build a fresh, normalized JSON store from a list of extracted rule dicts."""
        store: dict[str, Any] = {
            "version": "1.0",
            "source": "Transnet National Ports Authority Tariff Book 2024-2025",
            "vat_rate": settings.vat_rate,
            "rules_by_type": {},
            "all_rules": [],
        }

        for rule in raw_rules:
            due_type = rule.get("due_type") or "other"
            store["rules_by_type"].setdefault(due_type, []).append(rule)
            store["all_rules"].append(rule)

        self._save(store)
        log.info(
            "JSON store built",
            total_rules=len(raw_rules),
            due_types=list(store["rules_by_type"].keys()),
        )
        return store

    def load(self) -> dict[str, Any]:
        """Load the store from disk. Raises if not found."""
        if not self.store_path.exists():
            raise FileNotFoundError(
                f"JSON store not found at {self.store_path}. "
                "Run ingestion first: python main.py ingest --pdf <path>"
            )
        with open(self.store_path) as f:
            data: dict[str, Any] = json.load(f)
        log.info("JSON store loaded", rules=len(data.get("all_rules", [])))
        return data

    def upsert_rule(self, rule: dict[str, Any]) -> None:
        """Insert or update a single rule (useful for incremental updates)."""
        store = self.load()
        due_type = rule.get("due_type") or "other"
        bucket = store["rules_by_type"].setdefault(due_type, [])

        for i, existing in enumerate(bucket):
            if existing.get("section_id") == rule.get("section_id"):
                bucket[i] = rule
                log.info("Rule updated", section_id=rule.get("section_id"))
                self._save(store)
                return

        bucket.append(rule)
        store["all_rules"].append(rule)
        self._save(store)
        log.info("Rule inserted", section_id=rule.get("section_id"))

    def _save(self, store: dict[str, Any]) -> None:
        with open(self.store_path, "w") as f:
            json.dump(store, f, indent=2)
        log.info("JSON store saved", path=str(self.store_path))
