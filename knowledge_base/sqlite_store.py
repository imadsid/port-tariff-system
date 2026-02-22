"""
knowledge_base/sqlite_store.py
Hybrid Knowledge Base — SQLite layer.
"""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from config.settings import settings
from monitoring import get_logger

log = get_logger(__name__)

DB_PATH = Path(settings.sqlite_db_path)


SCHEMA = """
-- ─────────────────────────────────────────────────────────────────
-- rates: base rate per 100GT (or per GT, per call, etc.)
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS rates (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    due_type    TEXT NOT NULL,        -- light_dues, port_dues, vts_dues, ...
    port        TEXT DEFAULT 'ALL',   -- Durban | Cape Town | ALL
    vessel_type TEXT DEFAULT 'ALL',   -- Bulk Carrier | ALL
    rate        REAL NOT NULL,        -- numeric rate value
    unit        TEXT NOT NULL,        -- per_100gt | per_gt | per_call | flat
    section     TEXT,                 -- e.g. "1.1"
    notes       TEXT,
    ingested_at TEXT DEFAULT (datetime('now'))
);

-- ─────────────────────────────────────────────────────────────────
-- tiers: GT bracket tiers (used by towage, pilotage, running lines)
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tiers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    due_type    TEXT NOT NULL,
    port        TEXT DEFAULT 'ALL',
    gt_min      REAL NOT NULL,        -- lower bound of GT bracket (inclusive)
    gt_max      REAL,                 -- upper bound (NULL = no upper limit)
    base_fee    REAL NOT NULL,        -- fixed fee for this bracket
    rate_per_unit REAL DEFAULT 0,     -- rate per 100GT above gt_min
    section     TEXT,
    notes       TEXT,
    ingested_at TEXT DEFAULT (datetime('now'))
);

-- ─────────────────────────────────────────────────────────────────
-- surcharges: percentage additions (e.g. OWH, weekend)
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS surcharges (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    due_type    TEXT NOT NULL,
    name        TEXT NOT NULL,        -- e.g. "outside_working_hours"
    pct         REAL NOT NULL,        -- e.g. 25.0 means 25%
    applies_to  TEXT DEFAULT 'ALL',   -- ALL | specific port
    condition   TEXT,                 -- plain-text condition description
    section     TEXT,
    ingested_at TEXT DEFAULT (datetime('now'))
);

-- ─────────────────────────────────────────────────────────────────
-- reductions: percentage discounts
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS reductions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    due_type    TEXT NOT NULL,
    name        TEXT NOT NULL,        -- e.g. "coaster", "double_hull_tanker"
    pct         REAL NOT NULL,        -- e.g. 35.0 means 35% discount
    applies_to  TEXT DEFAULT 'ALL',
    condition   TEXT,
    section     TEXT,
    ingested_at TEXT DEFAULT (datetime('now'))
);

-- ─────────────────────────────────────────────────────────────────
-- minimums: minimum charge thresholds
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS minimums (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    due_type    TEXT NOT NULL,
    port        TEXT DEFAULT 'ALL',
    amount      REAL NOT NULL,
    condition   TEXT,
    section     TEXT,
    ingested_at TEXT DEFAULT (datetime('now'))
);

-- ─────────────────────────────────────────────────────────────────
-- ingestion_log: tracks what has been ingested
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ingestion_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    pdf_path    TEXT NOT NULL,
    pdf_hash    TEXT,
    ingested_at TEXT DEFAULT (datetime('now')),
    rows_inserted INTEGER DEFAULT 0,
    status      TEXT DEFAULT 'success'
);
"""


class SQLiteStore:
    """
    Read/write interface to the SQLite tariff database.
    Used by:
      - IngestionPipeline (write): populates tables from LLM extraction
      - Calculators (read):        queries rates, tiers, surcharges
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        log.info("SQLite store ready", path=str(self.db_path))

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Write (ingestion) ─────────────────────────────────────────────────────

    def clear_all(self) -> None:
        """Wipe all tariff data (used before re-ingestion)."""
        tables = ["rates", "tiers", "surcharges", "reductions", "minimums"]
        with self._conn() as conn:
            for t in tables:
                conn.execute(f"DELETE FROM {t}")
        log.info("SQLite store cleared")

    def insert_rates(self, rows: list[dict]) -> int:
        sql = """INSERT INTO rates (due_type, port, vessel_type, rate, unit, section, notes)
                 VALUES (:due_type, :port, :vessel_type, :rate, :unit, :section, :notes)"""
        with self._conn() as conn:
            conn.executemany(sql, rows)
        return len(rows)

    def insert_tiers(self, rows: list[dict]) -> int:
        sql = """INSERT INTO tiers (due_type, port, gt_min, gt_max, base_fee, rate_per_unit, section, notes)
                 VALUES (:due_type, :port, :gt_min, :gt_max, :base_fee, :rate_per_unit, :section, :notes)"""
        with self._conn() as conn:
            conn.executemany(sql, rows)
        return len(rows)

    def insert_surcharges(self, rows: list[dict]) -> int:
        sql = """INSERT INTO surcharges (due_type, name, pct, applies_to, condition, section)
                 VALUES (:due_type, :name, :pct, :applies_to, :condition, :section)"""
        with self._conn() as conn:
            conn.executemany(sql, rows)
        return len(rows)

    def insert_reductions(self, rows: list[dict]) -> int:
        sql = """INSERT INTO reductions (due_type, name, pct, applies_to, condition, section)
                 VALUES (:due_type, :name, :pct, :applies_to, :condition, :section)"""
        with self._conn() as conn:
            conn.executemany(sql, rows)
        return len(rows)

    def insert_minimums(self, rows: list[dict]) -> int:
        sql = """INSERT INTO minimums (due_type, port, amount, condition, section)
                 VALUES (:due_type, :port, :amount, :condition, :section)"""
        with self._conn() as conn:
            conn.executemany(sql, rows)
        return len(rows)

    def log_ingestion(self, pdf_path: str, pdf_hash: str, rows_inserted: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO ingestion_log (pdf_path, pdf_hash, rows_inserted) VALUES (?, ?, ?)",
                (pdf_path, pdf_hash, rows_inserted),
            )

    # ── Read (calculators) ────────────────────────────────────────────────────

    def get_rate(self, due_type: str, port: str = "ALL", unit: str = "per_100gt") -> Optional[float]:
        """Get the best-matching rate for a due type and port."""
        with self._conn() as conn:
            # Try exact port match first, fall back to ALL
            row = conn.execute("""
                SELECT rate FROM rates
                WHERE due_type = ?
                  AND unit = ?
                  AND (port = ? OR port = 'ALL')
                ORDER BY CASE WHEN port = ? THEN 0 ELSE 1 END
                LIMIT 1
            """, (due_type, unit, port, port)).fetchone()
            return float(row["rate"]) if row else None

    def get_tiers(self, due_type: str, port: str = "ALL") -> list[dict]:
        """Get GT tiers for towage/pilotage/running lines, ordered by gt_min."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM tiers
                WHERE due_type = ?
                  AND (port = ? OR port = 'ALL')
                ORDER BY CASE WHEN port = ? THEN 0 ELSE 1 END, gt_min
            """, (due_type, port, port)).fetchall()
            return [dict(r) for r in rows]

    def get_surcharges(self, due_type: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM surcharges WHERE due_type = ? OR due_type = 'ALL'",
                (due_type,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_reductions(self, due_type: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM reductions WHERE due_type = ? OR due_type = 'ALL'",
                (due_type,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_minimum(self, due_type: str, port: str = "ALL") -> Optional[float]:
        with self._conn() as conn:
            row = conn.execute("""
                SELECT amount FROM minimums
                WHERE due_type = ?
                  AND (port = ? OR port = 'ALL')
                ORDER BY CASE WHEN port = ? THEN 0 ELSE 1 END
                LIMIT 1
            """, (due_type, port, port)).fetchone()
            return float(row["amount"]) if row else None

    def stats(self) -> dict:
        """Return row counts per table."""
        tables = ["rates", "tiers", "surcharges", "reductions", "minimums"]
        result = {}
        with self._conn() as conn:
            for t in tables:
                result[t] = conn.execute(f"SELECT COUNT(*) as n FROM {t}").fetchone()["n"]
        return result

    def count(self) -> int:
        s = self.stats()
        return sum(s.values())
