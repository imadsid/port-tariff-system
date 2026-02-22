"""
calculation_engine/calculators.py
Component 4 — Pure-Math Due Type Calculators

Lookup priority for every rate/tier/surcharge:
  1. SQLite (populated by ingestion pipeline from the actual PDF)
  2. Hardcoded constants (TNPA 2024-25 fallback — used if ingestion hasn't run)

This means:
  - Ingest a new PDF → SQLite updates → calculators automatically use new rates
  - No ingestion yet → hardcoded fallback ensures system still works
"""
import math
from dataclasses import dataclass, field
from typing import Optional

from knowledge_base.json_store import JSONStore
from knowledge_base.sqlite_store import SQLiteStore
from monitoring import get_logger
from query_processor.models import VesselQuery

log = get_logger(__name__)

# Shared SQLite store instance (lazy singleton)
_sqlite: Optional[SQLiteStore] = None

def _get_sqlite() -> SQLiteStore:
    global _sqlite
    if _sqlite is None:
        _sqlite = SQLiteStore()
    return _sqlite


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DueResult:
    due_type: str
    base_amount: float
    surcharge_amount: float
    reduction_amount: float
    net_amount: float
    vat_amount: float
    total_with_vat: float
    breakdown: list[dict] = field(default_factory=list)
    formula_applied: str = ""
    confidence: float = 1.0
    warnings: list[str] = field(default_factory=list)
    exempted: bool = False
    exemption_reason: str = ""
    rate_source: str = "hardcoded"   # "sqlite" | "hardcoded"


# ─────────────────────────────────────────────────────────────────────────────
# Shared base class
# ─────────────────────────────────────────────────────────────────────────────
class _Base:
    """Shared base: SQLite-first lookup, finalisation, and utility methods."""

    def __init__(self, store: JSONStore) -> None:
        self.store  = store                # legacy JSON store (VAT rate, exemptions)
        self.sqlite = _get_sqlite()        # NEW: SQLite structured store

    def _finalise(
        self,
        due_type: str,
        base: float,
        surcharges: float,
        reductions: float,
        breakdown: list[dict],
        formula: str = "",
        confidence: float = 1.0,
        warnings: Optional[list[str]] = None,
        exempted: bool = False,
        exemption_reason: str = "",
        rate_source: str = "hardcoded",
    ) -> DueResult:
        vat_rate = self.store.get_vat_rate()
        net = max(0.0, base + surcharges - reductions)
        vat = net * vat_rate
        return DueResult(
            due_type=due_type,
            base_amount=round(base, 2),
            surcharge_amount=round(surcharges, 2),
            reduction_amount=round(reductions, 2),
            net_amount=round(net, 2),
            vat_amount=round(vat, 2),
            total_with_vat=round(net + vat, 2),
            breakdown=breakdown,
            formula_applied=formula,
            confidence=confidence,
            warnings=warnings or [],
            exempted=exempted,
            exemption_reason=exemption_reason,
            rate_source=rate_source,
        )

    @staticmethod
    def _units_of_100gt(gt: float) -> int:
        return math.ceil(gt / 100)

    @staticmethod
    def _exempt_result(due_type: str, reason: str, store: JSONStore) -> DueResult:
        return DueResult(
            due_type=due_type,
            base_amount=0, surcharge_amount=0, reduction_amount=0,
            net_amount=0, vat_amount=0, total_with_vat=0,
            formula_applied="Exempt",
            exempted=True, exemption_reason=reason,
        )

    def _is_exempt(self, query: VesselQuery, due_type: str) -> tuple[bool, str]:
        flag = (query.vessel_flag or "").lower()
        name = (query.vessel_name or "").lower()
        exempt_kw = ["saps", "sandf", "police", "defence", "defense", "navy", "samsa"]
        if any(k in flag or k in name for k in exempt_kw):
            return True, "Government/SAPS/SANDF vessel"
        if query.vessel_type.lower() in ("pleasure",) and due_type in ("light_dues", "port_dues"):
            return True, "Pleasure vessel at registered port"
        return False, ""

    def _owh_surcharge(self, base: float, pct: float = 25.0) -> float:
        return base * (pct / 100)

    def _log_source(self, due_type: str, source: str) -> None:
        log.debug("Rate source", due_type=due_type, source=source)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LIGHT DUES  (Section 1.1)
# ─────────────────────────────────────────────────────────────────────────────
class LightDuesCalculator(_Base):
    """
    Section 1.1 — Light Dues
    R117.08 per 100 GT per port call (hardcoded fallback).
    Verified: ceil(51300/100) × R117.08 = 513 × R117.08 = R60,062.04 ✅
    """
    DUE_TYPE = "light_dues"
    _FALLBACK_RATE: float = 117.08

    def calculate(self, query: VesselQuery) -> DueResult:
        exempt, reason = self._is_exempt(query, self.DUE_TYPE)
        if exempt:
            return self._exempt_result(self.DUE_TYPE, reason, self.store)

        gt    = query.gross_tonnage
        units = self._units_of_100gt(gt)

        # ── SQLite first ──────────────────────────────────────────────────────
        source = "hardcoded"
        rate   = self._FALLBACK_RATE
        sqlite_rate = self.sqlite.get_rate(self.DUE_TYPE, query.port, "per_100gt")
        if sqlite_rate:
            rate   = sqlite_rate
            source = "sqlite"
        else:
            # Try per_gt unit as fallback
            sqlite_rate = self.sqlite.get_rate(self.DUE_TYPE, query.port, "per_gt")
            if sqlite_rate:
                rate   = sqlite_rate * 100   # convert to per_100gt
                source = "sqlite"

        self._log_source(self.DUE_TYPE, source)
        base = units * rate

        # Apply minimum if stored
        minimum = self.sqlite.get_minimum(self.DUE_TYPE, query.port)
        if minimum and base < minimum:
            base = minimum

        bd = [{"item": f"Light dues: {units} × 100GT units × R{rate:.2f}", "amount": base}]
        formula = f"ceil({gt}/100)={units} × R{rate:.2f}/100GT [{source}]"
        return self._finalise(self.DUE_TYPE, base, 0.0, 0.0, bd, formula, rate_source=source)


# ─────────────────────────────────────────────────────────────────────────────
# 2. VTS DUES  (Section 2.1)
# ─────────────────────────────────────────────────────────────────────────────
class VTSDuesCalculator(_Base):
    """
    Section 2.1 — VTS Charges
    Durban/Richards Bay/Saldanha: R0.65/GT | Others: R0.54/GT | Min: R235.52
    Verified: 51300 × R0.65 = R33,345.00 ✅
    """
    DUE_TYPE = "vts_dues"
    _RATE_HIGH: float  = 0.65   # Durban, Richards Bay, Saldanha
    _RATE_STD:  float  = 0.54
    _MINIMUM:   float  = 235.52
    _HIGH_PORTS = ("durban", "richards bay", "saldanha")

    def calculate(self, query: VesselQuery) -> DueResult:
        exempt, reason = self._is_exempt(query, self.DUE_TYPE)
        if exempt:
            return self._exempt_result(self.DUE_TYPE, reason, self.store)

        port_l = query.port.lower()
        gt     = query.gross_tonnage

        # ── SQLite first ──────────────────────────────────────────────────────
        source = "hardcoded"
        rate   = self._RATE_HIGH if any(p in port_l for p in self._HIGH_PORTS) else self._RATE_STD
        sqlite_rate = self.sqlite.get_rate(self.DUE_TYPE, query.port, "per_gt")
        if sqlite_rate:
            rate   = sqlite_rate
            source = "sqlite"

        minimum = self.sqlite.get_minimum(self.DUE_TYPE, query.port) or self._MINIMUM

        raw  = gt * rate
        base = max(raw, minimum)

        bd = (
            [{"item": f"VTS minimum applied (raw={raw:.2f})", "amount": base}]
            if raw < minimum
            else [{"item": f"VTS: {gt:,.0f} GT × R{rate}/GT", "amount": base}]
        )
        formula = f"{gt} GT × R{rate}/GT, min R{minimum} [{source}]"
        return self._finalise(self.DUE_TYPE, base, 0.0, 0.0, bd, formula, rate_source=source)


# ─────────────────────────────────────────────────────────────────────────────
# 3. PORT DUES  (Section 4.1.1)
# ─────────────────────────────────────────────────────────────────────────────
class PortDuesCalculator(_Base):
    """
    Section 4.1.1 — Port Dues
    Basic: ceil(GT/100) × R192.73  +  Incremental: ceil(GT/100) × R57.79 × days
    Verified: 513×192.73 + 513×57.79×3.39 = R199,371 ✅
    """
    DUE_TYPE = "port_dues"
    _BASE_PER_100GT:     float = 192.73
    _INCR_PER_100GT_DAY: float = 57.79
    _RED_COASTER:        float = 0.35
    _RED_SHORT_STAY:     float = 0.15
    _RED_DOUBLE_HULL:    float = 0.10

    def calculate(self, query: VesselQuery) -> DueResult:
        exempt, reason = self._is_exempt(query, self.DUE_TYPE)
        if exempt:
            return self._exempt_result(self.DUE_TYPE, reason, self.store)

        gt    = query.gross_tonnage
        units = self._units_of_100gt(gt)

        # ── SQLite first ──────────────────────────────────────────────────────
        source    = "hardcoded"
        base_rate = self._BASE_PER_100GT
        incr_rate = self._INCR_PER_100GT_DAY

        sqlite_base = self.sqlite.get_rate(self.DUE_TYPE, query.port, "per_100gt")
        if sqlite_base:
            base_rate = sqlite_base
            source    = "sqlite"
            # Try to get incremental rate too
            sqlite_incr = self.sqlite.get_rate(self.DUE_TYPE, query.port, "per_100gt_per_day")
            if sqlite_incr:
                incr_rate = sqlite_incr

        # ── Reductions from SQLite ────────────────────────────────────────────
        red_coaster     = self._RED_COASTER
        red_short_stay  = self._RED_SHORT_STAY
        red_double_hull = self._RED_DOUBLE_HULL
        for r in self.sqlite.get_reductions(self.DUE_TYPE):
            name = (r.get("name") or "").lower()
            pct  = float(r.get("pct") or 0) / 100
            if "coaster" in name and pct:
                red_coaster = pct
            elif "short" in name and pct:
                red_short_stay = pct
            elif "double" in name and pct:
                red_double_hull = pct

        basic = units * base_rate
        incr  = units * incr_rate * query.days_in_port
        base  = basic + incr

        bd = [
            {"item": f"Basic: {units} units × R{base_rate:.2f}/100GT [{source}]", "amount": basic},
            {"item": f"Incremental: {units} × R{incr_rate:.2f} × {query.days_in_port:.2f}d", "amount": incr},
        ]

        # Reductions
        total_red = 0.0
        if query.is_coaster:
            red = base * red_coaster
            total_red += red
            bd.append({"item": f"Coaster {red_coaster*100:.0f}% reduction", "amount": -red})
        if query.days_in_port < 0.5:
            red = base * red_short_stay
            total_red += red
            bd.append({"item": f"Short stay {red_short_stay*100:.0f}% reduction", "amount": -red})
        if query.is_double_hull_tanker:
            red = base * red_double_hull
            total_red += red
            bd.append({"item": f"Double-hull {red_double_hull*100:.0f}% reduction", "amount": -red})

        surcharge = 0.0
        if query.days_in_port > 30 and not query.activity:
            sc = incr * 0.20
            surcharge += sc
            bd.append({"item": "20% surcharge (>30 days, no cargo)", "amount": sc})

        formula = f"({units}×R{base_rate}) + ({units}×R{incr_rate}×{query.days_in_port:.2f}d) [{source}]"
        return self._finalise(self.DUE_TYPE, base, surcharge, total_red, bd, formula, rate_source=source)


# ─────────────────────────────────────────────────────────────────────────────
# 4. TOWAGE DUES  (Section 3.6)
# ─────────────────────────────────────────────────────────────────────────────
class TowageDuesCalculator(_Base):
    """
    Section 3.6 — Tug / Vessel Assistance
    Tiered: base_fee + ceil((GT - tier_min)/100) × rate_per_unit, × num_operations
    Verified (Durban 50001-100000): R73,118.07 + ceil(1299/100)×R32.24 × 2 = R147,074.38 ✅
    """
    DUE_TYPE = "towage_dues"

    # Hardcoded fallback tiers per port
    _TIERS: dict[str, list[dict]] = {
        "durban": [
            {"tier_min": 0,      "tier_max": 2000,   "base_fee": 8140.00,   "rate_per_unit": 0.00},
            {"tier_min": 2001,   "tier_max": 10000,  "base_fee": 12633.99,  "rate_per_unit": 268.99},
            {"tier_min": 10001,  "tier_max": 50000,  "base_fee": 38494.51,  "rate_per_unit": 84.95},
            {"tier_min": 50001,  "tier_max": 100000, "base_fee": 73118.07,  "rate_per_unit": 32.24},
            {"tier_min": 100001, "tier_max": None,   "base_fee": 93548.13,  "rate_per_unit": 23.65},
        ],
        "richards bay": [
            {"tier_min": 0,      "tier_max": 2000,   "base_fee": 7001.67,   "rate_per_unit": 0.00},
            {"tier_min": 2001,   "tier_max": 10000,  "base_fee": 13020.67,  "rate_per_unit": 275.32},
            {"tier_min": 10001,  "tier_max": 50000,  "base_fee": 39999.88,  "rate_per_unit": 101.08},
            {"tier_min": 50001,  "tier_max": 100000, "base_fee": 79999.76,  "rate_per_unit": 30.11},
            {"tier_min": 100001, "tier_max": None,   "base_fee": 103999.70, "rate_per_unit": 21.50},
        ],
        "east london": [
            {"tier_min": 0,      "tier_max": 2000,   "base_fee": 5622.16,   "rate_per_unit": 0.00},
            {"tier_min": 2001,   "tier_max": 10000,  "base_fee": 8152.14,   "rate_per_unit": 200.97},
            {"tier_min": 10001,  "tier_max": 50000,  "base_fee": 27956.91,  "rate_per_unit": 66.67},
            {"tier_min": 50001,  "tier_max": 100000, "base_fee": 55913.82,  "rate_per_unit": 25.80},
            {"tier_min": 100001, "tier_max": None,   "base_fee": 72682.97,  "rate_per_unit": 25.80},
        ],
        "port elizabeth": [
            {"tier_min": 0,      "tier_max": 2000,   "base_fee": 7206.98,   "rate_per_unit": 0.00},
            {"tier_min": 2001,   "tier_max": 10000,  "base_fee": 11168.45,  "rate_per_unit": 237.53},
            {"tier_min": 10001,  "tier_max": 50000,  "base_fee": 32257.98,  "rate_per_unit": 73.10},
            {"tier_min": 50001,  "tier_max": 100000, "base_fee": 64515.95,  "rate_per_unit": 21.50},
            {"tier_min": 100001, "tier_max": None,   "base_fee": 82542.46,  "rate_per_unit": 21.50},
        ],
        "ngqura": [
            {"tier_min": 0,      "tier_max": 2000,   "base_fee": 7206.98,   "rate_per_unit": 0.00},
            {"tier_min": 2001,   "tier_max": 10000,  "base_fee": 11168.45,  "rate_per_unit": 237.53},
            {"tier_min": 10001,  "tier_max": 50000,  "base_fee": 32257.98,  "rate_per_unit": 73.10},
            {"tier_min": 50001,  "tier_max": 100000, "base_fee": 64515.95,  "rate_per_unit": 21.50},
            {"tier_min": 100001, "tier_max": None,   "base_fee": 82542.46,  "rate_per_unit": 21.50},
        ],
        "mossel bay": [
            {"tier_min": 0,      "tier_max": 2000,   "base_fee": 6316.53,   "rate_per_unit": 0.00},
            {"tier_min": 2001,   "tier_max": 10000,  "base_fee": 8152.14,   "rate_per_unit": 173.37},
            {"tier_min": 10001,  "tier_max": 50000,  "base_fee": 25806.37,  "rate_per_unit": 60.21},
            {"tier_min": 50001,  "tier_max": None,   "base_fee": 49959.21,  "rate_per_unit": 23.65},
        ],
        "cape town": [
            {"tier_min": 0,      "tier_max": 2000,   "base_fee": 5411.47,   "rate_per_unit": 0.00},
            {"tier_min": 2001,   "tier_max": 10000,  "base_fee": 7898.57,   "rate_per_unit": 194.63},
            {"tier_min": 10001,  "tier_max": 50000,  "base_fee": 27741.85,  "rate_per_unit": 64.52},
            {"tier_min": 50001,  "tier_max": 100000, "base_fee": 53978.33,  "rate_per_unit": 47.32},
            {"tier_min": 100001, "tier_max": None,   "base_fee": 79569.67,  "rate_per_unit": 38.71},
        ],
        "saldanha": [
            {"tier_min": 0,      "tier_max": 2000,   "base_fee": 9038.42,   "rate_per_unit": 0.00},
            {"tier_min": 2001,   "tier_max": 10000,  "base_fee": 15378.78,  "rate_per_unit": 327.43},
            {"tier_min": 10001,  "tier_max": 50000,  "base_fee": 47311.70,  "rate_per_unit": 103.23},
            {"tier_min": 50001,  "tier_max": 100000, "base_fee": 90322.33,  "rate_per_unit": 27.97},
            {"tier_min": 100001, "tier_max": None,   "base_fee": 111827.63, "rate_per_unit": 47.32},
        ],
    }

    def calculate(self, query: VesselQuery) -> DueResult:
        exempt, reason = self._is_exempt(query, self.DUE_TYPE)
        if exempt:
            return self._exempt_result(self.DUE_TYPE, reason, self.store)

        gt      = query.gross_tonnage
        port_l  = query.port.lower()
        num_ops = max(query.num_tug_operations, 1)
        bd: list[dict] = []

        tiers, source = self._get_tiers(port_l)
        tier = self._find_tier(gt, tiers)
        if not tier:
            return self._finalise(
                self.DUE_TYPE, 0.0, 0.0, 0.0, [],
                formula="No tier found", confidence=0.3,
                warnings=["No towage tier found for this GT/port"],
            )

        base_fee      = float(tier.get("base_fee") or 0)
        rate_per_unit = float(tier.get("rate_per_unit") or 0)
        tier_min      = float(tier.get("tier_min") or tier.get("gt_min") or 0)
        tier_max_str  = f"{tier.get('tier_max') or tier.get('gt_max'):,}" if (tier.get("tier_max") or tier.get("gt_max")) else "∞"

        above       = max(0.0, gt - tier_min)
        incr        = math.ceil(above / 100) * rate_per_unit
        per_service = base_fee + incr

        bd.append({"item": f"Tier [{tier_min:,.0f}–{tier_max_str}] GT: base R{base_fee:,.2f} [{source}]", "amount": base_fee})
        if incr > 0:
            bd.append({"item": f"Incr: ceil(({gt:,.0f}-{tier_min:,.0f})/100)×R{rate_per_unit:.2f}=R{incr:,.2f}", "amount": incr})

        total_base = per_service * num_ops
        if num_ops > 1:
            bd.append({"item": f"× {num_ops} tug operations", "amount": total_base})

        surcharge = 0.0
        if query.outside_working_hours:
            # Check SQLite for OWH pct first
            owh_pct = 25.0
            for sc in self.sqlite.get_surcharges(self.DUE_TYPE):
                if "working" in (sc.get("name") or "").lower() or "owh" in (sc.get("name") or "").lower():
                    owh_pct = float(sc.get("pct") or 25.0)
                    break
            surcharge = self._owh_surcharge(total_base, owh_pct)
            bd.append({"item": f"{owh_pct:.0f}% OWH surcharge", "amount": surcharge})

        formula = f"Port={query.port}, GT={gt:,.0f}, base=R{base_fee:,.2f}+incr=R{incr:,.2f}, ×{num_ops} [{source}]"
        return self._finalise(self.DUE_TYPE, total_base, surcharge, 0.0, bd, formula, rate_source=source)

    def _get_tiers(self, port_l: str) -> tuple[list[dict], str]:
        """Returns (tiers, source) — SQLite first, hardcoded fallback."""
        sqlite_tiers = self.sqlite.get_tiers(self.DUE_TYPE, port_l)
        if sqlite_tiers:
            # Normalise SQLite column names to match hardcoded format
            tiers = [
                {
                    "tier_min":      float(t.get("gt_min") or 0),
                    "tier_max":      t.get("gt_max"),
                    "base_fee":      float(t.get("base_fee") or 0),
                    "rate_per_unit": float(t.get("rate_per_unit") or 0),
                }
                for t in sqlite_tiers
            ]
            return tiers, "sqlite"
        # Hardcoded fallback
        for key, tiers in self._TIERS.items():
            if key in port_l or port_l in key:
                return tiers, "hardcoded"
        return self._TIERS["durban"], "hardcoded"

    @staticmethod
    def _find_tier(gt: float, tiers: list[dict]) -> Optional[dict]:
        for tier in sorted(tiers, key=lambda t: t.get("tier_min") or 0):
            lo = tier.get("tier_min") or 0
            hi = tier.get("tier_max")
            if gt >= lo and (hi is None or gt <= hi):
                return tier
        return tiers[-1] if tiers else None


# ─────────────────────────────────────────────────────────────────────────────
# 5. PILOTAGE DUES  (Section 3.3)
# ─────────────────────────────────────────────────────────────────────────────
class PilotageDuesCalculator(_Base):
    """
    Section 3.3 — Pilotage Services
    Per service: basic_fee + ceil(GT/100) × rate_per_100GT, × num_operations (min 2)
    Verified (Durban): (R18,608.61 + 513×R9.72) × 2 = R47,189.94 ✅
    """
    DUE_TYPE = "pilotage_dues"

    _RATES: dict[str, dict[str, float]] = {
        "durban":         {"basic": 18608.61, "per_100gt":  9.72},
        "richards bay":   {"basic": 30960.46, "per_100gt": 10.93},
        "port elizabeth": {"basic":  8970.00, "per_100gt": 14.33},
        "ngqura":         {"basic":  8970.00, "per_100gt": 14.33},
        "cape town":      {"basic":  6342.39, "per_100gt": 10.20},
        "saldanha":       {"basic":  9673.57, "per_100gt": 13.66},
        "east london":    {"basic":  6547.45, "per_100gt": 10.49},
        "mossel bay":     {"basic":  6547.45, "per_100gt": 10.49},
    }
    _DEFAULT = {"basic": 6547.45, "per_100gt": 10.49}

    def calculate(self, query: VesselQuery) -> DueResult:
        exempt, reason = self._is_exempt(query, self.DUE_TYPE)
        if exempt:
            return self._exempt_result(self.DUE_TYPE, reason, self.store)

        port_l  = query.port.lower()
        gt      = query.gross_tonnage
        units   = self._units_of_100gt(gt)
        num_ops = max(query.num_tug_operations, 2)
        bd: list[dict] = []

        rate, source = self._get_rate(port_l)
        basic     = rate["basic"]
        per_100gt = rate["per_100gt"]

        per_service = basic + units * per_100gt
        bd.append({"item": f"Basic pilotage ({query.port}) [{source}]: R{basic:,.2f}", "amount": basic})
        bd.append({"item": f"GT: {units} × R{per_100gt:.2f}/100GT = R{units*per_100gt:,.2f}", "amount": units * per_100gt})

        total_base = per_service * num_ops
        if num_ops > 1:
            bd.append({"item": f"× {num_ops} pilotage operations", "amount": total_base})

        surcharge = 0.0
        if query.outside_working_hours:
            owh_pct = 50.0
            for sc in self.sqlite.get_surcharges(self.DUE_TYPE):
                if "working" in (sc.get("name") or "").lower():
                    owh_pct = float(sc.get("pct") or 50.0)
                    break
            surcharge = self._owh_surcharge(total_base, owh_pct)
            bd.append({"item": f"{owh_pct:.0f}% OWH surcharge", "amount": surcharge})

        formula = f"(R{basic:,.2f} + {units}×R{per_100gt:.2f}) × {num_ops} [{source}]"
        return self._finalise(self.DUE_TYPE, total_base, surcharge, 0.0, bd, formula, rate_source=source)

    def _get_rate(self, port_l: str) -> tuple[dict[str, float], str]:
        """Returns (rate_dict, source)."""
        # SQLite tiers first
        sqlite_tiers = self.sqlite.get_tiers(self.DUE_TYPE, port_l)
        if sqlite_tiers:
            t = sqlite_tiers[0]
            return {"basic": float(t.get("base_fee") or 0), "per_100gt": float(t.get("rate_per_unit") or 0)}, "sqlite"
        # Hardcoded fallback
        for key, rates in self._RATES.items():
            if key in port_l or port_l in key:
                return rates, "hardcoded"
        return self._DEFAULT, "hardcoded"


# ─────────────────────────────────────────────────────────────────────────────
# 6. RUNNING LINES DUES  (Section 3.8)
# ─────────────────────────────────────────────────────────────────────────────
class RunningLinesDuesCalculator(_Base):
    """
    Section 3.8 — Berthing Services (Running Lines)
    Per service: basic_fee + ceil(GT/100) × rate_per_100GT, × num_operations (min 2)
    Verified (Durban): (R2,801.91 + 513×R13.68) × 2 = R19,639.50 ✅
    """
    DUE_TYPE = "running_lines_dues"

    _RATES: dict[str, dict[str, float]] = {
        "richards bay":   {"basic": 3175.89, "per_100gt": 13.46},
        "port elizabeth": {"basic": 3838.62, "per_100gt": 18.72},
        "ngqura":         {"basic": 3838.62, "per_100gt": 18.72},
        "cape town":      {"basic": 3052.33, "per_100gt": 14.92},
        "saldanha":       {"basic": 4006.34, "per_100gt": 16.97},
    }
    _DEFAULT = {"basic": 2801.91, "per_100gt": 13.68}

    def calculate(self, query: VesselQuery) -> DueResult:
        port_l  = query.port.lower()
        gt      = query.gross_tonnage
        units   = self._units_of_100gt(gt)
        num_ops = max(query.num_tug_operations, 2)
        bd: list[dict] = []

        rate, source = self._get_rate(port_l)
        basic     = rate["basic"]
        per_100gt = rate["per_100gt"]

        per_service = basic + units * per_100gt
        bd.append({"item": f"Berthing basic ({query.port}) [{source}]: R{basic:,.2f}", "amount": basic})
        bd.append({"item": f"GT: {units} × R{per_100gt:.2f}/100GT = R{units*per_100gt:,.2f}", "amount": units * per_100gt})

        total_base = per_service * num_ops
        if num_ops > 1:
            bd.append({"item": f"× {num_ops} berthing operations", "amount": total_base})

        surcharge = 0.0
        if query.outside_working_hours:
            owh_pct = 50.0
            for sc in self.sqlite.get_surcharges(self.DUE_TYPE):
                if "working" in (sc.get("name") or "").lower():
                    owh_pct = float(sc.get("pct") or 50.0)
                    break
            surcharge = self._owh_surcharge(total_base, owh_pct)
            bd.append({"item": f"{owh_pct:.0f}% OWH surcharge", "amount": surcharge})

        formula = f"(R{basic:,.2f} + {units}×R{per_100gt:.2f}) × {num_ops} [{source}]"
        return self._finalise(self.DUE_TYPE, total_base, surcharge, 0.0, bd, formula, rate_source=source)

    def _get_rate(self, port_l: str) -> tuple[dict[str, float], str]:
        """Returns (rate_dict, source)."""
        sqlite_tiers = self.sqlite.get_tiers(self.DUE_TYPE, port_l)
        if sqlite_tiers:
            t = sqlite_tiers[0]
            return {"basic": float(t.get("base_fee") or 0), "per_100gt": float(t.get("rate_per_unit") or 0)}, "sqlite"
        for key, rates in self._RATES.items():
            if key in port_l or port_l in key:
                return rates, "hardcoded"
        return self._DEFAULT, "hardcoded"
