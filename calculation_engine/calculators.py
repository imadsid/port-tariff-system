"""
calculation_engine/calculators.py

All rate constants are sourced from Tariff Book and verified
against the SUDESTADA reference vessel ground-truth values.
"""
import math
from dataclasses import dataclass, field
from typing import Optional

from knowledge_base.json_store import JSONStore
from monitoring import get_logger
from query_processor.models import VesselQuery

log = get_logger(__name__)


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



class _Base:
    """Shared base: store access, finalisation, and utility methods."""

    def __init__(self, store: JSONStore) -> None:
        self.store = store

    # Finalise: compute VAT and wrap into DueResult
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
        )

    @staticmethod
    def _units_of_100gt(gt: float) -> int:
        """Round up GT to the nearest 100-GT unit."""
        return math.ceil(gt / 100)

    @staticmethod
    def _exempt_result(due_type: str, reason: str, store: JSONStore) -> DueResult:
        vat_rate = store.get_vat_rate()
        return DueResult(
            due_type=due_type,
            base_amount=0, surcharge_amount=0, reduction_amount=0,
            net_amount=0, vat_amount=0, total_with_vat=0,
            formula_applied="Exempt",
            exempted=True, exemption_reason=reason,
        )

    def _is_exempt(self, query: VesselQuery, due_type: str) -> tuple[bool, str]:
        """Return (is_exempt, reason)."""
        flag = (query.vessel_flag or "").lower()
        name = (query.vessel_name or "").lower()
        exempt_kw = ["saps", "sandf", "police", "defence", "defense", "navy", "samsa"]
        if any(k in flag or k in name for k in exempt_kw):
            return True, "Government/SAPS/SANDF vessel"
        if query.vessel_type.lower() in ("pleasure",) and due_type in ("light_dues", "port_dues"):
            return True, "Pleasure vessel at registered port"
        return False, ""

    def _first_extracted_rate(self, due_type: str, port: str) -> Optional[dict]:
        """Return first LLM-extracted rate entry for this due_type + port, if any."""
        rates = self.store.get_rates_for_port(due_type, port)
        return rates[0] if rates else None

    def _owh_surcharge(self, base: float, pct: float = 25.0) -> float:
        return base * (pct / 100)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LIGHT DUES  (Section 1.1)
# ─────────────────────────────────────────────────────────────────────────────
class LightDuesCalculator(_Base):
    """
    Section 1.1 — Light Dues
    """
    DUE_TYPE = "light_dues"
    RATE_PER_100GT: float = 117.08   # All other vessels (visiting)

    def calculate(self, query: VesselQuery) -> DueResult:
        exempt, reason = self._is_exempt(query, self.DUE_TYPE)
        if exempt:
            return self._exempt_result(self.DUE_TYPE, reason, self.store)

        gt = query.gross_tonnage
        units = self._units_of_100gt(gt)

        # Prefer LLM-extracted rate
        rate = self.RATE_PER_100GT
        for rule in self.store.get_rules_by_type(self.DUE_TYPE):
            for r in rule.get("rates", []):
                extracted = r.get("rate_per_unit")
                if extracted and "100" in (r.get("unit") or ""):
                    rate = float(extracted)
                    break

        base = units * rate
        breakdown = [{"item": f"Light dues: {units} × 100GT units × R{rate:.2f}", "amount": base}]
        formula = f"ceil({gt}/100) = {units} units × R{rate}/100GT"
        return self._finalise(self.DUE_TYPE, base, 0.0, 0.0, breakdown, formula)


# ─────────────────────────────────────────────────────────────────────────────
# 2. VTS DUES  (Section 2.1)
# ─────────────────────────────────────────────────────────────────────────────
class VTSDuesCalculator(_Base):
    """
    Section 2.1 — Vessel Traffic Services (VTS) Charges
    Durban & Saldanha: R0.65/GT | All other ports: R0.54/GT
    Minimum fee: R235.52
    """
    DUE_TYPE = "vts_dues"
    RATE_DURBAN_SALDANHA: float = 0.65
    RATE_STANDARD: float = 0.54
    MINIMUM: float = 235.52

    def calculate(self, query: VesselQuery) -> DueResult:
        exempt, reason = self._is_exempt(query, self.DUE_TYPE)
        if exempt:
            return self._exempt_result(self.DUE_TYPE, reason, self.store)

        port_l = query.port.lower()
        gt = query.gross_tonnage

        # Port-specific rate
        rate = self.RATE_DURBAN_SALDANHA if port_l in ("durban", "saldanha") else self.RATE_STANDARD

        # LLM override
        for rule in self.store.get_rules_by_type(self.DUE_TYPE):
            for r in rule.get("rates", []):
                rp = (r.get("port") or "ALL").lower()
                if (port_l in rp or rp in port_l or rp == "all") and r.get("rate_per_unit"):
                    rate = float(r["rate_per_unit"])
                    break

        raw = gt * rate
        base = max(raw, self.MINIMUM)
        if raw < self.MINIMUM:
            bd = [{"item": f"VTS minimum fee applied (raw={raw:.2f})", "amount": base}]
        else:
            bd = [{"item": f"VTS: {gt:,.0f} GT × R{rate}/GT", "amount": base}]

        formula = f"{gt} GT × R{rate}/GT (port: {query.port}), min R{self.MINIMUM}"
        return self._finalise(self.DUE_TYPE, base, 0.0, 0.0, bd, formula)


# ─────────────────────────────────────────────────────────────────────────────
# 3. PORT DUES  (Section 4.1.1)
# ─────────────────────────────────────────────────────────────────────────────
class PortDuesCalculator(_Base):
    """
    Section 4.1.1 — Port Dues
    """
    DUE_TYPE = "port_dues"
    BASE_PER_100GT: float = 192.73
    INCR_PER_100GT_PER_24H: float = 57.79
    REDUCTION_COASTER: float = 0.35
    REDUCTION_SHORT_STAY: float = 0.15   # < 12 hours
    REDUCTION_DOUBLE_HULL: float = 0.10

    def calculate(self, query: VesselQuery) -> DueResult:
        exempt, reason = self._is_exempt(query, self.DUE_TYPE)
        if exempt:
            return self._exempt_result(self.DUE_TYPE, reason, self.store)

        gt = query.gross_tonnage
        units = self._units_of_100gt(gt)
        base_rate = self.BASE_PER_100GT
        incr_rate = self.INCR_PER_100GT_PER_24H

        # LLM override
        for rule in self.store.get_rules_by_type(self.DUE_TYPE):
            for r in rule.get("rates", []):
                if r.get("base_fee"):
                    extracted = r.get("rate_per_unit")
                    if extracted:
                        base_rate = float(extracted)
                        break

        basic = units * base_rate
        incr  = units * incr_rate * query.days_in_port
        base  = basic + incr

        bd = [
            {"item": f"Basic: {units} units × R{base_rate:.2f}/100GT", "amount": basic},
            {"item": f"Incremental: {units} units × R{incr_rate:.2f} × {query.days_in_port:.2f}d", "amount": incr},
        ]

        # Reductions
        total_red = 0.0
        if query.is_coaster:
            red = base * self.REDUCTION_COASTER
            total_red += red
            bd.append({"item": "Coaster 35% reduction", "amount": -red})
        if query.days_in_port < 0.5:   # < 12 hours
            red = base * self.REDUCTION_SHORT_STAY
            total_red += red
            bd.append({"item": "< 12h stay 15% reduction", "amount": -red})
        if query.is_double_hull_tanker:
            red = base * self.REDUCTION_DOUBLE_HULL
            total_red += red
            bd.append({"item": "Double-hull tanker 10% reduction", "amount": -red})

        # Surcharge: vessel idle > 30 days
        surcharge = 0.0
        if query.days_in_port > 30 and not query.activity:
            sc = incr * 0.20
            surcharge += sc
            bd.append({"item": "20% surcharge (> 30 days, no cargo)", "amount": sc})

        formula = (
            f"(ceil({gt}/100)×R{base_rate}) + "
            f"(ceil({gt}/100)×R{incr_rate}×{query.days_in_port:.2f}d)"
        )
        return self._finalise(self.DUE_TYPE, base, surcharge, total_red, bd, formula)


# ─────────────────────────────────────────────────────────────────────────────
# 4. TOWAGE DUES  (Section 3.6)
# ─────────────────────────────────────────────────────────────────────────────
class TowageDuesCalculator(_Base):
    """
    Section 3.6 — Tug / Vessel Assistance
    Port-specific tiered schedule: base_fee + ceil((GT – tier_min)/100) × rate_per_unit.
        """
    DUE_TYPE = "towage_dues"

    # Fallback tiers per port — sourced directly from TNPA Tariff Book 2024-2025
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

        gt       = query.gross_tonnage
        port_l   = query.port.lower()
        num_ops  = max(query.num_tug_operations, 1)
        bd: list[dict] = []

        tiers = self._get_tiers(port_l)
        tier  = self._find_tier(gt, tiers)
        if not tier:
            return self._finalise(
                self.DUE_TYPE, 0.0, 0.0, 0.0, [],
                formula="No tier found",
                confidence=0.3,
                warnings=["No towage tier could be determined for this GT/port combination"],
            )

        base_fee       = float(tier.get("base_fee") or 0)
        rate_per_unit  = float(tier.get("rate_per_unit") or 0)
        tier_min       = float(tier.get("tier_min") or 0)
        tier_max_str   = f"{tier.get('tier_max'):,}" if tier.get("tier_max") else "∞"

        above = max(0.0, gt - tier_min)
        incr  = math.ceil(above / 100) * rate_per_unit
        per_service = base_fee + incr

        bd.append({"item": f"Tier [{tier_min:,.0f}–{tier_max_str}] GT: base R{base_fee:,.2f}", "amount": base_fee})
        if incr > 0:
            bd.append({"item": f"Incr: ceil(({gt:,.0f}-{tier_min:,.0f})/100)×R{rate_per_unit:.2f}=R{incr:,.2f}", "amount": incr})

        total_base = per_service * num_ops
        if num_ops > 1:
            bd.append({"item": f"× {num_ops} tug operations", "amount": total_base})

        surcharge = 0.0
        if query.outside_working_hours:
            sc = self._owh_surcharge(total_base, 25.0)
            surcharge = sc
            bd.append({"item": "25% OWH surcharge", "amount": sc})

        formula = (
            f"Port={query.port}, GT={gt:,.0f}, "
            f"base=R{base_fee:,.2f}+incr=R{incr:,.2f}, ×{num_ops} ops"
        )
        return self._finalise(self.DUE_TYPE, total_base, surcharge, 0.0, bd, formula)

    def _get_tiers(self, port_l: str) -> list[dict]:
        # Try LLM-extracted first
        extracted = self.store.get_rates_for_port(self.DUE_TYPE, port_l)
        if extracted and any(r.get("base_fee") for r in extracted):
            tiers = [
                {
                    "tier_min": r.get("tier_min") or 0,
                    "tier_max": r.get("tier_max"),
                    "base_fee": float(r.get("base_fee") or 0),
                    "rate_per_unit": float(r.get("rate_per_unit") or 0),
                }
                for r in extracted
            ]
            return sorted(tiers, key=lambda t: t["tier_min"])
        # Hardcoded fallback
        for key, tiers in self._TIERS.items():
            if key in port_l or port_l in key:
                return tiers
        return self._TIERS["durban"]   # Default

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
    Per service: basic_fee + ceil(GT/100) × rate_per_100GT.
    × number_of_operations (min 2: inward + outward).
    50% surcharge for outside ordinary working hours.
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

        port_l   = query.port.lower()
        gt       = query.gross_tonnage
        units    = self._units_of_100gt(gt)
        num_ops  = max(query.num_tug_operations, 2)   # min: entry + departure
        bd: list[dict] = []

        rate = self._get_rate(port_l)
        basic      = rate["basic"]
        per_100gt  = rate["per_100gt"]

        per_service = basic + units * per_100gt
        bd.append({"item": f"Basic pilotage ({query.port}): R{basic:,.2f}", "amount": basic})
        bd.append({"item": f"GT: {units} × R{per_100gt:.2f}/100GT = R{units * per_100gt:,.2f}", "amount": units * per_100gt})

        total_base = per_service * num_ops
        if num_ops > 1:
            bd.append({"item": f"× {num_ops} pilotage operations", "amount": total_base})

        surcharge = 0.0
        if query.outside_working_hours:
            sc = self._owh_surcharge(total_base, 50.0)
            surcharge = sc
            bd.append({"item": "50% OWH surcharge", "amount": sc})

        formula = f"(R{basic:,.2f} + {units}×R{per_100gt:.2f}) × {num_ops} ops [{query.port}]"
        return self._finalise(self.DUE_TYPE, total_base, surcharge, 0.0, bd, formula)

    def _get_rate(self, port_l: str) -> dict[str, float]:
        # Try LLM-extracted
        extracted = self.store.get_rates_for_port(self.DUE_TYPE, port_l)
        for r in extracted:
            if r.get("base_fee") and r.get("rate_per_unit"):
                return {"basic": float(r["base_fee"]), "per_100gt": float(r["rate_per_unit"])}
        # Hardcoded fallback
        for key, rates in self._RATES.items():
            if key in port_l or port_l in key:
                return rates
        return self._DEFAULT


# ─────────────────────────────────────────────────────────────────────────────
# 6. BERTHING SERVICES / RUNNING LINES  (Section 3.8)
# ─────────────────────────────────────────────────────────────────────────────
class RunningLinesDuesCalculator(_Base):
    """
    Section 3.8 — Berthing Services
    (Labelled "Running Lines Dues" in the take-home test expected outputs)
    Per service: basic_fee + ceil(GT/100) × rate_per_100GT.
    × number_of_operations (min 2: berthing + unberthing).
    """
    DUE_TYPE = "running_lines_dues"

    _RATES: dict[str, dict[str, float]] = {
        "richards bay":   {"basic": 3175.89, "per_100gt": 13.46},
        "port elizabeth": {"basic": 3838.62, "per_100gt": 18.72},
        "ngqura":         {"basic": 3838.62, "per_100gt": 18.72},
        "cape town":      {"basic": 3052.33, "per_100gt": 14.92},
        "saldanha":       {"basic": 4006.34, "per_100gt": 16.97},
    }
    _DEFAULT = {"basic": 2801.91, "per_100gt": 13.68}   # Durban + all other ports

    def calculate(self, query: VesselQuery) -> DueResult:
        port_l  = query.port.lower()
        gt      = query.gross_tonnage
        units   = self._units_of_100gt(gt)
        num_ops = max(query.num_tug_operations, 2)
        bd: list[dict] = []

        rate = self._get_rate(port_l)
        basic     = rate["basic"]
        per_100gt = rate["per_100gt"]

        per_service = basic + units * per_100gt
        bd.append({"item": f"Berthing basic ({query.port}): R{basic:,.2f}", "amount": basic})
        bd.append({"item": f"GT: {units} × R{per_100gt:.2f}/100GT = R{units * per_100gt:,.2f}", "amount": units * per_100gt})

        total_base = per_service * num_ops
        if num_ops > 1:
            bd.append({"item": f"× {num_ops} berthing operations", "amount": total_base})

        surcharge = 0.0
        if query.outside_working_hours:
            sc = self._owh_surcharge(total_base, 50.0)
            surcharge = sc
            bd.append({"item": "50% OWH surcharge", "amount": sc})

        formula = f"(R{basic:,.2f} + {units}×R{per_100gt:.2f}) × {num_ops} ops [Sec. 3.8, {query.port}]"
        return self._finalise(self.DUE_TYPE, total_base, surcharge, 0.0, bd, formula)

    def _get_rate(self, port_l: str) -> dict[str, float]:
        extracted = self.store.get_rates_for_port(self.DUE_TYPE, port_l)
        for r in extracted:
            if r.get("base_fee") and r.get("rate_per_unit"):
                return {"basic": float(r["base_fee"]), "per_100gt": float(r["rate_per_unit"])}
        for key, rates in self._RATES.items():
            if key in port_l or port_l in key:
                return rates
        return self._DEFAULT
