"""
guardrails/guardrail_layer.py
Component 5 — Guardrail Layer
Multi-stage quality and safety checks:
  1. InputValidator        — validates VesselQuery before calculation
  2. HallucinationDetector — checks amounts against known-reasonable ranges
  3. ConfidenceScorer      — computes holistic confidence score
  4. BusinessRulesEnforcer — enforces TNPA-specific domain rules
  5. OutputValidator       — final sanity check on totals
"""
from dataclasses import dataclass, field
from typing import Any

from calculation_engine.engine import CalculationResult
from monitoring import get_logger
from query_processor.models import VesselQuery

log = get_logger(__name__)

# Reasonable net-amount ranges (ZAR excl. VAT) for a commercial vessel port call
_SANITY: dict[str, tuple[float, float]] = {
    "light_dues":        (1_000,   500_000),
    "vts_dues":          (235,     300_000),
    "port_dues":         (5_000, 2_000_000),
    "towage_dues":       (5_000, 1_500_000),
    "pilotage_dues":     (3_000,   800_000),
    "running_lines_dues":(500,      80_000),
}

_VALID_PORTS = {
    "durban", "cape town", "richards bay", "port elizabeth",
    "ngqura", "east london", "saldanha", "mossel bay",
}


@dataclass
class ValidationReport:
    passed: bool
    confidence_score: float
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    hallucination_flags: list[str] = field(default_factory=list)
    business_violations: list[str] = field(default_factory=list)


# ── 1. Input Validator ────────────────────────────────────────────────────────

class InputValidator:

    def validate(self, query: VesselQuery) -> ValidationReport:
        issues: list[str] = []
        warnings: list[str] = []
        score = 1.0

        if query.gross_tonnage <= 0:
            issues.append("gross_tonnage must be > 0")
            score -= 0.5
        elif query.gross_tonnage > 600_000:
            warnings.append(f"GT {query.gross_tonnage:,.0f} is unusually large — please verify")
            score -= 0.1

        if not query.port:
            issues.append("port is required")
            score -= 0.3
        elif query.port.lower() not in _VALID_PORTS:
            # Unknown port is a hard failure — not a warning.
            # Calculating for an unknown port would silently use wrong rates.
            valid_list = ", ".join(sorted(p.title() for p in _VALID_PORTS))
            issues.append(
                f"Port '{query.port}' is not a recognised TNPA port. "
                f"Valid ports are: {valid_list}."
            )
            score -= 0.5

        if query.days_in_port < 0:
            issues.append("days_in_port cannot be negative")
            score -= 0.2

        if query.loa_meters < 0:
            issues.append("loa_meters cannot be negative")

        return ValidationReport(
            passed=len(issues) == 0,
            confidence_score=max(0.0, score),
            issues=issues,
            warnings=warnings,
        )


# ── 2. Hallucination Detector ─────────────────────────────────────────────────

class HallucinationDetector:

    def detect(self, result: CalculationResult) -> ValidationReport:
        flags: list[str] = []
        warnings: list[str] = []
        score = 1.0

        for due_type, dr in result.dues.items():
            if dr.exempted:
                continue
            lo, hi = _SANITY.get(due_type, (0, 1e9))
            if dr.net_amount < lo:
                flags.append(f"{due_type}: R{dr.net_amount:,.2f} is below expected min R{lo:,.0f}")
                score -= 0.15
            elif dr.net_amount > hi:
                flags.append(f"{due_type}: R{dr.net_amount:,.2f} exceeds expected max R{hi:,.0f}")
                score -= 0.10
            if dr.confidence < 0.5:
                warnings.append(f"{due_type}: low extraction confidence ({dr.confidence:.0%})")

        return ValidationReport(
            passed=len(flags) == 0,
            confidence_score=max(0.0, score),
            hallucination_flags=flags,
            warnings=warnings,
        )


# ── 3. Confidence Scorer ──────────────────────────────────────────────────────

class ConfidenceScorer:

    def score(self, query: VesselQuery, result: CalculationResult) -> float:
        base = 1.0
        if not query.arrival_time:
            base -= 0.05
        if not query.departure_time:
            base -= 0.05
        if query.gross_tonnage == 0:
            base -= 0.5

        calc_confs = [dr.confidence for dr in result.dues.values()]
        avg_calc = sum(calc_confs) / len(calc_confs) if calc_confs else 1.0
        combined = base * 0.5 + avg_calc * 0.5
        return round(max(0.0, min(1.0, combined)), 3)


# ── 4. Business Rules Enforcer ────────────────────────────────────────────────

class BusinessRulesEnforcer:

    VAT_RATE = 0.15
    VTS_MINIMUM = 235.52
    COMPULSORY_PILOTAGE_PORTS = {
        "durban", "richards bay", "cape town", "port elizabeth",
        "ngqura", "saldanha", "east london", "mossel bay",
    }

    def enforce(self, query: VesselQuery, result: CalculationResult) -> list[str]:
        violations: list[str] = []

        # VAT must be 15% on every non-zero due
        for dt, dr in result.dues.items():
            if dr.net_amount > 0:
                expected_vat = round(dr.net_amount * self.VAT_RATE, 2)
                if abs(dr.vat_amount - expected_vat) > 1.0:
                    violations.append(
                        f"{dt}: VAT should be R{expected_vat:.2f} but got R{dr.vat_amount:.2f}"
                    )

        # VTS minimum
        if "vts_dues" in result.dues:
            vts = result.dues["vts_dues"]
            if not vts.exempted and vts.net_amount < self.VTS_MINIMUM * 0.9:
                violations.append(
                    f"VTS net amount R{vts.net_amount:.2f} is below mandatory minimum R{self.VTS_MINIMUM}"
                )

        # Pilotage compulsory at all TNPA ports
        if query.port.lower() in self.COMPULSORY_PILOTAGE_PORTS:
            if "pilotage_dues" not in result.dues:
                violations.append(
                    f"Pilotage is compulsory at {query.port} but no pilotage dues were calculated"
                )

        return violations


# ── 5. Output Validator ───────────────────────────────────────────────────────

class OutputValidator:

    def validate(self, result: CalculationResult) -> ValidationReport:
        issues: list[str] = []

        if result.grand_total_incl_vat < 0:
            issues.append("Grand total cannot be negative")
        if not result.dues:
            issues.append("No dues were calculated")
        if result.grand_total_incl_vat > 15_000_000:
            issues.append(f"Grand total R{result.grand_total_incl_vat:,.2f} exceeds R15M — please verify")

        return ValidationReport(
            passed=len(issues) == 0,
            confidence_score=1.0 if not issues else 0.5,
            issues=issues,
        )


# ── Guardrail Orchestrator ────────────────────────────────────────────────────

class GuardrailLayer:
    """
    Runs all guardrail components and returns a unified ValidationReport.
    """

    def __init__(self) -> None:
        self._input_validator    = InputValidator()
        self._hallucination_det  = HallucinationDetector()
        self._confidence_scorer  = ConfidenceScorer()
        self._business_enforcer  = BusinessRulesEnforcer()
        self._output_validator   = OutputValidator()

    def validate_input(self, query: VesselQuery) -> ValidationReport:
        report = self._input_validator.validate(query)
        if not report.passed:
            log.warning("Input validation failed", issues=report.issues)
        return report

    def validate_output(
        self, query: VesselQuery, result: CalculationResult
    ) -> dict[str, Any]:
        """
        Runs all post-calculation checks and returns a serialisable summary dict.
        """
        hall_report      = self._hallucination_det.detect(result)
        business_viols   = self._business_enforcer.enforce(query, result)
        output_report    = self._output_validator.validate(result)
        confidence       = self._confidence_scorer.score(query, result)

        all_warnings = (
            hall_report.warnings
            + hall_report.hallucination_flags
            + output_report.issues
        )
        passed = (
            hall_report.passed
            and output_report.passed
            and len(business_viols) == 0
        )

        log.info(
            "Guardrail output check",
            passed=passed,
            confidence=confidence,
            warnings=len(all_warnings),
            business_violations=len(business_viols),
        )

        if not passed:
            from monitoring import GUARDRAIL_FAILURES
            GUARDRAIL_FAILURES.labels(check_type="output").inc()

        return {
            "passed": passed,
            "confidence_score": confidence,
            "warnings": all_warnings,
            "hallucination_report": {"flagged": hall_report.hallucination_flags},
            "business_violations": business_viols,
        }
