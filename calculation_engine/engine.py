"""
calculation_engine/engine.py
Coordinates all six due-type calculators and produces an aggregated result.
"""
from dataclasses import dataclass, field
from typing import Optional

from calculation_engine.calculators import (
    DueResult,
    LightDuesCalculator,
    PortDuesCalculator,
    PilotageDuesCalculator,
    RunningLinesDuesCalculator,
    TowageDuesCalculator,
    VTSDuesCalculator,
)
from knowledge_base.json_store import JSONStore
from monitoring import get_logger
from query_processor.models import VesselQuery

log = get_logger(__name__)


@dataclass
class CalculationResult:
    """Aggregated output from all requested due-type calculators."""
    vessel_name: str
    port: str
    gross_tonnage: float
    days_in_port: float
    dues: dict[str, DueResult] = field(default_factory=dict)
    grand_total_excl_vat: float = 0.0
    grand_total_vat: float = 0.0
    grand_total_incl_vat: float = 0.0
    warnings: list[str] = field(default_factory=list)
    calculation_metadata: dict = field(default_factory=dict)


class CalculationEngine:
    """
    Orchestrates calculation of all requested due types.
    Each calculator is independent, stateless, and directly testable.
    """

    _CALCULATOR_MAP: dict[str, type] = {
        "light_dues":        LightDuesCalculator,
        "vts_dues":          VTSDuesCalculator,
        "port_dues":         PortDuesCalculator,
        "towage_dues":       TowageDuesCalculator,
        "pilotage_dues":     PilotageDuesCalculator,
        "running_lines_dues": RunningLinesDuesCalculator,
    }

    def __init__(self, store: Optional[JSONStore] = None) -> None:
        self._store = store or JSONStore()
        self._calculators = {
            dt: cls(self._store) for dt, cls in self._CALCULATOR_MAP.items()
        }

    def calculate(self, query: VesselQuery) -> CalculationResult:
        """
        Run all requested calculators and aggregate results.

        Args:
            query: Parsed VesselQuery containing vessel and voyage parameters.

        Returns:
            CalculationResult with per-due-type results and grand totals.
        """
        log.info(
            "Starting calculation",
            vessel=query.vessel_name,
            port=query.port,
            gt=query.gross_tonnage,
            due_types=query.requested_due_types,
        )

        result = CalculationResult(
            vessel_name=query.vessel_name,
            port=query.port,
            gross_tonnage=query.gross_tonnage,
            days_in_port=query.days_in_port,
        )

        for due_type in query.requested_due_types:
            calculator = self._calculators.get(due_type)
            if not calculator:
                msg = f"No calculator registered for due_type='{due_type}'"
                log.warning(msg)
                result.warnings.append(msg)
                continue

            try:
                dr = calculator.calculate(query)
                result.dues[due_type] = dr
                log.info(
                    "Due calculated",
                    due_type=due_type,
                    net=dr.net_amount,
                    vat=dr.vat_amount,
                    total=dr.total_with_vat,
                    exempted=dr.exempted,
                )
            except Exception as exc:
                msg = f"Calculator failed for {due_type}: {exc}"
                log.error(msg)
                result.warnings.append(msg)

        # Aggregate (skip exempted dues)
        non_exempt = [dr for dr in result.dues.values() if not dr.exempted]
        result.grand_total_excl_vat = round(sum(dr.net_amount for dr in non_exempt), 2)
        result.grand_total_vat      = round(sum(dr.vat_amount for dr in non_exempt), 2)
        result.grand_total_incl_vat = round(sum(dr.total_with_vat for dr in non_exempt), 2)

        result.calculation_metadata = {
            "vat_rate": self._store.get_vat_rate(),
            "calculators_run": list(result.dues.keys()),
            "exempted": [dt for dt, dr in result.dues.items() if dr.exempted],
        }

        log.info(
            "Calculation complete",
            total_excl_vat=result.grand_total_excl_vat,
            total_incl_vat=result.grand_total_incl_vat,
        )
        return result
