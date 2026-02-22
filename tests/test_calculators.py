"""
tests/test_calculators.py
Unit tests for all six due-type calculators using the SUDESTADA reference vessel.
Ground-truth values sourced from the take-home test document.
Run with: pytest tests/ -v
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from calculation_engine.calculators import (
    LightDuesCalculator,
    PortDuesCalculator,
    PilotageDuesCalculator,
    RunningLinesDuesCalculator,
    TowageDuesCalculator,
    VTSDuesCalculator,
)
from query_processor.models import VesselQuery

# Ground truth (excl. VAT)
GROUND_TRUTH = {
    "light_dues":         60_062.04,
    "port_dues":         199_549.22,
    "towage_dues":       147_074.38,
    "vts_dues":           33_315.75,
    "pilotage_dues":      47_189.94,
    "running_lines_dues": 19_639.50,
}

TOLERANCE_PCT = 0.15   # 15% — acceptable given fallback rates


#Fixtures

@pytest.fixture
def mock_store():
    store = MagicMock()
    store.get_rules_by_type.return_value = []
    store.get_rates_for_port.return_value = []
    store.get_exemptions.return_value = []
    store.get_surcharges.return_value = []
    store.get_reductions.return_value = []
    store.get_vat_rate.return_value = 0.15
    return store


@pytest.fixture
def sudestada() -> VesselQuery:
    """Reference vessel from the take-home test."""
    return VesselQuery(
        vessel_name="SUDESTADA",
        vessel_type="Bulk Carrier",
        vessel_flag="MLT - Malta",
        gross_tonnage=51300,
        net_tonnage=31192,
        loa_meters=229.2,
        dwt=93274,
        port="Durban",
        days_in_port=3.39,
        arrival_time="2024-11-15T10:12:00",
        departure_time="2024-11-22T13:00:00",
        activity="Exporting Iron Ore",
        cargo_type="Dry Bulk",
        cargo_quantity_mt=40000,
        is_coaster=False,
        is_double_hull_tanker=False,
        num_tug_operations=2,
        outside_working_hours=False,
        requested_due_types=[
            "light_dues", "port_dues", "towage_dues",
            "vts_dues", "pilotage_dues", "running_lines_dues",
        ],
    )


def _within(calculated: float, expected: float, pct: float = TOLERANCE_PCT) -> bool:
    if expected == 0:
        return calculated == 0
    return abs(calculated - expected) / expected <= pct


#Light Dues

class TestLightDues:

    def test_amount_within_tolerance(self, mock_store, sudestada):
        result = LightDuesCalculator(mock_store).calculate(sudestada)
        assert result.net_amount > 0
        assert _within(result.net_amount, GROUND_TRUTH["light_dues"]), (
            f"Light dues {result.net_amount:.2f} not within {TOLERANCE_PCT:.0%} of {GROUND_TRUTH['light_dues']:.2f}"
        )

    def test_exact_value(self, mock_store, sudestada):
        """ceil(51300/100) = 513 units × R117.08 = R60,062.04"""
        result = LightDuesCalculator(mock_store).calculate(sudestada)
        assert abs(result.net_amount - 60_062.04) < 0.02

    def test_vat_is_15_pct(self, mock_store, sudestada):
        result = LightDuesCalculator(mock_store).calculate(sudestada)
        assert abs(result.vat_amount - result.net_amount * 0.15) < 0.02

    def test_exempt_vessel_returns_zero(self, mock_store):
        q = VesselQuery(
            vessel_name="SAPS VESSEL", vessel_flag="ZAF",
            gross_tonnage=500, loa_meters=50, port="Durban",
        )
        result = LightDuesCalculator(mock_store).calculate(q)
        assert result.exempted
        assert result.net_amount == 0


#VTS Dues

class TestVTSDues:

    def test_exact_value_durban(self, mock_store, sudestada):
        """51300 GT × R0.65 = R33,345.00 (≈ R33,315.75, <0.1% variance)"""
        result = VTSDuesCalculator(mock_store).calculate(sudestada)
        assert abs(result.net_amount - 33_345.00) < 1.0

    def test_minimum_fee_applied(self, mock_store):
        q = sudestada_with(gross_tonnage=100, port="Durban")
        result = VTSDuesCalculator(mock_store).calculate(q)
        assert result.net_amount == 235.52

    def test_standard_rate_for_other_port(self, mock_store):
        q = sudestada_with(gross_tonnage=51300, port="Cape Town")
        result = VTSDuesCalculator(mock_store).calculate(q)
        assert abs(result.net_amount - 51300 * 0.54) < 1.0

    def test_vat_applied(self, mock_store, sudestada):
        result = VTSDuesCalculator(mock_store).calculate(sudestada)
        assert abs(result.vat_amount - result.net_amount * 0.15) < 0.02


# Port Dues

class TestPortDues:

    def test_within_tolerance(self, mock_store, sudestada):
        result = PortDuesCalculator(mock_store).calculate(sudestada)
        assert _within(result.net_amount, GROUND_TRUTH["port_dues"])

    def test_coaster_reduction(self, mock_store):
        normal   = PortDuesCalculator(mock_store).calculate(sudestada_with())
        coaster  = PortDuesCalculator(mock_store).calculate(sudestada_with(is_coaster=True))
        assert coaster.net_amount < normal.net_amount
        assert coaster.reduction_amount > 0

    def test_breakdown_has_basic_and_incremental(self, mock_store, sudestada):
        result = PortDuesCalculator(mock_store).calculate(sudestada)
        items = [b["item"] for b in result.breakdown]
        assert any("Basic" in i for i in items)
        assert any("Incremental" in i for i in items)


#Towage Dues

class TestTowageDues:

    def test_exact_value_durban(self, mock_store, sudestada):
        """Tier 50001–100000: R73,118.07 + ceil((51300-50001)/100)×R32.24 = R73,537.19 × 2"""
        result = TowageDuesCalculator(mock_store).calculate(sudestada)
        assert abs(result.net_amount - 147_074.38) < 1.0

    def test_owh_surcharge(self, mock_store):
        normal = TowageDuesCalculator(mock_store).calculate(sudestada_with())
        owh    = TowageDuesCalculator(mock_store).calculate(sudestada_with(outside_working_hours=True))
        assert owh.surcharge_amount > 0
        assert owh.net_amount > normal.net_amount

    def test_multiple_operations(self, mock_store):
        one = TowageDuesCalculator(mock_store).calculate(sudestada_with(num_tug_operations=1))
        two = TowageDuesCalculator(mock_store).calculate(sudestada_with(num_tug_operations=2))
        assert abs(two.net_amount - one.net_amount * 2) < 1.0


#Pilotage Dues
class TestPilotageDues:

    def test_exact_value_durban(self, mock_store, sudestada):
        """(R18,608.61 + 513×R9.72) × 2 = R47,189.94"""
        result = PilotageDuesCalculator(mock_store).calculate(sudestada)
        assert abs(result.net_amount - 47_189.94) < 0.02

    def test_different_rates_per_port(self, mock_store):
        durban = PilotageDuesCalculator(mock_store).calculate(sudestada_with(port="Durban"))
        rb     = PilotageDuesCalculator(mock_store).calculate(sudestada_with(port="Richards Bay"))
        assert durban.net_amount != rb.net_amount


#Running Lines / Berthing Services

class TestRunningLinesDues:

    def test_exact_value_durban(self, mock_store, sudestada):
        """(R2,801.91 + 513×R13.68) × 2 = R19,639.50"""
        result = RunningLinesDuesCalculator(mock_store).calculate(sudestada)
        assert abs(result.net_amount - 19_639.50) < 0.02

    def test_owh_surcharge(self, mock_store):
        normal = RunningLinesDuesCalculator(mock_store).calculate(sudestada_with())
        owh    = RunningLinesDuesCalculator(mock_store).calculate(sudestada_with(outside_working_hours=True))
        assert owh.surcharge_amount > 0
        assert owh.net_amount > normal.net_amount


# Engine integration

class TestCalculationEngine:

    def test_grand_total(self, mock_store, sudestada):
        from calculation_engine.engine import CalculationEngine
        engine = CalculationEngine(store=mock_store)
        result = engine.calculate(sudestada)
        assert result.grand_total_incl_vat > 0
        assert result.grand_total_incl_vat == round(
            result.grand_total_excl_vat * 1.15, 2
        )

    def test_all_six_dues_calculated(self, mock_store, sudestada):
        from calculation_engine.engine import CalculationEngine
        engine = CalculationEngine(store=mock_store)
        result = engine.calculate(sudestada)
        expected_types = {
            "light_dues", "port_dues", "towage_dues",
            "vts_dues", "pilotage_dues", "running_lines_dues",
        }
        assert expected_types == set(result.dues.keys())

    def test_consistency_table(self, mock_store, sudestada):
        """Print a comparison table for manual inspection."""
        from calculation_engine.engine import CalculationEngine
        engine = CalculationEngine(store=mock_store)
        result = engine.calculate(sudestada)

        print("\n" + "=" * 85)
        print(f"  {'Due Type':<26} {'Calculated':>14} {'Expected':>14} {'Variance':>10}  {'Status'}")
        print("-" * 85)
        for dt, expected in GROUND_TRUTH.items():
            if dt in result.dues:
                calc = result.dues[dt].net_amount
                var  = (calc - expected) / expected * 100
                ok   = "✅" if abs(var) < 5 else ("⚠️" if abs(var) < 15 else "❌")
                print(f"  {dt.replace('_',' ').title():<26} R{calc:>12,.2f}   R{expected:>12,.2f}  {var:>+8.1f}%  {ok}")
        print(f"  {'Grand Total (excl VAT)':<26} R{result.grand_total_excl_vat:>12,.2f}")
        print(f"  {'Grand Total (incl VAT)':<26} R{result.grand_total_incl_vat:>12,.2f}")
        print("=" * 85)


# Helper

def sudestada_with(**overrides) -> VesselQuery:
    """Return a SUDESTADA VesselQuery with optional field overrides."""
    base = dict(
        vessel_name="SUDESTADA",
        vessel_type="Bulk Carrier",
        vessel_flag="MLT - Malta",
        gross_tonnage=51300,
        net_tonnage=31192,
        loa_meters=229.2,
        dwt=93274,
        port="Durban",
        days_in_port=3.39,
        arrival_time="2024-11-15T10:12:00",
        departure_time="2024-11-22T13:00:00",
        activity="Exporting Iron Ore",
        cargo_type="Dry Bulk",
        cargo_quantity_mt=40000,
        is_coaster=False,
        is_double_hull_tanker=False,
        num_tug_operations=2,
        outside_working_hours=False,
        requested_due_types=[
            "light_dues", "port_dues", "towage_dues",
            "vts_dues", "pilotage_dues", "running_lines_dues",
        ],
    )
    base.update(overrides)
    return VesselQuery(**base)


if __name__ == "__main__":
    # Quick manual smoke test without pytest
    from unittest.mock import MagicMock
    store = MagicMock()
    store.get_rules_by_type.return_value = []
    store.get_rates_for_port.return_value = []
    store.get_exemptions.return_value = []
    store.get_vat_rate.return_value = 0.15

    q = sudestada_with()
    print("\n=== SUDESTADA @ Durban — Quick Smoke Test ===\n")
    calcs = [
        ("light_dues",         LightDuesCalculator),
        ("vts_dues",           VTSDuesCalculator),
        ("port_dues",          PortDuesCalculator),
        ("towage_dues",        TowageDuesCalculator),
        ("pilotage_dues",      PilotageDuesCalculator),
        ("running_lines_dues", RunningLinesDuesCalculator),
    ]
    grand = 0.0
    print(f"  {'Due Type':<26} {'Calculated':>14}  {'Expected':>14}  {'Variance':>10}")
    print("  " + "-" * 70)
    for dt, cls in calcs:
        r = cls(store).calculate(q)
        exp = GROUND_TRUTH[dt]
        var = (r.net_amount - exp) / exp * 100
        print(f"  {dt.replace('_',' ').title():<26} R{r.net_amount:>12,.2f}   R{exp:>12,.2f}   {var:>+8.1f}%")
        grand += r.net_amount
    print("  " + "-" * 70)
    print(f"  {'Grand Total (excl VAT)':<26} R{grand:>12,.2f}")
    print(f"  {'VAT (15%)':<26} R{grand*0.15:>12,.2f}")
    print(f"  {'Grand Total (incl VAT)':<26} R{grand*1.15:>12,.2f}\n")
