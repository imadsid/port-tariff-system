"""
main.py
CLI entry point for the Port Tariff Calculation System.

Usage:
  python main.py ingest --pdf /path/to/Port_Tariff.pdf
  python main.py demo   --port Durban
  python main.py api
"""
import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path when running directly
sys.path.insert(0, str(Path(__file__).parent))

# ── Reference vessel from the take-home test ──────────────────────────────────
SUDESTADA_PROFILE = {
    "vessel_metadata": {
        "name": "SUDESTADA",
        "built_year": 2010,
        "flag": "MLT - Malta",
        "classification_society": "Registro Italiano Navale",
    },
    "technical_specs": {
        "type": "Bulk Carrier",
        "dwt": 93274,
        "gross_tonnage": 51300,
        "net_tonnage": 31192,
        "loa_meters": 229.2,
        "beam_meters": 38.0,
        "moulded_depth_meters": 20.7,
    },
    "operational_data": {
        "cargo_quantity_mt": 40000,
        "days_alongside": 3.39,
        "arrival_time": "2024-11-15T10:12:00",
        "departure_time": "2024-11-22T13:00:00",
        "activity": "Exporting Iron Ore",
        "num_operations": 2,
    },
}

GROUND_TRUTH = {
    "light_dues":        60_062.04,
    "port_dues":        199_549.22,
    "towage_dues":      147_074.38,
    "vts_dues":          33_315.75,
    "pilotage_dues":     47_189.94,
    "running_lines_dues":19_639.50,
}


# Demo mode

def run_demo(port: str = "Durban") -> None:
    from rich import box
    from rich.console import Console
    from rich.table import Table

    from calculation_engine.engine import CalculationEngine
    from guardrails.guardrail_layer import GuardrailLayer
    from knowledge_base.json_store import JSONStore
    from query_processor.parser import QueryProcessor

    console = Console()
    console.print("\n[bold blue]═══ PORT TARIFF CALCULATION SYSTEM — DEMO ═══[/bold blue]\n")

    store = JSONStore()
    qp    = QueryProcessor()
    engine    = CalculationEngine(store=store)
    guardrail = GuardrailLayer()

    # Build VesselQuery directly from the structured profile (no LLM needed)
    query = qp.from_vessel_profile(SUDESTADA_PROFILE, port)

    console.print(f"  [bold]Vessel:[/bold] {query.vessel_name}")
    console.print(f"  [bold]Port:[/bold]   {query.port}")
    console.print(f"  [bold]GT:[/bold]     {query.gross_tonnage:,.0f}")
    console.print(f"  [bold]LOA:[/bold]    {query.loa_meters} m")
    console.print(f"  [bold]Days:[/bold]   {query.days_in_port}")
    console.print()

    result    = engine.calculate(query)
    gr_report = guardrail.validate_output(query, result)

    # Results table
    table = Table(
        title=f"Port Dues — {query.vessel_name} @ {query.port}",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("Due Type",            style="cyan",  width=26)
    table.add_column("Calculated (excl VAT)", justify="right", width=20)
    table.add_column("Expected (excl VAT)",   justify="right", style="yellow", width=20)
    table.add_column("Variance",              justify="right", width=10)
    table.add_column("Total (incl 15% VAT)", justify="right", style="green", width=20)

    for due_type, expected in GROUND_TRUTH.items():
        if due_type in result.dues:
            dr = result.dues[due_type]
            calc = dr.net_amount
            var  = (calc - expected) / expected * 100 if expected else 0.0
            vstyle = "green" if abs(var) < 5 else ("yellow" if abs(var) < 15 else "red")
            table.add_row(
                due_type.replace("_", " ").title(),
                f"R{calc:>14,.2f}",
                f"R{expected:>14,.2f}",
                f"[{vstyle}]{var:+.1f}%[/{vstyle}]",
                f"R{dr.total_with_vat:>14,.2f}",
            )
        else:
            table.add_row(
                due_type.replace("_", " ").title(),
                "— not calculated —", f"R{expected:>14,.2f}", "—", "—",
            )

    table.add_section()
    table.add_row(
        "[bold]GRAND TOTAL[/bold]",
        f"[bold]R{result.grand_total_excl_vat:>14,.2f}[/bold]",
        "", "",
        f"[bold green]R{result.grand_total_incl_vat:>14,.2f}[/bold green]",
    )
    console.print(table)

    # Guardrail summary
    console.print(f"\n  [bold]Confidence Score:[/bold] {gr_report['confidence_score']:.0%}")
    status_str = "[green]PASSED[/green]" if gr_report["passed"] else "[red]FLAGGED[/red]"
    console.print(f"  [bold]Guardrail Status:[/bold] {status_str}")
    for w in gr_report.get("warnings", []):
        console.print(f"  [yellow]⚠  {w}[/yellow]")

    # Breakdown
    console.print("\n  [bold]Formula Breakdown:[/bold]")
    for due_type, dr in result.dues.items():
        console.print(f"\n  [{due_type}]  formula: {dr.formula_applied}")
        for item in dr.breakdown:
            amt = item.get("amount", 0)
            console.print(f"    • {item.get('item',''):<60}  R{amt:>12,.2f}")

    console.print(f"\n  [bold]Grand Total (excl VAT):[/bold]  R{result.grand_total_excl_vat:>14,.2f}")
    console.print(f"  [bold]VAT (15%):[/bold]               R{result.grand_total_vat:>14,.2f}")
    console.print(f"  [bold green]Grand Total (incl VAT):[/bold green]  R{result.grand_total_incl_vat:>14,.2f}\n")


# Ingest mode

def run_ingest(pdf_path: str, force: bool = False) -> None:
    from ingestion.pipeline import IngestionPipeline
    pipeline = IngestionPipeline()
    summary  = pipeline.run(pdf_path, force_reingest=force)
    print(json.dumps(summary, indent=2))


# ── API mode ──────────────────────────────────────────────────────────────────

def run_api() -> None:
    import uvicorn
    from config.settings import settings
    uvicorn.run(
        "api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
        # For debugging execute demo mode
        # run_demo()
        run_api()
