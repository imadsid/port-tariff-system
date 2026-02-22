"""
query_processor/models.py
Shared VesselQuery dataclass used by the parser, calculators, guardrails,
and explanation generator.  Kept in a separate module to avoid circular imports.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VesselQuery:
    """
    Normalised vessel & voyage parameters extracted from a natural language query
    or a structured vessel profile dict.  Used as the single source of truth
    throughout the calculation pipeline.
    """
    # Vessel identity
    vessel_name: str = ""
    vessel_type: str = "general"
    vessel_flag: str = ""

    # Tonnage
    gross_tonnage: float = 0.0
    net_tonnage: float = 0.0
    dwt: float = 0.0
    loa_meters: float = 0.0

    # Port & timing
    port: str = ""
    days_in_port: float = 0.0
    arrival_time: Optional[str] = None
    departure_time: Optional[str] = None

    # Operations
    activity: str = ""
    cargo_type: Optional[str] = None
    cargo_quantity_mt: Optional[float] = None
    num_tug_operations: int = 2        # entry + departure
    outside_working_hours: bool = False

    # Vessel characteristics
    is_coaster: bool = False
    is_double_hull_tanker: bool = False

    # Requested due types
    requested_due_types: list[str] = field(default_factory=lambda: [
        "light_dues", "port_dues", "towage_dues",
        "vts_dues", "pilotage_dues", "running_lines_dues",
    ])

    # Meta
    raw_query: str = ""
    extraction_confidence: float = 1.0
    warnings: list[str] = field(default_factory=list)
