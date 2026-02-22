"""
api/models.py
Pydantic request/response models.

Two modes for the calculate endpoint:
  1. Structured:  provide vessel_data (no LLM call)
  2. Natural language: provide query string only (Groq LLM extracts parameters)
"""
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator



class VesselMetadata(BaseModel):
    name:                    Optional[str] = None
    built_year:              Optional[int] = None
    flag:                    Optional[str] = None
    classification_society:  Optional[str] = None


class TechnicalSpecs(BaseModel):
    type:                   Optional[str]   = "General"
    dwt:                    Optional[float] = None
    gross_tonnage:          Optional[float] = Field(default=None, description="Gross tonnage (GT)", gt=0)
    net_tonnage:            Optional[float] = None
    loa_meters:             Optional[float] = Field(default=None, description="Length Overall in metres", gt=0)
    beam_meters:            Optional[float] = None
    moulded_depth_meters:   Optional[float] = None


class OperationalData(BaseModel):
    cargo_quantity_mt:  Optional[float] = None
    days_alongside:     Optional[float] = Field(default=None, ge=0)
    arrival_time:       Optional[str]   = None
    departure_time:     Optional[str]   = None
    activity:           Optional[str]   = None
    num_operations:     int             = Field(default=2, ge=1)


class VesselProfile(BaseModel):
    vessel_metadata:  VesselMetadata  = Field(default_factory=VesselMetadata)
    technical_specs:  TechnicalSpecs  = Field(default_factory=TechnicalSpecs)
    operational_data: OperationalData = Field(default_factory=OperationalData)



class CalculationRequest(BaseModel):
    query: Optional[str] = Field(
        default=None,
        description=(
            "Natural language query describing the vessel and voyage. "
            "Example: 'Calculate dues for a 51300 GT bulk carrier at Durban for 3.39 days'. "
            "When provided, vessel_data is optional — the LLM extracts all parameters from the query."
        ),
    )
    port: Optional[str] = Field(
        default=None,
        description="TNPA port name e.g. 'Durban'. Can be omitted if mentioned in the query.",
    )
    vessel_data: Optional[VesselProfile] = Field(
        default=None,
        description=(
            "Structured vessel profile. Required when query is not provided. "
            "When query is provided, this supplements or overrides the LLM extraction."
        ),
    )
    due_types: Optional[list[str]] = Field(
        default=None,
        description=(
            "Specific due types to calculate. "
            "Defaults to all 6: light_dues, port_dues, towage_dues, "
            "vts_dues, pilotage_dues, running_lines_dues."
        ),
    )
    outside_working_hours:  bool = Field(default=False, description="Apply OWH surcharges")
    is_coaster:             bool = Field(default=False, description="Apply coaster reduction (35%)")
    is_double_hull_tanker:  bool = Field(default=False, description="Apply double-hull tanker reduction (10%)")
    include_explanation:    bool = Field(
        default=False,
        description="Generate a plain-English explanation of the calculation (uses Groq).",
    )

    @model_validator(mode="after")
    def require_query_or_vessel_data(self):
        if not self.query and not self.vessel_data:
            raise ValueError(
                "Either 'query' (natural language) or 'vessel_data' (structured) must be provided."
            )
        # If structured mode, gross_tonnage is required
        if not self.query:
            gt = self.vessel_data.technical_specs.gross_tonnage if self.vessel_data else None
            if not gt:
                raise ValueError("vessel_data.technical_specs.gross_tonnage is required in structured mode.")
        return self



class IngestionRequest(BaseModel):
    pdf_path:       str  = Field(..., description="Absolute path to the port tariff PDF file")
    force_reingest: bool = Field(default=False, description="Re-run ingestion even if KB already exists")



class DueTypeResult(BaseModel):
    due_type:         str
    base_amount:      float
    surcharge_amount: float
    reduction_amount: float
    net_amount:       float
    vat_amount:       float
    total_with_vat:   float
    formula_applied:  str
    breakdown:        list[dict[str, Any]]
    confidence:       float
    exempted:         bool
    exemption_reason: str
    warnings:         list[str]


class GuardrailReport(BaseModel):
    passed:               bool
    confidence_score:     float
    warnings:             list[str]
    hallucination_report: dict[str, Any]
    business_violations:  list[str]


class TariffSummaryItem(BaseModel):
    tariff_item: str
    value:       str   # formatted e.g. "60,062.04"


class CalculationResponse(BaseModel):
    success:              bool
    request_id:           Optional[str] = None
    timestamp:            str           = Field(default_factory=lambda: datetime.utcnow().isoformat())
    vessel_name:          str
    port:                 str
    gross_tonnage:        float
    days_in_port:         float
    dues:                 dict[str, DueTypeResult]
    grand_total_excl_vat: float
    grand_total_vat:      float
    grand_total_incl_vat: float
    guardrail_report:     GuardrailReport
    explanation:          Optional[str]              = None
    citations:            Optional[list[dict[str, Any]]] = None
    tariff_summary:        list[TariffSummaryItem]     = []
    warnings:             list[str]                  = []
    calculation_metadata: dict[str, Any]             = {}


class IngestionResponse(BaseModel):
    success: bool
    status:  str
    summary: dict[str, Any] = {}
    error:   Optional[str]  = None


class ErrorResponse(BaseModel):
    success:    bool         = False
    error:      str
    detail:     Optional[str] = None
    request_id: Optional[str] = None
