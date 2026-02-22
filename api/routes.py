"""
api/routes.py
REST endpoints.
  """
import time
import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.concurrency import run_in_threadpool

from api.models import (
    CalculationRequest,
    CalculationResponse,
    DueTypeResult,
    GuardrailReport,
    IngestionRequest,
    IngestionResponse,
    TariffSummaryItem,
)
from calculation_engine.engine import CalculationEngine
from guardrails.guardrail_layer import GuardrailLayer
from knowledge_base.json_store import JSONStore
from query_processor.parser import QueryProcessor

router = APIRouter()

_store     = JSONStore()
_engine    = CalculationEngine(store=_store)
_qp        = QueryProcessor()
_guardrail = GuardrailLayer()


def _log():
    from monitoring import get_logger
    return get_logger(__name__)


def _build_query_from_request(request: CalculationRequest):
    """
    Build a query from the request.

    Three paths:
      A) NL only     — Groq extracts everything from request.query
      B) Structured  — pure Python mapping from request.vessel_data (no LLM)
      C) Mixed       — Groq extracts from query, then structured data overrides
    """
    has_query   = bool(request.query and request.query.strip())
    has_vessel  = request.vessel_data is not None
    has_gt      = has_vessel and bool(
        request.vessel_data.technical_specs.gross_tonnage
    )

    if has_query:
        # Build a flat vessel dict from whatever structured data is available
        flat: dict = {}
        if has_vessel:
            meta = request.vessel_data.vessel_metadata
            tech = request.vessel_data.technical_specs
            ops  = request.vessel_data.operational_data
            flat = {
                "name":            meta.name or "",
                "flag":            meta.flag or "",
                "gross_tonnage":   tech.gross_tonnage or 0,
                "net_tonnage":     tech.net_tonnage or 0,
                "loa_meters":      tech.loa_meters or 0,
                "dwt":             tech.dwt or 0,
                "type":            tech.type or "",
                "days_in_port":    ops.days_alongside or 0,
                "arrival_time":    ops.arrival_time,
                "departure_time":  ops.departure_time,
                "activity":        ops.activity or "",
                "num_operations":  ops.num_operations,
            }

        # Add port hint to query if provided separately
        query_text = request.query
        if request.port and request.port.lower() not in query_text.lower():
            query_text = f"{query_text}. Port: {request.port}"

        # LLM call — run in thread pool
        return "nl", query_text, flat

    else:
        # Structured path — pure Python, no LLM
        return "structured", None, None



@router.post(
    "/calculate",
    response_model=CalculationResponse,
    summary="Calculate port dues for a vessel",
    description="""
Calculate all TNPA port dues for a vessel.

**Two input modes:**

**1. Natural language** — just describe the vessel in plain English:
```json
{ "query": "Calculate dues for a 51300 GT bulk carrier called SUDESTADA at Durban for 3.39 days" }
```

**2. Structured** — pass the full vessel profile:
```json
{ "port": "Durban", "vessel_data": { ... } }
```

**3. Mixed** — provide a query and supplement with structured data for precision.
""",
)
async def calculate_dues(request: CalculationRequest) -> CalculationResponse:
    log = _log()
    request_id = str(uuid.uuid4())[:8]
    t0 = time.perf_counter()

    has_query = bool(request.query and request.query.strip())
    log.info(
        "Calculation request",
        request_id=request_id,
        mode="nl" if has_query else "structured",
        port=request.port,
    )

    try:
        # ── Build VesselQuery
        mode, query_text, flat_vessel = _build_query_from_request(request)

        if mode == "nl":
            # Groq LLM call — must run in thread pool
            query = await run_in_threadpool(_qp.parse, query_text, flat_vessel)

            # Port override
            if request.port:
                query.port = _qp._normalise_port(request.port)

        else:
            # Pure Python — instant, no LLM
            query = _qp.from_vessel_profile(
                request.vessel_data.model_dump(), request.port or ""
            )

        # Apply explicit flag overrides from request body
        if request.outside_working_hours:
            query.outside_working_hours = True
        if request.is_coaster:
            query.is_coaster = True
        if request.is_double_hull_tanker:
            query.is_double_hull_tanker = True

        # Override num_operations from vessel_data if provided
        if request.vessel_data:
            query.num_tug_operations = request.vessel_data.operational_data.num_operations
            if request.vessel_data.operational_data.days_alongside:
                query.days_in_port = request.vessel_data.operational_data.days_alongside

        if request.due_types:
            query.requested_due_types = request.due_types

        log.info(
            "VesselQuery built",
            vessel=query.vessel_name,
            port=query.port,
            gt=query.gross_tonnage,
            days=query.days_in_port,
        )

        # Input guardrail
        input_report = _guardrail.validate_input(query)
        if not input_report.passed:
            raise HTTPException(
                status_code=422,
                detail={"errors": input_report.issues, "warnings": input_report.warnings},
            )

        result = _engine.calculate(query)

        # Output guardrail
        gr = _guardrail.validate_output(query, result)

        # Explanation
        explanation: Optional[str] = None
        citations:   Optional[list] = None

        if request.include_explanation:
            try:
                from explanation.generator import ExplanationGenerator
                exp_data    = await run_in_threadpool(
                    ExplanationGenerator().generate, query, result
                )
                explanation = exp_data["explanation"]
                citations   = exp_data["citations"]
            except Exception as exp_exc:
                log.warning("Explanation generation failed", error=str(exp_exc))
                explanation = f"Explanation unavailable: {exp_exc}"
                citations   = []

        # Build response
        dues_out = {
            dt: DueTypeResult(
                due_type         =dt,
                base_amount      =dr.base_amount,
                surcharge_amount =dr.surcharge_amount,
                reduction_amount =dr.reduction_amount,
                net_amount       =dr.net_amount,
                vat_amount       =dr.vat_amount,
                total_with_vat   =dr.total_with_vat,
                formula_applied  =dr.formula_applied,
                breakdown        =dr.breakdown,
                confidence       =dr.confidence,
                exempted         =dr.exempted,
                exemption_reason =dr.exemption_reason,
                warnings         =dr.warnings,
            )
            for dt, dr in result.dues.items()
        }

        elapsed = time.perf_counter() - t0
        log.info(
            "Calculation complete",
            request_id=request_id,
            elapsed_ms=round(elapsed * 1000),
            total_incl_vat=result.grand_total_incl_vat,
        )

        #Tariff summary table
        DISPLAY_NAMES = {
            "light_dues":         "Light Dues",
            "port_dues":          "Port Dues",
            "towage_dues":        "Towage Dues",
            "vts_dues":           "VTS Dues",
            "pilotage_dues":      "Pilotage Dues",
            "running_lines_dues": "Running Lines",
        }
        tariff_summary = [
            TariffSummaryItem(
                tariff_item=DISPLAY_NAMES.get(dt, dt.replace("_", " ").title()),
                value=f"{dr.net_amount:,.2f}",
            )
            for dt, dr in result.dues.items()
            if not dr.exempted
        ]

        return CalculationResponse(
            success              =True,
            request_id           =request_id,
            vessel_name          =result.vessel_name,
            port                 =result.port,
            gross_tonnage        =result.gross_tonnage,
            days_in_port         =result.days_in_port,
            dues                 =dues_out,
            grand_total_excl_vat =result.grand_total_excl_vat,
            grand_total_vat      =result.grand_total_vat,
            grand_total_incl_vat =result.grand_total_incl_vat,
            guardrail_report     =GuardrailReport(**gr),
            explanation          =explanation,
            citations            =citations,
            tariff_summary        =tariff_summary,
            warnings             =result.warnings,
            calculation_metadata =result.calculation_metadata,
        )

    except HTTPException:
        raise
    except Exception as exc:
        log.error("Calculation failed", error=str(exc), request_id=request_id)
        raise HTTPException(status_code=500, detail=str(exc))


#POST /ingest

@router.post("/ingest", response_model=IngestionResponse, summary="Ingest a port tariff PDF")
async def ingest_pdf(request: IngestionRequest, background_tasks: BackgroundTasks) -> IngestionResponse:
    try:
        from ingestion.pipeline import IngestionPipeline
        pipeline = IngestionPipeline()
        summary  = await run_in_threadpool(pipeline.run, request.pdf_path, request.force_reingest)
        # Reload JSON store cache after ingestion
        _store.reload()
        return IngestionResponse(success=True, status=summary.get("status", "done"), summary=summary)
    except Exception as exc:
        return IngestionResponse(success=False, status="error", error=str(exc))


#GET /health

@router.get("/health", summary="Health check")
async def health() -> dict:
    from knowledge_base.sqlite_store import SQLiteStore
    from knowledge_base.vector_store import VectorStore
    sqlite = SQLiteStore()
    chroma = VectorStore()
    sqlite_stats = sqlite.stats()
    chroma_stats = chroma.stats()
    return {
        "status": "healthy",
        "nl_query_supported": True,
        "knowledge_base": {
            "sqlite": {
                "loaded":   sqlite.count() > 0,
                "rows":     sqlite.count(),
                "breakdown": sqlite_stats,
            },
            "chromadb": {
                "loaded":        chroma.count() > 0,
                "total_chunks":  chroma.count(),
                "semantic_rules": chroma_stats.get("semantic_rule", 0),
                "raw_chunks":    chroma_stats.get("raw", 0),
            },
        },
        "supported_ports": [
            "Durban", "Cape Town", "Richards Bay", "Port Elizabeth",
            "Ngqura", "East London", "Saldanha", "Mossel Bay",
        ],
    }


# GET /ports

@router.get("/ports", summary="List supported TNPA ports")
async def list_ports() -> dict:
    return {
        "ports": [
            "Durban", "Cape Town", "Richards Bay", "Port Elizabeth",
            "Ngqura", "East London", "Saldanha", "Mossel Bay",
        ]
    }



