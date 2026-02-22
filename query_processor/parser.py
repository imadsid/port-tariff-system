"""
query_processor/parser.py
Query Processor using Groq LLM.
Converts a natural language query or structured vessel dict into a VesselQuery.
"""
import json
import re
from typing import Any, Optional

from config.settings import settings
from monitoring import get_logger
from query_processor.models import VesselQuery

log = get_logger(__name__)

PORT_ALIASES: dict[str, str] = {
    "durban": "Durban", "dbn": "Durban",
    "richards bay": "Richards Bay", "richardsbay": "Richards Bay", "rbay": "Richards Bay",
    "east london": "East London", "eastlondon": "East London", "el": "East London",
    "ngqura": "Ngqura", "coega": "Ngqura",
    "port elizabeth": "Port Elizabeth", "portelizabeth": "Port Elizabeth",
    "pe": "Port Elizabeth", "gqeberha": "Port Elizabeth",
    "mossel bay": "Mossel Bay", "mosselbay": "Mossel Bay",
    "cape town": "Cape Town", "capetown": "Cape Town", "cpt": "Cape Town",
    "saldanha": "Saldanha", "saldanha bay": "Saldanha",
}

_PROMPT = """You are a port tariff expert. Extract vessel and voyage parameters from the information below.

Return ONLY a valid JSON object with these exact keys (use null for missing values):
{
  "vessel_name": "string",
  "vessel_type": "Bulk Carrier | Container | Tanker | Passenger | Fishing | General",
  "vessel_flag": "string",
  "gross_tonnage": number,
  "net_tonnage": number,
  "dwt": number,
  "loa_meters": number,
  "port": "string",
  "days_in_port": number,
  "arrival_time": "ISO datetime or null",
  "departure_time": "ISO datetime or null",
  "activity": "string",
  "cargo_type": "Dry Bulk | Liquid Bulk | Breakbulk | Container | null",
  "cargo_quantity_mt": number_or_null,
  "num_tug_operations": number,
  "outside_working_hours": boolean,
  "is_coaster": boolean,
  "is_double_hull_tanker": boolean,
  "requested_due_types": ["light_dues","port_dues","towage_dues","vts_dues","pilotage_dues","running_lines_dues"]
}

Return ONLY the JSON object. No explanation, no markdown, no code fences.

Natural language query:
%s

Vessel profile (structured data):
%s
"""


class QueryProcessor:
    """
    Parses vessel/voyage parameters into a VesselQuery.
    Uses Groq LLM for natural language queries; pure Python for structured dicts.
    """

    def __init__(self) -> None:
        self._client = None  # lazy

    @property
    def client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=settings.groq_api_key)
        return self._client

    def parse(self, query: str, vessel_data: Optional[dict[str, Any]] = None) -> VesselQuery:
        """Parse a natural language query into a VesselQuery using Groq."""
        if not query.strip() and vessel_data:
            return self._from_dict(vessel_data)

        log.info("Parsing NL query with Groq", chars=len(query))
        vessel_json = json.dumps(vessel_data or {}, indent=2)
        prompt = _PROMPT % (query, vessel_json)

        response = self.client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```json\s*|\s*```$", "", raw).strip()
        params: dict[str, Any] = json.loads(raw)
        vq = self._build_vessel_query(params, query)
        log.info("Query parsed", port=vq.port, gt=vq.gross_tonnage)
        return vq

    def from_vessel_profile(self, profile: dict[str, Any], port: str) -> VesselQuery:
        """Build VesselQuery directly from a structured vessel profile dict (no LLM)."""
        meta = profile.get("vessel_metadata", {})
        tech = profile.get("technical_specs", {})
        ops  = profile.get("operational_data", {})
        days = float(ops.get("days_alongside") or 1.0)
        return VesselQuery(
            vessel_name=meta.get("name", ""),
            vessel_type=tech.get("type", "General"),
            vessel_flag=meta.get("flag", ""),
            gross_tonnage=float(tech.get("gross_tonnage") or 0),
            net_tonnage=float(tech.get("net_tonnage") or 0),
            dwt=float(tech.get("dwt") or 0),
            loa_meters=float(tech.get("loa_meters") or 0),
            port=self._normalise_port(port),
            days_in_port=days,
            arrival_time=ops.get("arrival_time"),
            departure_time=ops.get("departure_time"),
            activity=ops.get("activity", ""),
            cargo_quantity_mt=ops.get("cargo_quantity_mt"),
            num_tug_operations=int(ops.get("num_operations") or 2),
            requested_due_types=[
                "light_dues","port_dues","towage_dues",
                "vts_dues","pilotage_dues","running_lines_dues",
            ],
            extraction_confidence=1.0,
        )

    def _from_dict(self, d: dict[str, Any]) -> VesselQuery:
        return VesselQuery(
            vessel_name=d.get("name",""), vessel_type=d.get("type","General"),
            vessel_flag=d.get("flag",""),
            gross_tonnage=float(d.get("gross_tonnage") or 0),
            net_tonnage=float(d.get("net_tonnage") or 0),
            dwt=float(d.get("dwt") or 0),
            loa_meters=float(d.get("loa_meters") or 0),
            port=self._normalise_port(d.get("port","")),
            days_in_port=float(d.get("days_in_port") or 1),
            arrival_time=d.get("arrival_time"), departure_time=d.get("departure_time"),
            activity=d.get("activity",""),
            num_tug_operations=int(d.get("num_tug_operations") or 2),
            outside_working_hours=bool(d.get("outside_working_hours",False)),
            is_coaster=bool(d.get("is_coaster",False)),
            is_double_hull_tanker=bool(d.get("is_double_hull_tanker",False)),
            requested_due_types=d.get("requested_due_types",[
                "light_dues","port_dues","towage_dues",
                "vts_dues","pilotage_dues","running_lines_dues",
            ]),
            extraction_confidence=1.0,
        )

    def _build_vessel_query(self, params: dict[str, Any], raw_query: str) -> VesselQuery:
        return VesselQuery(
            vessel_name=params.get("vessel_name") or "",
            vessel_type=params.get("vessel_type") or "General",
            vessel_flag=params.get("vessel_flag") or "",
            gross_tonnage=float(params.get("gross_tonnage") or 0),
            net_tonnage=float(params.get("net_tonnage") or 0),
            dwt=float(params.get("dwt") or 0),
            loa_meters=float(params.get("loa_meters") or 0),
            port=self._normalise_port(params.get("port") or ""),
            days_in_port=float(params.get("days_in_port") or 1),
            arrival_time=params.get("arrival_time"), departure_time=params.get("departure_time"),
            activity=params.get("activity") or "",
            cargo_type=params.get("cargo_type"),
            cargo_quantity_mt=params.get("cargo_quantity_mt"),
            num_tug_operations=int(params.get("num_tug_operations") or 2),
            outside_working_hours=bool(params.get("outside_working_hours",False)),
            is_coaster=bool(params.get("is_coaster",False)),
            is_double_hull_tanker=bool(params.get("is_double_hull_tanker",False)),
            requested_due_types=params.get("requested_due_types") or [
                "light_dues","port_dues","towage_dues",
                "vts_dues","pilotage_dues","running_lines_dues",
            ],
            raw_query=raw_query, extraction_confidence=1.0,
        )

    @staticmethod
    def _normalise_port(raw: str) -> str:
        if not raw: return ""
        key = raw.lower().strip()
        if key in PORT_ALIASES: return PORT_ALIASES[key]
        for alias, canonical in PORT_ALIASES.items():
            if alias in key or key in alias: return canonical
        return raw.strip().title()
