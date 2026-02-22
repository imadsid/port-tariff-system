"""
query_processor/parser.py
Query Processor using LangChain + ChatGroq.

Uses LangChain's ChatGroq wrapper and PromptTemplate to convert
natural language queries into structured VesselQuery objects.

LangChain benefits here:
- PromptTemplate for clean, reusable prompt management
- ChatGroq wrapper — swap to ChatOpenAI or ChatAnthropic in one line
- JsonOutputParser for reliable structured output parsing
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

# LangChain PromptTemplate — uses {query} and {vessel_json} as variables
_TEMPLATE = """You are a port tariff expert. Extract vessel and voyage parameters from the information below.

Return ONLY a valid JSON object with these exact keys (use null for missing values):
{{
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
}}

Return ONLY the JSON object. No explanation, no markdown, no code fences.

Natural language query:
{query}

Vessel profile (structured data):
{vessel_json}
"""


class QueryProcessor:
    """
    Parses vessel/voyage parameters into a VesselQuery.

    LangChain pipeline:
      PromptTemplate → ChatGroq → JsonOutputParser → VesselQuery
    """

    def __init__(self) -> None:
        self._llm    = None   # lazy ChatGroq instance
        self._chain  = None   # lazy LangChain chain

    def _get_llm(self):
        """Lazy-init LangChain ChatGroq."""
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=0,
            max_tokens=1024,
        )

    def _get_chain(self):
        """
        Build a LangChain chain:
          PromptTemplate | ChatGroq | JsonOutputParser
        The | operator is LangChain's LCEL (LangChain Expression Language)
        for composing chains declaratively.
        """
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import JsonOutputParser

        if self._llm is None:
            self._llm = self._get_llm()

        prompt = PromptTemplate(
            template=_TEMPLATE,
            input_variables=["query", "vessel_json"],
        )
        # LCEL chain: prompt → LLM → JSON parser
        return prompt | self._llm | JsonOutputParser()

    @property
    def chain(self):
        if self._chain is None:
            self._chain = self._get_chain()
        return self._chain

    def parse(self, query: str, vessel_data: Optional[dict[str, Any]] = None) -> VesselQuery:
        """
        Parse a natural language query into a VesselQuery using LangChain.
        Falls back to direct dict parsing if no query text provided.
        """
        if not query.strip() and vessel_data:
            return self._from_dict(vessel_data)

        log.info("Parsing NL query via LangChain ChatGroq", chars=len(query))
        vessel_json = json.dumps(vessel_data or {}, indent=2)

        try:
            # LangChain chain handles prompt formatting + LLM call + JSON parsing
            params = self.chain.invoke({
                "query":       query,
                "vessel_json": vessel_json,
            })
        except Exception:
            # JsonOutputParser failed — fall back to manual parsing
            log.warning("JsonOutputParser failed — falling back to manual parse")
            params = self._manual_parse(query, vessel_json)

        vq = self._build_vessel_query(params, query)
        log.info("Query parsed", port=vq.port, gt=vq.gross_tonnage)
        return vq

    def _manual_parse(self, query: str, vessel_json: str) -> dict:
        """Fallback: call LLM directly and strip fences manually."""
        from langchain_core.messages import HumanMessage
        if self._llm is None:
            self._llm = self._get_llm()
        filled = _TEMPLATE.replace("{query}", query).replace("{vessel_json}", vessel_json)
        response = self._llm.invoke([HumanMessage(content=filled)])
        raw = response.content.strip()
        raw = re.sub(r"^```json\s*|\s*```$", "", raw).strip()
        return json.loads(raw)

    def from_vessel_profile(self, profile: dict[str, Any], port: str) -> VesselQuery:
        """Build VesselQuery directly from structured vessel profile — no LLM needed."""
        meta = profile.get("vessel_metadata", {})
        tech = profile.get("technical_specs", {})
        ops  = profile.get("operational_data", {})
        return VesselQuery(
            vessel_name        =meta.get("name", ""),
            vessel_type        =tech.get("type", "General"),
            vessel_flag        =meta.get("flag", ""),
            gross_tonnage      =float(tech.get("gross_tonnage") or 0),
            net_tonnage        =float(tech.get("net_tonnage") or 0),
            dwt                =float(tech.get("dwt") or 0),
            loa_meters         =float(tech.get("loa_meters") or 0),
            port               =self._normalise_port(port),
            days_in_port       =float(ops.get("days_alongside") or 1.0),
            arrival_time       =ops.get("arrival_time"),
            departure_time     =ops.get("departure_time"),
            activity           =ops.get("activity", ""),
            cargo_quantity_mt  =ops.get("cargo_quantity_mt"),
            num_tug_operations =int(ops.get("num_operations") or 2),
            requested_due_types=[
                "light_dues", "port_dues", "towage_dues",
                "vts_dues", "pilotage_dues", "running_lines_dues",
            ],
            extraction_confidence=1.0,
        )

    def _from_dict(self, d: dict[str, Any]) -> VesselQuery:
        return VesselQuery(
            vessel_name          =d.get("name", ""),
            vessel_type          =d.get("type", "General"),
            vessel_flag          =d.get("flag", ""),
            gross_tonnage        =float(d.get("gross_tonnage") or 0),
            net_tonnage          =float(d.get("net_tonnage") or 0),
            dwt                  =float(d.get("dwt") or 0),
            loa_meters           =float(d.get("loa_meters") or 0),
            port                 =self._normalise_port(d.get("port", "")),
            days_in_port         =float(d.get("days_in_port") or 1),
            arrival_time         =d.get("arrival_time"),
            departure_time       =d.get("departure_time"),
            activity             =d.get("activity", ""),
            num_tug_operations   =int(d.get("num_tug_operations") or 2),
            outside_working_hours=bool(d.get("outside_working_hours", False)),
            is_coaster           =bool(d.get("is_coaster", False)),
            is_double_hull_tanker=bool(d.get("is_double_hull_tanker", False)),
            requested_due_types  =d.get("requested_due_types", [
                "light_dues", "port_dues", "towage_dues",
                "vts_dues", "pilotage_dues", "running_lines_dues",
            ]),
            extraction_confidence=1.0,
        )

    def _build_vessel_query(self, params: dict[str, Any], raw_query: str) -> VesselQuery:
        return VesselQuery(
            vessel_name          =params.get("vessel_name") or "",
            vessel_type          =params.get("vessel_type") or "General",
            vessel_flag          =params.get("vessel_flag") or "",
            gross_tonnage        =float(params.get("gross_tonnage") or 0),
            net_tonnage          =float(params.get("net_tonnage") or 0),
            dwt                  =float(params.get("dwt") or 0),
            loa_meters           =float(params.get("loa_meters") or 0),
            port                 =self._normalise_port(params.get("port") or ""),
            days_in_port         =float(params.get("days_in_port") or 1),
            arrival_time         =params.get("arrival_time"),
            departure_time       =params.get("departure_time"),
            activity             =params.get("activity") or "",
            cargo_type           =params.get("cargo_type"),
            cargo_quantity_mt    =params.get("cargo_quantity_mt"),
            num_tug_operations   =int(params.get("num_tug_operations") or 2),
            outside_working_hours=bool(params.get("outside_working_hours", False)),
            is_coaster           =bool(params.get("is_coaster", False)),
            is_double_hull_tanker=bool(params.get("is_double_hull_tanker", False)),
            requested_due_types  =params.get("requested_due_types") or [
                "light_dues", "port_dues", "towage_dues",
                "vts_dues", "pilotage_dues", "running_lines_dues",
            ],
            raw_query            =raw_query,
            extraction_confidence=1.0,
        )

    @staticmethod
    def _normalise_port(raw: str) -> str:
        if not raw:
            return ""
        key = raw.lower().strip()
        if key in PORT_ALIASES:
            return PORT_ALIASES[key]
        for alias, canonical in PORT_ALIASES.items():
            if alias in key or key in alias:
                return canonical
        return raw.strip().title()
