"""
config/settings.py
Central configuration — reads from environment variables and .env file.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings:

    def __init__(self):
        self._load()

    def _load(self):
        import os
        env_file = BASE_DIR / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip())

        self.groq_api_key   = os.environ.get("GROQ_API_KEY", "")
        self.groq_model     = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

        # Embedding model (local, no API key needed)
        self.embedding_model = os.environ.get(
            "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )

        self.chroma_persist_dir     = str(BASE_DIR / "data" / "chroma_db")
        self.chroma_collection_name = "port_tariff_chunks"
        self.json_store_path        = str(BASE_DIR / "data" / "tariff_rules.json")
        self.sqlite_db_path         = str(BASE_DIR / "data" / "tariff.db")

        self.api_host    = os.environ.get("API_HOST", "0.0.0.0")
        self.api_port    = int(os.environ.get("API_PORT", "8000"))
        self.api_title   = "Port Tariff Calculation API"
        self.api_version = "1.0.0"

        self.chunk_size    = 800
        self.chunk_overlap = 150

        self.metrics_port       = int(os.environ.get("METRICS_PORT", "9090"))
        self.log_level          = os.environ.get("LOG_LEVEL", "INFO")
        self.alert_threshold_ms = 5000.0

        self.min_confidence_threshold = 0.6
        self.vat_rate = 0.15

        self.supported_ports = [
            "Durban", "Cape Town", "Richards Bay", "Port Elizabeth",
            "Ngqura", "East London", "Saldanha", "Mossel Bay",
        ]


settings = Settings()
