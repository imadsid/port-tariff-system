"""
monitoring/logger.py
Structured logging, Prometheus metrics, timing decorators.
"""
import functools
import logging
import time
from typing import Any, Callable


def get_logger(name: str):
    """Return a structlog logger. Imports structlog on first call only."""
    import structlog
    return structlog.get_logger(name)


def _ensure_configured():
    """Configure structlog once. Called lazily."""
    import structlog
    if structlog.is_configured():
        return
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(level=logging.INFO, format="%(message)s")


_ensure_configured()


# ── Prometheus Metrics (created lazily on first access) ───────────────────────

class _LazyMetric:
    """Wraps a Prometheus metric and creates it on first use."""
    def __init__(self, factory, *args, **kwargs):
        self._factory = factory
        self._args = args
        self._kwargs = kwargs
        self._metric = None

    def _get(self):
        if self._metric is None:
            self._metric = self._factory(*self._args, **self._kwargs)
        return self._metric

    def labels(self, **kw):
        return self._get().labels(**kw)

    def inc(self):
        self._get().inc()

    def observe(self, v):
        self._get().observe(v)

    def set(self, v):
        self._get().set(v)


def _make_counter(name, desc, labels=()):
    from prometheus_client import Counter
    return _LazyMetric(Counter, name, desc, list(labels))

def _make_histogram(name, desc, labels=(), buckets=None):
    from prometheus_client import Histogram
    kw = {}
    if buckets:
        kw["buckets"] = buckets
    return _LazyMetric(Histogram, name, desc, list(labels), **kw)

def _make_gauge(name, desc):
    from prometheus_client import Gauge
    return _LazyMetric(Gauge, name, desc)


CALC_REQUESTS      = _make_counter("port_tariff_calculation_requests_total", "Total calculation requests", ["port", "status"])
CALC_LATENCY       = _make_histogram("port_tariff_calculation_duration_seconds", "Latency", ["due_type"], [0.1, 0.5, 1.0, 2.5, 5.0, 10.0])
INGESTION_RUNS     = _make_counter("port_tariff_ingestion_runs_total", "Ingestion runs", ["status"])
CONFIDENCE_GAUGE   = _make_gauge("port_tariff_last_confidence_score", "Last confidence score")
GUARDRAIL_FAILURES = _make_counter("port_tariff_guardrail_failures_total", "Guardrail failures", ["check_type"])

metrics = {
    "calc_requests":      CALC_REQUESTS,
    "calc_latency":       CALC_LATENCY,
    "ingestion_runs":     INGESTION_RUNS,
    "confidence":         CONFIDENCE_GAUGE,
    "guardrail_failures": GUARDRAIL_FAILURES,
}


def start_metrics_server(port: int = 9090) -> None:
    try:
        from prometheus_client import start_http_server
        start_http_server(port)
        get_logger("monitoring").info("Prometheus metrics server started", port=port)
    except Exception as exc:
        get_logger("monitoring").warning("Could not start metrics server", error=str(exc))


def timed(label: str = "unknown") -> Callable:
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                CALC_LATENCY.labels(due_type=label).observe(time.perf_counter() - t0)
        return wrapper
    return decorator


def async_timed(label: str = "unknown") -> Callable:
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return await fn(*args, **kwargs)
            finally:
                CALC_LATENCY.labels(due_type=label).observe(time.perf_counter() - t0)
        return wrapper
    return decorator


logger = get_logger("port_tariff")
