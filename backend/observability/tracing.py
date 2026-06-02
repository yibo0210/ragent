"""OpenTelemetry 分布式追踪初始化。开发环境用 Console 导出器，生产环境切 OTLP gRPC。"""
import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

TRACING_ENABLED = os.getenv("OTEL_ENABLED", "false").lower() != "false"
OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
OTLP_HEADERS = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")

_provider = None


def init_tracing(app=None):
    global _provider
    if not TRACING_ENABLED:
        return None

    _provider = TracerProvider()

    if OTLP_ENDPOINT:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            headers = {}
            if OTLP_HEADERS:
                for pair in OTLP_HEADERS.split(","):
                    k, v = pair.strip().split("=", 1)
                    headers[k.strip()] = v.strip()
            exporter = OTLPSpanExporter(endpoint=OTLP_ENDPOINT, headers=headers or None)
        except ImportError:
            exporter = ConsoleSpanExporter()
    else:
        exporter = ConsoleSpanExporter()

    _provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(_provider)
    return _provider


def get_tracer(name: str = "ragent"):
    return trace.get_tracer(name)


def current_span():
    return trace.get_current_span()
