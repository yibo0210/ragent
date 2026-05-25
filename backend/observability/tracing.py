"""OpenTelemetry 分布式追踪初始化。使用 Console 导出器（开发环境），生产环境切换 OTLP。"""
import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

TRACING_ENABLED = os.getenv("OTEL_ENABLED", "false").lower() != "false"

_provider = None


def init_tracing(app=None):
    global _provider
    if not TRACING_ENABLED:
        return None

    _provider = TracerProvider()
    exporter = ConsoleSpanExporter()
    _provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(_provider)

    return _provider


def get_tracer(name: str = "ragent"):
    return trace.get_tracer(name)


def current_span():
    return trace.get_current_span()
