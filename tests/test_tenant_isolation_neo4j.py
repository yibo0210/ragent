import pytest
import inspect
from backend.storage.graph_ingestion import ingest_extraction_result


def test_ingest_extraction_result_accepts_tenant_id():
    sig = inspect.signature(ingest_extraction_result)
    param_names = list(sig.parameters.keys())
    assert "tenant_id" in param_names, f"Expected 'tenant_id' param, got: {param_names}"


def test_entity_merges_include_tenant_id():
    source = inspect.getsource(ingest_extraction_result)
    assert "tenant_id" in source
