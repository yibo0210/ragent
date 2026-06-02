import pytest
import inspect
from backend.milvus.client import MilvusManager


def test_milvus_schema_has_tenant_id():
    source = inspect.getsource(MilvusManager.init_collection)
    assert "tenant_id" in source


def test_hybrid_retrieve_accepts_filter_expr():
    sig = inspect.signature(MilvusManager.hybrid_retrieve)
    assert "filter_expr" in sig.parameters
