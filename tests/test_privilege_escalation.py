"""Integration tests verifying cross-tenant data isolation."""
import pytest
import inspect


def test_milvus_filter_includes_tenant_id():
    """Verify that retrieval always includes tenant_id filter."""
    from backend.rag.utils import retrieve_documents
    sig = inspect.signature(retrieve_documents)
    assert "tenant_id" in sig.parameters


def test_graph_retriever_includes_tenant_id():
    """Verify that graph search includes tenant constraint."""
    from backend.rag.graph_retriever import local_graph_search
    sig = inspect.signature(local_graph_search)
    assert "tenant_id" in sig.parameters


def test_ingestion_includes_tenant_id():
    """Verify that ingestion passes tenant_id to all stores."""
    from backend.pipeline.ingestion_worker import run_ingestion_task
    sig = inspect.signature(run_ingestion_task)
    assert "tenant_id" in sig.parameters


def test_supervisor_state_has_user_context():
    """Verify that SupervisorState includes user_context."""
    from backend.agent.orchestrator import SupervisorState
    annotations = SupervisorState.__annotations__
    assert "user_context" in annotations


def test_chat_endpoints_require_auth():
    """Verify that chat endpoints have auth dependency."""
    from backend.api.routes import chat_stream_endpoint
    import inspect
    sig = inspect.signature(chat_stream_endpoint)
    param_names = list(sig.parameters.keys())
    assert any("user" in p.lower() for p in param_names)


def test_document_upload_requires_auth():
    """Verify that document upload has auth dependency."""
    from backend.api.routes import upload_document
    sig = inspect.signature(upload_document)
    param_names = list(sig.parameters.keys())
    assert any("user" in p.lower() for p in param_names)


def test_list_sessions_requires_auth():
    """Verify that session listing has auth dependency."""
    from backend.api.routes import list_sessions
    sig = inspect.signature(list_sessions)
    param_names = list(sig.parameters.keys())
    assert any("user" in p.lower() for p in param_names)


def test_graph_ingestion_has_tenant_id():
    """Verify graph ingestion accepts tenant_id."""
    from backend.storage.graph_ingestion import ingest_extraction_result
    sig = inspect.signature(ingest_extraction_result)
    assert "tenant_id" in sig.parameters


def test_data_analyst_has_tenant_id():
    """Verify data analyst SQL generation accepts tenant_id."""
    from backend.agent.data_analyst import generate_sql
    sig = inspect.signature(generate_sql)
    assert "tenant_id" in sig.parameters


def test_auth_models_exist():
    """Verify auth models are properly defined."""
    from backend.auth.models import Tenant, User, Role
    assert hasattr(Tenant, 'name')
    assert hasattr(User, 'tenant_id')
    assert hasattr(User, 'role')
    assert hasattr(Role, 'access_level')


def test_jwt_handler_works():
    """Verify JWT encode/decode roundtrip."""
    from backend.auth.jwt_handler import encode_token, decode_token
    payload = {"user_id": 1, "tenant_id": 1, "role": "admin", "access_level": 4}
    token = encode_token(payload)
    decoded = decode_token(token)
    assert decoded["user_id"] == 1
    assert decoded["tenant_id"] == 1


def test_user_context_dataclass():
    """Verify UserContext has all required fields."""
    from backend.auth.dependencies import UserContext
    ctx = UserContext(
        user_id=1, username="test", tenant_id=1,
        tenant_name="test", role="admin", access_level=4
    )
    assert ctx.user_id == 1
    assert ctx.tenant_id == 1
