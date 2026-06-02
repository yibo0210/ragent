import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.storage.database import Base
from backend.storage.models import DocumentIndex, ParentChunk, ChatSession
from backend.auth.models import Tenant, User


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def two_tenants(db_session):
    t1 = Tenant(name="acme", display_name="Acme")
    t2 = Tenant(name="globex", display_name="Globex")
    db_session.add_all([t1, t2])
    db_session.flush()
    return t1, t2


def test_document_index_has_tenant_id(db_session, two_tenants):
    t1, t2 = two_tenants
    doc = DocumentIndex(filename="report.pdf", file_hash="abc", tenant_id=t1.id)
    db_session.add(doc)
    db_session.commit()
    assert doc.tenant_id == t1.id


def test_document_index_tenant_filter(db_session, two_tenants):
    t1, t2 = two_tenants
    db_session.add(DocumentIndex(filename="a.pdf", file_hash="a", tenant_id=t1.id))
    db_session.add(DocumentIndex(filename="b.pdf", file_hash="b", tenant_id=t2.id))
    db_session.commit()
    t1_docs = db_session.query(DocumentIndex).filter(DocumentIndex.tenant_id == t1.id).all()
    assert len(t1_docs) == 1
    assert t1_docs[0].filename == "a.pdf"


def test_chat_session_has_tenant_id(db_session, two_tenants):
    t1, _ = two_tenants
    session = ChatSession(session_id="s1", tenant_id=t1.id)
    db_session.add(session)
    db_session.commit()
    assert session.tenant_id == t1.id
