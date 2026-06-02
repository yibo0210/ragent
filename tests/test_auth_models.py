import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.storage.database import Base
from backend.auth.models import Tenant, User, Role


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_create_tenant(db_session):
    tenant = Tenant(name="acme_corp", display_name="Acme Corporation")
    db_session.add(tenant)
    db_session.commit()
    assert tenant.id is not None
    assert tenant.name == "acme_corp"


def test_create_user_with_tenant(db_session):
    tenant = Tenant(name="acme", display_name="Acme")
    db_session.add(tenant)
    db_session.flush()
    user = User(
        username="alice",
        hashed_password="hashed",
        tenant_id=tenant.id,
        role="admin",
    )
    db_session.add(user)
    db_session.commit()
    assert user.tenant_id == tenant.id
    assert user.role == "admin"


def test_create_role(db_session):
    tenant = Tenant(name="acme", display_name="Acme")
    db_session.add(tenant)
    db_session.flush()
    role = Role(name="hr", tenant_id=tenant.id, access_level=2)
    db_session.add(role)
    db_session.commit()
    assert role.access_level == 2


def test_user_tenant_relationship(db_session):
    tenant = Tenant(name="acme", display_name="Acme")
    db_session.add(tenant)
    db_session.flush()
    user = User(username="bob", hashed_password="h", tenant_id=tenant.id, role="viewer")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    assert user.tenant.name == "acme"
