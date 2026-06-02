import pytest
from unittest.mock import patch, MagicMock
from backend.auth.dependencies import UserContext, _get_current_user
from backend.auth.jwt_handler import encode_token


@pytest.fixture
def mock_db():
    return MagicMock()


@pytest.fixture
def valid_token():
    return encode_token({
        "sub": "alice",
        "user_id": 1,
        "tenant_id": 1,
        "tenant_name": "acme",
        "role": "admin",
        "access_level": 4,
    })


def test_user_context_from_token(valid_token, mock_db):
    ctx = _get_current_user(valid_token, mock_db)
    assert ctx.user_id == 1
    assert ctx.tenant_id == 1
    assert ctx.tenant_name == "acme"
    assert ctx.role == "admin"
    assert ctx.access_level == 4


def test_user_context_missing_token(mock_db):
    with pytest.raises(Exception):
        _get_current_user(None, mock_db)


def test_user_context_invalid_token(mock_db):
    with pytest.raises(Exception):
        _get_current_user("bad.token.here", mock_db)
