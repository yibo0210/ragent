import pytest
from backend.auth.jwt_handler import encode_token, decode_token, hash_password, verify_password


def test_encode_decode_token():
    payload = {"user_id": 1, "tenant_id": 1, "role": "admin", "access_level": 4}
    token = encode_token(payload)
    decoded = decode_token(token)
    assert decoded["user_id"] == 1
    assert decoded["tenant_id"] == 1
    assert decoded["role"] == "admin"


def test_decode_expired_token():
    import time
    payload = {"user_id": 1, "tenant_id": 1, "role": "admin", "access_level": 4}
    token = encode_token(payload, expires_seconds=0)
    time.sleep(1)
    with pytest.raises(Exception):
        decode_token(token)


def test_decode_invalid_token():
    with pytest.raises(Exception):
        decode_token("invalid.token.here")


def test_password_hash_and_verify():
    hashed = hash_password("mypassword")
    assert verify_password("mypassword", hashed) is True
    assert verify_password("wrongpassword", hashed) is False
