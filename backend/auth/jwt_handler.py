import jwt
from datetime import datetime, timezone, timedelta
from passlib.context import CryptContext
from backend.config import get_settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _get_jwt_config():
    s = get_settings()
    secret = s.jwt_secret
    if not secret:
        raise RuntimeError("JWT_SECRET is not set. Refusing to start with insecure default.")
    return secret, s.jwt_algorithm, s.jwt_expire_minutes


def encode_token(payload: dict, expires_seconds: int = None) -> str:
    secret, algorithm, expire_minutes = _get_jwt_config()
    to_encode = payload.copy()
    expire = datetime.now(timezone.utc) + timedelta(
        seconds=expires_seconds if expires_seconds is not None else expire_minutes * 60
    )
    to_encode["exp"] = expire
    return jwt.encode(to_encode, secret, algorithm=algorithm)


def decode_token(token: str) -> dict:
    secret, algorithm, _ = _get_jwt_config()
    return jwt.decode(token, secret, algorithms=[algorithm])


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)
