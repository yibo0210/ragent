import os
import jwt
from datetime import datetime, timezone, timedelta
from passlib.context import CryptContext

JWT_SECRET = os.getenv("JWT_SECRET", "ragent-ai-dev-secret-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "24"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def encode_token(payload: dict, expires_seconds: int = None) -> str:
    to_encode = payload.copy()
    expire = datetime.now(timezone.utc) + timedelta(
        seconds=expires_seconds if expires_seconds is not None else JWT_EXPIRE_HOURS * 3600
    )
    to_encode["exp"] = expire
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)
