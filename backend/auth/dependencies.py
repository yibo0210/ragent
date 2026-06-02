from dataclasses import dataclass
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from backend.auth.jwt_handler import decode_token
from backend.storage.database import get_db

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)


@dataclass
class UserContext:
    user_id: int
    username: str
    tenant_id: int
    tenant_name: str
    role: str
    access_level: int  # 1=public, 2=internal, 3=confidential, 4=secret


def _get_current_user(
    token: str,
    db: Session,
) -> UserContext:
    """Core auth logic — called directly in tests, wrapped by get_current_user for FastAPI."""
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = decode_token(token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return UserContext(
        user_id=payload["user_id"],
        username=payload.get("sub", ""),
        tenant_id=payload["tenant_id"],
        tenant_name=payload.get("tenant_name", ""),
        role=payload["role"],
        access_level=payload["access_level"],
    )


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> UserContext:
    """FastAPI dependency: extract and validate the current user from JWT token."""
    return _get_current_user(token, db)
