from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel
from backend.storage.database import get_db
from backend.auth.models import Tenant, User
from backend.auth.jwt_handler import hash_password, verify_password, encode_token

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    username: str
    password: str
    tenant_name: str
    display_name: str = ""
    role: str = "viewer"


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/register", status_code=201)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == req.username).first():
        raise HTTPException(status_code=409, detail="Username already exists")
    tenant = db.query(Tenant).filter(Tenant.name == req.tenant_name).first()
    if not tenant:
        tenant = Tenant(name=req.tenant_name, display_name=req.display_name or req.tenant_name)
        db.add(tenant)
        db.flush()
    access_level_map = {"admin": 4, "editor": 3, "viewer": 1}
    user = User(
        username=req.username,
        hashed_password=hash_password(req.password),
        tenant_id=tenant.id,
        role=req.role,
        access_level=access_level_map.get(req.role, 1),
    )
    db.add(user)
    db.commit()
    return {"username": user.username, "tenant": tenant.name, "role": user.role}


@router.post("/token", response_model=TokenResponse)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form.username).first()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = encode_token({
        "sub": user.username,
        "user_id": user.id,
        "tenant_id": user.tenant_id,
        "tenant_name": user.tenant.name,
        "role": user.role,
        "access_level": user.access_level,
    })
    return TokenResponse(access_token=token)
