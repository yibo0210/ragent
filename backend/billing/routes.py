from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from backend.storage.database import get_db
from backend.auth.dependencies import UserContext, get_current_user
from backend.billing.token_tracker import get_usage_summary
from backend.billing.models import AuditLog
from backend.schemas import TokenUsageSummary, AuditLogListResponse, AuditLogEntry
from datetime import datetime as dt

router = APIRouter(prefix="/billing", tags=["billing"])


@router.get("/usage", response_model=TokenUsageSummary)
def get_usage(
    days: int = Query(30, ge=1, le=365),
    user: UserContext = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return get_usage_summary(db, tenant_id=user.tenant_id, days=days)


@router.get("/audit", response_model=AuditLogListResponse)
def get_audit_logs(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    action: str = Query(None),
    user: UserContext = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(AuditLog).filter(AuditLog.tenant_id == user.tenant_id)
    if action:
        query = query.filter(AuditLog.action == action)
    total = query.count()
    logs = query.order_by(AuditLog.created_at.desc()).offset(offset).limit(limit).all()
    return AuditLogListResponse(
        logs=[AuditLogEntry(
            id=l.id, tenant_id=l.tenant_id, user_id=l.user_id,
            action=l.action, target=l.target, result_summary=l.result_summary,
            risk_level=l.risk_level, created_at=l.created_at,
        ) for l in logs],
        total=total,
    )
