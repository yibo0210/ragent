from sqlalchemy.orm import Session
from backend.billing.models import AuditLog


def log_audit_event(
    db: Session, tenant_id: int, user_id: int,
    action: str, target: str = None, arguments: str = None,
    result_summary: str = None, risk_level: str = "low",
    session_id: str = None, ip_address: str = None,
) -> AuditLog:
    log = AuditLog(
        tenant_id=tenant_id, user_id=user_id, session_id=session_id,
        action=action, target=target, arguments=arguments,
        result_summary=result_summary, risk_level=risk_level,
        ip_address=ip_address,
    )
    db.add(log)
    db.commit()
    return log


class AuditContext:
    """Context manager for audit logging with before/after semantics."""

    def __init__(self, db: Session, tenant_id: int, user_id: int,
                 action: str, target: str = None, arguments: str = None,
                 session_id: str = None, ip_address: str = None):
        self.db = db
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.action = action
        self.target = target
        self.arguments = arguments
        self.session_id = session_id
        self.ip_address = ip_address
        self.result_summary = None
        self.risk_level = "low"

    def set_result(self, summary: str, risk_level: str = "low"):
        self.result_summary = summary
        self.risk_level = risk_level

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.result_summary = f"ERROR: {exc_val}"
            self.risk_level = "high"
        log_audit_event(
            db=self.db, tenant_id=self.tenant_id, user_id=self.user_id,
            action=self.action, target=self.target, arguments=self.arguments,
            result_summary=self.result_summary, risk_level=self.risk_level,
            session_id=self.session_id, ip_address=self.ip_address,
        )
        return False
