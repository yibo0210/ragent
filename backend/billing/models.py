from sqlalchemy import Column, Integer, String, DateTime, Text, BigInteger, Index
from datetime import datetime, timezone
from backend.storage.database import Base


class TokenUsageLog(Base):
    __tablename__ = "token_usage_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(Integer, nullable=False, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    session_id = Column(String(120), nullable=True)
    model_name = Column(String(100), nullable=False)
    prompt_tokens = Column(Integer, nullable=False, default=0)
    completion_tokens = Column(Integer, nullable=False, default=0)
    agent_name = Column(String(50), nullable=True)
    request_type = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens

    __table_args__ = (
        Index("ix_token_usage_tenant_created", "tenant_id", "created_at"),
    )


class RateLimitRule(Base):
    __tablename__ = "rate_limit_rules"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(Integer, nullable=False, unique=True, index=True)
    tier = Column(String(50), nullable=False, default="free")
    qps_limit = Column(Integer, nullable=False, default=10)
    daily_token_limit = Column(BigInteger, nullable=False, default=100000)
    concurrent_limit = Column(Integer, nullable=False, default=5)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(Integer, nullable=False, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    session_id = Column(String(120), nullable=True)
    action = Column(String(100), nullable=False)
    target = Column(String(255), nullable=True)
    arguments = Column(Text, nullable=True)
    result_summary = Column(String(500), nullable=True)
    risk_level = Column(String(20), nullable=False, default="low")
    ip_address = Column(String(45), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_audit_tenant_created", "tenant_id", "created_at"),
    )
