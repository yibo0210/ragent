"""Research ORM models for MySQL persistence."""

from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, String, Text, Float, DateTime, ForeignKey, JSON, Index,
)
from sqlalchemy.orm import relationship

from backend.storage.database import Base


class ResearchExecution(Base):
    """Tracks a full research execution lifecycle."""

    __tablename__ = "research_executions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    execution_id = Column(String(64), unique=True, nullable=False, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    session_id = Column(String(120), nullable=True)
    goal = Column(Text, nullable=False)
    plan_json = Column(JSON, nullable=True)
    status = Column(String(20), nullable=False, default="pending", index=True)
    progress = Column(Float, default=0.0)
    review_count = Column(Integer, default=0)
    reviewer_result_json = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    evidence_items = relationship(
        "ResearchEvidence", back_populates="execution", cascade="all, delete-orphan"
    )
    reports = relationship(
        "ResearchReportRecord", back_populates="execution", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_research_exec_tenant_status", "tenant_id", "status"),
    )


class ResearchEvidence(Base):
    """Persisted evidence item collected during research."""

    __tablename__ = "research_evidence"

    id = Column(Integer, primary_key=True, autoincrement=True)
    evidence_id = Column(String(64), unique=True, nullable=False, index=True)
    execution_id = Column(
        Integer, ForeignKey("research_executions.id"), nullable=False, index=True
    )
    task_id = Column(String(32), nullable=False, index=True)
    source = Column(String(30), nullable=False)
    content = Column(Text, nullable=True)
    citation = Column(String(2048), nullable=True)
    confidence = Column(String(10), nullable=False, default="medium")
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    execution = relationship("ResearchExecution", back_populates="evidence_items")

    __table_args__ = (
        Index("ix_research_evidence_exec_task", "execution_id", "task_id"),
    )


class ResearchReportRecord(Base):
    """Persisted research report."""

    __tablename__ = "research_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(String(64), unique=True, nullable=False, index=True)
    execution_id = Column(
        Integer, ForeignKey("research_executions.id"), nullable=False, index=True
    )
    title = Column(String(500), nullable=False)
    format = Column(String(20), nullable=False, default="markdown")
    content = Column(Text, nullable=True)
    file_path = Column(String(1024), nullable=True)
    evidence_map_json = Column(JSON, nullable=True)
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    execution = relationship("ResearchExecution", back_populates="reports")

    __table_args__ = (
        Index("ix_research_report_exec", "execution_id"),
    )
