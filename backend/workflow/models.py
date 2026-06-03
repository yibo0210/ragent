"""Workflow ORM models for MySQL persistence."""

from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, String, Text, Float, DateTime, ForeignKey, JSON, Index,
)
from sqlalchemy.orm import relationship

from backend.storage.database import Base


class WorkflowDefinition(Base):
    """Stored workflow plan — can be from Planner or a saved template."""

    __tablename__ = "workflow_definitions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, default="")
    goal = Column(Text, nullable=False)
    steps_json = Column(JSON, nullable=False)
    reasoning = Column(Text, default="")
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    created_by = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_wf_def_tenant", "tenant_id"),
    )


class WorkflowExecution(Base):
    """Runtime instance of a workflow execution."""

    __tablename__ = "workflow_executions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    execution_id = Column(String(64), unique=True, nullable=False, index=True)
    definition_id = Column(
        Integer, ForeignKey("workflow_definitions.id"), nullable=False
    )
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    session_id = Column(String(120), nullable=True)
    status = Column(
        String(20), nullable=False, default="pending", index=True
    )
    current_step_id = Column(String(64), nullable=True)
    progress = Column(Float, default=0.0)
    state_json = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    definition = relationship("WorkflowDefinition", lazy="joined")
    artifacts = relationship(
        "WorkflowArtifact", back_populates="execution", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_wf_exec_tenant_status", "tenant_id", "status"),
        Index("ix_wf_exec_execution_id", "execution_id"),
    )


class WorkflowArtifact(Base):
    """Output artifact from a workflow execution step."""

    __tablename__ = "workflow_artifacts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    execution_id = Column(
        Integer, ForeignKey("workflow_executions.id"), nullable=False, index=True
    )
    step_id = Column(String(64), nullable=False)
    artifact_type = Column(
        String(30), nullable=False
    )
    title = Column(String(500), nullable=False)
    mime_type = Column(String(100), default="text/markdown")
    content = Column(Text, nullable=True)
    file_path = Column(String(1024), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    execution = relationship("WorkflowExecution", back_populates="artifacts")

    __table_args__ = (
        Index("ix_wf_artifact_exec", "execution_id"),
    )
