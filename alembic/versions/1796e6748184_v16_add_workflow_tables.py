"""v16 add workflow tables

Revision ID: 1796e6748184
Revises:
Create Date: 2026-06-03 22:08:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "1796e6748184"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "workflow_definitions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("goal", sa.Text(), nullable=False),
        sa.Column("steps_json", sa.JSON(), nullable=False),
        sa.Column("reasoning", sa.Text(), nullable=True),
        sa.Column("tenant_id", sa.Integer(), nullable=False),
        sa.Column("created_by", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_wf_def_tenant", "workflow_definitions", ["tenant_id"])
    op.create_index(op.f("ix_workflow_definitions_name"), "workflow_definitions", ["name"])

    op.create_table(
        "workflow_executions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("execution_id", sa.String(64), nullable=False),
        sa.Column("definition_id", sa.Integer(), nullable=False),
        sa.Column("tenant_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.String(120), nullable=True),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("current_step_id", sa.String(64), nullable=True),
        sa.Column("progress", sa.Float(), nullable=True),
        sa.Column("state_json", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["definition_id"], ["workflow_definitions.id"]),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_wf_exec_tenant_status", "workflow_executions", ["tenant_id", "status"])
    op.create_index("ix_wf_exec_execution_id", "workflow_executions", ["execution_id"], unique=True)
    op.create_index(op.f("ix_workflow_executions_execution_id"), "workflow_executions", ["execution_id"])
    op.create_index(op.f("ix_workflow_executions_status"), "workflow_executions", ["status"])
    op.create_index(op.f("ix_workflow_executions_tenant_id"), "workflow_executions", ["tenant_id"])
    op.create_index(op.f("ix_workflow_executions_user_id"), "workflow_executions", ["user_id"])

    op.create_table(
        "workflow_artifacts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("execution_id", sa.Integer(), nullable=False),
        sa.Column("step_id", sa.String(64), nullable=False),
        sa.Column("artifact_type", sa.String(30), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("mime_type", sa.String(100), nullable=True),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("file_path", sa.String(1024), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["execution_id"], ["workflow_executions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_wf_artifact_exec", "workflow_artifacts", ["execution_id"])


def downgrade() -> None:
    op.drop_table("workflow_artifacts")
    op.drop_table("workflow_executions")
    op.drop_table("workflow_definitions")
