"""Memory Graph data models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    TASK = "task"
    RELATION = "relation"


class MemoryNode(BaseModel):
    memory_id: str = ""
    memory_type: MemoryType = MemoryType.FACT
    content: str = ""
    subject: str = ""
    object_entity: str = ""
    predicate: str = ""
    importance: float = 0.5
    session_id: str = ""
    tenant_id: int = 0
    user_id: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class MemoryExtraction(BaseModel):
    memories: list[MemoryNode] = Field(default_factory=list)
    summary: str = ""
