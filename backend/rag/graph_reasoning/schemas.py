"""Reasoning data models for the 5-stage pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ReasoningStrategy(str, Enum):
    FACTOID = "factoid"
    ENTITY_RELATION = "entity_relation"
    MULTI_HOP = "multi_hop"
    TEMPORAL = "temporal"
    COMPARISON = "comparison"


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    UNSUPPORTED = "UNSUPPORTED"


class ReasoningPlan(BaseModel):
    query_type: str = "factoid"
    start_entities: list[str] = Field(default_factory=list)
    target_relations: list[str] = Field(default_factory=list)
    max_hops: int = 3
    reasoning_strategy: ReasoningStrategy = ReasoningStrategy.FACTOID
    temporal_year: str = ""
    need_reasoning: bool = False


class ReasoningPath(BaseModel):
    nodes: list[str] = Field(default_factory=list)
    edges: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    hop_count: int = 0
    semantic_score: float = 0.0
    relation_confidence: float = 0.0
    temporal_consistency: float = 1.0
    path_score: float = 0.0


class VerificationResult(BaseModel):
    verdict: Verdict = Verdict.UNSUPPORTED
    confidence: float = 0.0
    explanation: str = ""
    supporting_paths: list[int] = Field(default_factory=list)
