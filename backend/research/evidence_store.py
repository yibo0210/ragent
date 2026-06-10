# backend/research/evidence_store.py
"""EvidenceStore: persists and queries research evidence."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import and_

from backend.storage.database import SessionLocal
from backend.research.schemas import Evidence, EvidenceSource, EvidenceConfidence
from backend.research.models import ResearchEvidence


class EvidenceStore:
    """Central evidence repository with persistence and query capabilities."""

    def save(self, evidence: Evidence, execution_record_id: int) -> bool:
        """Persist a single evidence item to MySQL."""
        if not evidence.id:
            evidence.id = f"ev_{uuid.uuid4().hex[:12]}"

        db = SessionLocal()
        try:
            record = ResearchEvidence(
                evidence_id=evidence.id,
                execution_id=execution_record_id,
                task_id=evidence.task_id,
                source=evidence.source.value,
                content=evidence.content,
                citation=evidence.citation,
                confidence=evidence.confidence.value,
                metadata_json=evidence.metadata,
            )
            db.add(record)
            db.commit()
            return True
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def save_batch(self, evidence_list: list[Evidence], execution_record_id: int) -> int:
        """Persist multiple evidence items. Returns count saved."""
        count = 0
        db = SessionLocal()
        try:
            for evidence in evidence_list:
                if not evidence.id:
                    evidence.id = f"ev_{uuid.uuid4().hex[:12]}"
                record = ResearchEvidence(
                    evidence_id=evidence.id,
                    execution_id=execution_record_id,
                    task_id=evidence.task_id,
                    source=evidence.source.value,
                    content=evidence.content,
                    citation=evidence.citation,
                    confidence=evidence.confidence.value,
                    metadata_json=evidence.metadata,
                )
                db.add(record)
                count += 1
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
        return count

    def get_by_execution(self, execution_record_id: int) -> list[Evidence]:
        """Retrieve all evidence for a research execution."""
        db = SessionLocal()
        try:
            rows = (
                db.query(ResearchEvidence)
                .filter(ResearchEvidence.execution_id == execution_record_id)
                .all()
            )
            return [self._to_domain(r) for r in rows]
        finally:
            db.close()

    def get_by_task(self, execution_record_id: int, task_id: str) -> list[Evidence]:
        """Retrieve evidence for a specific task."""
        db = SessionLocal()
        try:
            rows = (
                db.query(ResearchEvidence)
                .filter(
                    and_(
                        ResearchEvidence.execution_id == execution_record_id,
                        ResearchEvidence.task_id == task_id,
                    )
                )
                .all()
            )
            return [self._to_domain(r) for r in rows]
        finally:
            db.close()

    def get_stats(self, execution_record_id: int) -> dict:
        """Get evidence statistics for an execution."""
        db = SessionLocal()
        try:
            rows = (
                db.query(ResearchEvidence)
                .filter(ResearchEvidence.execution_id == execution_record_id)
                .all()
            )
            total = len(rows)
            sources = {}
            confidences = {"high": 0, "medium": 0, "low": 0}
            tasks_with_evidence = set()
            for r in rows:
                sources[r.source] = sources.get(r.source, 0) + 1
                confidences[r.confidence] = confidences.get(r.confidence, 0) + 1
                tasks_with_evidence.add(r.task_id)
            return {
                "total": total,
                "by_source": sources,
                "by_confidence": confidences,
                "tasks_covered": len(tasks_with_evidence),
            }
        finally:
            db.close()

    def _to_domain(self, record: ResearchEvidence) -> Evidence:
        return Evidence(
            id=record.evidence_id,
            task_id=record.task_id,
            source=EvidenceSource(record.source),
            content=record.content or "",
            citation=record.citation or "",
            confidence=EvidenceConfidence(record.confidence),
            metadata=record.metadata_json or {},
            created_at=record.created_at.isoformat() if record.created_at else "",
        )


_store: EvidenceStore | None = None


def get_evidence_store() -> EvidenceStore:
    global _store
    if _store is None:
        _store = EvidenceStore()
    return _store
