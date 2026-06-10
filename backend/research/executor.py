# backend/research/executor.py
"""ResearchExecutor: executes a ResearchPlan DAG with evidence collection and review loop."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from backend.config import get_settings
from backend.research.schemas import (
    ResearchPlan, ResearchTask, ResearchTaskStatus,
    ResearchState, Evidence, ReviewResult, GapAnalysis,
)
from backend.research.research_agents import AGENT_MAP
from backend.research.evidence_store import get_evidence_store
from backend.research.models import ResearchExecution


class ResearchExecutor:
    """Executes a research plan with review-loop and evidence collection."""

    def __init__(self):
        self._settings = get_settings()

    async def execute(
        self,
        plan: ResearchPlan,
        tenant_id: int,
        user_id: int,
        session_id: str = "",
    ) -> ResearchState:
        """Execute a full research plan: collect -> review -> gap -> collect loop."""
        execution_id = f"rx_{uuid.uuid4().hex[:16]}"
        state = ResearchState(
            execution_id=execution_id,
            plan=plan,
            status=ResearchTaskStatus.RUNNING,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        # Persist execution record
        db_record_id = self._create_execution_record(
            execution_id, plan, tenant_id, user_id, session_id,
        )

        try:
            # --- Phase 1: Initial evidence collection ---
            await self._execute_all_tasks(state, tenant_id, user_id)

            # --- Phase 2: Review -> Gap -> Collect loop ---
            max_rounds = self._settings.research_max_review_rounds

            # Lazy imports for not-yet-created modules (Tasks 7, 8)
            from backend.research.reviewer import get_research_reviewer  # noqa: E402
            from backend.research.gap_analyzer import get_gap_analyzer  # noqa: E402

            reviewer = get_research_reviewer()
            gap_analyzer = get_gap_analyzer()

            for round_num in range(max_rounds):
                state.review_count = round_num + 1

                # Review evidence sufficiency
                review_result = await reviewer.review(state, plan)
                if review_result.is_sufficient:
                    break

                # Gap analysis
                gap = await gap_analyzer.analyze(state, review_result)
                state.gap_analyses.append(gap)

                # If no meaningful gaps found, stop looping
                if not gap or not any(
                    g.missing_aspect for g in [gap] if g and g.missing_aspect
                ):
                    break

            # --- Phase 3: Finalize ---
            state.status = ResearchTaskStatus.COMPLETED
            state.completed_at = datetime.now(timezone.utc).isoformat()
            state.progress = 100.0

            # Persist evidence to DB
            store = get_evidence_store()
            if state.evidence:
                store.save_batch(state.evidence, db_record_id)

            state = await self._generate_report(state, tenant_id, user_id, db_record_id)

            # Update execution record
            self._update_execution_record(db_record_id, state)

        except Exception as e:
            state.status = ResearchTaskStatus.FAILED
            state.error_message = str(e)
            self._update_execution_record(db_record_id, state)

        return state

    async def _execute_all_tasks(
        self, state: ResearchState, tenant_id: int, user_id: int,
    ) -> None:
        """Execute tasks respecting DAG dependencies (parallel where possible)."""
        plan = state.plan
        if not plan:
            return

        total_tasks = len(plan.tasks)
        completed: set[str] = set()

        while len(completed) < total_tasks:
            # Find tasks whose dependencies are all completed
            ready = [
                t for t in plan.tasks
                if t.task_id not in completed
                and all(dep in completed for dep in t.dependencies)
            ]

            if not ready:
                break  # circular dependency guard

            # Execute ready tasks in parallel
            results = await asyncio.gather(
                *[self._execute_single_task(t, state.task_results, tenant_id, user_id)
                  for t in ready],
                return_exceptions=True,
            )

            for task, result in zip(ready, results):
                if isinstance(result, Exception):
                    task.status = ResearchTaskStatus.FAILED
                    state.task_results[task.task_id] = {"error": str(result)}
                else:
                    findings, evidence_list = result
                    task.status = ResearchTaskStatus.COMPLETED
                    state.task_results[task.task_id] = {"finding": findings}
                    for ev in evidence_list:
                        ev.task_id = task.task_id
                    state.evidence.extend(evidence_list)

                completed.add(task.task_id)

            state.progress = (len(completed) / total_tasks) * 80.0  # 80% for collection
            state.completed_tasks = list(completed)

    async def _execute_single_task(
        self,
        task: ResearchTask,
        task_results: dict[str, dict],
        tenant_id: int,
        user_id: int,
    ) -> tuple[str, list[Evidence]]:
        """Execute one research task via the appropriate agent."""
        agent_func = AGENT_MAP.get(task.agent)
        if not agent_func:
            return f"Unknown agent: {task.agent}", []

        try:
            result = await asyncio.wait_for(
                agent_func(task.name, task.query, task_results),
                timeout=task.timeout,
            )
            return result
        except asyncio.TimeoutError:
            return f"Task timed out after {task.timeout}s", []
        except Exception as e:
            return f"Task failed: {str(e)}", []

    async def _generate_report(
        self, state: ResearchState, tenant_id: int, user_id: int, db_record_id: int,
    ) -> ResearchState:
        """Generate research report after evidence collection is complete."""
        try:
            # Lazy import for not-yet-created module (Task 9)
            from backend.research.report_generator import get_report_generator  # noqa: E402

            generator = get_report_generator()
            report = await generator.generate(state, tenant_id, user_id)
            # Persist report
            self._save_report(db_record_id, report)
        except Exception:
            pass  # Report generation is best-effort
        return state

    def _create_execution_record(
        self, execution_id: str, plan: ResearchPlan,
        tenant_id: int, user_id: int, session_id: str,
    ) -> int:
        from backend.storage.database import SessionLocal
        db = SessionLocal()
        try:
            record = ResearchExecution(
                execution_id=execution_id,
                tenant_id=tenant_id,
                user_id=user_id,
                session_id=session_id,
                goal=plan.goal,
                plan_json=plan.model_dump(),
                status="running",
            )
            db.add(record)
            db.commit()
            return record.id
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _update_execution_record(self, record_id: int, state: ResearchState):
        from backend.storage.database import SessionLocal
        db = SessionLocal()
        try:
            record = db.query(ResearchExecution).filter(
                ResearchExecution.id == record_id
            ).first()
            if record:
                record.status = state.status.value
                record.progress = state.progress
                record.review_count = state.review_count
                record.error_message = state.error_message
                if state.completed_at:
                    record.completed_at = datetime.fromisoformat(
                        state.completed_at.replace("Z", "+00:00")
                    )
                db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()

    def _save_report(self, db_record_id: int, report):
        from backend.storage.database import SessionLocal
        from backend.research.models import ResearchReportRecord

        # Handle both Pydantic model and dict
        if hasattr(report, "model_dump"):
            report_dict = report.model_dump()
        elif isinstance(report, dict):
            report_dict = report
        else:
            return

        db = SessionLocal()
        try:
            record = ResearchReportRecord(
                report_id=report_dict.get("report_id", ""),
                execution_id=db_record_id,
                title=report_dict.get("title", ""),
                format=report_dict.get("format", "markdown"),
                content=report_dict.get("content", ""),
                evidence_map_json=report_dict.get("evidence_map", {}),
                summary=report_dict.get("executive_summary", ""),
            )
            db.add(record)
            db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()


_executor: ResearchExecutor | None = None


def get_research_executor() -> ResearchExecutor:
    global _executor
    if _executor is None:
        _executor = ResearchExecutor()
    return _executor
