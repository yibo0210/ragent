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
    Hypothesis, HypothesisStatus, EvidenceNode, EvidenceRelationType,
    ConflictDetection, ExpandedQuestion,
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
        execution_id: str,
        plan: ResearchPlan,
        tenant_id: int,
        user_id: int,
        session_id: str = "",
    ) -> ResearchState:
        """Execute a full research plan: collect -> review -> gap -> collect loop."""
        state = ResearchState(
            execution_id=execution_id,
            plan=plan,
            status=ResearchTaskStatus.RUNNING,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        # Look up the DB record created by the route handler
        db_record_id = self._get_execution_record_id(execution_id)

        try:
            # --- Phase 1: Initial evidence collection ---
            await self._execute_all_tasks(state, tenant_id, user_id, db_record_id)

            # --- Phase 1.5: Hypothesis-Driven Dynamic Research (v21) ---
            if plan and state.evidence:
                from backend.research.hypothesis_generator import get_hypothesis_generator
                from backend.research.evidence_graph import get_evidence_graph
                from backend.research.conflict_detector import get_conflict_detector
                from backend.research.question_expander import get_question_expander
                from backend.research.confidence_estimator import get_confidence_estimator

                hg = get_hypothesis_generator()
                eg = get_evidence_graph()
                cd = get_conflict_detector()
                qe = get_question_expander()
                ce = get_confidence_estimator()

                # 1. Generate hypotheses
                existing_findings = " ".join(
                    r.get("finding", "")[:500] for r in state.task_results.values()
                )[:3000]
                state.hypotheses = await hg.generate(plan.goal, context=existing_findings)
                for h in state.hypotheses:
                    eg.add_hypothesis(h)

                # 2. Convert evidence to graph nodes
                evidence_nodes = []
                for ev in state.evidence:
                    node = EvidenceNode(
                        content=ev.content, source=ev.source.value,
                        citation=ev.citation,
                        confidence={"high": 0.9, "medium": 0.6, "low": 0.3}.get(ev.confidence.value, 0.5),
                        task_id=ev.task_id, execution_id=state.execution_id,
                    )
                    eg.add_evidence(node)
                    evidence_nodes.append(node)

                # 3. Link evidence to hypotheses (simple keyword overlap)
                for node in evidence_nodes:
                    for hyp in state.hypotheses:
                        keywords = hyp.statement[:30].replace("是", " ").replace("的", " ").split()
                        if any(kw in node.content for kw in keywords if len(kw) >= 2):
                            eg.link_to_hypothesis(node.node_id, hyp.hypothesis_id, EvidenceRelationType.SUPPORTS)

                # 4. Detect conflicts (only across different hypotheses, skip all-pairs)
                if len(evidence_nodes) >= 2 and len(state.hypotheses) >= 2:
                    evidence_by_hyp = {}
                    for node in evidence_nodes:
                        hid = node.hypothesis_id or "unknown"
                        evidence_by_hyp.setdefault(hid, []).append(node)
                    hyp_ids = list(evidence_by_hyp.keys())
                    conflicts = []
                    for i in range(len(hyp_ids)):
                        for j in range(i + 1, len(hyp_ids)):
                            for ev_a in evidence_by_hyp[hyp_ids[i]]:
                                for ev_b in evidence_by_hyp[hyp_ids[j]]:
                                    if len(conflicts) >= 10:  # max 10 conflicts total
                                        break
                                    c = await cd.detect(ev_a, ev_b)
                                    if c.has_conflict:
                                        conflicts.append(c)
                                        eg.link_evidence_pair(c.evidence_a, c.evidence_b, EvidenceRelationType.REFUTES, c.explanation)
                                if len(conflicts) >= 10:
                                    break
                    state.conflicts = conflicts

                # 5. Re-estimate confidence
                conflict_pairs = {(c.evidence_a, c.evidence_b) for c in state.conflicts if c.has_conflict}
                evidence_nodes = ce.batch_estimate(evidence_nodes, conflict_pairs)
                state.evidence_graph = evidence_nodes

                # 6. Generate expanded questions + dynamic follow-up (1 round for MVP)
                expanded = await qe.expand(
                    goal=plan.goal, hypotheses=state.hypotheses,
                    conflicts=state.conflicts, verified_hypotheses=state.completed_tasks,
                )
                state.expanded_questions = expanded

                if expanded and state.dynamic_round < state.max_dynamic_rounds:
                    state.dynamic_round += 1
                    top_q = max(expanded, key=lambda q: q.priority)
                    task = ResearchTask(
                        task_id=f"DX{state.dynamic_round}",
                        name=f"追问: {top_q.question[:40]}",
                        agent="web", query=top_q.question, dependencies=[], timeout=60,
                    )
                    findings, new_evidence = await self._execute_single_task(
                        task, state.task_results, tenant_id, user_id,
                    )
                    state.task_results[task.task_id] = {"finding": findings}
                    for ev in new_evidence:
                        node = EvidenceNode(
                            content=ev.content, source=ev.source.value,
                            citation=ev.citation,
                            confidence={"high": 0.9, "medium": 0.6, "low": 0.3}.get(ev.confidence.value, 0.5),
                            task_id=task.task_id, execution_id=state.execution_id,
                        )
                        eg.add_evidence(node)
                        evidence_nodes.append(node)
                        if top_q.target_hypothesis:
                            eg.link_to_hypothesis(node.node_id, top_q.target_hypothesis, EvidenceRelationType.SUPPORTS)
                    state.evidence_graph = evidence_nodes

                state.progress = 85.0
                self._update_execution_record(db_record_id, state)

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
        self, state: ResearchState, tenant_id: int, user_id: int, db_record_id: int,
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
            self._update_execution_record(db_record_id, state)  # persist progress immediately

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

    def _get_execution_record_id(self, execution_id: str) -> int:
        from backend.storage.database import SessionLocal
        db = SessionLocal()
        try:
            record = db.query(ResearchExecution).filter(
                ResearchExecution.execution_id == execution_id
            ).first()
            if record:
                return record.id
            return 0
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
