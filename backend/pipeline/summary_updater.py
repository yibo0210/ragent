"""定向摘要更新器 — 只对 dirty 社区重新生成摘要。

由增量聚类触发或定时执行，扫描 is_dirty=True 的社区，
调用 LLM 生成摘要并更新 Milvus 向量。
"""
from backend.observability import get_logger
from backend.graph.community import update_dirty_summaries, get_community_count

log = get_logger("pipeline.summary_updater")


def run_summary_update_cycle() -> dict:
    """Run one cycle of dirty summary updates. Returns stats."""
    counts = get_community_count()
    log.info("summary_cycle_start", **counts)

    if counts["dirty"] == 0:
        return {"updated": 0, "skipped": True, **counts}

    updated = update_dirty_summaries()

    log.info("summary_cycle_complete", updated=updated, remaining_dirty=counts["dirty"] - updated)
    return {"updated": updated, "skipped": False, **counts}
