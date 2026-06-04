"""MemoryImportance: scoring with time decay for memory prioritization."""

from __future__ import annotations

import math
from datetime import datetime, timezone


class MemoryImportance:
    """Scores memory importance with recency + frequency + confidence."""

    def __init__(self, decay_days: float = 30.0):
        self.decay_days = decay_days

    def compute(self, base_score: float, created_at: str, access_count: int = 1) -> float:
        try:
            s = created_at.replace("Z", "+00:00")
            created = datetime.fromisoformat(s)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            days_elapsed = (now - created).total_seconds() / 86400.0
        except Exception:
            days_elapsed = 0

        recency_weight = math.exp(-days_elapsed / self.decay_days)
        frequency_weight = min(1.0, math.log(1 + access_count) / math.log(5))
        return base_score * (0.5 * recency_weight + 0.3 * frequency_weight + 0.2)
