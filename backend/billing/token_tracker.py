from sqlalchemy.orm import Session
from backend.billing.models import TokenUsageLog
from datetime import datetime, timezone, timedelta


def record_token_usage(
    db: Session, tenant_id: int, user_id: int,
    model_name: str, prompt_tokens: int, completion_tokens: int,
    session_id: str = None, agent_name: str = None, request_type: str = None,
) -> TokenUsageLog:
    log = TokenUsageLog(
        tenant_id=tenant_id, user_id=user_id, session_id=session_id,
        model_name=model_name, prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens, agent_name=agent_name,
        request_type=request_type,
    )
    db.add(log)
    db.commit()
    return log


def get_usage_summary(db: Session, tenant_id: int, days: int = 30) -> dict:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    logs = db.query(TokenUsageLog).filter(
        TokenUsageLog.tenant_id == tenant_id,
        TokenUsageLog.created_at >= since,
    ).all()
    total_prompt = sum(l.prompt_tokens for l in logs)
    total_completion = sum(l.completion_tokens for l in logs)
    return {
        "tenant_id": tenant_id,
        "period_days": days,
        "request_count": len(logs),
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
    }
