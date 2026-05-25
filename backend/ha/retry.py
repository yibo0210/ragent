"""指数退避重试装饰器。"""
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from backend.observability import get_logger

log = get_logger("ragent.ha")


def with_retry(max_attempts: int = 3, min_wait: float = 1.0, max_wait: float = 10.0):
    """指数退避重试：适用于网络抖动场景。"""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(min=min_wait, max=max_wait),
        retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
        before_sleep=lambda retry_state: log.warning(
            "retry_attempt",
            attempt=retry_state.attempt_number,
            exception=str(retry_state.outcome.exception())[:200],
        ),
    )
