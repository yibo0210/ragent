"""熔断器模式：外部 API 调用保护。"""
import time
import functools
from enum import Enum
from backend.observability import get_logger, Metrics

log = get_logger("ragent.ha")


class State(Enum):
    CLOSED = 0
    OPEN = 1
    HALF_OPEN = 2


class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 3,
                 recovery_timeout: float = 60.0, half_open_max: int = 1):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max
        self.state = State.CLOSED
        self.failures = 0
        self.last_failure_time = 0.0
        self._half_open_count = 0

    def _transition(self, new_state: State):
        self.state = new_state
        Metrics.set_circuit_breaker(self.name, new_state.value)
        log.warning("circuit_breaker_transition", service=self.name,
                     state=new_state.name)

    def call(self, func, *args, **kwargs):
        if self.state == State.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self._transition(State.HALF_OPEN)
                self._half_open_count = 0
            else:
                raise CircuitBreakerOpenError(self.name)

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        if self.state == State.HALF_OPEN:
            self._half_open_count += 1
            if self._half_open_count >= self.half_open_max:
                self._transition(State.CLOSED)
                self.failures = 0
        self.failures = max(0, self.failures - 1)

    def _on_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self._transition(State.OPEN)


class CircuitBreakerOpenError(Exception):
    def __init__(self, service: str):
        super().__init__(f"Circuit breaker OPEN for '{service}'")


llm_breaker = CircuitBreaker("llm", failure_threshold=3, recovery_timeout=60.0)
tavily_breaker = CircuitBreaker("tavily", failure_threshold=3, recovery_timeout=60.0)


def with_circuit_breaker(breaker: CircuitBreaker, fallback=None):
    """装饰器：为函数包裹熔断保护。"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return breaker.call(func, *args, **kwargs)
            except CircuitBreakerOpenError:
                if fallback:
                    return fallback(*args, **kwargs)
                raise
        return wrapper
    return decorator
