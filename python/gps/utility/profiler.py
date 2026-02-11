"""Performance profiling utilities."""
import time
import functools
import logging
from typing import Callable, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def profile_function(func: Callable) -> Callable:
    """Decorator to profile function execution time."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.debug(f"{func.__name__} took {duration:.4f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.4f}s: {e}")
            raise
    return wrapper


@contextmanager
def profile_section(name: str):
    """Context manager to profile code sections."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        logger.debug(f"{name} took {duration:.4f}s")


class PerformanceMonitor:
    """Monitor and track performance metrics."""

    def __init__(self, name: str):
        self.name = name
        self.start_time: float = 0
        self.end_time: float = 0
        self.duration: float = 0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        logger.info(f"{self.name}: {self.duration:.4f}s")
