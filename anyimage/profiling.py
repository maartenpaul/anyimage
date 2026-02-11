"""Performance profiling tools for anyimage.

Provides decorators, context managers, and a profiler class to measure
and report performance of key operations in BioImageViewer.

Usage:
    from anyimage.profiling import Profiler, profile_operation, timer

    # As a context manager
    with timer("composite_channels"):
        result = composite_channels(...)

    # As a decorator
    @profile_operation("tile_generation")
    def _get_tile(self, ...):
        ...

    # Full profiling session
    profiler = Profiler()
    profiler.enable()
    # ... do work ...
    profiler.disable()
    profiler.report()
"""

import functools
import statistics
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class OperationStats:
    """Statistics for a single profiled operation."""

    name: str
    call_count: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float("inf")
    max_time_ms: float = 0.0
    times_ms: list[float] = field(default_factory=list)

    def record(self, elapsed_ms: float) -> None:
        self.call_count += 1
        self.total_time_ms += elapsed_ms
        self.min_time_ms = min(self.min_time_ms, elapsed_ms)
        self.max_time_ms = max(self.max_time_ms, elapsed_ms)
        self.times_ms.append(elapsed_ms)

    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / self.call_count if self.call_count > 0 else 0.0

    @property
    def median_time_ms(self) -> float:
        return statistics.median(self.times_ms) if self.times_ms else 0.0

    @property
    def p95_time_ms(self) -> float:
        if len(self.times_ms) < 2:
            return self.max_time_ms
        sorted_times = sorted(self.times_ms)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    def summary(self) -> dict:
        return {
            "name": self.name,
            "calls": self.call_count,
            "total_ms": round(self.total_time_ms, 2),
            "avg_ms": round(self.avg_time_ms, 2),
            "median_ms": round(self.median_time_ms, 2),
            "min_ms": round(self.min_time_ms, 2) if self.min_time_ms != float("inf") else 0.0,
            "max_ms": round(self.max_time_ms, 2),
            "p95_ms": round(self.p95_time_ms, 2),
        }


@dataclass
class CacheStats:
    """Statistics for cache hit/miss tracking."""

    name: str
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    current_size: int = 0
    max_size: int = 0

    def record_hit(self) -> None:
        self.hits += 1

    def record_miss(self) -> None:
        self.misses += 1

    def record_eviction(self) -> None:
        self.evictions += 1

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def summary(self) -> dict:
        return {
            "name": self.name,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate * 100, 1),
            "evictions": self.evictions,
            "current_size": self.current_size,
            "max_size": self.max_size,
        }


class Profiler:
    """Collects and reports performance metrics for anyimage operations.

    Usage:
        profiler = Profiler()
        profiler.enable()

        with profiler.track("my_operation"):
            do_something()

        profiler.report()
    """

    _instance: "Profiler | None" = None

    def __init__(self) -> None:
        self._operations: OrderedDict[str, OperationStats] = OrderedDict()
        self._caches: OrderedDict[str, CacheStats] = OrderedDict()
        self._enabled: bool = False

    @classmethod
    def get_instance(cls) -> "Profiler":
        """Get the global singleton profiler instance."""
        if cls._instance is None:
            cls._instance = Profiler()
        return cls._instance

    def enable(self) -> None:
        """Enable profiling."""
        self._enabled = True

    def disable(self) -> None:
        """Disable profiling (stops collecting, preserves data)."""
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def reset(self) -> None:
        """Clear all collected data."""
        self._operations.clear()
        self._caches.clear()

    def _get_op(self, name: str) -> OperationStats:
        if name not in self._operations:
            self._operations[name] = OperationStats(name=name)
        return self._operations[name]

    def _get_cache(self, name: str) -> CacheStats:
        if name not in self._caches:
            self._caches[name] = CacheStats(name=name)
        return self._caches[name]

    def record_operation(self, name: str, elapsed_ms: float) -> None:
        """Record a single operation timing."""
        if self._enabled:
            self._get_op(name).record(elapsed_ms)

    def record_cache_hit(self, cache_name: str) -> None:
        """Record a cache hit."""
        if self._enabled:
            self._get_cache(cache_name).record_hit()

    def record_cache_miss(self, cache_name: str) -> None:
        """Record a cache miss."""
        if self._enabled:
            self._get_cache(cache_name).record_miss()

    def record_cache_eviction(self, cache_name: str) -> None:
        """Record a cache eviction."""
        if self._enabled:
            self._get_cache(cache_name).record_eviction()

    def update_cache_size(self, cache_name: str, current_size: int, max_size: int) -> None:
        """Update current cache size info."""
        if self._enabled:
            cache = self._get_cache(cache_name)
            cache.current_size = current_size
            cache.max_size = max_size

    @contextmanager
    def track(self, name: str):
        """Context manager to time an operation.

        Usage:
            with profiler.track("my_operation"):
                do_something()
        """
        if not self._enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._get_op(name).record(elapsed_ms)

    def report(self) -> str:
        """Generate a formatted performance report.

        Returns:
            Multi-line string with performance summary.
        """
        lines = []
        lines.append("=" * 80)
        lines.append("ANYIMAGE PERFORMANCE REPORT")
        lines.append("=" * 80)

        if self._operations:
            lines.append("")
            lines.append("OPERATIONS:")
            lines.append(
                f"  {'Name':<35} {'Calls':>6} {'Total':>9} {'Avg':>8} "
                f"{'Med':>8} {'Min':>8} {'Max':>8} {'P95':>8}"
            )
            lines.append("  " + "-" * 92)
            for op in self._operations.values():
                s = op.summary()
                lines.append(
                    f"  {s['name']:<35} {s['calls']:>6} "
                    f"{s['total_ms']:>8.1f}ms {s['avg_ms']:>7.2f}ms "
                    f"{s['median_ms']:>7.2f}ms {s['min_ms']:>7.2f}ms "
                    f"{s['max_ms']:>7.2f}ms {s['p95_ms']:>7.2f}ms"
                )

        if self._caches:
            lines.append("")
            lines.append("CACHES:")
            lines.append(
                f"  {'Name':<25} {'Hits':>8} {'Misses':>8} "
                f"{'Hit Rate':>9} {'Evictions':>10} {'Size':>12}"
            )
            lines.append("  " + "-" * 74)
            for cache in self._caches.values():
                s = cache.summary()
                lines.append(
                    f"  {s['name']:<25} {s['hits']:>8} {s['misses']:>8} "
                    f"{s['hit_rate']:>8.1f}% {s['evictions']:>10} "
                    f"{s['current_size']:>5}/{s['max_size']:<5}"
                )

        lines.append("")
        lines.append("=" * 80)
        return "\n".join(lines)

    def summary_dict(self) -> dict:
        """Return profiling data as a dictionary for programmatic access.

        Returns:
            Dictionary with 'operations' and 'caches' keys.
        """
        return {
            "operations": {
                name: op.summary() for name, op in self._operations.items()
            },
            "caches": {
                name: cache.summary() for name, cache in self._caches.items()
            },
        }


@contextmanager
def timer(name: str):
    """Simple context manager that prints elapsed time.

    Usage:
        with timer("my_operation"):
            do_something()
    """
    start = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"[perf] {name}: {elapsed_ms:.2f}ms")
    # Also record in global profiler if enabled
    profiler = Profiler.get_instance()
    if profiler.enabled:
        profiler.record_operation(name, elapsed_ms)


def profile_operation(name: str | None = None):
    """Decorator to profile a function's execution time.

    Args:
        name: Operation name for reporting. Defaults to function name.

    Usage:
        @profile_operation("tile_generation")
        def _get_tile(self, ...):
            ...
    """
    def decorator(func):
        op_name = name or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = Profiler.get_instance()
            if not profiler.enabled:
                return func(*args, **kwargs)
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                profiler.record_operation(op_name, elapsed_ms)
            return result

        return wrapper

    return decorator
