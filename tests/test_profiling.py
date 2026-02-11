"""Tests for the profiling module."""

import time

from anyimage.profiling import (
    CacheStats,
    OperationStats,
    Profiler,
    profile_operation,
)


class TestOperationStats:
    def test_empty_stats(self):
        stats = OperationStats(name="test")
        assert stats.call_count == 0
        assert stats.total_time_ms == 0.0
        assert stats.avg_time_ms == 0.0
        assert stats.median_time_ms == 0.0

    def test_single_record(self):
        stats = OperationStats(name="test")
        stats.record(10.0)
        assert stats.call_count == 1
        assert stats.total_time_ms == 10.0
        assert stats.avg_time_ms == 10.0
        assert stats.min_time_ms == 10.0
        assert stats.max_time_ms == 10.0

    def test_multiple_records(self):
        stats = OperationStats(name="test")
        for t in [5.0, 10.0, 15.0, 20.0]:
            stats.record(t)
        assert stats.call_count == 4
        assert stats.total_time_ms == 50.0
        assert stats.avg_time_ms == 12.5
        assert stats.min_time_ms == 5.0
        assert stats.max_time_ms == 20.0

    def test_p95(self):
        stats = OperationStats(name="test")
        for i in range(100):
            stats.record(float(i))
        # P95 should be around 95
        assert 90 <= stats.p95_time_ms <= 99

    def test_summary(self):
        stats = OperationStats(name="my_op")
        stats.record(10.0)
        s = stats.summary()
        assert s["name"] == "my_op"
        assert s["calls"] == 1
        assert s["total_ms"] == 10.0


class TestCacheStats:
    def test_empty(self):
        stats = CacheStats(name="test")
        assert stats.hit_rate == 0.0

    def test_hit_rate(self):
        stats = CacheStats(name="test")
        stats.record_hit()
        stats.record_hit()
        stats.record_miss()
        assert abs(stats.hit_rate - 2 / 3) < 0.001

    def test_summary(self):
        stats = CacheStats(name="cache")
        stats.record_hit()
        stats.record_miss()
        stats.record_eviction()
        stats.current_size = 10
        stats.max_size = 100
        s = stats.summary()
        assert s["name"] == "cache"
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["evictions"] == 1
        assert abs(s["hit_rate"] - 50.0) < 0.1


class TestProfiler:
    def test_default_disabled(self):
        profiler = Profiler()
        assert not profiler.enabled

    def test_enable_disable(self):
        profiler = Profiler()
        profiler.enable()
        assert profiler.enabled
        profiler.disable()
        assert not profiler.enabled

    def test_track_context_manager(self):
        profiler = Profiler()
        profiler.enable()
        with profiler.track("test_op"):
            time.sleep(0.005)
        summary = profiler.summary_dict()
        assert "test_op" in summary["operations"]
        assert summary["operations"]["test_op"]["calls"] == 1
        assert summary["operations"]["test_op"]["total_ms"] > 1  # at least 1ms

    def test_track_disabled(self):
        profiler = Profiler()
        # Disabled by default
        with profiler.track("test_op"):
            pass
        summary = profiler.summary_dict()
        assert "test_op" not in summary["operations"]

    def test_record_cache(self):
        profiler = Profiler()
        profiler.enable()
        profiler.record_cache_hit("test_cache")
        profiler.record_cache_hit("test_cache")
        profiler.record_cache_miss("test_cache")
        profiler.record_cache_eviction("test_cache")
        profiler.update_cache_size("test_cache", 10, 100)

        summary = profiler.summary_dict()
        cache = summary["caches"]["test_cache"]
        assert cache["hits"] == 2
        assert cache["misses"] == 1
        assert cache["evictions"] == 1
        assert cache["current_size"] == 10
        assert cache["max_size"] == 100

    def test_reset(self):
        profiler = Profiler()
        profiler.enable()
        profiler.record_operation("op", 10.0)
        profiler.record_cache_hit("cache")
        profiler.reset()
        summary = profiler.summary_dict()
        assert len(summary["operations"]) == 0
        assert len(summary["caches"]) == 0

    def test_report_format(self):
        profiler = Profiler()
        profiler.enable()
        profiler.record_operation("op1", 10.0)
        profiler.record_operation("op2", 20.0)
        profiler.record_cache_hit("cache1")
        profiler.record_cache_miss("cache1")
        report = profiler.report()
        assert "ANYIMAGE PERFORMANCE REPORT" in report
        assert "op1" in report
        assert "op2" in report
        assert "cache1" in report

    def test_singleton(self):
        """get_instance should return the same object."""
        # Save and restore
        old = Profiler._instance
        Profiler._instance = None
        try:
            p1 = Profiler.get_instance()
            p2 = Profiler.get_instance()
            assert p1 is p2
        finally:
            Profiler._instance = old


class TestProfileOperation:
    def test_decorator(self):
        old = Profiler._instance
        profiler = Profiler()
        profiler.enable()
        Profiler._instance = profiler

        try:
            @profile_operation("my_func")
            def my_func(x):
                return x * 2

            result = my_func(5)
            assert result == 10
            summary = profiler.summary_dict()
            assert "my_func" in summary["operations"]
            assert summary["operations"]["my_func"]["calls"] == 1
        finally:
            Profiler._instance = old

    def test_decorator_disabled(self):
        old = Profiler._instance
        profiler = Profiler()
        # disabled
        Profiler._instance = profiler

        try:
            @profile_operation("my_func")
            def my_func(x):
                return x * 2

            result = my_func(5)
            assert result == 10
            summary = profiler.summary_dict()
            assert "my_func" not in summary["operations"]
        finally:
            Profiler._instance = old

    def test_decorator_default_name(self):
        old = Profiler._instance
        profiler = Profiler()
        profiler.enable()
        Profiler._instance = profiler

        try:
            @profile_operation()
            def some_function():
                return 42

            some_function()
            summary = profiler.summary_dict()
            # Should use qualname
            assert any("some_function" in name for name in summary["operations"])
        finally:
            Profiler._instance = old
