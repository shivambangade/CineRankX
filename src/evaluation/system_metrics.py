"""System KPIs — Response Time, Memory Usage, CPU Usage, Throughput.

Deliberately generic: `measure()` instruments any callable rather than
importing the engines. That keeps this module free of a dependency cycle
(strategy_selector imports src.evaluation.ml_metrics, so src.evaluation must
not import strategy_selector back) and lets the report instrument IR search,
strategy prediction and ranking through one code path.
"""

import gc
import time

import numpy as np
import psutil

from src.evaluation.metric_names import (
    CPU_USAGE,
    MEMORY_USAGE,
    RESPONSE_TIME,
    THROUGHPUT,
)

_BYTES_PER_MB = 1024 * 1024

# Below this the CPU figure is timer noise rather than a measurement: CPU-time
# accounting is granular (typically ~10ms), so a window of a few milliseconds
# can report anything from 0% to several hundred percent.
_MIN_CPU_WINDOW_SECONDS = 0.05


def measure(operation, n_calls: int = 20, warmup: int = 2, label: str = "") -> dict:
    """Time `n_calls` invocations of `operation()` and report the system KPIs.

    Args:
        operation: zero-argument callable performing one unit of work.
        n_calls: measured invocations (excludes warmup).
        warmup: unmeasured invocations run first. Without these the first call
            pays for lazy imports, cache population and CPU frequency scaling,
            which would land entirely in the reported p95.

    Returns:
        Dict keyed by the exact eval_config.yaml metric names:
        Response Time (mean ms/call), Memory Usage (MB RSS after the run),
        CPU Usage (process CPU % during the run), Throughput (calls/second).
    """
    process = psutil.Process()

    for _ in range(warmup):
        operation()

    # Collect before the baseline so garbage from warmup is not misattributed
    # to the measured run's memory delta.
    gc.collect()
    rss_before_mb = process.memory_info().rss / _BYTES_PER_MB

    # CPU is derived from cpu_times() deltas rather than psutil.cpu_percent().
    # cpu_percent(interval=None) depends on a primed internal counter that any
    # other call in the process can disturb, and silently reports against a
    # single-core denominator -- which is how an 8-core box produces "1134%".
    cpu_before = process.cpu_times()

    latencies = np.empty(n_calls, dtype=float)
    wall_start = time.perf_counter()
    for i in range(n_calls):
        call_start = time.perf_counter()
        operation()
        latencies[i] = (time.perf_counter() - call_start) * 1000.0
    wall_elapsed = time.perf_counter() - wall_start

    cpu_after = process.cpu_times()
    rss_after_mb = process.memory_info().rss / _BYTES_PER_MB

    cpu_seconds = (
        (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)
    )
    n_cores = psutil.cpu_count() or 1
    if wall_elapsed >= _MIN_CPU_WINDOW_SECONDS:
        cpu_percent_of_core = 100.0 * cpu_seconds / wall_elapsed
        cpu_reliable = True
    else:
        cpu_percent_of_core = float("nan")
        cpu_reliable = False

    return {
        "label": label,
        RESPONSE_TIME: float(latencies.mean()),
        MEMORY_USAGE: float(rss_after_mb),
        # Percent of ONE core: 100% = one core saturated, and values above 100%
        # mean genuine multi-core parallelism (sklearn/numpy use n_jobs=-1).
        CPU_USAGE: cpu_percent_of_core,
        THROUGHPUT: float(n_calls / wall_elapsed) if wall_elapsed > 0 else 0.0,
        "cpu_percent_of_machine": cpu_percent_of_core / n_cores if cpu_reliable else float("nan"),
        "cpu_seconds": float(cpu_seconds),
        "cpu_reliable": cpu_reliable,
        "n_cores": n_cores,
        "response_time_p50_ms": float(np.percentile(latencies, 50)),
        "response_time_p95_ms": float(np.percentile(latencies, 95)),
        "memory_delta_mb": float(rss_after_mb - rss_before_mb),
        "n_calls": n_calls,
        "wall_seconds": float(wall_elapsed),
    }


def aggregate(measurements: list[dict]) -> dict:
    """Roll per-operation measurements into one system-level KPI row.

    Response Time is the mean across operations, Memory Usage the peak RSS
    observed (a process-wide figure, so summing would double-count), CPU Usage
    the total CPU seconds over total wall seconds, and Throughput the combined
    work rate. CPU is recomputed from the totals rather than averaged, so a
    long expensive operation is not given the same weight as a short cheap one.
    """
    if not measurements:
        return {}
    total_calls = sum(m["n_calls"] for m in measurements)
    total_seconds = sum(m["wall_seconds"] for m in measurements)
    reliable = [m for m in measurements if m.get("cpu_reliable")]
    total_cpu_seconds = sum(m["cpu_seconds"] for m in reliable)
    reliable_wall = sum(m["wall_seconds"] for m in reliable)
    n_cores = measurements[0].get("n_cores", 1)

    cpu_of_core = 100.0 * total_cpu_seconds / reliable_wall if reliable_wall > 0 else float("nan")
    return {
        RESPONSE_TIME: float(np.mean([m[RESPONSE_TIME] for m in measurements])),
        MEMORY_USAGE: float(max(m[MEMORY_USAGE] for m in measurements)),
        CPU_USAGE: cpu_of_core,
        THROUGHPUT: float(total_calls / total_seconds) if total_seconds > 0 else 0.0,
        "cpu_percent_of_machine": cpu_of_core / n_cores if reliable_wall > 0 else float("nan"),
        "n_cores": n_cores,
        "per_operation": measurements,
    }
