"""Medición de tiempo y memoria pico (CPU + GPU) durante la inferencia."""
import time
import tracemalloc
from contextlib import contextmanager

try:
    import torch
except Exception:
    torch = None


@contextmanager
def measure(label: str = ""):
    record = {"label": label, "elapsed_s": 0.0, "cpu_peak_mb": 0.0, "gpu_peak_mb": 0.0}
    tracemalloc.start()
    gpu_available = torch is not None and torch.cuda.is_available()
    if gpu_available:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        yield record
    finally:
        record["elapsed_s"] = float(time.perf_counter() - t0)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        record["cpu_peak_mb"] = float(peak / (1024 * 1024))
        if gpu_available:
            record["gpu_peak_mb"] = float(
                torch.cuda.max_memory_allocated() / (1024 * 1024)
            )
        else:
            record["gpu_peak_mb"] = 0.0


def per_minute_audio(record: dict, audio_seconds: float) -> dict:
    if audio_seconds <= 0:
        return {
            "elapsed_s_per_min": None,
            "real_time_factor": None,
            "cpu_peak_mb": record["cpu_peak_mb"],
            "gpu_peak_mb": record["gpu_peak_mb"],
        }
    minutes = audio_seconds / 60.0
    rtf = record["elapsed_s"] / audio_seconds
    return {
        "elapsed_s_per_min": float(record["elapsed_s"] / minutes),
        "real_time_factor": float(rtf),
        "cpu_peak_mb": record["cpu_peak_mb"],
        "gpu_peak_mb": record["gpu_peak_mb"],
    }
