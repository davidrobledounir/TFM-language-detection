import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import bootstrap, cost, desagregate, segment_metrics


def test_bootstrap_ci_is_deterministic_with_fixed_seed():
    items = list(range(20))
    a = bootstrap.bootstrap_ci(items, statistic=lambda xs: float(np.mean(xs)), n=50, seed=1)
    b = bootstrap.bootstrap_ci(items, statistic=lambda xs: float(np.mean(xs)), n=50, seed=1)
    assert a == b
    assert a["low"] <= a["point"] <= a["high"]


def test_bootstrap_wer_cer_returns_point_and_ci():
    res = bootstrap.bootstrap_wer_cer(
        hypotheses=["hola mundo", "buenos dias"],
        references=["hola mundo", "buenos días"],
        n=20,
        seed=7,
    )
    assert res["wer"]["point"] >= 0.0
    assert res["wer"]["low"] <= res["wer"]["point"] <= res["wer"]["high"]


def test_cost_measure_records_time_and_memory():
    with cost.measure("dummy") as rec:
        x = np.zeros(1_000_000, dtype=np.float32)
        x.sum()
    assert rec["elapsed_s"] >= 0.0
    assert rec["cpu_peak_mb"] >= 0.0


def test_per_minute_audio_computes_rtf():
    rec = {"elapsed_s": 12.0, "cpu_peak_mb": 100.0, "gpu_peak_mb": 0.0}
    out = cost.per_minute_audio(rec, audio_seconds=6.0)
    assert abs(out["real_time_factor"] - 2.0) < 1e-9
    assert abs(out["elapsed_s_per_min"] - 120.0) < 1e-9


def test_segment_metrics_perfect_alignment():
    gold = [
        {"start": 0.0, "end": 2.0, "lang": "spa"},
        {"start": 2.0, "end": 4.0, "lang": "cat"},
    ]
    pred = list(gold)
    res = segment_metrics.score(pred, gold, labels=["spa", "cat", "eus"])
    assert res["f1_macro"] > 0.6
    assert res["n"] == 2


def test_desagregate_by_density_buckets_correctly():
    entries = [
        {"id": "a", "duration_s": 60.0, "segments": [{"lang": "spa"}], "hyp_text": "hola", "ref_text": "hola"},
        {"id": "b", "duration_s": 60.0,
         "segments": [{"lang": "spa"}, {"lang": "cat"}, {"lang": "eus"}, {"lang": "spa"}, {"lang": "cat"}],
         "hyp_text": "hola", "ref_text": "hola"},
    ]
    out = desagregate.by_density(entries)
    assert len(out) >= 1
    for stats in out.values():
        assert "wer" in stats
        assert "n" in stats
