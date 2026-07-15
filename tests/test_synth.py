import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.common_voice import CVRow
from src.evaluation.frame_lid import sample_labels, score as frame_lid_score
from src.synth_codeswitch import generate


def _fake_pool(
    tmp_path: Path, lang: str, n: int, dur_s: float, sample_rate: int = 16000
) -> list[CVRow]:
    import soundfile as sf

    rows: list[CVRow] = []
    rng = np.random.default_rng(hash(lang) % (2**32))
    for i in range(n):
        path = tmp_path / f"{lang}_{i:03d}.wav"
        audio = (rng.standard_normal(int(dur_s * sample_rate)) * 0.05).astype(np.float32)
        sf.write(path, audio, sample_rate, subtype="PCM_16")
        rows.append(CVRow(path=path, sentence=f"frase {lang} {i}", locale=lang))
    return rows


def test_generate_single_language_no_switches(tmp_path):
    pool = _fake_pool(tmp_path, "eus", n=10, dur_s=2.0)
    sample = generate({"eus": pool}, target_duration_s=8.0, density_per_min=10.0, seed=1)
    assert sample.observed_switches == 0
    assert all(s["lang"] == "eus" for s in sample.segments)
    assert sample.audio.dtype == np.float32


def test_generate_multilingual_emits_switches(tmp_path):
    eus = _fake_pool(tmp_path, "eus", n=10, dur_s=1.5)
    spa = _fake_pool(tmp_path, "spa", n=10, dur_s=1.5)
    sample = generate(
        {"eus": eus, "spa": spa},
        target_duration_s=30.0,
        density_per_min=10.0,
        seed=7,
    )
    langs = {s["lang"] for s in sample.segments}
    assert langs == {"eus", "spa"}
    assert sample.observed_switches >= 1


def test_segments_are_temporally_consistent(tmp_path):
    pool = _fake_pool(tmp_path, "cat", n=5, dur_s=2.0)
    sample = generate({"cat": pool}, target_duration_s=6.0, density_per_min=0.0, seed=2)
    for a, b in zip(sample.segments, sample.segments[1:]):
        assert a["end"] <= b["start"]
    last = sample.segments[-1]
    assert last["end"] <= len(sample.audio) / sample.sample_rate + 1e-3


def test_frame_lid_perfect_match():
    gold = [
        {"start": 0.0, "end": 2.0, "lang": "spa"},
        {"start": 2.0, "end": 4.0, "lang": "cat"},
    ]
    pred = list(gold)
    res = frame_lid_score(pred, gold, duration_s=4.0, labels=["spa", "cat"], frame_hz=10.0)
    assert res["f1_macro"] == 1.0
    assert res["frames"] == 40


def test_frame_lid_handles_shift():
    gold = [
        {"start": 0.0, "end": 2.0, "lang": "spa"},
        {"start": 2.0, "end": 4.0, "lang": "cat"},
    ]
    pred = [
        {"start": 0.0, "end": 2.3, "lang": "spa"},
        {"start": 2.3, "end": 4.0, "lang": "cat"},
    ]
    res = frame_lid_score(pred, gold, duration_s=4.0, labels=["spa", "cat"], frame_hz=10.0)
    assert 0.85 < res["f1_macro"] < 1.0
