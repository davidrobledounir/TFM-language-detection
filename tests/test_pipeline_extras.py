import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import _fill_nonspeech
from src.segmentation import Window, refine_windows_by_lid


@dataclass
class _FakePred:
    confidence: float
    margin: float


def test_refine_subdivides_unstable_window():
    sr = 16000
    audio = np.zeros(sr * 10, dtype=np.float32)
    windows = [Window(0.0, 8.0, audio[: sr * 8])]
    preds = [_FakePred(confidence=0.4, margin=0.05)]
    refined = refine_windows_by_lid(
        audio, sr, windows, preds,
        min_confidence=0.6, min_margin=0.15, min_refined_s=1.0,
    )
    assert len(refined) == 2
    assert abs((refined[0].end - refined[0].start) - 4.0) < 1e-6


def test_refine_keeps_stable_window():
    sr = 16000
    audio = np.zeros(sr * 4, dtype=np.float32)
    windows = [Window(0.0, 4.0, audio)]
    preds = [_FakePred(confidence=0.95, margin=0.8)]
    refined = refine_windows_by_lid(
        audio, sr, windows, preds,
        min_confidence=0.6, min_margin=0.15, min_refined_s=1.0,
    )
    assert len(refined) == 1
    assert refined[0].end == 4.0


def test_refine_does_not_subdivide_short_window():
    sr = 16000
    audio = np.zeros(sr * 2, dtype=np.float32)
    windows = [Window(0.0, 1.5, audio[: int(sr * 1.5)])]
    preds = [_FakePred(confidence=0.1, margin=0.0)]
    refined = refine_windows_by_lid(
        audio, sr, windows, preds,
        min_confidence=0.6, min_margin=0.15, min_refined_s=1.0,
    )
    assert len(refined) == 1


def test_fill_nonspeech_fills_gaps():
    segs = [{"start": 1.0, "end": 3.0, "lang": "spa", "text": "x", "lid_conf": 0.9}]
    out = _fill_nonspeech(segs, total_duration_s=5.0)
    assert [s["lang"] for s in out] == ["nonspeech", "spa", "nonspeech"]
    assert out[0]["start"] == 0.0
    assert out[-1]["end"] == 5.0
