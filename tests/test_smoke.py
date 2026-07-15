import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.boundaries import boundaries_from_labels, boundary_points
from src.evaluation.boundary_metrics import score as boundary_score
from src.evaluation.lid_metrics import score as lid_score
from src.io_audio import rms_normalize
from src.postprocess import consolidate


def test_rms_normalize_targets_expected_level():
    audio = np.full(16000, 0.01, dtype=np.float32)
    out = rms_normalize(audio, target_dbfs=-20.0)
    rms = float(np.sqrt(np.mean(out ** 2)))
    assert abs(rms - 0.1) < 1e-3


def test_boundaries_collapse_consecutive_labels():
    labels = ["spa", "spa", "cat", "cat", "eus"]
    starts = [0.0, 3.0, 6.0, 9.0, 12.0]
    ends = [3.0, 6.0, 9.0, 12.0, 15.0]
    segs = boundaries_from_labels(labels, starts, ends)
    assert [s["lang"] for s in segs] == ["spa", "cat", "eus"]
    assert segs[0]["end"] == 6.0
    assert segs[1]["start"] == 6.0
    assert boundary_points(segs) == [6.0, 12.0]


def test_consolidate_merges_same_lang():
    segs = [
        {"start": 0.0, "end": 3.0, "lang": "spa", "text": "hola"},
        {"start": 3.0, "end": 6.0, "lang": "spa", "text": "mundo"},
        {"start": 6.0, "end": 9.0, "lang": "cat", "text": "bon dia"},
    ]
    out = consolidate(segs)
    assert len(out) == 2
    assert out[0]["text"] == "hola mundo"
    assert out[1]["lang"] == "cat"


def test_lid_metrics_perfect_score():
    res = lid_score(
        pred_labels=["spa", "cat", "eus"],
        true_labels=["spa", "cat", "eus"],
        labels=["spa", "cat", "eus"],
    )
    assert res["f1_macro"] == 1.0


def test_boundary_metrics_within_tolerance():
    res = boundary_score(pred=[1.0, 5.0], gold=[1.1, 4.6], tolerances_s=[0.2, 0.5])
    assert res["tol_200ms"]["tp"] == 1
    assert res["tol_500ms"]["tp"] == 2
