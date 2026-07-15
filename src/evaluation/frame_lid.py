"""F1 de LID muestreado a frecuencia fija sobre la línea temporal."""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def _label_at(segments: list[dict], t: float) -> str | None:
    for seg in segments:
        if seg["start"] <= t < seg["end"]:
            return seg["lang"]
    return None


def sample_labels(
    segments: list[dict],
    duration_s: float,
    frame_hz: float = 10.0,
) -> tuple[np.ndarray, list[str | None]]:
    n = max(1, int(round(duration_s * frame_hz)))
    times = (np.arange(n) + 0.5) / frame_hz
    labels = [_label_at(segments, float(t)) for t in times]
    return times, labels


def score(
    pred_segments: list[dict],
    gold_segments: list[dict],
    duration_s: float,
    labels: list[str],
    frame_hz: float = 10.0,
) -> dict:
    _, pred = sample_labels(pred_segments, duration_s, frame_hz)
    _, gold = sample_labels(gold_segments, duration_s, frame_hz)
    pairs = [(p, g) for p, g in zip(pred, gold) if g is not None]
    if not pairs:
        return {"f1_macro": 0.0, "confusion": [], "labels": labels, "frames": 0}
    pred_arr = [p if p in labels else "OOV" for p, _ in pairs]
    gold_arr = [g for _, g in pairs]
    f1 = float(
        f1_score(
            gold_arr, pred_arr, labels=labels, average="macro", zero_division=0
        )
    )
    cm = confusion_matrix(gold_arr, pred_arr, labels=labels).tolist()
    return {
        "f1_macro": f1,
        "confusion": cm,
        "labels": labels,
        "frames": len(pairs),
    }
