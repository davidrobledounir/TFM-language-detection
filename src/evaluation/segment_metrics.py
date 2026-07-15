"""F1 macro de LID por segmento, emparejando hipótesis y referencia por mayor solape temporal."""
from sklearn.metrics import confusion_matrix, f1_score


def _overlap(a: dict, b: dict) -> float:
    lo = max(a["start"], b["start"])
    hi = min(a["end"], b["end"])
    return max(0.0, hi - lo)


def align_segments(
    pred_segments: list[dict], gold_segments: list[dict]
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for g in gold_segments:
        if g["end"] <= g["start"]:
            continue
        best = None
        best_ov = 0.0
        for p in pred_segments:
            ov = _overlap(p, g)
            if ov > best_ov:
                best_ov = ov
                best = p
        if best is None:
            continue
        pairs.append((best["lang"], g["lang"]))
    return pairs


def score(
    pred_segments: list[dict],
    gold_segments: list[dict],
    labels: list[str],
) -> dict:
    pairs = align_segments(pred_segments, gold_segments)
    if not pairs:
        return {"f1_macro": 0.0, "confusion": [], "labels": labels, "n": 0}
    pred = [p if p in labels else "OOV" for p, _ in pairs]
    gold = [g for _, g in pairs]
    f1 = float(
        f1_score(gold, pred, labels=labels, average="macro", zero_division=0)
    )
    cm = confusion_matrix(gold, pred, labels=labels).tolist()
    return {"f1_macro": f1, "confusion": cm, "labels": labels, "n": len(pairs)}
