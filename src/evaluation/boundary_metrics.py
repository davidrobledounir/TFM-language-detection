def _match(
    pred: list[float], gold: list[float], tol_s: float
) -> tuple[int, int, int, list[float]]:
    used: set[int] = set()
    tp = 0
    errors: list[float] = []
    for p in pred:
        best = -1
        best_d = tol_s
        for i, g in enumerate(gold):
            if i in used:
                continue
            d = abs(p - g)
            if d <= best_d:
                best_d = d
                best = i
        if best >= 0:
            used.add(best)
            tp += 1
            errors.append(best_d)
    fp = len(pred) - tp
    fn = len(gold) - tp
    return tp, fp, fn, errors


def score(
    pred: list[float], gold: list[float], tolerances_s: list[float]
) -> dict:
    out: dict = {}
    for tol in tolerances_s:
        tp, fp, fn, errors = _match(pred, gold, tol)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        out[f"tol_{int(tol * 1000)}ms"] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "mean_error_s": float(sum(errors) / len(errors)) if errors else None,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return out
