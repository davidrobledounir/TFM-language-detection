"""Bootstrap por muestreo con reemplazo para IC 95 % sobre métricas agregadas."""
from typing import Callable, Sequence

import numpy as np
from jiwer import cer, wer
from sklearn.metrics import f1_score

from .normalize import normalize


def bootstrap_ci(
    items: Sequence,
    statistic: Callable[[Sequence], float],
    n: int = 1000,
    alpha: float = 0.05,
    seed: int = 1337,
) -> dict:
    if not items:
        return {"point": None, "low": None, "high": None, "n_replicates": 0}
    rng = np.random.default_rng(seed)
    idx = np.arange(len(items))
    point = float(statistic(items))
    replicates = np.empty(n, dtype=np.float64)
    for i in range(n):
        sample_idx = rng.integers(0, len(items), size=len(items))
        sample = [items[j] for j in sample_idx]
        replicates[i] = float(statistic(sample))
    low = float(np.quantile(replicates, alpha / 2))
    high = float(np.quantile(replicates, 1 - alpha / 2))
    return {"point": point, "low": low, "high": high, "n_replicates": int(n)}


def bootstrap_wer_cer(
    hypotheses: list[str],
    references: list[str],
    n: int = 1000,
    alpha: float = 0.05,
    seed: int = 1337,
) -> dict:
    pairs = list(
        zip([normalize(h) for h in hypotheses], [normalize(r) for r in references])
    )

    def _wer(pairs_):
        if not pairs_:
            return 0.0
        return float(wer([r for _, r in pairs_], [h for h, _ in pairs_]))

    def _cer(pairs_):
        if not pairs_:
            return 0.0
        return float(cer([r for _, r in pairs_], [h for h, _ in pairs_]))

    return {
        "wer": bootstrap_ci(pairs, _wer, n=n, alpha=alpha, seed=seed),
        "cer": bootstrap_ci(pairs, _cer, n=n, alpha=alpha, seed=seed + 1),
    }


def bootstrap_f1_macro(
    pred_labels: list[str],
    true_labels: list[str],
    labels: list[str],
    n: int = 1000,
    alpha: float = 0.05,
    seed: int = 1337,
) -> dict:
    pairs = list(zip(pred_labels, true_labels))

    def _f1(pairs_):
        if not pairs_:
            return 0.0
        return float(
            f1_score(
                [t for _, t in pairs_],
                [p for p, _ in pairs_],
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )

    return bootstrap_ci(pairs, _f1, n=n, alpha=alpha, seed=seed)
