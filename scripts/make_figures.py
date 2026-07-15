"""Genera figuras PDF para el capítulo de resultados del TFM."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parents[2] / "Documento" / "figuras"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA = Path(__file__).resolve().parents[1] / "data"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fig_wer_per_lang() -> None:
    """WER por idioma agregando todas las semillas disponibles en data/runs/.

    Prefiere data/runs/<dataset>_s<seed>.json (formato consolidado por semilla).
    Si no encuentra ningún fichero ahí, cae a los results.json legacy.
    """
    runs_dir = DATA / "runs"
    datasets = ["eu_sample", "es_sample"]
    base, base_err, pipe, pipe_err, labels = [], [], [], [], []
    for ds in datasets:
        files = sorted(runs_dir.glob(f"{ds}_s*.json")) if runs_dir.exists() else []
        if not files:
            legacy = DATA / ds / "results.json"
            if legacy.exists():
                files = [legacy]
        if not files:
            continue
        b_vals, p_vals = [], []
        n_clips = 0
        for f in files:
            d = _load(f)
            b_vals.append(d["baseline"]["wer"])
            p_vals.append(d["pipeline"]["wer"])
            n_clips = d.get("n") or n_clips
        base.append(float(np.mean(b_vals)))
        base_err.append(float(np.std(b_vals)) if len(b_vals) > 1 else 0.0)
        pipe.append(float(np.mean(p_vals)))
        pipe_err.append(float(np.std(p_vals)) if len(p_vals) > 1 else 0.0)
        seeds_lbl = f" ({len(files)} seeds)" if len(files) > 1 else ""
        labels.append(f"{ds.replace('_sample', '')} (n={n_clips}){seeds_lbl}")
    if not labels:
        return
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    b1 = ax.bar(x - w / 2, base, w, yerr=base_err, capsize=3,
                label="Línea base Whisper-large-v3", color="#888888")
    b2 = ax.bar(x + w / 2, pipe, w, yerr=pipe_err, capsize=3,
                label="Pipeline propuesto", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("WER")
    ax.set_ylim(0, max(max(base), max(pipe)) * 1.25)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for rects in (b1, b2):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "wer_per_lang.pdf")
    plt.close(fig)


def fig_boundary_tolerance() -> None:
    """F1 de frontera. Combina semillas si hay synth_es_eu_s*.json en data/runs."""
    runs_dir = DATA / "runs"
    files = sorted(runs_dir.glob("*synth_es_eu*_s*.json")) if runs_dir.exists() else []
    if files:
        # Promediar sobre seeds.
        tols = ["tol_200ms", "tol_500ms", "tol_1000ms"]
        agg = {t: {"f1": [], "precision": [], "recall": []} for t in tols}
        for f in files:
            d = _load(f)
            for t in tols:
                v = d.get("boundary", {}).get(t, {})
                if v:
                    for k in ("f1", "precision", "recall"):
                        agg[t][k].append(v.get(k, 0.0))
        f1 = [float(np.mean(agg[t]["f1"])) if agg[t]["f1"] else 0.0 for t in tols]
        p = [float(np.mean(agg[t]["precision"])) if agg[t]["precision"] else 0.0 for t in tols]
        r = [float(np.mean(agg[t]["recall"])) if agg[t]["recall"] else 0.0 for t in tols]
    else:
        synth_path = DATA / "synth" / "results.json"
        if not synth_path.exists():
            return
        synth = _load(synth_path)
        bnd = synth["boundary"]
        tols = ["tol_200ms", "tol_500ms", "tol_1000ms"]
        f1 = [bnd[t]["f1"] for t in tols]
        p = [bnd[t]["precision"] for t in tols]
        r = [bnd[t]["recall"] for t in tols]
    labels = ["200 ms", "500 ms", "1000 ms"]

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    ax.bar(x - w, p, w, label="Precisión", color="#cccccc")
    ax.bar(x, r, w, label="Recall", color="#888888")
    ax.bar(x + w, f1, w, label="F1", color="#1f77b4")
    ax.axhline(0.75, linestyle="--", color="#c0392b", linewidth=1.2,
               label="Umbral SMART (F1 ≥ 0,75)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Tolerancia temporal τ")
    ax.set_ylabel("Métrica")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "boundary_f1.pdf")
    plt.close(fig)


def fig_wins_ties_losses() -> None:
    from jiwer import wer as _wer
    from src.evaluation.normalize import normalize

    def _wer_pair(ref: str, hyp: str) -> float:
        r = normalize(ref)
        h = normalize(hyp)
        if not r.strip():
            return 0.0 if not h.strip() else 1.0
        return float(_wer(r, h))

    runs_dir = DATA / "runs"
    runs: dict[str, dict] = {}
    for code, ds in [("eu", "eu_sample"), ("es", "es_sample")]:
        files = sorted(runs_dir.glob(f"{ds}_s*.json")) if runs_dir.exists() else []
        if not files:
            legacy = DATA / ds / "results.json"
            if legacy.exists():
                files = [legacy]
        if files:
            # Concatenar filas de todas las seeds para acumular conteos.
            merged_rows = []
            for f in files:
                merged_rows.extend(_load(f).get("rows", []))
            runs[code] = {"rows": merged_rows}
    if not runs:
        return
    labels = list(runs.keys())
    wins, ties, losses = [], [], []
    for run in runs.values():
        w_count = t_count = l_count = 0
        for r in run["rows"]:
            d = _wer_pair(r["ref"], r["baseline"]) - _wer_pair(r["ref"], r["pipeline"])
            if d > 1e-6:
                w_count += 1
            elif d < -1e-6:
                l_count += 1
            else:
                t_count += 1
        wins.append(w_count)
        ties.append(t_count)
        losses.append(l_count)

    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    x = np.arange(len(labels))
    p_w = ax.bar(x, wins, 0.55, label="Pipeline mejor", color="#1f77b4")
    p_t = ax.bar(x, ties, 0.55, bottom=wins, label="Empate", color="#cccccc")
    p_l = ax.bar(x, losses, 0.55, bottom=np.array(wins) + np.array(ties),
                 label="Línea base mejor", color="#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Clips")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for bars, values in [(p_w, wins), (p_t, ties), (p_l, losses)]:
        for rect, v in zip(bars, values):
            if v > 0:
                y = rect.get_y() + rect.get_height() / 2
                ax.annotate(str(v), xy=(rect.get_x() + rect.get_width() / 2, y),
                            ha="center", va="center", fontsize=8, color="white" if v > 3 else "black")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "wins_ties_losses.pdf")
    plt.close(fig)


def fig_wer_per_variant() -> None:
    """Comparativa WER baseline-large vs. baseline-medium vs. pipeline-MMS vs. pipeline-XLSR.

    Requiere ``data/runs/<variante>_<dataset>_s<seed>.json``. Si no existe, omite la figura.
    """
    runs_dir = DATA / "runs"
    if not runs_dir.exists():
        return
    variants = ["baseline_large", "baseline_medium", "pipeline_mms", "pipeline_xlsr"]
    datasets = ["eu_sample", "es_sample", "ca_sample"]
    colors = {"baseline_large": "#888888", "baseline_medium": "#bbbbbb",
              "pipeline_mms": "#1f77b4", "pipeline_xlsr": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    x = np.arange(len(datasets))
    width = 0.2
    for i, var in enumerate(variants):
        wers, errs = [], []
        for ds in datasets:
            files = list(runs_dir.glob(f"{var}_{ds}_s*.json"))
            vals = []
            for f in files:
                d = _load(f)
                w = (d.get("pipeline") or d.get("baseline") or {}).get("wer")
                if w is not None:
                    vals.append(w)
            if vals:
                wers.append(np.mean(vals))
                errs.append(np.std(vals))
            else:
                wers.append(0.0)
                errs.append(0.0)
        ax.bar(x + (i - 1.5) * width, wers, width, yerr=errs, capsize=3,
               label=var.replace("_", "-"), color=colors[var])
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("WER (media ± std, semillas)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "wer_per_variant.pdf")
    plt.close(fig)


def fig_wer_by_density() -> None:
    """WER por banda de densidad de cambio sobre el corpus sintético (todas las semillas combinadas)."""
    runs_dir = DATA / "runs"
    candidates = list(runs_dir.glob("pipeline_mms_synth*.json")) if runs_dir.exists() else []
    if not candidates:
        synth = DATA / "synth" / "results.json"
        if synth.exists():
            candidates = [synth]
    if not candidates:
        return
    buckets: dict[str, list[float]] = {}
    for f in candidates:
        d = _load(f)
        by = d.get("wer_by_density") or {}
        for k, v in by.items():
            buckets.setdefault(k, []).append(v.get("wer", 0.0))
    if not buckets:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    labels = list(buckets.keys())
    means = [np.mean(buckets[l]) for l in labels]
    errs = [np.std(buckets[l]) for l in labels]
    x = np.arange(len(labels))
    ax.bar(x, means, 0.6, yerr=errs, capsize=3, color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("WER")
    ax.set_xlabel("Densidad de cambio")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "wer_by_density.pdf")
    plt.close(fig)


def fig_cost_per_minute() -> None:
    runs_dir = DATA / "runs"
    if not runs_dir.exists():
        return
    rows: list[tuple[str, float, float]] = []  # (variant, rtf, gpu_mb)
    for f in sorted(runs_dir.glob("*.json")):
        d = _load(f)
        cost = (d.get("cost") or {}).get("pipeline") or d.get("cost")
        if not cost:
            continue
        rtf = cost.get("real_time_factor")
        gpu = cost.get("gpu_peak_mb")
        if rtf is None:
            continue
        variant = f.stem.split("_s")[0]
        rows.append((variant, rtf, gpu or 0.0))
    if not rows:
        return
    agg: dict[str, list[float]] = {}
    gpu_agg: dict[str, list[float]] = {}
    for v, r, g in rows:
        agg.setdefault(v, []).append(r)
        gpu_agg.setdefault(v, []).append(g)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.8))
    labels = list(agg.keys())
    ax1.bar(labels, [np.mean(agg[l]) for l in labels],
            yerr=[np.std(agg[l]) for l in labels], capsize=3, color="#1f77b4")
    ax1.axhline(1.0, color="#c0392b", linestyle="--", label="tiempo real")
    ax1.set_ylabel("RTF (tiempo / audio)")
    ax1.tick_params(axis="x", rotation=20)
    ax1.legend(fontsize=8, frameon=False)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax2.bar(labels, [np.mean(gpu_agg[l]) for l in labels],
            yerr=[np.std(gpu_agg[l]) for l in labels], capsize=3, color="#888888")
    ax2.set_ylabel("Memoria GPU pico (MB)")
    ax2.tick_params(axis="x", rotation=20)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cost.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_wer_per_lang()
    fig_boundary_tolerance()
    fig_wins_ties_losses()
    try:
        fig_wer_per_variant()
    except Exception as exc:
        print(f"  fig_wer_per_variant: {exc}")
    try:
        fig_wer_by_density()
    except Exception as exc:
        print(f"  fig_wer_by_density: {exc}")
    try:
        fig_cost_per_minute()
    except Exception as exc:
        print(f"  fig_cost_per_minute: {exc}")
    print(f"Figuras escritas en {OUT_DIR}")
