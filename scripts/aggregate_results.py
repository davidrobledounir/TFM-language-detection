"""Agrega resultados de varios analyze_sample.py / evaluate_synth.py en una tabla comparativa."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _parse_runs(spec: str) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"esperado label=ruta, recibido {piece!r}")
        label, path = piece.split("=", 1)
        out.append((label.strip(), Path(path.strip())))
    return out


def _diagonal_correct(confusion: list[list[int]], labels: list[str]) -> dict:
    out: dict = {}
    for i, lab in enumerate(labels):
        row_total = sum(confusion[i])
        if row_total:
            out[lab] = {"correct": confusion[i][i], "total": row_total,
                        "acc": confusion[i][i] / row_total}
    return out


def _ci(boot: dict | None, key: str = "wer") -> tuple[float | None, float | None]:
    if not boot:
        return (None, None)
    entry = boot.get(key) if key in boot else boot
    if not isinstance(entry, dict):
        return (None, None)
    return (entry.get("low"), entry.get("high"))


def _row_analyze(label: str, data: dict) -> dict:
    base = data.get("baseline", {})
    pipe = data.get("pipeline", {})
    delta = None
    if base.get("wer") is not None and pipe.get("wer") is not None and base["wer"] > 0:
        delta = (base["wer"] - pipe["wer"]) / base["wer"]
    lid = data.get("lid") or {}
    diag = _diagonal_correct(lid.get("confusion", []), lid.get("labels", []))
    base_boot = base.get("bootstrap") or {}
    pipe_boot = pipe.get("bootstrap") or {}
    cost = data.get("cost") or {}
    return {
        "label": label,
        "kind": "analyze",
        "seed": data.get("seed"),
        "n": data.get("n"),
        "elapsed_s": data.get("elapsed_s"),
        "base_wer": base.get("wer"),
        "base_wer_ci": _ci(base_boot, "wer"),
        "base_cer": base.get("cer"),
        "pipe_wer": pipe.get("wer"),
        "pipe_wer_ci": _ci(pipe_boot, "wer"),
        "pipe_cer": pipe.get("cer"),
        "delta_rel": delta,
        "lid_f1_macro": lid.get("f1_macro"),
        "lid_per_class": diag,
        "cost_pipeline_rtf": (cost.get("pipeline") or {}).get("real_time_factor"),
        "cost_baseline_rtf": (cost.get("baseline") or {}).get("real_time_factor"),
        "cost_pipeline_gpu_mb": (cost.get("pipeline") or {}).get("gpu_peak_mb"),
        "cost_pipeline_cpu_mb": (cost.get("pipeline") or {}).get("cpu_peak_mb"),
    }


def _row_synth(label: str, data: dict) -> dict:
    tr = data.get("transcription", {})
    flid = data.get("lid_frame") or {}
    slid = data.get("lid_segment") or {}
    bnd = data.get("boundary", {})
    cost = data.get("cost", {})
    tr_boot = (tr.get("bootstrap") or {}) if isinstance(tr.get("bootstrap"), dict) else {}
    return {
        "label": label,
        "kind": "synth",
        "seed": data.get("seed"),
        "n": data.get("n"),
        "elapsed_s": data.get("elapsed_s"),
        "wer": tr.get("wer"),
        "wer_ci": _ci(tr_boot, "wer"),
        "cer": tr.get("cer"),
        "cer_ci": _ci(tr_boot, "cer"),
        "frame_lid_f1": flid.get("f1_macro"),
        "segment_lid_f1": slid.get("f1_macro"),
        "boundary": {
            tol: {"f1": v.get("f1"), "p": v.get("precision"), "r": v.get("recall")}
            for tol, v in bnd.items()
        },
        "cost_rtf": cost.get("real_time_factor"),
        "cost_gpu_mb": cost.get("gpu_peak_mb"),
        "cost_cpu_mb": cost.get("cpu_peak_mb"),
        "wer_by_length": data.get("wer_by_segment_length") or {},
        "wer_by_density": data.get("wer_by_density") or {},
    }


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _render(rows: list[dict]) -> str:
    out: list[str] = []
    out.append("# Comparativa agregada de resultados")
    out.append("")

    analyze = [r for r in rows if r["kind"] == "analyze"]
    synth = [r for r in rows if r["kind"] == "synth"]

    if analyze:
        out.append("## Análisis monolingüe (analyze_sample)")
        out.append("")
        out.append("| Run | Seed | N | Base WER (IC95) | Pipe WER (IC95) | ΔWER rel | LID F1 | RTF pipe | GPU MB |")
        out.append("|---|---:|---:|---|---|---:|---:|---:|---:|")
        for r in analyze:
            d = r["delta_rel"]
            d_str = f"{d * 100:+.1f} %" if d is not None else "—"
            bw_ci = r.get("base_wer_ci") or (None, None)
            pw_ci = r.get("pipe_wer_ci") or (None, None)
            bw = f"{_fmt(r['base_wer'])}" + (f" ({_fmt(bw_ci[0])}–{_fmt(bw_ci[1])})" if bw_ci[0] is not None else "")
            pw = f"{_fmt(r['pipe_wer'])}" + (f" ({_fmt(pw_ci[0])}–{_fmt(pw_ci[1])})" if pw_ci[0] is not None else "")
            out.append(
                f"| {r['label']} | {r.get('seed', '—')} | {r['n']} | {bw} | {pw} | "
                f"{d_str} | {_fmt(r['lid_f1_macro'])} | {_fmt(r.get('cost_pipeline_rtf'))} | "
                f"{_fmt(r.get('cost_pipeline_gpu_mb'), 0)} |"
            )
        out.append("")
        out.append("### Acierto LID en clase presente")
        out.append("")
        out.append("| Run | Clase | Aciertos / Total |")
        out.append("|---|---|---:|")
        for r in analyze:
            for cls, stats in r["lid_per_class"].items():
                out.append(
                    f"| {r['label']} | {cls} | {stats['correct']}/{stats['total']} "
                    f"({stats['acc'] * 100:.1f} %) |"
                )
        out.append("")

    if synth:
        out.append("## Evaluación sobre corpus sintético (evaluate_synth)")
        out.append("")
        out.append("| Run | Seed | N | WER (IC95) | Frame LID F1 | Seg LID F1 | F1 frontera @ 500 ms | RTF |")
        out.append("|---|---:|---:|---|---:|---:|---:|---:|")
        for r in synth:
            f1_500 = (r.get("boundary") or {}).get("tol_500ms", {}).get("f1")
            wc = r.get("wer_ci") or (None, None)
            wer_str = _fmt(r.get("wer")) + (f" ({_fmt(wc[0])}–{_fmt(wc[1])})" if wc[0] is not None else "")
            out.append(
                f"| {r['label']} | {r.get('seed', '—')} | {r['n']} | {wer_str} | "
                f"{_fmt(r.get('frame_lid_f1'))} | {_fmt(r.get('segment_lid_f1'))} | "
                f"{_fmt(f1_500)} | {_fmt(r.get('cost_rtf'))} |"
            )
        out.append("")
        # Desagregado por densidad y longitud (último run sintético como referencia).
        last = synth[-1]
        if last.get("wer_by_density"):
            out.append("### WER por banda de densidad de cambio")
            out.append("")
            out.append("| Banda | N | WER | CER |")
            out.append("|---|---:|---:|---:|")
            for bucket, stats in last["wer_by_density"].items():
                out.append(
                    f"| {bucket} | {stats.get('n', '—')} | "
                    f"{_fmt(stats.get('wer'))} | {_fmt(stats.get('cer'))} |"
                )
            out.append("")
        if last.get("wer_by_length"):
            out.append("### WER por banda de longitud de segmento")
            out.append("")
            out.append("| Banda | N | WER | CER |")
            out.append("|---|---:|---:|---:|")
            for bucket, stats in last["wer_by_length"].items():
                out.append(
                    f"| {bucket} | {stats.get('n', '—')} | "
                    f"{_fmt(stats.get('wer'))} | {_fmt(stats.get('cer'))} |"
                )
            out.append("")

    out.append("## Umbrales SMART")
    out.append("")
    out.append("| Métrica | Umbral | Cobertura actual |")
    out.append("|---|---:|---|")
    out.append("| ΔWER relativo vs. referencia | ≥ 8 % | ver tabla |")
    out.append("| ΔF1 macro LID por segmento | ≥ 5 pp | pendiente de runs multi-lengua |")
    out.append("| F1 frontera @ τ = 500 ms | ≥ 0,75 | ver tabla synth |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        default=None,
        help="Comma-separated label=ruta. Ej: eu=data/eu_sample/results.json,synth=data/synth/results.json",
    )
    parser.add_argument(
        "--runs-dir",
        default=None,
        help="Directorio de results.json producido por run_seeds.py. Etiqueta = nombre de fichero sin extensión.",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    runs: list[tuple[str, Path]] = []
    if args.runs_dir:
        for p in sorted(Path(args.runs_dir).glob("*.json")):
            runs.append((p.stem, p))
    if args.runs:
        runs.extend(_parse_runs(args.runs))
    if not runs:
        print("ERROR: indica --runs o --runs-dir")
        sys.exit(2)
    rows: list[dict] = []
    for label, path in runs:
        if not path.exists():
            print(f"  AVISO: {path} no existe, salto {label}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if "baseline" in data and "pipeline" in data:
            rows.append(_row_analyze(label, data))
        elif "transcription" in data:
            rows.append(_row_synth(label, data))
        else:
            print(f"  AVISO: formato desconocido en {path}, salto")

    if not rows:
        print("ERROR: ningún run válido")
        sys.exit(1)

    md = _render(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"Agregado: {args.out}  ({len(rows)} runs)")


if __name__ == "__main__":
    main()
