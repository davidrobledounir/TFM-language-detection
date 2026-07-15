"""Genera un informe Markdown a partir del JSON de salida de analyze_sample.py o evaluate_synth.py."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.normalize import normalize


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "—"


def _render_analyze_sample(data: dict) -> str:
    lines: list[str] = []
    lines.append("# Informe — análisis Common Voice")
    lines.append("")
    lines.append(f"- Clips procesados: **{data.get('n', 0)}**")
    lines.append(f"- Tiempo total: **{data.get('elapsed_s', 0):.1f} s**")
    lines.append("")
    base = data.get("baseline", {})
    pipe = data.get("pipeline", {})
    lines.append("## Métricas globales")
    lines.append("")
    lines.append("| Sistema | WER | CER |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Baseline Whisper-large-v3 | {_fmt(base.get('wer'))} | {_fmt(base.get('cer'))} |")
    lines.append(f"| Pipeline propuesto | {_fmt(pipe.get('wer'))} | {_fmt(pipe.get('cer'))} |")
    if base.get("wer") is not None and pipe.get("wer") is not None and base.get("wer") > 0:
        delta = (base["wer"] - pipe["wer"]) / base["wer"]
        lines.append("")
        lines.append(f"ΔWER relativo (pipeline vs baseline): **{delta * 100:+.1f} %**")
    lid = data.get("lid")
    if lid:
        lines.append("")
        lines.append("## LID por segmento")
        lines.append("")
        lines.append(f"- F1 macro: **{_fmt(lid.get('f1_macro'))}**")
        labels = lid.get("labels", [])
        cm = lid.get("confusion", [])
        if labels and cm:
            lines.append("")
            header = "| gold \\ pred | " + " | ".join(labels) + " |"
            sep = "|---|" + "|".join(["---:"] * len(labels)) + "|"
            lines.append(header)
            lines.append(sep)
            for lab, row in zip(labels, cm):
                lines.append(f"| **{lab}** | " + " | ".join(str(v) for v in row) + " |")
    rows = data.get("rows", [])
    if rows:
        lines.append("")
        lines.append("## Por clip")
        lines.append("")
        lines.append("| # | id | langs | baseline (s) | pipeline (s) |")
        lines.append("|---:|---|---|---:|---:|")
        for i, r in enumerate(rows, 1):
            lines.append(
                f"| {i} | {r['id']} | {','.join(r.get('pipeline_langs', []))} | "
                f"{r.get('baseline_s', '?')} | {r.get('pipeline_s', '?')} |"
            )
        lines.append("")
        lines.append("## Muestras de transcripción (5 primeros)")
        lines.append("")
        for r in rows[:5]:
            lines.append(f"### `{r['id']}`")
            lines.append("")
            lines.append(f"- **Ref:** {normalize(r['ref'])}")
            lines.append(f"- **Baseline:** {normalize(r['baseline'])}")
            lines.append(f"- **Pipeline:** {normalize(r['pipeline'])}")
            lines.append("")
    return "\n".join(lines)


def _render_evaluate_synth(data: dict) -> str:
    lines: list[str] = []
    lines.append("# Informe — evaluación sobre corpus sintético")
    lines.append("")
    lines.append(f"- Muestras: **{data.get('n', 0)}**")
    lines.append(f"- Tiempo total: **{data.get('elapsed_s', 0):.1f} s**")
    lines.append("")
    tr = data.get("transcription", {})
    lines.append("## Transcripción agregada")
    lines.append("")
    lines.append(f"- WER: **{_fmt(tr.get('wer'))}**  ·  CER: **{_fmt(tr.get('cer'))}**")
    flid = data.get("lid_frame")
    if flid:
        lines.append("")
        lines.append("## LID por frame")
        lines.append("")
        lines.append(
            f"- F1 macro: **{_fmt(flid.get('f1_macro'))}**  ·  frames: {flid.get('frames')}"
        )
        labels = flid.get("labels", [])
        cm = flid.get("confusion", [])
        if labels and cm:
            header = "| gold \\ pred | " + " | ".join(labels) + " |"
            sep = "|---|" + "|".join(["---:"] * len(labels)) + "|"
            lines.append("")
            lines.append(header)
            lines.append(sep)
            for lab, row in zip(labels, cm):
                lines.append(f"| **{lab}** | " + " | ".join(str(v) for v in row) + " |")
    bnd = data.get("boundary", {})
    if bnd:
        lines.append("")
        lines.append("## Fronteras de cambio de idioma")
        lines.append("")
        lines.append("| Tolerancia | F1 | Precisión | Recall | err. medio (s) |")
        lines.append("|---|---:|---:|---:|---:|")
        for tol_key, vals in bnd.items():
            me = vals.get("mean_error_s")
            me_str = f"{me:.3f}" if me is not None else "—"
            lines.append(
                f"| {tol_key} | {_fmt(vals.get('f1'))} | {_fmt(vals.get('precision'))} | "
                f"{_fmt(vals.get('recall'))} | {me_str} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="JSON de analyze_sample o evaluate_synth")
    parser.add_argument("--out", required=True, help="Markdown de salida")
    args = parser.parse_args()

    data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    if "baseline" in data and "pipeline" in data:
        md = _render_analyze_sample(data)
    elif "lid_frame" in data or "boundary" in data:
        md = _render_evaluate_synth(data)
    else:
        print("ERROR: formato de resultados no reconocido")
        sys.exit(2)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"Informe escrito: {args.out}")


if __name__ == "__main__":
    main()
