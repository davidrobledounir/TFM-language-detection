"""Análisis de error por clip: WER baseline vs pipeline, deltas, top wins / top losses."""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jiwer import wer as _wer

from src.evaluation.normalize import normalize


def _wer_pair(ref: str, hyp: str) -> float:
    r = normalize(ref)
    h = normalize(hyp)
    if not r.strip():
        return 0.0 if not h.strip() else 1.0
    return float(_wer(r, h))


def _truncate(text: str, width: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="results.json de analyze_sample.py")
    parser.add_argument("--out", required=True, help="Markdown de salida")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    if not rows:
        print("ERROR: sin filas en results.json")
        sys.exit(1)

    enriched: list[dict] = []
    for r in rows:
        wer_b = _wer_pair(r["ref"], r["baseline"])
        wer_p = _wer_pair(r["ref"], r["pipeline"])
        enriched.append(
            {
                "id": r["id"],
                "ref": r["ref"],
                "baseline": r["baseline"],
                "pipeline": r["pipeline"],
                "pipeline_langs": r.get("pipeline_langs", []),
                "wer_baseline": wer_b,
                "wer_pipeline": wer_p,
                "delta": wer_b - wer_p,
            }
        )

    deltas = [e["delta"] for e in enriched]
    wins = [e for e in enriched if e["delta"] > 0]
    ties = [e for e in enriched if e["delta"] == 0]
    losses = [e for e in enriched if e["delta"] < 0]

    enriched_sorted = sorted(enriched, key=lambda e: e["delta"], reverse=True)
    top_wins = enriched_sorted[: args.top_n]
    top_losses = list(reversed(enriched_sorted[-args.top_n:]))

    lines: list[str] = []
    lines.append("# Análisis de error por clip")
    lines.append("")
    lines.append(f"- Clips: **{len(enriched)}**")
    lines.append(
        f"- ΔWER (baseline − pipeline) media: **{statistics.mean(deltas):+.3f}**  "
        f"·  mediana: **{statistics.median(deltas):+.3f}**  ·  "
        f"std: **{statistics.pstdev(deltas):.3f}**"
    )
    lines.append(
        f"- Reparto: **wins {len(wins)}** ({len(wins) * 100 / len(enriched):.0f} %)  "
        f"·  empates {len(ties)}  ·  losses {len(losses)} ({len(losses) * 100 / len(enriched):.0f} %)"
    )
    lines.append(
        f"- WER medio  ·  baseline: **{statistics.mean(e['wer_baseline'] for e in enriched):.3f}**  "
        f"·  pipeline: **{statistics.mean(e['wer_pipeline'] for e in enriched):.3f}**"
    )
    lines.append("")

    lines.append(f"## Top {args.top_n} clips donde el pipeline mejora más")
    lines.append("")
    lines.append("| id | langs | WER base | WER pipe | Δ |")
    lines.append("|---|---|---:|---:|---:|")
    for e in top_wins:
        lines.append(
            f"| `{e['id']}` | {','.join(e['pipeline_langs'])} | "
            f"{e['wer_baseline']:.3f} | {e['wer_pipeline']:.3f} | **{e['delta']:+.3f}** |"
        )
    lines.append("")
    for e in top_wins:
        lines.append(f"### `{e['id']}` (Δ={e['delta']:+.3f})")
        lines.append("")
        lines.append(f"- **Ref:** {_truncate(normalize(e['ref']))}")
        lines.append(f"- **Baseline:** {_truncate(normalize(e['baseline']))}")
        lines.append(f"- **Pipeline:** {_truncate(normalize(e['pipeline']))}")
        lines.append("")

    if top_losses:
        lines.append(f"## Top {args.top_n} clips donde el pipeline pierde más")
        lines.append("")
        lines.append("| id | langs | WER base | WER pipe | Δ |")
        lines.append("|---|---|---:|---:|---:|")
        for e in top_losses:
            lines.append(
                f"| `{e['id']}` | {','.join(e['pipeline_langs'])} | "
                f"{e['wer_baseline']:.3f} | {e['wer_pipeline']:.3f} | **{e['delta']:+.3f}** |"
            )
        lines.append("")
        for e in top_losses:
            lines.append(f"### `{e['id']}` (Δ={e['delta']:+.3f})")
            lines.append("")
            lines.append(f"- **Ref:** {_truncate(normalize(e['ref']))}")
            lines.append(f"- **Baseline:** {_truncate(normalize(e['baseline']))}")
            lines.append(f"- **Pipeline:** {_truncate(normalize(e['pipeline']))}")
            lines.append("")

    lines.append("## Tabla completa")
    lines.append("")
    lines.append("| # | id | WER base | WER pipe | Δ |")
    lines.append("|---:|---|---:|---:|---:|")
    for i, e in enumerate(enriched_sorted, 1):
        lines.append(
            f"| {i} | `{e['id']}` | {e['wer_baseline']:.3f} | "
            f"{e['wer_pipeline']:.3f} | {e['delta']:+.3f} |"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"Analisis escrito: {args.out}")
    print(
        f"  wins {len(wins)} / ties {len(ties)} / losses {len(losses)}  "
        f"mean delta {statistics.mean(deltas):+.3f}"
    )


if __name__ == "__main__":
    main()
