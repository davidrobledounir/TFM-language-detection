"""Ficha técnica de un dump de Common Voice (scripted o spontaneous): tamaño, duraciones, hablantes, demografía."""
import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import soundfile as sf

from src.datasets.common_voice import _resolve_tsv, _SCHEMAS
import csv


def _read_tsv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _duration_seconds(path: Path) -> float | None:
    try:
        info = sf.info(str(path))
        return float(info.duration)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--locale", required=True, help="Código ISO 639-1 (eu, es, ca)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--measure-clip-durations", action="store_true",
                        help="Medir duración real de cada wav/mp3 en disco (más lento)")
    args = parser.parse_args()

    root = Path(args.sample_dir)
    tsv_path, schema = _resolve_tsv(root, args.locale, split="validated")
    path_col, text_col = _SCHEMAS[schema]
    rows = _read_tsv(tsv_path)

    clips_dir = root / "clips"
    on_disk = {p.name for p in clips_dir.glob("*.mp3")} | {p.name for p in clips_dir.glob("*.wav")}
    rows_present = [r for r in rows if r.get(path_col) in on_disk]
    rows_with_text = [r for r in rows_present if r.get(text_col, "").strip()]
    sentences = {r.get(text_col, "") for r in rows_with_text}

    speakers = Counter(r.get("client_id", "_unknown") for r in rows_with_text)
    age_dist = Counter(r.get("age", "") or "_blank" for r in rows_with_text)
    gender_dist = Counter(r.get("gender", "") or "_blank" for r in rows_with_text)
    accents = Counter(r.get("accents", "") or "_blank" for r in rows_with_text)
    variants = Counter(r.get("variant", "") or "_blank" for r in rows_with_text)

    durations_from_tsv: list[float] = []
    for r in rows_with_text:
        ms = r.get("duration_ms")
        if ms:
            try:
                durations_from_tsv.append(int(ms) / 1000.0)
            except ValueError:
                pass

    measured_durations: list[float] = []
    if args.measure_clip_durations:
        for r in rows_with_text:
            name = r.get(path_col)
            if not name:
                continue
            d = _duration_seconds(clips_dir / name)
            if d is not None:
                measured_durations.append(d)

    durations = measured_durations if measured_durations else durations_from_tsv

    def _stats(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0}
        return {
            "n": len(xs),
            "total_s": round(sum(xs), 1),
            "min_s": round(min(xs), 2),
            "median_s": round(statistics.median(xs), 2),
            "mean_s": round(statistics.mean(xs), 2),
            "max_s": round(max(xs), 2),
        }

    summary = {
        "sample_dir": str(root),
        "locale": args.locale,
        "schema": schema,
        "tsv": tsv_path.name,
        "rows_in_tsv": len(rows),
        "clips_on_disk": len(on_disk),
        "rows_present_in_both": len(rows_present),
        "rows_with_text": len(rows_with_text),
        "rows_without_text": len(rows_present) - len(rows_with_text),
        "unique_sentences": len(sentences),
        "unique_speakers": len(speakers),
        "duration_stats": _stats(durations),
        "duration_source": "measured_from_audio" if measured_durations else (
            "duration_ms_field" if durations_from_tsv else "n/a"
        ),
    }

    md: list[str] = []
    md.append(f"# Ficha técnica — corpus {args.locale}")
    md.append("")
    md.append(f"- Directorio: `{root}`")
    md.append(f"- Schema detectado: `{schema}`  ·  TSV: `{tsv_path.name}`")
    md.append(f"- Filas en TSV: **{len(rows)}**")
    md.append(f"- Clips en disco: **{len(on_disk)}**")
    md.append(f"- Clips referenciados en TSV con archivo: **{len(rows_present)}**")
    md.append(f"- Clips con transcripción no vacía: **{len(rows_with_text)}**")
    if len(rows_without_text := rows_present) and len(rows_with_text) < len(rows_present):
        md.append(
            f"  - Sin transcripción usable: **{len(rows_present) - len(rows_with_text)}**"
        )
    md.append(f"- Oraciones únicas: **{len(sentences)}**")
    md.append(f"- Hablantes únicos (client_id): **{len(speakers)}**")
    md.append("")
    stats = summary["duration_stats"]
    if stats.get("n"):
        md.append("## Duraciones por clip")
        md.append("")
        md.append(f"- Fuente: `{summary['duration_source']}`  ·  n: **{stats['n']}**")
        md.append(
            f"- Total: **{stats['total_s']} s** ({stats['total_s'] / 60:.1f} min)  ·  "
            f"min/mediana/media/max: **{stats['min_s']} / {stats['median_s']} / "
            f"{stats['mean_s']} / {stats['max_s']} s**"
        )
        md.append("")

    def _table(name: str, counter: Counter, top: int = 10) -> list[str]:
        if not counter:
            return []
        out = [f"## {name}", "", "| valor | n |", "|---|---:|"]
        for val, n in counter.most_common(top):
            out.append(f"| {val or '_vacío_'} | {n} |")
        out.append("")
        return out

    md.extend(_table("Edad declarada", age_dist))
    md.extend(_table("Género declarado", gender_dist))
    md.extend(_table("Acentos declarados", accents))
    md.extend(_table("Variantes declaradas", variants))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(md), encoding="utf-8")
    Path(args.out).with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Ficha: {args.out}")
    print(f"Resumen: {Path(args.out).with_suffix('.json')}")


if __name__ == "__main__":
    main()
