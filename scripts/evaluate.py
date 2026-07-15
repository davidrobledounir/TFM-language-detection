import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import boundary_metrics, lid_metrics
from src.evaluation import wer as wer_mod


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluación WER/CER, LID por segmento y fronteras de cambio."
    )
    parser.add_argument(
        "--hyp", required=True, help="JSONL con {id, text, segments?} por línea"
    )
    parser.add_argument(
        "--ref", required=True, help="JSONL con {id, text, segments?} por línea"
    )
    parser.add_argument("--lid-labels", default="spa,cat,eus")
    parser.add_argument("--boundary-tolerances", default="0.2,0.5,1.0")
    args = parser.parse_args()

    hyps = {row["id"]: row for row in _read_jsonl(Path(args.hyp))}
    refs = {row["id"]: row for row in _read_jsonl(Path(args.ref))}
    ids = sorted(set(hyps) & set(refs))
    if not ids:
        print(json.dumps({"error": "sin IDs comunes entre hyp y ref"}))
        sys.exit(1)

    hyp_texts = [hyps[i].get("text", "") for i in ids]
    ref_texts = [refs[i].get("text", "") for i in ids]
    out = {"items": len(ids), "transcription": wer_mod.score(hyp_texts, ref_texts)}

    lid_h: list[str] = []
    lid_r: list[str] = []
    bnd_h: list[float] = []
    bnd_r: list[float] = []
    for i in ids:
        h_segs = hyps[i].get("segments") or []
        r_segs = refs[i].get("segments") or []
        for h, r in zip(h_segs, r_segs):
            lid_h.append(h["lang"])
            lid_r.append(r["lang"])
        bnd_h.extend(s["start"] for s in h_segs[1:])
        bnd_r.extend(s["start"] for s in r_segs[1:])

    if lid_h:
        labels = args.lid_labels.split(",")
        out["lid"] = lid_metrics.score(lid_h, lid_r, labels)
    if bnd_h or bnd_r:
        tols = [float(t) for t in args.boundary_tolerances.split(",")]
        out["boundary"] = boundary_metrics.score(bnd_h, bnd_r, tols)

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
