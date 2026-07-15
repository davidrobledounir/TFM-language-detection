"""Descarga splits de FLEURS desde HuggingFace `datasets` y los vuelca al layout local que `src/datasets/fleurs.py` espera.

```
python scripts/fetch_fleurs.py --langs es_419,ca,eu --split test --out data/fleurs
```
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _dump_lang(lang: str, split: str, out_dir: Path, max_rows: int | None) -> int:
    from datasets import load_dataset  # importación tardía
    import soundfile as sf

    ds = load_dataset("google/fleurs", lang, split=split, streaming=False)
    lang_dir = out_dir / lang
    clips_dir = lang_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    tsv = lang_dir / f"{split}.tsv"
    n_written = 0
    with tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["path", "sentence"])
        for i, row in enumerate(ds):
            if max_rows is not None and n_written >= max_rows:
                break
            audio = row.get("audio") or {}
            arr = audio.get("array")
            sr = audio.get("sampling_rate", 16000)
            sent = row.get("raw_transcription") or row.get("transcription")
            if arr is None or not sent:
                continue
            name = f"{lang}_{i:06d}.wav"
            sf.write(clips_dir / name, arr, sr, subtype="PCM_16")
            writer.writerow([name, sent])
            n_written += 1
    return n_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--langs", default="es_419,ca,eu")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="data/fleurs")
    parser.add_argument("--max-rows", type=int, default=200,
                        help="Máximo por idioma. 0 = todos.")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = args.max_rows if args.max_rows > 0 else None
    for lang in [l.strip() for l in args.langs.split(",") if l.strip()]:
        try:
            n = _dump_lang(lang, args.split, out, cap)
            print(f"  {lang}: {n} clips en {out / lang}")
        except Exception as exc:
            print(f"  {lang}: ERROR ({exc.__class__.__name__}: {exc})")


if __name__ == "__main__":
    main()
