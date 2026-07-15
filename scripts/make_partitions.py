"""Particiona un manifest JSONL en train/dev/test estratificado por idioma o densidad de cambio."""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def _stratum_key(entry: dict, mode: str) -> str:
    if mode == "lang":
        langs = entry.get("pipeline_langs") or [
            s.get("lang") for s in entry.get("segments", [])
        ]
        if not langs:
            return "_unknown"
        return ",".join(sorted(set(langs)))
    if mode == "density":
        n_segments = len(entry.get("segments", []))
        switches = max(0, n_segments - 1)
        dur = entry.get("duration_s", 0) or 0
        density = (switches / dur * 60) if dur > 0 else 0
        if density < 0.5:
            return "d_0"
        if density < 3:
            return "d_low"
        if density < 8:
            return "d_med"
        return "d_high"
    raise ValueError(f"mode desconocido: {mode}")


def _split_stratum(items: list, ratios: tuple[float, float, float], rng: random.Random) -> dict:
    rng.shuffle(items)
    n = len(items)
    n_train = int(round(n * ratios[0]))
    n_dev = int(round(n * ratios[1]))
    train = items[:n_train]
    dev = items[n_train: n_train + n_dev]
    test = items[n_train + n_dev:]
    return {"train": train, "dev": dev, "test": test}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--by",
        choices=["lang", "density"],
        default="lang",
        help="Estratificación: 'lang' o 'density'",
    )
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--dev", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    total = args.train + args.dev + args.test
    if abs(total - 1.0) > 1e-6:
        print(f"ERROR: ratios suman {total}, deben sumar 1.0")
        sys.exit(2)

    entries = [
        json.loads(line)
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not entries:
        print("ERROR: manifest vacío")
        sys.exit(1)

    rng = random.Random(args.seed)
    strata: dict[str, list] = defaultdict(list)
    for e in entries:
        strata[_stratum_key(e, args.by)].append(e)

    out: dict[str, list] = {"train": [], "dev": [], "test": []}
    summary: dict[str, dict] = {}
    for key, items in strata.items():
        split = _split_stratum(items, (args.train, args.dev, args.test), rng)
        for k, v in split.items():
            out[k].extend(v)
        summary[key] = {k: len(v) for k, v in split.items()}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, items in out.items():
        rng.shuffle(items)
        path = out_dir / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"  {split_name}: {len(items)} -> {path}")
    (out_dir / "strata_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Resumen por estrato: {out_dir / 'strata_summary.json'}")


if __name__ == "__main__":
    main()
