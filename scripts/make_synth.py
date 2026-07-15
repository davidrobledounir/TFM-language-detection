"""Genera muestras sintéticas de code-switching a partir de pools por idioma de Common Voice."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.common_voice import load as load_cv
from src.synth_codeswitch import generate, write_sample


def parse_pool_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"esperado <locale>=<dir>, recibido {spec!r}")
    locale, root = spec.split("=", 1)
    return locale, Path(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool",
        action="append",
        required=True,
        help="Pool por idioma: <locale>=<dir_cv>. Repetir por cada idioma.",
    )
    parser.add_argument("--out", default="data/synth")
    parser.add_argument("--n", type=int, default=5, help="Número de muestras a generar")
    parser.add_argument("--target-s", type=float, default=20.0)
    parser.add_argument(
        "--density",
        type=float,
        default=4.0,
        help="Cambios de idioma esperados por minuto",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Lista de semillas separadas por coma (p.ej. 1337,2025,7). "
             "Si se pasa, anula --seed y emite un manifiesto por semilla en out/seed_<n>/.",
    )
    parser.add_argument("--max-rows-per-pool", type=int, default=200)
    parser.add_argument(
        "--max-clip-s",
        type=float,
        default=5.0,
        help="Recorta cada clip fuente a este máximo (s) antes de concatenar. 0 = sin recorte.",
    )
    args = parser.parse_args()

    pools = {}
    for spec in args.pool:
        locale, root = parse_pool_spec(spec)
        rows = load_cv(root, locale, max_rows=args.max_rows_per_pool)
        if not rows:
            print(f"  AVISO: pool vacío para {locale} en {root}")
            continue
        pools[locale] = rows
        print(f"  pool {locale}: {len(rows)} clips desde {root}")

    if len(pools) < 1:
        print("ERROR: ningún pool válido")
        sys.exit(2)
    if len(pools) < 2 and args.density > 0:
        print("AVISO: density>0 con un solo idioma equivale a 0 (no hay con quién cambiar)")

    out_base = Path(args.out)
    out_base.mkdir(parents=True, exist_ok=True)
    seeds = (
        [int(s) for s in args.seeds.split(",") if s.strip()]
        if args.seeds
        else [args.seed]
    )

    for seed in seeds:
        out = out_base / f"seed_{seed}" if len(seeds) > 1 else out_base
        out.mkdir(parents=True, exist_ok=True)
        manifest: list[dict] = []
        for i in range(args.n):
            sample_id = f"synth_{i:04d}_d{int(args.density)}_s{seed}"
            sample = generate(
                pools,
                target_duration_s=args.target_s,
                density_per_min=args.density,
                seed=seed + i,
                max_clip_s=args.max_clip_s if args.max_clip_s > 0 else None,
            )
            meta = write_sample(sample, out, sample_id)
            manifest.append(meta)
            print(
                f"  [seed={seed} {i + 1}/{args.n}] {sample_id}.wav  dur={meta['duration_s']}s  "
                f"switches={meta['observed_switches']}  langs={meta['languages']}"
            )

        manifest_path = out / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as f:
            for m in manifest:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
