"""Orquestador multi-semilla y multi-variante.

Para cada (variante × dataset × semilla) ejecuta el script correspondiente
(`analyze_sample.py` para muestras monolingües, `evaluate_synth.py` para
material con manifiesto temporal) y agrega los resultados.

Las variantes son combinaciones de baseline y pipeline definidas en `configs/`:

- `baseline_large` → configs/baseline.yaml
- `baseline_medium` → configs/baseline_medium.yaml
- `pipeline_mms` → configs/pipeline.yaml
- `pipeline_xlsr` → configs/pipeline_xlsr.yaml

```
python scripts/run_seeds.py --seeds 1337,2025,7 --datasets eu_sample,es_sample,ca_sample,synth,fleurs --variants baseline_large,baseline_medium,pipeline_mms,pipeline_xlsr
```
"""
import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


VARIANT_CONFIGS = {
    "baseline_large": ("baseline", "configs/baseline.yaml"),
    "baseline_medium": ("baseline", "configs/baseline_medium.yaml"),
    "pipeline_mms": ("pipeline", "configs/pipeline.yaml"),
    "pipeline_xlsr": ("pipeline", "configs/pipeline_xlsr.yaml"),
}


MONO_DATASETS = {
    "eu_sample": ("eu", "data/eu_sample"),
    "es_sample": ("es", "data/es_sample"),
    "ca_sample": ("ca", "data/ca_sample"),
    "fleurs_eu": ("eu", "data/fleurs/eu"),
    "fleurs_es": ("es_419", "data/fleurs/es_419"),
    "fleurs_ca": ("ca", "data/fleurs/ca"),
}

SYNTH_DATASETS = {"synth", "synth_es_ca", "synth_es_eu", "synth_ca_eu"}


def _run_mono(
    variant: str, baseline_cfg: str, pipeline_cfg: str, sample_dir: Path,
    locale: str, seed: int, out_runs: Path,
) -> Path:
    out = out_runs / f"{variant}_{sample_dir.name}_s{seed}.json"
    cmd = [
        sys.executable, str(ROOT / "scripts" / "analyze_sample.py"),
        "--sample-dir", str(sample_dir),
        "--locale", locale,
        "--pipeline-config", pipeline_cfg,
        "--baseline-config", baseline_cfg,
        "--out", str(out),
        "--seed", str(seed),
    ]
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return out


def _run_synth(
    pipeline_cfg: str, manifest: Path, seed: int, out_runs: Path, variant: str,
) -> Path:
    out = out_runs / f"{variant}_{manifest.parent.name}_s{seed}.json"
    cmd = [
        sys.executable, str(ROOT / "scripts" / "evaluate_synth.py"),
        "--manifest", str(manifest),
        "--pipeline-config", pipeline_cfg,
        "--out", str(out),
        "--seed", str(seed),
    ]
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="1337,2025,7")
    parser.add_argument(
        "--datasets",
        default="eu_sample,es_sample,ca_sample,synth",
        help="Coma-separados de la lista en MONO_DATASETS o SYNTH_DATASETS",
    )
    parser.add_argument(
        "--variants",
        default="baseline_large,baseline_medium,pipeline_mms,pipeline_xlsr",
    )
    parser.add_argument("--out-runs", default="data/runs")
    parser.add_argument(
        "--synth-manifest-template",
        default="data/synth/seed_{seed}/manifest.jsonl",
        help="Plantilla con {seed} para localizar el manifiesto sintético por semilla.",
    )
    parser.add_argument(
        "--synth-pair-template",
        default="data/synth_{combo}/seed_{seed}/manifest.jsonl",
        help="Plantilla para sub-corpus por combinación (es_ca, es_eu, ca_eu).",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    out_runs = Path(args.out_runs)
    out_runs.mkdir(parents=True, exist_ok=True)

    # Pre-cargar configs por variante.
    baseline_cfg = "configs/baseline.yaml"
    pipeline_cfg = "configs/pipeline.yaml"

    for variant in variants:
        if variant not in VARIANT_CONFIGS:
            print(f"  AVISO: variante desconocida {variant}")
            continue
        kind, cfg = VARIANT_CONFIGS[variant]
        if kind == "baseline":
            baseline_cfg = cfg
        else:
            pipeline_cfg = cfg

        for seed in seeds:
            for ds in datasets:
                if ds in MONO_DATASETS:
                    locale, root = MONO_DATASETS[ds]
                    sample_dir = Path(root)
                    if not sample_dir.exists():
                        print(f"  AVISO: {sample_dir} no existe, salto")
                        continue
                    _run_mono(variant, baseline_cfg, pipeline_cfg, sample_dir, locale, seed, out_runs)
                elif ds == "synth":
                    manifest = Path(args.synth_manifest_template.format(seed=seed))
                    if not manifest.exists():
                        # Compatibilidad con layout sin seed (manifest único en data/synth/).
                        fallback = Path("data/synth/manifest.jsonl")
                        if fallback.exists():
                            manifest = fallback
                    if not manifest.exists():
                        print(f"  AVISO: manifiesto sintético no existe para seed {seed}, salto")
                        continue
                    _run_synth(pipeline_cfg, manifest, seed, out_runs, variant)
                elif ds.startswith("synth_"):
                    combo = ds.replace("synth_", "")
                    manifest = Path(args.synth_pair_template.format(combo=combo, seed=seed))
                    if not manifest.exists():
                        print(f"  AVISO: {manifest} no existe, salto")
                        continue
                    _run_synth(pipeline_cfg, manifest, seed, out_runs, variant)
                else:
                    print(f"  AVISO: dataset desconocido {ds}")


if __name__ == "__main__":
    main()
