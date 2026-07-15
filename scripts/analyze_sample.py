"""Corre baseline y pipeline sobre la muestra de Common Voice descargada y reporta métricas."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import baseline_whisper, pipeline
from src.datasets.common_voice import load_transcripts_map
from src.evaluation import bootstrap as bootstrap_mod, cost as cost_mod, lid_metrics, wer as wer_mod
from src.evaluation.normalize import normalize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", default="data/eu_sample")
    parser.add_argument("--locale", default="eu", help="ISO 639-1 (eu, es, ca)")
    parser.add_argument("--expected-lid", default=None,
                        help="Etiqueta MMS esperada (eus/spa/cat). Si se omite, se infiere de --locale")
    parser.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    parser.add_argument("--baseline-config", default="configs/baseline.yaml")
    parser.add_argument("--out", default=None,
                        help="JSON de salida. Por defecto <sample-dir>/results.json")
    parser.add_argument("--max-clips", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1337,
                        help="Sobrescribe la semilla declarada en los YAMLs.")
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--lid-labels", default="spa,eus",
                        help="Etiquetas LID a usar en F1 macro y matriz de confusión.")
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    clips_dir = sample_dir / "clips"
    transcripts = load_transcripts_map(sample_dir, args.locale)
    locale_to_mms = {"eu": "eus", "es": "spa", "ca": "cat"}
    expected_lid = args.expected_lid or locale_to_mms.get(args.locale, args.locale)
    out_path = Path(args.out) if args.out else sample_dir / "results.json"

    clips = sorted(p for p in clips_dir.glob("*.mp3") if p.name in transcripts)[
        : args.max_clips
    ]
    if not clips:
        clips = sorted(p for p in clips_dir.glob("*.wav") if p.name in transcripts)[
            : args.max_clips
        ]
    if not clips:
        print("ERROR: ningún clip con transcripción válida encontrado")
        sys.exit(1)
    print(f"Procesando {len(clips)} clips")

    pcfg = pipeline.load_config(args.pipeline_config)
    bcfg = baseline_whisper.load_config(args.baseline_config)
    pcfg["seed"] = args.seed
    bcfg["seed"] = args.seed

    rows: list[dict] = []
    t0 = time.time()
    baseline_cost_rec = {"elapsed_s": 0.0, "cpu_peak_mb": 0.0, "gpu_peak_mb": 0.0}
    pipeline_cost_rec = {"elapsed_s": 0.0, "cpu_peak_mb": 0.0, "gpu_peak_mb": 0.0}
    audio_total_s = 0.0
    for idx, clip in enumerate(clips, 1):
        ref = transcripts[clip.name]
        with cost_mod.measure(f"baseline:{clip.name}") as cb:
            baseline_text = baseline_whisper.run(clip, bcfg)
        with cost_mod.measure(f"pipeline:{clip.name}") as cp:
            segments = pipeline.run(clip, pcfg)
        t_b = cb["elapsed_s"]
        t_p = cp["elapsed_s"]
        baseline_cost_rec["elapsed_s"] += cb["elapsed_s"]
        baseline_cost_rec["cpu_peak_mb"] = max(baseline_cost_rec["cpu_peak_mb"], cb["cpu_peak_mb"])
        baseline_cost_rec["gpu_peak_mb"] = max(baseline_cost_rec["gpu_peak_mb"], cb["gpu_peak_mb"])
        pipeline_cost_rec["elapsed_s"] += cp["elapsed_s"]
        pipeline_cost_rec["cpu_peak_mb"] = max(pipeline_cost_rec["cpu_peak_mb"], cp["cpu_peak_mb"])
        pipeline_cost_rec["gpu_peak_mb"] = max(pipeline_cost_rec["gpu_peak_mb"], cp["gpu_peak_mb"])
        try:
            import soundfile as sf

            audio_total_s += float(sf.info(str(clip)).duration)
        except Exception:
            pass
        pipeline_text = " ".join(s.get("text", "") for s in segments).strip()
        pipeline_langs = [s.get("lang") for s in segments]
        rows.append(
            {
                "id": clip.name,
                "ref": ref,
                "baseline": baseline_text,
                "baseline_s": round(t_b, 2),
                "pipeline": pipeline_text,
                "pipeline_s": round(t_p, 2),
                "pipeline_langs": pipeline_langs,
                "pipeline_segments": segments,
            }
        )
        print(
            f"[{idx}/{len(clips)}] {clip.name}  b={t_b:.1f}s  p={t_p:.1f}s  langs={pipeline_langs}"
        )

    elapsed = time.time() - t0

    baseline_score = wer_mod.score(
        [r["baseline"] for r in rows], [r["ref"] for r in rows]
    )
    pipeline_score = wer_mod.score(
        [r["pipeline"] for r in rows], [r["ref"] for r in rows]
    )

    lid_pred = []
    lid_true = []
    for r in rows:
        for lang in r["pipeline_langs"]:
            lid_pred.append(lang)
            lid_true.append(expected_lid)
    lid = lid_metrics.score(lid_pred, lid_true, args.lid_labels.split(",")) if lid_pred else None

    baseline_boot = bootstrap_mod.bootstrap_wer_cer(
        [r["baseline"] for r in rows], [r["ref"] for r in rows], n=args.bootstrap_n, seed=args.seed,
    )
    pipeline_boot = bootstrap_mod.bootstrap_wer_cer(
        [r["pipeline"] for r in rows], [r["ref"] for r in rows], n=args.bootstrap_n, seed=args.seed + 100,
    )
    lid_boot = (
        bootstrap_mod.bootstrap_f1_macro(lid_pred, lid_true, args.lid_labels.split(","),
                                          n=args.bootstrap_n, seed=args.seed + 200)
        if lid_pred else None
    )

    summary = {
        "n": len(rows),
        "elapsed_s": round(elapsed, 1),
        "seed": args.seed,
        "baseline": baseline_score | {"bootstrap": baseline_boot},
        "pipeline": pipeline_score | {"bootstrap": pipeline_boot},
        "lid": (lid | {"bootstrap": lid_boot}) if lid else None,
        "cost": {
            "baseline": cost_mod.per_minute_audio(baseline_cost_rec, audio_total_s),
            "pipeline": cost_mod.per_minute_audio(pipeline_cost_rec, audio_total_s),
            "audio_total_s": audio_total_s,
        },
        "samples": [
            {
                "id": r["id"],
                "ref": normalize(r["ref"]),
                "baseline": normalize(r["baseline"]),
                "pipeline": normalize(r["pipeline"]),
                "pipeline_langs": r["pipeline_langs"],
            }
            for r in rows[:5]
        ],
        "rows": rows,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print()
    print(f"--- Resumen ({len(rows)} clips, {elapsed:.1f}s totales) ---")
    print(f"Baseline  WER={baseline_score['wer']:.3f}  CER={baseline_score['cer']:.3f}")
    print(f"Pipeline  WER={pipeline_score['wer']:.3f}  CER={pipeline_score['cer']:.3f}")
    if lid:
        print(f"LID       F1-macro={lid['f1_macro']:.3f}")
        print(f"          confusión (filas={lid['labels']}): {lid['confusion']}")
    print(f"Resultados completos: {out_path}")


if __name__ == "__main__":
    main()
