"""Evalúa el pipeline sobre el manifest de un corpus sintético de code-switching."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import pipeline
from src.boundaries import boundary_points
from src.evaluation import (
    boundary_metrics,
    bootstrap as bootstrap_mod,
    cost as cost_mod,
    desagregate,
    frame_lid,
    segment_metrics,
)
from src.evaluation import wer as wer_mod


def _read_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Ruta a synth/manifest.jsonl")
    parser.add_argument("--audio-dir", default=None, help="Si los wavs no están junto al manifest")
    parser.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    parser.add_argument("--out", default=None, help="JSON de salida")
    parser.add_argument(
        "--lid-labels",
        default="spa,eus",
        help="Etiquetas usadas en el F1 macro (coma)",
    )
    parser.add_argument(
        "--boundary-tolerances",
        default="0.2,0.5,1.0",
        help="Tolerancias de frontera en s (coma)",
    )
    parser.add_argument("--frame-hz", type=float, default=10.0)
    parser.add_argument("--max", type=int, default=0, help="0 = todos")
    parser.add_argument("--seed", type=int, default=1337,
                        help="Sobrescribe la semilla declarada en el YAML del pipeline.")
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    audio_dir = Path(args.audio_dir) if args.audio_dir else manifest_path.parent
    entries = _read_manifest(manifest_path)
    if args.max:
        entries = entries[: args.max]
    if not entries:
        print("ERROR: manifest vacío")
        sys.exit(1)

    pcfg = pipeline.load_config(args.pipeline_config)
    pcfg["seed"] = args.seed
    lid_labels = args.lid_labels.split(",")
    tols = [float(t) for t in args.boundary_tolerances.split(",")]
    segment_lid_pred: list[str] = []
    segment_lid_gold: list[str] = []
    length_buckets: dict[str, dict[str, list[str]]] = {}
    density_entries: list[dict] = []

    per_clip: list[dict] = []
    all_hyp_text: list[str] = []
    all_ref_text: list[str] = []
    all_bnd_h: list[float] = []
    all_bnd_r: list[float] = []
    confusion_acc = None
    frames_total = 0
    cost_rec = {"elapsed_s": 0.0, "cpu_peak_mb": 0.0, "gpu_peak_mb": 0.0}
    audio_total_s = 0.0

    print(f"Evaluando {len(entries)} muestras")
    t0 = time.time()
    for i, entry in enumerate(entries, 1):
        wav = audio_dir / f"{entry['id']}.wav"
        if not wav.exists():
            print(f"  AVISO: falta {wav}, salto")
            continue
        with cost_mod.measure(f"pipeline:{entry['id']}") as c:
            pred_segments = pipeline.run(wav, pcfg)
        t_p = c["elapsed_s"]
        cost_rec["elapsed_s"] += c["elapsed_s"]
        cost_rec["cpu_peak_mb"] = max(cost_rec["cpu_peak_mb"], c["cpu_peak_mb"])
        cost_rec["gpu_peak_mb"] = max(cost_rec["gpu_peak_mb"], c["gpu_peak_mb"])
        audio_total_s += float(entry["duration_s"])

        hyp_text = " ".join(s.get("text", "") for s in pred_segments).strip()
        ref_text = entry["text"]
        all_hyp_text.append(hyp_text)
        all_ref_text.append(ref_text)

        gold_segs = entry["segments"]
        hyp_bnd = boundary_points(pred_segments) if pred_segments else []
        gold_bnd = entry.get("boundaries_s")
        if gold_bnd is None:
            gold_bnd = boundary_points(gold_segs)
        all_bnd_h.extend(hyp_bnd)
        all_bnd_r.extend(gold_bnd)

        flid = frame_lid.score(
            pred_segments,
            gold_segs,
            duration_s=entry["duration_s"],
            labels=lid_labels,
            frame_hz=args.frame_hz,
        )
        seg_align = segment_metrics.align_segments(pred_segments, gold_segs)
        for p, g in seg_align:
            segment_lid_pred.append(p)
            segment_lid_gold.append(g)
        density_entries.append({
            "id": entry["id"],
            "duration_s": entry["duration_s"],
            "segments": gold_segs,
            "hyp_text": hyp_text,
            "ref_text": ref_text,
        })
        # Acumular pares por bucket de longitud de segmento gold para WER global por banda.
        for g in gold_segs:
            length = float(g["end"] - g["start"])
            text_ref = g.get("text") or ""
            if length <= 0 or not text_ref:
                continue
            # Hipótesis: mejor solape con un segmento del pipeline.
            best, best_ov = None, 0.0
            for p in pred_segments:
                lo, hi = max(p["start"], g["start"]), min(p["end"], g["end"])
                ov = max(0.0, hi - lo)
                if ov > best_ov:
                    best, best_ov = p, ov
            text_hyp = (best.get("text") if best else "") or ""
            for lo_b, hi_b in desagregate.DEFAULT_LENGTH_BANDS_S:
                if lo_b <= length < hi_b:
                    name = f">={int(lo_b)}s" if hi_b >= 1e9 else f"{int(lo_b)}-{int(hi_b)}s"
                    bucket = length_buckets.setdefault(name, {"hyp": [], "ref": []})
                    bucket["hyp"].append(text_hyp)
                    bucket["ref"].append(text_ref)
                    break
        if confusion_acc is None:
            import numpy as np

            confusion_acc = np.array(flid["confusion"]) if flid["confusion"] else None
        else:
            import numpy as np

            if flid["confusion"]:
                confusion_acc = confusion_acc + np.array(flid["confusion"])
        frames_total += flid["frames"]

        per_clip.append(
            {
                "id": entry["id"],
                "duration_s": entry["duration_s"],
                "elapsed_pipeline_s": round(t_p, 2),
                "ref_text": ref_text,
                "hyp_text": hyp_text,
                "ref_segments": gold_segs,
                "hyp_segments": pred_segments,
                "frame_lid_f1": flid["f1_macro"],
                "frames": flid["frames"],
            }
        )
        print(
            f"  [{i}/{len(entries)}] {entry['id']}  "
            f"dur={entry['duration_s']}s  pipeline={t_p:.1f}s  "
            f"frame_lid_f1={flid['f1_macro']:.3f}"
        )

    elapsed = time.time() - t0

    transcription = wer_mod.score(all_hyp_text, all_ref_text)
    transcription["bootstrap"] = bootstrap_mod.bootstrap_wer_cer(
        all_hyp_text, all_ref_text, n=args.bootstrap_n, seed=args.seed,
    )
    boundary = boundary_metrics.score(all_bnd_h, all_bnd_r, tols)
    seg_lid = (
        segment_metrics.score([], [], lid_labels) if not segment_lid_pred else {
            "f1_macro": float(
                __import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(
                    segment_lid_gold, segment_lid_pred, labels=lid_labels,
                    average="macro", zero_division=0,
                )
            ),
            "n": len(segment_lid_pred),
            "labels": lid_labels,
            "bootstrap": bootstrap_mod.bootstrap_f1_macro(
                segment_lid_pred, segment_lid_gold, lid_labels,
                n=args.bootstrap_n, seed=args.seed + 200,
            ),
        }
    )

    length_wer = {
        bucket: wer_mod.score(v["hyp"], v["ref"]) | {"n": len(v["hyp"])}
        for bucket, v in sorted(length_buckets.items())
    }
    density_wer = desagregate.by_density(density_entries)
    if confusion_acc is not None and frames_total:
        from sklearn.metrics import f1_score
        import numpy as np

        cm = confusion_acc
        per_class_f1 = []
        for i, _ in enumerate(lid_labels):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            per_class_f1.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
        macro_f1 = float(np.mean(per_class_f1))
        lid_global = {
            "f1_macro": macro_f1,
            "confusion": cm.tolist(),
            "labels": lid_labels,
            "frames": frames_total,
        }
    else:
        lid_global = None

    summary = {
        "n": len(per_clip),
        "elapsed_s": round(elapsed, 1),
        "seed": args.seed,
        "transcription": transcription,
        "lid_frame": lid_global,
        "lid_segment": seg_lid,
        "boundary": boundary,
        "wer_by_segment_length": length_wer,
        "wer_by_density": density_wer,
        "cost": cost_mod.per_minute_audio(cost_rec, audio_total_s) | {"audio_total_s": audio_total_s},
        "per_clip": per_clip,
    }

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Resultados: {args.out}")

    print()
    print(f"--- Resumen sintético ({len(per_clip)} muestras, {elapsed:.1f}s) ---")
    print(f"WER={transcription['wer']:.3f}  CER={transcription['cer']:.3f}")
    if lid_global:
        print(
            f"LID por frame F1-macro={lid_global['f1_macro']:.3f} "
            f"({lid_global['frames']} frames @ {args.frame_hz} Hz)"
        )
    for tol_key, vals in boundary.items():
        me = vals["mean_error_s"]
        me_str = f"{me:.3f}s" if me is not None else "—"
        print(
            f"Frontera {tol_key}: F1={vals['f1']:.3f}  "
            f"P={vals['precision']:.3f}  R={vals['recall']:.3f}  err_med={me_str}"
        )


if __name__ == "__main__":
    main()
