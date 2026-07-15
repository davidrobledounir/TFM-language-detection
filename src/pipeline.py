from pathlib import Path

import yaml

from .seed import set_seed
from .io_audio import load_audio, rms_normalize
from .segmentation import detect_speech, adaptive_windows, refine_windows_by_lid
from .lid import get_lid_model, smooth_labels
from .boundaries import boundaries_from_labels
from .asr_conditioned import get_conditioned_asr
from .postprocess import consolidate


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(audio_path: str | Path, config: dict) -> list[dict]:
    set_seed(config["seed"])
    sr = config["audio"]["sample_rate"]
    audio = load_audio(audio_path, sr)
    audio = rms_normalize(audio, config["audio"]["rms_target_dbfs"])

    seg_cfg = config["segmentation"]
    vad_cfg = seg_cfg["vad"]
    win_cfg = seg_cfg["window"]
    speech = detect_speech(
        audio,
        sr,
        threshold=vad_cfg["threshold"],
        min_speech_ms=vad_cfg["min_speech_ms"],
        min_silence_ms=vad_cfg["min_silence_ms"],
        backend=vad_cfg.get("backend", "auto"),
    )
    windows = adaptive_windows(
        audio,
        sr,
        speech,
        min_s=win_cfg["min_s"],
        max_s=win_cfg["max_s"],
        target_s=win_cfg["target_s"],
    )
    if not windows:
        return []

    lid_cfg = config["lid"]
    lid = get_lid_model(
        lid_cfg["model_id"],
        lid_cfg["languages"],
        backend=lid_cfg.get("backend", "mms"),
    )
    preds = [lid.predict(w.audio, sr) for w in windows]

    refine_cfg = seg_cfg.get("refine") or {}
    if refine_cfg.get("enabled", False):
        windows = refine_windows_by_lid(
            audio,
            sr,
            windows,
            preds,
            min_confidence=float(refine_cfg.get("min_confidence", 0.6)),
            min_margin=float(refine_cfg.get("min_margin", 0.15)),
            min_refined_s=float(refine_cfg.get("min_refined_s", 1.0)),
        )
        preds = [lid.predict(w.audio, sr) for w in windows]

    labels = smooth_labels(
        preds,
        window=lid_cfg["smoothing"]["window"],
        hysteresis=lid_cfg["smoothing"]["hysteresis"],
    )

    starts = [w.start for w in windows]
    ends = [w.end for w in windows]
    pre_segments = boundaries_from_labels(labels, starts, ends)

    asr_cfg = config["asr"]
    asr = get_conditioned_asr(
        model_size=asr_cfg["model_size"],
        compute_type_gpu=asr_cfg["compute_type_gpu"],
        compute_type_cpu=asr_cfg["compute_type_cpu"],
        beam_size=asr_cfg["beam_size"],
        condition_on_previous_text=asr_cfg["condition_on_previous_text"],
    )

    enriched: list[dict] = []
    for seg in pre_segments:
        s_idx = int(seg["start"] * sr)
        e_idx = int(seg["end"] * sr)
        text = asr.transcribe(audio[s_idx:e_idx], seg["lang"])
        confs = [
            p.confidence
            for w, p in zip(windows, preds)
            if w.start >= seg["start"] - 1e-6 and w.end <= seg["end"] + 1e-6
        ]
        lid_conf = float(sum(confs) / len(confs)) if confs else 0.0
        enriched.append({**seg, "text": text, "lid_conf": lid_conf})

    pp_cfg = config.get("postprocess", {})
    if pp_cfg.get("merge_consecutive_same_lang", True):
        enriched = consolidate(enriched)

    if pp_cfg.get("emit_nonspeech", False):
        enriched = _fill_nonspeech(enriched, total_duration_s=len(audio) / sr)

    return enriched


def _fill_nonspeech(segments: list[dict], total_duration_s: float) -> list[dict]:
    if not segments:
        if total_duration_s > 0:
            return [{"start": 0.0, "end": float(total_duration_s), "lang": "nonspeech", "text": "", "lid_conf": 0.0}]
        return []
    out: list[dict] = []
    cursor = 0.0
    for seg in segments:
        if seg["start"] > cursor + 1e-3:
            out.append({"start": cursor, "end": seg["start"], "lang": "nonspeech", "text": "", "lid_conf": 0.0})
        out.append(seg)
        cursor = seg["end"]
    if total_duration_s > cursor + 1e-3:
        out.append({"start": cursor, "end": float(total_duration_s), "lang": "nonspeech", "text": "", "lid_conf": 0.0})
    return out
