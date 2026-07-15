"""Generador sintético de code-switching por concatenación controlada de clips monolingües."""
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .datasets.common_voice import CVRow
from .io_audio import load_audio, rms_normalize


_ISO1_TO_MMS = {"eu": "eus", "es": "spa", "ca": "cat"}


def _to_mms(code: str) -> str:
    return _ISO1_TO_MMS.get(code, code)


@dataclass
class SynthSample:
    audio: np.ndarray
    sample_rate: int
    segments: list[dict]
    text: str
    languages: list[str]
    requested_density_per_min: float
    observed_switches: int


def _maybe_switch(
    rng: random.Random,
    current: str,
    langs: list[str],
    last_dur_s: float,
    density_per_min: float,
) -> str:
    if len(langs) <= 1 or density_per_min <= 0:
        return current
    p_switch = min(1.0, density_per_min * last_dur_s / 60.0)
    if rng.random() < p_switch:
        return rng.choice([l for l in langs if l != current])
    return current


def generate(
    pools: dict[str, list[CVRow]],
    target_duration_s: float,
    density_per_min: float,
    seed: int = 1337,
    pad_silence_ms: int = 120,
    sample_rate: int = 16000,
    rms_dbfs: float = -20.0,
    max_clip_s: float | None = None,
) -> SynthSample:
    if not pools:
        raise ValueError("pools vacío")
    rng = random.Random(seed)
    langs = sorted(pools.keys())
    current = rng.choice(langs)

    pad_samples = int(sample_rate * pad_silence_ms / 1000)
    pad = np.zeros(pad_samples, dtype=np.float32)

    chunks: list[np.ndarray] = []
    segments: list[dict] = []
    pieces_of_text: list[str] = []
    cursor_samples = 0
    switches = 0
    last_dur = 0.0

    while cursor_samples / sample_rate < target_duration_s:
        next_lang = _maybe_switch(rng, current, langs, last_dur, density_per_min)
        if next_lang != current:
            switches += 1
        current = next_lang

        pool = pools[current]
        if not pool:
            raise ValueError(f"pool vacío para {current}")
        row = rng.choice(pool)
        clip = load_audio(row.path, sample_rate)
        if clip.size == 0:
            continue
        if max_clip_s is not None and max_clip_s > 0:
            max_samples = int(max_clip_s * sample_rate)
            if clip.size > max_samples:
                start = rng.randint(0, clip.size - max_samples)
                clip = clip[start : start + max_samples]
        clip = rms_normalize(clip, rms_dbfs)

        start_s = cursor_samples / sample_rate
        chunks.append(clip)
        end_samples = cursor_samples + clip.size
        end_s = end_samples / sample_rate
        segments.append(
            {
                "start": round(start_s, 3),
                "end": round(end_s, 3),
                "lang": _to_mms(current),
                "text": row.sentence,
            }
        )
        pieces_of_text.append(row.sentence)

        chunks.append(pad)
        cursor_samples = end_samples + pad_samples
        last_dur = end_s - start_s

    audio = np.concatenate(chunks).astype(np.float32)
    return SynthSample(
        audio=audio,
        sample_rate=sample_rate,
        segments=segments,
        text=" ".join(pieces_of_text),
        languages=[_to_mms(l) for l in langs],
        requested_density_per_min=density_per_min,
        observed_switches=switches,
    )


def _real_boundaries(segments: list[dict]) -> list[float]:
    out: list[float] = []
    prev_lang = None
    for s in segments:
        if prev_lang is not None and s["lang"] != prev_lang:
            out.append(s["start"])
        prev_lang = s["lang"]
    return out


def write_sample(sample: SynthSample, out_dir: Path, sample_id: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{sample_id}.wav"
    sf.write(wav_path, sample.audio, sample.sample_rate, subtype="PCM_16")
    meta = {
        "id": sample_id,
        "duration_s": round(len(sample.audio) / sample.sample_rate, 3),
        "languages": sample.languages,
        "requested_density_per_min": sample.requested_density_per_min,
        "observed_switches": sample.observed_switches,
        "segments": sample.segments,
        "boundaries_s": _real_boundaries(sample.segments),
        "text": sample.text,
    }
    return meta
