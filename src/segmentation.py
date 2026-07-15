from dataclasses import dataclass

import numpy as np


@dataclass
class Window:
    start: float
    end: float
    audio: np.ndarray


def _silero_speech(
    audio: np.ndarray,
    sample_rate: int,
    threshold: float,
    min_speech_ms: int,
    min_silence_ms: int,
) -> list[tuple[float, float]]:
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad()
    tensor = torch.from_numpy(audio).float()
    raw = get_speech_timestamps(
        tensor,
        model,
        sampling_rate=sample_rate,
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
    )
    return [(d["start"] / sample_rate, d["end"] / sample_rate) for d in raw]


def _energy_speech(
    audio: np.ndarray,
    sample_rate: int,
    min_speech_ms: int,
    min_silence_ms: int,
    frame_ms: int = 30,
    energy_percentile: float = 60.0,
) -> list[tuple[float, float]]:
    frame = max(1, int(sample_rate * frame_ms / 1000))
    n_frames = len(audio) // frame
    if n_frames == 0:
        return []
    trimmed = audio[: n_frames * frame].reshape(n_frames, frame)
    energy = np.sqrt(np.mean(trimmed.astype(np.float64) ** 2, axis=1))
    if energy.max() <= 1e-9:
        return []
    threshold = max(np.percentile(energy, energy_percentile), 1e-4)
    active = energy > threshold

    min_speech_frames = max(1, int(min_speech_ms / frame_ms))
    min_silence_frames = max(1, int(min_silence_ms / frame_ms))

    segments: list[tuple[int, int]] = []
    i = 0
    while i < n_frames:
        if active[i]:
            j = i
            silence_run = 0
            while j < n_frames:
                if active[j]:
                    silence_run = 0
                    j += 1
                else:
                    silence_run += 1
                    if silence_run >= min_silence_frames:
                        break
                    j += 1
            end = j - silence_run
            if end - i >= min_speech_frames:
                segments.append((i, end))
            i = j
        else:
            i += 1

    return [
        (s * frame / sample_rate, e * frame / sample_rate) for s, e in segments
    ]


def detect_speech(
    audio: np.ndarray,
    sample_rate: int,
    threshold: float,
    min_speech_ms: int,
    min_silence_ms: int,
    backend: str = "auto",
) -> list[tuple[float, float]]:
    if backend in ("auto", "silero"):
        try:
            return _silero_speech(
                audio, sample_rate, threshold, min_speech_ms, min_silence_ms
            )
        except Exception as exc:
            if backend == "silero":
                raise
            import warnings

            warnings.warn(
                f"silero-vad no disponible ({exc.__class__.__name__}), usando VAD energético"
            )
    return _energy_speech(audio, sample_rate, min_speech_ms, min_silence_ms)


def refine_windows_by_lid(
    audio: np.ndarray,
    sample_rate: int,
    windows: list["Window"],
    preds: list,
    min_confidence: float,
    min_margin: float,
    min_refined_s: float,
) -> list["Window"]:
    """Subdivide ventanas inestables (LID con baja confianza o margen estrecho).

    Cada ventana cuya predicción tenga ``confidence < min_confidence`` o
    ``margin < min_margin`` se parte en dos mitades del tamaño igual o hasta
    que la duración resultante quede por debajo de ``min_refined_s``.
    Las ventanas estables se conservan sin cambios.
    """
    if not windows or not preds or len(windows) != len(preds):
        return list(windows)
    refined: list[Window] = []
    for w, p in zip(windows, preds):
        conf = getattr(p, "confidence", 1.0)
        margin = getattr(p, "margin", 1.0)
        dur = w.end - w.start
        unstable = conf < min_confidence or margin < min_margin
        if unstable and dur >= 2 * min_refined_s:
            mid = (w.start + w.end) / 2.0
            a_idx = int(w.start * sample_rate)
            m_idx = int(mid * sample_rate)
            e_idx = int(w.end * sample_rate)
            refined.append(Window(w.start, mid, audio[a_idx:m_idx]))
            refined.append(Window(mid, w.end, audio[m_idx:e_idx]))
        else:
            refined.append(w)
    return refined


def adaptive_windows(
    audio: np.ndarray,
    sample_rate: int,
    speech: list[tuple[float, float]],
    min_s: float,
    max_s: float,
    target_s: float,
) -> list[Window]:
    windows: list[Window] = []
    for s, e in speech:
        duration = e - s
        if duration <= 0:
            continue
        n_chunks = max(1, int(round(duration / target_s)))
        chunk = duration / n_chunks
        if chunk < min_s and n_chunks > 1:
            n_chunks = max(1, int(np.floor(duration / min_s)))
            chunk = duration / max(n_chunks, 1)
        if chunk > max_s:
            n_chunks = int(np.ceil(duration / max_s))
            chunk = duration / n_chunks
        for i in range(n_chunks):
            cs = s + i * chunk
            ce = e if i == n_chunks - 1 else min(e, cs + chunk)
            a = audio[int(cs * sample_rate): int(ce * sample_rate)]
            if a.size > 0:
                windows.append(Window(cs, ce, a))
    return windows
