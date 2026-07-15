from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


TARGET_SR = 16000


def _load_with_faster_whisper(path: str, sample_rate: int) -> np.ndarray:
    from faster_whisper.audio import decode_audio

    return np.asarray(decode_audio(path, sampling_rate=sample_rate), dtype=np.float32)


def load_audio(path: str | Path, sample_rate: int = TARGET_SR) -> np.ndarray:
    path_str = str(path)
    try:
        audio, sr = sf.read(path_str, dtype="float32", always_2d=False)
    except Exception:
        return _load_with_faster_whisper(path_str, sample_rate)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        gcd = int(np.gcd(sr, sample_rate))
        audio = resample_poly(audio, sample_rate // gcd, sr // gcd).astype(np.float32)
    return audio.astype(np.float32, copy=False)


def rms_normalize(audio: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)) + 1e-12)
    target = 10.0 ** (target_dbfs / 20.0)
    return (audio * (target / rms)).astype(np.float32)
