from pathlib import Path

import torch
import yaml
from faster_whisper import WhisperModel

from .seed import set_seed
from .io_audio import load_audio, rms_normalize


def _device_and_compute(
    compute_type_gpu: str, compute_type_cpu: str
) -> tuple[str, str]:
    if torch.cuda.is_available():
        return "cuda", compute_type_gpu
    return "cpu", compute_type_cpu


_BASELINE_CACHE: dict[tuple, WhisperModel] = {}


def _get_model(model_size: str, device: str, compute: str) -> WhisperModel:
    key = (model_size, device, compute)
    if key not in _BASELINE_CACHE:
        _BASELINE_CACHE[key] = WhisperModel(model_size, device=device, compute_type=compute)
    return _BASELINE_CACHE[key]


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(audio_path: str | Path, config: dict) -> str:
    set_seed(config["seed"])
    sr = config["audio"]["sample_rate"]
    audio = load_audio(audio_path, sr)
    audio = rms_normalize(audio, config["audio"]["rms_target_dbfs"])

    asr_cfg = config["asr"]
    device, compute = _device_and_compute(
        asr_cfg["compute_type_gpu"], asr_cfg["compute_type_cpu"]
    )
    model = _get_model(asr_cfg["model_size"], device, compute)
    segments, _ = model.transcribe(
        audio,
        language=asr_cfg["language"],
        beam_size=asr_cfg["beam_size"],
        condition_on_previous_text=asr_cfg["condition_on_previous_text"],
        vad_filter=False,
    )
    return " ".join(s.text.strip() for s in segments).strip()
