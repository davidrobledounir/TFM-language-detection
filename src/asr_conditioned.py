import numpy as np
import torch
from faster_whisper import WhisperModel


MMS_TO_WHISPER = {"spa": "es", "cat": "ca", "eus": "eu"}


def _device_and_compute(
    compute_type_gpu: str, compute_type_cpu: str
) -> tuple[str, str]:
    if torch.cuda.is_available():
        return "cuda", compute_type_gpu
    return "cpu", compute_type_cpu


class ConditionedAsr:
    def __init__(
        self,
        model_size: str,
        compute_type_gpu: str,
        compute_type_cpu: str,
        beam_size: int,
        condition_on_previous_text: bool,
    ):
        device, compute = _device_and_compute(compute_type_gpu, compute_type_cpu)
        self.model = WhisperModel(model_size, device=device, compute_type=compute)
        self.beam_size = beam_size
        self.condition_on_previous_text = condition_on_previous_text

    def transcribe(self, audio: np.ndarray, lang: str) -> str:
        whisper_lang = MMS_TO_WHISPER.get(lang, lang)
        segments, _ = self.model.transcribe(
            audio,
            language=whisper_lang,
            beam_size=self.beam_size,
            condition_on_previous_text=self.condition_on_previous_text,
            vad_filter=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()


_ASR_CACHE: dict[tuple, "ConditionedAsr"] = {}


def get_conditioned_asr(
    model_size: str,
    compute_type_gpu: str,
    compute_type_cpu: str,
    beam_size: int,
    condition_on_previous_text: bool,
) -> "ConditionedAsr":
    key = (model_size, compute_type_gpu, compute_type_cpu, beam_size, condition_on_previous_text)
    if key not in _ASR_CACHE:
        _ASR_CACHE[key] = ConditionedAsr(
            model_size, compute_type_gpu, compute_type_cpu,
            beam_size, condition_on_previous_text,
        )
    return _ASR_CACHE[key]
