"""Identificación de idioma por ventana con backends seleccionables (MMS, XLS-R)."""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


@dataclass
class LidPrediction:
    label: str
    logp: dict[str, float]
    confidence: float

    @property
    def margin(self) -> float:
        ordered = sorted(self.logp.values(), reverse=True)
        if len(ordered) < 2:
            return 0.0
        return float(np.exp(ordered[0]) - np.exp(ordered[1]))


class LidBackend(ABC):
    @abstractmethod
    def predict(self, audio: np.ndarray, sample_rate: int) -> LidPrediction: ...


class _HFClassifierLid(LidBackend):
    """Backend genérico para modelos HF de clasificación de audio con `id2label`.

    Se usa tanto para MMS-LID-126 (`facebook/mms-lid-126`) como para variantes XLS-R
    ajustadas a clasificación de idioma (e.g. `facebook/mms-lid-256`,
    `Mr-MMS/Wav2Vec2-XLSR-LID`, etc.).
    """

    def __init__(self, model_id: str, languages: list[str], device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = (
            AutoModelForAudioClassification.from_pretrained(model_id)
            .to(self.device)
            .eval()
        )
        id2label = self.model.config.id2label
        keep_ids = [i for i, lab in id2label.items() if lab in languages]
        if not keep_ids:
            available = sorted(set(id2label.values()))
            raise ValueError(
                f"Ninguna etiqueta de {languages} disponible en {model_id}. "
                f"Ejemplos disponibles: {available[:10]}"
            )
        self.keep_ids = keep_ids
        self.id_to_label = {i: id2label[i] for i in keep_ids}

    @torch.inference_mode()
    def predict(self, audio: np.ndarray, sample_rate: int) -> LidPrediction:
        inputs = self.feature_extractor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits[0]
        sub = logits[self.keep_ids]
        log_probs = torch.log_softmax(sub, dim=-1).cpu().numpy()
        ordered_labels = [self.id_to_label[i] for i in self.keep_ids]
        best = int(np.argmax(log_probs))
        return LidPrediction(
            label=ordered_labels[best],
            logp={lab: float(log_probs[i]) for i, lab in enumerate(ordered_labels)},
            confidence=float(np.exp(log_probs[best])),
        )


class MmsLid(_HFClassifierLid):
    """Backend MMS-LID (Facebook AI) — equivalente al `LidModel` original."""


class XlsrLid(_HFClassifierLid):
    """Backend XLS-R ajustado a clasificación de idioma.

    La interfaz es idéntica a `MmsLid`: cualquier modelo HF con cabecera
    `AudioClassification` y `id2label` que contenga las etiquetas declaradas
    es válido. El `model_id` se configura desde el YAML.
    """


# Alias para retro-compatibilidad con imports antiguos.
LidModel = MmsLid


_BACKENDS: dict[str, type[LidBackend]] = {
    "mms": MmsLid,
    "xlsr": XlsrLid,
}


_LID_CACHE: dict[tuple, LidBackend] = {}


def get_lid_model(
    model_id: str,
    languages: list[str],
    device: str | None = None,
    backend: str = "mms",
) -> LidBackend:
    backend = backend.lower()
    if backend not in _BACKENDS:
        raise ValueError(f"backend LID desconocido: {backend}. Opciones: {list(_BACKENDS)}")
    key = (backend, model_id, tuple(sorted(languages)), device or "")
    if key not in _LID_CACHE:
        _LID_CACHE[key] = _BACKENDS[backend](model_id, languages, device)
    return _LID_CACHE[key]


def smooth_labels(
    preds: list[LidPrediction],
    window: int,
    hysteresis: float,
) -> list[str]:
    if not preds:
        return []
    labels = list(preds[0].logp.keys())
    n = len(preds)
    logp_matrix = np.array([[p.logp[l] for l in labels] for p in preds])
    smoothed: list[str] = []
    current_idx = int(np.argmax(logp_matrix[0]))
    half = max(1, window // 2)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        avg = logp_matrix[lo:hi].mean(axis=0)
        order = np.argsort(avg)[::-1]
        top_idx = int(order[0])
        if top_idx != current_idx:
            margin = float(avg[top_idx] - avg[current_idx])
            if margin >= hysteresis:
                current_idx = top_idx
        smoothed.append(labels[current_idx])
    return smoothed
