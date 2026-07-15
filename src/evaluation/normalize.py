from transformers.models.whisper.english_normalizer import BasicTextNormalizer

_normalizer = BasicTextNormalizer()


def normalize(text: str) -> str:
    return _normalizer(text)
