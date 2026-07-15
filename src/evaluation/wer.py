from jiwer import cer, wer

from .normalize import normalize


def score(hypotheses: list[str], references: list[str]) -> dict:
    hyps = [normalize(h) for h in hypotheses]
    refs = [normalize(r) for r in references]
    return {"wer": float(wer(refs, hyps)), "cer": float(cer(refs, hyps))}
