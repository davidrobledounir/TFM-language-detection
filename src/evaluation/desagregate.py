"""Desagregación de WER/CER por banda de longitud de segmento y densidad de cambio."""
from .wer import score as wer_score


DEFAULT_LENGTH_BANDS_S = ((0.0, 5.0), (5.0, 15.0), (15.0, 30.0), (30.0, 1e9))
DEFAULT_DENSITY_BANDS_PER_MIN = (0.0, 0.5, 3.0, 8.0, 1e9)


def _density(entry: dict) -> float:
    segs = entry.get("segments") or []
    if not segs:
        return 0.0
    switches = max(0, sum(1 for i in range(1, len(segs)) if segs[i]["lang"] != segs[i - 1]["lang"]))
    dur = entry.get("duration_s") or 0
    if dur <= 0:
        return 0.0
    return switches / dur * 60.0


def _length_bucket(length_s: float, bands: tuple) -> str:
    for lo, hi in bands:
        if lo <= length_s < hi:
            if hi >= 1e9:
                return f">={int(lo)}s"
            return f"{int(lo)}-{int(hi)}s"
    return ">=max"


def _density_bucket(d: float, bands: tuple) -> str:
    edges = list(bands)
    for i in range(len(edges) - 1):
        if edges[i] <= d < edges[i + 1]:
            lo, hi = edges[i], edges[i + 1]
            if hi >= 1e9:
                return f">={int(lo)}/min"
            return f"{lo:.1f}-{hi:.1f}/min"
    return ">=max"


def by_segment_length(
    pred_segments: list[dict],
    gold_segments: list[dict],
    bands: tuple = DEFAULT_LENGTH_BANDS_S,
) -> dict:
    buckets: dict[str, dict[str, list[str]]] = {}
    used = [False] * len(pred_segments)
    for g in gold_segments:
        length = g["end"] - g["start"]
        if length <= 0 or not g.get("text"):
            continue
        bucket = _length_bucket(length, bands)
        best, best_ov = None, 0.0
        for i, p in enumerate(pred_segments):
            lo = max(p["start"], g["start"])
            hi = min(p["end"], g["end"])
            ov = max(0.0, hi - lo)
            if ov > best_ov:
                best, best_ov = i, ov
        if best is None:
            continue
        d = buckets.setdefault(bucket, {"hyp": [], "ref": []})
        d["hyp"].append(pred_segments[best].get("text", ""))
        d["ref"].append(g["text"])
        used[best] = True
    return {b: wer_score(v["hyp"], v["ref"]) | {"n": len(v["hyp"])} for b, v in buckets.items()}


def by_density(
    entries: list[dict],
    bands: tuple = DEFAULT_DENSITY_BANDS_PER_MIN,
) -> dict:
    buckets: dict[str, dict[str, list[str]]] = {}
    for e in entries:
        d = _density(e)
        bucket = _density_bucket(d, bands)
        if "hyp_text" not in e or "ref_text" not in e:
            continue
        buckets.setdefault(bucket, {"hyp": [], "ref": []})["hyp"].append(e["hyp_text"])
        buckets[bucket]["ref"].append(e["ref_text"])
    return {b: wer_score(v["hyp"], v["ref"]) | {"n": len(v["hyp"])} for b, v in buckets.items()}
