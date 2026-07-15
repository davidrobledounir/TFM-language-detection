def consolidate(segments: list[dict]) -> list[dict]:
    if not segments:
        return []
    out = [dict(segments[0])]
    for seg in segments[1:]:
        if seg.get("lang") == out[-1].get("lang"):
            out[-1]["end"] = seg["end"]
            joined = (out[-1].get("text", "") + " " + seg.get("text", "")).strip()
            out[-1]["text"] = joined
            if "lid_conf" in seg and "lid_conf" in out[-1]:
                out[-1]["lid_conf"] = 0.5 * (out[-1]["lid_conf"] + seg["lid_conf"])
        else:
            out.append(dict(seg))
    return out
