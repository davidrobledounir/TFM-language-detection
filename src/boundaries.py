def boundaries_from_labels(
    labels: list[str],
    starts: list[float],
    ends: list[float],
) -> list[dict]:
    if not labels:
        return []
    segments: list[dict] = []
    cur_label = labels[0]
    cur_start = starts[0]
    cur_end = ends[0]
    for lab, s, e in zip(labels[1:], starts[1:], ends[1:]):
        if lab == cur_label:
            cur_end = e
        else:
            segments.append({"start": cur_start, "end": cur_end, "lang": cur_label})
            cur_label = lab
            cur_start = s
            cur_end = e
    segments.append({"start": cur_start, "end": cur_end, "lang": cur_label})
    return segments


def boundary_points(segments: list[dict]) -> list[float]:
    return [seg["start"] for seg in segments[1:]]
