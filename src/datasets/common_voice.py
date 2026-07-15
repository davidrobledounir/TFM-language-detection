import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CVRow:
    path: Path
    sentence: str
    locale: str


_SCHEMAS: dict[str, tuple[str, str]] = {
    "scripted": ("path", "sentence"),
    "spontaneous": ("audio_file", "transcription"),
}

_MMS_TO_ISO1 = {"spa": "es", "cat": "ca", "eus": "eu"}


def _resolve_tsv(root: Path, locale: str, split: str) -> tuple[Path, str]:
    iso1 = _MMS_TO_ISO1.get(locale, locale)
    candidates = [
        (root / f"{split}.tsv", "scripted"),
        (root / f"ss-corpus-{locale}.tsv", "spontaneous"),
        (root / f"ss-corpus-{iso1}.tsv", "spontaneous"),
    ]
    for path, schema in candidates:
        if path.exists():
            return path, schema
    # último intento: cualquier ss-corpus-*.tsv en root
    for p in sorted(root.glob("ss-corpus-*.tsv")):
        return p, "spontaneous"
    raise FileNotFoundError(
        f"No encuentro TSV de Common Voice en {root} para locale={locale} split={split}"
    )


def load(
    root: str | Path,
    locale: str,
    split: str = "validated",
    max_rows: int | None = None,
) -> list[CVRow]:
    root = Path(root)
    tsv, schema = _resolve_tsv(root, locale, split)
    path_col, text_col = _SCHEMAS[schema]
    clips_dir = root / "clips"

    rows: list[CVRow] = []
    with tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            name = r.get(path_col)
            sentence = r.get(text_col)
            if not name or not sentence:
                continue
            clip_path = clips_dir / name
            if not clip_path.exists():
                continue
            rows.append(CVRow(clip_path, sentence, locale))
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def load_transcripts_map(
    root: str | Path,
    locale: str,
    split: str = "validated",
) -> dict[str, str]:
    return {row.path.name: row.sentence for row in load(root, locale, split=split)}
