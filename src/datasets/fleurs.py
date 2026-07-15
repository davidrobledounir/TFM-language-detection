"""Loader para una copia local de FLEURS (Google) con manifest por idioma.

Se asume la estructura:
    <root>/<lang>/<split>.tsv     (campos: path<TAB>sentence)
    <root>/<lang>/clips/<archivo>

`<lang>` usa los códigos de FLEURS: `es_419`, `ca`, `eu` (y similares). El loader
admite también `<lang>/audio/<split>/<archivo>` por compatibilidad con dumps de
HuggingFace `datasets`.

`fetch_fleurs.py` produce este layout automáticamente.
"""
import csv
from dataclasses import dataclass
from pathlib import Path


_FLEURS_TO_MMS = {
    "es_419": "spa",
    "ca": "cat",
    "eu": "eus",
}


@dataclass(frozen=True)
class FleursRow:
    path: Path
    sentence: str
    lang: str  # código FLEURS (e.g. "eu")
    locale: str  # código MMS equivalente (e.g. "eus")


def _clip_path(root: Path, lang: str, split: str, name: str) -> Path | None:
    for cand in [root / lang / "clips" / name, root / lang / "audio" / split / name, root / lang / name]:
        if cand.exists():
            return cand
    return None


def load(
    root: str | Path,
    lang: str,
    split: str = "test",
    max_rows: int | None = None,
) -> list[FleursRow]:
    root = Path(root)
    tsv = root / lang / f"{split}.tsv"
    if not tsv.exists():
        raise FileNotFoundError(f"No encuentro {tsv}. ¿Has corrido fetch_fleurs.py?")
    locale = _FLEURS_TO_MMS.get(lang, lang)
    rows: list[FleursRow] = []
    with tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            name = r.get("path") or r.get("audio_file")
            sentence = r.get("sentence") or r.get("transcription")
            if not name or not sentence:
                continue
            clip = _clip_path(root, lang, split, name)
            if clip is None:
                continue
            rows.append(FleursRow(clip, sentence, lang, locale))
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def load_transcripts_map(
    root: str | Path,
    lang: str,
    split: str = "test",
) -> dict[str, str]:
    return {row.path.name: row.sentence for row in load(root, lang, split=split)}
