import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import fleurs


def _make_fake_fleurs(tmp_path: Path, lang: str, n: int = 3) -> Path:
    import soundfile as sf

    root = tmp_path / "fleurs"
    clips_dir = root / lang / "clips"
    clips_dir.mkdir(parents=True)
    tsv = root / lang / "test.tsv"
    with tsv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["path", "sentence"])
        for i in range(n):
            name = f"{lang}_{i:03d}.wav"
            sf.write(clips_dir / name, np.zeros(16000, dtype=np.float32), 16000, subtype="PCM_16")
            w.writerow([name, f"frase {i}"])
    return root


def test_fleurs_loader_reads_local_tsv(tmp_path):
    root = _make_fake_fleurs(tmp_path, "eu", n=4)
    rows = fleurs.load(root, "eu", split="test")
    assert len(rows) == 4
    assert rows[0].locale == "eus"


def test_fleurs_loader_respects_max_rows(tmp_path):
    root = _make_fake_fleurs(tmp_path, "ca", n=5)
    rows = fleurs.load(root, "ca", split="test", max_rows=2)
    assert len(rows) == 2
    assert rows[0].locale == "cat"
