import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import baseline_whisper


def main() -> None:
    parser = argparse.ArgumentParser(description="Línea base Whisper-large-v3 plana.")
    parser.add_argument("--audio", required=True, help="Ruta a un archivo de audio")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--out", default="-", help="Ruta de salida JSON o '-' para stdout")
    args = parser.parse_args()

    cfg = baseline_whisper.load_config(args.config)
    text = baseline_whisper.run(args.audio, cfg)
    payload = json.dumps({"text": text}, ensure_ascii=False, indent=2)
    if args.out == "-":
        print(payload)
    else:
        Path(args.out).write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
