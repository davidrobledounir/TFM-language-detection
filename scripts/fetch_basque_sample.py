"""Descarga por streaming una muestra del tar.gz de Common Voice y la guarda en data/eu_sample/."""
import argparse
import os
import random
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def fetch_download_url(dataset_id: str, api_key: str) -> dict:
    url = f"https://mozilladatacollective.com/api/datasets/{dataset_id}/download"
    req = urllib.request.Request(
        url,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=b"{}",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        import json

        return json.loads(r.read().decode())


def stream_sample(
    download_url: str,
    out_dir: Path,
    sample_size: int,
    max_bytes: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(exist_ok=True)

    saved_clips: list[str] = []
    saved_meta: list[str] = []
    bytes_read = 0

    class CountingReader:
        def __init__(self, fp):
            self.fp = fp
            self.read_count = 0

        def read(self, n: int = -1) -> bytes:
            data = self.fp.read(n)
            self.read_count += len(data)
            return data

    req = urllib.request.Request(download_url)
    with urllib.request.urlopen(req, timeout=120) as resp:
        counter = CountingReader(resp)
        tar = tarfile.open(fileobj=counter, mode="r|gz")
        try:
            for member in tar:
                bytes_read = counter.read_count
                if bytes_read > max_bytes:
                    print(f"  alcanzado límite max_bytes={max_bytes}, parando")
                    break
                if not member.isfile():
                    continue
                name = member.name
                base = Path(name).name
                if name.endswith(".tsv") or name.endswith(".txt"):
                    fp = tar.extractfile(member)
                    if fp is None:
                        continue
                    (out_dir / base).write_bytes(fp.read())
                    saved_meta.append(base)
                    print(f"  meta: {base}")
                elif name.endswith(".mp3") and len(saved_clips) < sample_size:
                    fp = tar.extractfile(member)
                    if fp is None:
                        continue
                    (clips_dir / base).write_bytes(fp.read())
                    saved_clips.append(base)
                    if len(saved_clips) % 5 == 0:
                        print(f"  clips: {len(saved_clips)}/{sample_size}")
                # corte una vez tenemos todo lo necesario y validated.tsv
                if "validated.tsv" in saved_meta and len(saved_clips) >= sample_size:
                    print("  todos los recursos necesarios obtenidos, parando")
                    break
        except (tarfile.ReadError, urllib.error.URLError) as exc:
            print(f"  stream interrumpido: {exc.__class__.__name__}")

    return {
        "bytes_read": counter.read_count,
        "clips": saved_clips,
        "meta": saved_meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default="cmn2hwe0d01n8mm07wug9r5he")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("MDC_API_KEY"),
        help="API key. Por defecto MDC_API_KEY del entorno.",
    )
    parser.add_argument("--out", default="data/eu_sample")
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--max-bytes", type=int, default=2_000_000_000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: pasa --api-key o MDC_API_KEY en el entorno")
        sys.exit(2)

    random.seed(args.seed)
    print(f"Solicitando URL de descarga (dataset {args.dataset_id})...")
    info = fetch_download_url(args.dataset_id, args.api_key)
    print(f"  archivo: {info['filename']}")
    print(f"  tamaño total: {int(info['sizeBytes']) / 1e9:.2f} GB")
    print(f"  expira: {info['expiresAt']}")

    out_dir = Path(args.out)
    print(f"Streaming a {out_dir} (max {args.max_bytes / 1e9:.2f} GB)...")
    summary = stream_sample(
        info["downloadUrl"], out_dir, args.sample_size, args.max_bytes
    )
    print(f"Listo. Bajados {summary['bytes_read'] / 1e6:.1f} MB.")
    print(f"  meta: {len(summary['meta'])} archivos")
    print(f"  clips: {len(summary['clips'])} mp3")


if __name__ == "__main__":
    main()
