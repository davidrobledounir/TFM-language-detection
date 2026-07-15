# Código del TFM

## Estructura

```
configs/        Configuración YAML del pipeline y de la línea base.
src/            Módulos del flujo propuesto, datasets, síntesis y evaluación.
scripts/        CLIs para ejecutar pipeline, línea base, síntesis y reporte.
tests/          Pruebas mínimas de humo (no requieren modelos).
data/           (gitignored) audios, manifests, resultados de ejecuciones.
```

## Instalación

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Hardware objetivo: GPU 8 GB (float16) o CPU x86-64 (int8). Se selecciona en runtime.

## Uso

Pipeline propuesto sobre un wav:

```
python scripts/run_pipeline.py --audio entrada.wav --out salida.json
```

Línea base Whisper-large-v3:

```
python scripts/run_baseline.py --audio entrada.wav --out salida_baseline.json
```

Evaluación contra una referencia (JSONL con `{id, text, segments?}` por línea):

```
python scripts/evaluate.py --hyp hipotesis.jsonl --ref referencia.jsonl
```

## Componentes del pipeline

1. `segmentation` — VAD silero más ventanas adaptativas de 3 a 30 s, con refinamiento por confianza del LID (subdivide ventanas inestables hasta `refine.min_refined_s`).
2. `lid` — backends seleccionables por configuración: MMS-LID-126 (por defecto) y XLS-R (`facebook/wav2vec2-xls-r-300m` ajustado a clasificación de idioma). Suavizado temporal con histéresis sobre las etiquetas estabilizadas.
3. `boundaries` — fronteras derivadas de las etiquetas estabilizadas.
4. `asr_conditioned` — Whisper invocado con `language=` fijado por el LID. Configurable a `large-v3` (referencia principal) o `medium` (referencia secundaria).
5. `postprocess` — consolida segmentos contiguos de la misma lengua. Opción `emit_nonspeech` para etiquetar regiones de silencio.

## Salida del pipeline

Lista de segmentos con esta forma:

```
{
  "start": 0.0,
  "end": 4.2,
  "lang": "spa",
  "lid_conf": 0.93,
  "text": "..."
}
```

## Corpus sintético de code-switching

Generación por concatenación controlada de clips monolingües con densidad paramétrica.

```
python scripts/make_synth.py --pool eus=data/eu_sample --pool spa=data/es_sample --pool cat=data/ca_sample --n 50 --target-s 30 --density 6 --out data/synth
```

Cada wav viene con su ground truth (`manifest.jsonl`) que incluye fronteras de cambio y texto por segmento. Evaluable con:

```
python scripts/evaluate_synth.py --manifest data/synth/manifest.jsonl --out data/synth/results.json
```

## Particiones

```
python scripts/make_partitions.py --manifest data/synth/manifest.jsonl --out-dir data/synth/splits --by lang
```

`--by lang` estratifica por conjunto de idiomas presentes. `--by density` por bandas de densidad de cambio (0 / baja / media / alta).

## Reportes

```
python scripts/report.py --results data/eu_sample/results.json --out data/eu_sample/report.md
```

Detecta automáticamente si el JSON proviene de `analyze_sample.py` o de `evaluate_synth.py`.

## Reproducibilidad

- Semillas fijadas (`src/seed.py`).
- `beam_size=1` por defecto.
- Misma normalización de texto que la línea base (`src/evaluation/normalize.py`).
- `requirements.txt` con versiones mínimas.

## Multi-semilla, bootstrap y coste

```
python scripts/run_seeds.py --seeds 1337,2025,7 --datasets eu_sample,es_sample,ca_sample,synth,fleurs --variants baseline_large,baseline_medium,pipeline_mms,pipeline_xlsr
```

Cada combinación se ejecuta bajo el context manager [src/evaluation/cost.py](src/evaluation/cost.py), que registra tiempo, RSS pico (CPU) y memoria GPU pico (`torch.cuda.max_memory_allocated`). El agregador aplica bootstrap (1000 réplicas, IC 95 %) sobre WER, CER y F1 macro.

## Corpus FLEURS

```
python scripts/fetch_fleurs.py --langs es_419,ca,eu --out data/fleurs
python scripts/evaluate_synth.py --manifest data/fleurs/manifest.jsonl --out data/fleurs/results.json
```

`src/datasets/fleurs.py` sigue el mismo patrón que `common_voice.py` y se consume desde los mismos scripts de evaluación.
