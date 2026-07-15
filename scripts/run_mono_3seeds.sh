#!/usr/bin/env bash
# Re-evalúa los 3 muestreos monolingües con baseline_large + pipeline_mms para 3 semillas.
# Output: data/runs/<dataset>_s<seed>.json
set -u
mkdir -p data/runs

for SEED in 1337 2025 7; do
  for PAIR in "data/eu_sample eu" "data/es_sample es"; do
    SAMPLE=$(echo "$PAIR" | cut -d' ' -f1)
    LOCALE=$(echo "$PAIR" | cut -d' ' -f2)
    NAME=$(basename "$SAMPLE")
    OUT="data/runs/${NAME}_s${SEED}.json"
    if [ -f "$OUT" ]; then
      echo "[skip] $OUT ya existe"
      continue
    fi
    echo "=== seed=$SEED dataset=$NAME ==="
    python scripts/analyze_sample.py \
      --sample-dir "$SAMPLE" \
      --locale "$LOCALE" \
      --max-clips 30 \
      --seed "$SEED" \
      --bootstrap-n 1000 \
      --lid-labels "spa,eus" \
      --out "$OUT" || echo "  FALLO en $NAME seed=$SEED"
  done
done
echo "Hecho. Resultados en data/runs/"
