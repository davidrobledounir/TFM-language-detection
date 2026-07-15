#!/usr/bin/env bash
# Evalúa el pipeline propuesto sobre el corpus sintético ES-EU con 3 semillas.
# Cada manifiesto vive en data/synth_es_eu/seed_<seed>/manifest.jsonl.
# Resultados: data/runs/synth_es_eu_s<seed>.json
set -u
mkdir -p data/runs

for SEED in 1337 2025 7; do
  MANIFEST="data/synth_es_eu/seed_${SEED}/manifest.jsonl"
  OUT="data/runs/synth_es_eu_s${SEED}.json"
  if [ ! -f "$MANIFEST" ]; then
    echo "AVISO: falta $MANIFEST, salto seed=$SEED"
    continue
  fi
  if [ -f "$OUT" ]; then
    echo "[skip] $OUT ya existe"
    continue
  fi
  echo "=== synth_es_eu seed=$SEED ==="
  python scripts/evaluate_synth.py \
    --manifest "$MANIFEST" \
    --pipeline-config configs/pipeline.yaml \
    --lid-labels "spa,eus" \
    --boundary-tolerances "0.2,0.5,1.0" \
    --bootstrap-n 1000 \
    --seed "$SEED" \
    --out "$OUT" || echo "  FALLO en seed=$SEED"
done
echo "Hecho. Resultados en data/runs/synth_es_eu_s*.json"
