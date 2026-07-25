#!/usr/bin/env bash
# Round 3 (Rota B) — desenho 2x2: backbone {EfficientNet-ImageNet, ViT-FER} x
# base de treino {com calm, sem calm}. Cada vencedor e medido nos DOIS benchmarks
# (teste com calm e teste sem calm), para que os quatro numeros sejam comparaveis
# e da para dizer qual fator trouxe o ganho.
set -euo pipefail
cd /workspace/t112

echo "[setup] deps"
pip install -q --break-system-packages scikit-learn pandas transformers

if [ ! -d faces ]; then
  echo "[setup] extraindo t112_faces.tar"
  tar -xf t112_faces.tar
fi
find faces -name '._*' -delete
python -c "
import pandas as pd, pathlib
df = pd.read_csv('faces/labels.csv')
n = len(list(pathlib.Path('faces').glob('*.jpg')))
assert len(df) == n, f'labels {len(df)} != jpgs {n}'
print(f'[setup] dataset ok: {len(df)} frames')
"

# --- treinos: mesma varredura, duas bases -----------------------------------
echo "[treino] regime COM calm"
python train_t112_fer.py --data-root . --out results_r3_com_calm \
  --sweep v3 --epochs 12 --patience 3 --batch-size 128 2>&1 | tee train_r3_com_calm.log

echo "[treino] regime SEM calm"
python train_t112_fer.py --data-root . --out results_r3_sem_calm \
  --sweep v3 --epochs 12 --patience 3 --batch-size 128 --drop-emotions calm 2>&1 | tee train_r3_sem_calm.log

# --- avaliacao cruzada: cada vencedor nos dois benchmarks -------------------
for reg in com_calm sem_calm; do
  ck="results_r3_${reg}/v3_fer_best.pt"
  [ -f "$ck" ] || { echo "[eval] sem checkpoint para $reg"; continue; }
  echo "[eval] treino=$reg -> benchmark COM calm"
  python eval_t112_clip.py --ckpt "$ck" --data-root . --tag _bench_com_calm
  echo "[eval] treino=$reg -> benchmark SEM calm"
  python eval_t112_clip.py --ckpt "$ck" --data-root . --drop-emotions calm --tag _bench_sem_calm
done

echo "[fim] resultados em results_r3_com_calm/ e results_r3_sem_calm/"
