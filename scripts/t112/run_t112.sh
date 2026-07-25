#!/usr/bin/env bash
# T112 no pod RunPod: extrai os dados, roda a varredura e (opcional) desliga o pod.
# Uso no pod:  AUTO_STOP=1 bash run_t112.sh
set -euo pipefail
cd /workspace/t112

echo "[setup] deps"
# Ubuntu 24.04 marca o Python do sistema como externally-managed (PEP 668); a imagem
# do pod é descartável, então instalar no ambiente do sistema é o caminho certo aqui.
pip install -q --break-system-packages scikit-learn pandas

if [ ! -d faces ]; then
  echo "[setup] extraindo t112_faces.tar"
  tar -xf t112_faces.tar
fi
# O tar do macOS carrega AppleDouble (._arquivo) junto; sem isso a contagem dobra.
find faces -name '._*' -delete
python - <<'EOF'
import pandas as pd
df = pd.read_csv("faces/labels.csv")
n_jpg = len(list(__import__("pathlib").Path("faces").glob("*.jpg")))
assert len(df) == n_jpg, f"labels {len(df)} != jpgs {n_jpg}"
print(f"[setup] dataset ok: {len(df)} frames")
EOF

python train_t112_fer.py --data-root /workspace/t112 --out /workspace/t112/results
status=$?

echo "[fim] status=$status — resultados em /workspace/t112/results"
if [ "${AUTO_STOP:-0}" = "1" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  echo "[fim] desligando o pod ($RUNPOD_POD_ID)"
  runpodctl stop pod "$RUNPOD_POD_ID" || true
fi
exit $status
