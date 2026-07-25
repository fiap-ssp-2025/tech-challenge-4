# Revisão da Etapa 1 (P1) — registro curto

**(a) Rótulos de postura do V2 — decisão do grupo.** Os rótulos `defensiva`/`neutra` seguem o
proxy de emoção do RAVDESS (`fearful`/`sad` → defensiva; `neutral`/`calm`/`happy` → neutra),
**sem sprint de validação manual**, por restrição de prazo. Consequência a declarar no relatório
(RNF-06): o V2 mede *postura típica de expressão encenada de medo/tristeza* como aproximação de
postura defensiva — não postura defensiva anotada por humano. Validação humana das anotações fica
como **trabalho futuro**. O F1 do V2 (0,6868, split por ator 8) deve ser lido sob essa ressalva.

**(b) CREMA-D (P4) pendente.** O pull LFS falhou com HTTP 502 no `download_cremad.py`; a base do
T104 hoje é só RAVDESS. Reexecução é **local do P4** (`download_cremad.py` + `extract_face_frames.py
--force`), não do P1. Impacta a diversidade de identidades do V3 (T112).

**(c) T103 (P3) bloqueado por infraestrutura.** Depende do recurso Azure Speech e do `.env`
criados **manualmente pelo P1** — enquanto não existir `AZURE_SPEECH_KEY`, o A1 resolve para stub
(hoje é o que o mapa `[resolve]` mostra).

**(d) Bugs vistos ao plugar (notas, sem refatorar código alheio).** `v2_pose/infer.py` apontava
`MODEL_PATH` para `parents[4]` (fora do repo) — corrigido no T114 por ser bloqueante; o default
`--raw-dir` de `extract_pose_frames.py` (`data/pose_posture/raw`) não bate com o destino do
`download_ravdess.py` (`data/video_consulta/raw/ravdess`); o V1 chama `model.track()` sem
`stream=True` e acumula resultados em RAM (alerta da ultralytics) — revisar antes da demo longa.
`pyproject.toml` declara `opencv-python-headless>=4.9`, mas o `uv.lock` resolveu
`opencv-python==5.0.0.93` (variante cheia, não headless); essa versão não expõe mais
`cv2.CascadeClassifier`, então `--detector haar` de `extract_face_frames.py` quebra neste ambiente.
Não bloqueia T112 (default é `--detector yolo`, usado no dataset atual) — revisar o pin do opencv
antes de reativar o caminho haar.
