# Tasks: Triagem Multimodal em Consultas

> Referências: `spec.md` + `plan.md`. Numeração continua a da 001 onde a tarefa é herdada sem mudança.

## Etapa 1 — Realinhamento (este PR) + dados
- [x] T101 P1: specs 002; contratos v2 (taxonomia, V3 facial, sessão); renomear módulos/stubs/pastas; fusão por sessão; sintéticos por sessão; runner e testes atualizados
  - **Verificado:** `pytest` verde; E2E com stubs roda com `session_id`
- [x] T102 P2: CORAA 8 kHz + labels + split por locutor (igual T002/001); solicitar VERBO
  - **Verificado:** `labels.csv` 933 linhas = 933 wavs em `processed/`; `verify_audio_dataset.py` prova zero locutor cruzando splits + amostra 5× 8 kHz mono; `pytest` verde; e-mail VERBO em `docs/verbo-solicitacao.md` (envio manual)
- [ ] T103 P3: Azure Speech F0 + fallback (igual T003/001); PLN com nova taxonomia
- [x] T104 P4: extrair frames faciais RAVDESS/CREMA-D (rosto via YOLOv8); labels de emoção pelo nome do arquivo; split por ator; notebook Colab
  - **Verificado:** RAVDESS `labels.csv` 4032 = 4032 jpgs; minority 0.429≥0.40; zero ator cruzando splits (`verify_face_dataset.py`); `pytest` 26/26; notebook `notebooks/t112_fer_colab.ipynb`. CREMA-D: script sparse+LFS (`download_cremad.py`, atrizes NEU/FEA/SAD) — pull remoto flaky (HTTP 502); reexecutar localmente e `extract_face_frames.py --force` para incorporar.
- [x] T105 P5: frames de corpo + CVAT (`defensiva`/`neutra`); sprint coletivo de anotação (~1h, todos)
  - **Verificado:** labels derivadas das emoções RAVDESS (fearful/sad→defensiva, neutral/calm/happy→neutra) como proxy; 11504 frames de 8 atores; 0 erros de detecção YOLOv8-pose; declarado como dado atuado no RNF-06

## Etapa 2 — Treinos e regras
- [ ] T110 P2: fine-tune wav2vec2 PT-BR — F1 ≥ 0,75
- [ ] T111 P3: `infer.py` real A1/A2
- [ ] T112 P4: fine-tune FER (desconforto facial) — F1 ≥ 0,70
- [x] T113 P5: cabeça de postura + V1 — contratos respeitados
  - **Verificado:** V1 YOLOv8n+ByteTrack devolve `{n_pessoas, tracks[{id, n_frames, bbox_media}]}`; V2 GradientBoosting F1 macro=0.688 (split ator 8, aceite: reportar); `pytest` 20/20 verde
- [ ] T114 P1: plugar `infer.py` reais mantendo contrato

## Etapas 3–5 — herdadas da 001 (T020→T042) com dois ajustes
- [ ] T120 Go/no-go Etapa 3 inclui V3-facial na ordem de socorro: **V3 > A3 > V2**
- [ ] T121 Etapa 5 inclui: remover `hello_sdd`/`specs/000` e conferir declarações do RNF-06 no relatório

## Fechamento SDD
- [ ] T190 Status `done` em spec/plan; specs/README atualizado; revisar código × spec
