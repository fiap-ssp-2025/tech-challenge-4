# Tasks: Triagem Multimodal em Consultas

> Referências: `spec.md` + `plan.md`. Numeração continua a da 001 onde a tarefa é herdada sem mudança.

## Etapa 1 — Realinhamento (este PR) + dados
- [x] T101 P1: specs 002; contratos v2 (taxonomia, V3 facial, sessão); renomear módulos/stubs/pastas; fusão por sessão; sintéticos por sessão; runner e testes atualizados
  - **Verificado:** `pytest` verde; E2E com stubs roda com `session_id`
- [x] T102 P2: CORAA 8 kHz + labels + split por locutor (igual T002/001); solicitar VERBO
  - **Verificado:** `labels.csv` 933 linhas = 933 wavs em `processed/`; `verify_audio_dataset.py` prova zero locutor cruzando splits + amostra 5× 8 kHz mono; `pytest` verde; e-mail VERBO em `docs/verbo-solicitacao.md` (envio manual)
- [ ] T103 P3: Azure Speech F0 + fallback (igual T003/001); PLN com nova taxonomia
- [x] T104 P4: extrair frames faciais RAVDESS/CREMA-D (rosto via YOLOv8); labels de emoção pelo nome do arquivo; split por ator; notebook Colab
  - **Verificado:** RAVDESS+CREMA-D via mirror GitLab; após balanceamento `labels.csv` 20612 = 20612 jpgs; minority 0.40; zero ator cruzando splits; notebook `notebooks/t112_fer_colab.ipynb`. Adendo: `download_cremad.py` default = GitLab mirror; `balance_binary_labels` no extract.
- [x] T105 P5: frames de corpo + CVAT (`defensiva`/`neutra`); sprint coletivo de anotação (~1h, todos)
  - **Verificado:** labels derivadas das emoções RAVDESS (fearful/sad→defensiva, neutral/calm/happy→neutra) como proxy; 11504 frames de 8 atores; 0 erros de detecção YOLOv8-pose; declarado como dado atuado no RNF-06

## Etapa 2 — Treinos e regras
- [x] T110 P2: fine-tune wav2vec2 PT-BR — F1 ≥ 0,75
  - **Verificado:** `wav2vec2-large-xlsr-53-portuguese`, 4 blocos superiores + cabeça treináveis, 8 épocas em CPU; split por locutor (train 666 / val 166 / test 101). F1 macro **0,7171** no limiar padrão 0,50 e **0,7705** no limiar 0,17 calibrado na validação (`scripts/eval_a3_threshold.py`) — **meta atingida no limiar calibrado**. AUC 0,92 val / 0,84 teste. O contrato A3 devolve `sofrimento` como score contínuo, então o limiar é decisão da fusão (T114) e do go/no-go (T120), não do A3. `pytest tests/test_a3_emotion.py` 2/2 verde. Ressalva: teste tem só 23 amostras `non_neutral`, estimativa com incerteza alta.
  - **Integração (P1):** rebase sobre a main pós-T114; `resolve.py` declara `packages`/`artifacts` do A3 (sem `models/a3_emotion/` o mapa mostra "artefato ausente" e cai para stub, em vez de anunciar real e degradar em execução); inferência em janelas de 6 s (mesma duração do treino), média dos scores.
  - **Publicado (P2):** pesos em `fiap-ssp-2025/tc4-a3-sofrimento-voz` (Hugging Face, **privado** sob a org do time, CC BY-NC-ND herdada do CORAA); `scripts/download_a3_model.py` restaura em `models/a3_emotion/`. Métricas versionadas em `models/a3_emotion_metrics.json` e `models/a3_threshold_metrics.json`. **Recalibrado no regime de janelamento** (o 0,7705 anterior era da truncagem): F1 macro teste **0,7171 @0,50** e **0,7896 @0,17**, AUC 0,9229 val / 0,8456 teste; `eval_a3_threshold.py` delega em `infer()` para não divergir do que roda. **Agregação revisada em 26/07:** áudio longo é pontuado pelo **máximo** das janelas de 6 s, não pela média — sofrimento é evento, não estado médio, e a média o diluía no restante da fala. Recalibrado: F1 macro **inalterado** (0,7171 @0,50 e 0,7896 @0,17), pois 87% dos clipes do CORAA cabem numa janela. **Verificado:** com `models/a3_emotion/` removido e restaurado só pelo download, `pytest` 59 passed / 4 skipped (os 4 são do `test_video_real.py`, pesos YOLO e RAVDESS ausentes) e `resolve` mostra `a3_emotion → real`.
- [x] T111 P3: `infer.py` real A1/A2
  - **Verificado:** PRs #1 e #6 mergeados — `src/audio/a1_stt/` (Azure Speech + fallback offline
    faster-whisper, via `stt.py`/`azure_stt.py`/`whisper_stt.py`) e `src/audio/a2_nlp/`
    (`rules.py` + `extractors.py`, taxonomia clínica). O merge adaptou o código à camada
    resolve do T114 — o runner não importa STT direto. `resolve` mostra `a1_stt → real` e
    `a2_nlp → real` mesmo sem `AZURE_SPEECH_KEY` (o A1 usa o modo offline). Sem áudio
    inteligível o A1 degrada para stub **em execução**, com o motivo logado.
- [x] T112 P4: fine-tune FER (desconforto facial) — F1 ≥ 0,70
  - **Verificado — meta atingida sem ajuste de limiar:** F1 macro teste **0,7045 por frame** e
    **0,7108 por clipe** (limiar padrão 0,50), AUC **0,8076**; split por ator do T104 preservado
    (train 14388 / val 2996 / test 3228 frames; 80/17/18 atores). Seleção sempre na validação;
    teste medido uma vez por rodada. Base: `trpakov/vit-face-expression` (ViT já treinado em
    expressão facial) — 3 rodadas em GPU RunPod, ~US$ 1,50.
  - **Desenho 2×2 (qual fator trouxe o ganho):** backbone {ImageNet, FER} × treino {com `calm`,
    sem `calm`}, todos medidos nos dois benchmarks. **O backbone FER trouxe o ganho**
    (AUC 0,7605 → 0,8076); **remover `calm` piorou** de forma consistente (val: 0,6815→0,6697
    na EfficientNet, 0,6880→0,6635 no ViT) — os 1152 frames de `calm` são negativos úteis.
  - **Inconsistência achada no `plan.md`:** a linha 51 define a meta como "fearful/sad vs
    **neutral**", mas a linha 76 monta a base com "neutro ← {neutral, **calm**}". O experimento
    mostra que manter `calm` é melhor ⇒ **alinhar a linha 51 à prática** (P1).
  - **Integração (P1):** `models/v3_face/` exportado como diretório auto-contido
    (`scripts/t112/export_v3_model.py`) — carrega offline, sem baixar o backbone; `infer.py`
    real amostra até 12 frames, aplica **o mesmo recorte YOLOv8 do T104** e devolve a média
    (mesma agregação que produziu o 0,7108); `resolve.py` declara `packages`/`artifacts`;
    `preprocess.json` versiona recorte e normalização junto do peso.
    **`pytest` 68 passed, 0 skipped**, incluindo `test_crop_matches_training_extractor`, que
    compara o recorte do `infer` com o do script de extração (guarda contra treino×produção
    divergirem em silêncio). E2E: os **6 módulos resolvem para real**; V3 em 152 ms.
  - **Publicado:** pesos em `fiap-ssp-2025/tc4-v3-desconforto-facial` (Hugging Face, **privado**
    sob a org do time; CC BY-NC-SA 4.0 herdada do RAVDESS), com model card declarando o proxy e a
    margem estreita. `scripts/t112/publish_v3_model.py` reproduz a publicação.
    **Verificado:** com `models/v3_face/` removido e restaurado só por `download_v3_model.py`,
    `resolve` mostra `v3_face → real` e `pytest` 68 passed, 0 skipped.
  - **Pendente RNF-06:** declarar no relatório que o rótulo é **proxy de expressão atuada**, não
    desconforto clínico anotado por humano.
- [x] T113 P5: cabeça de postura + V1 — contratos respeitados
  - **Verificado:** V1 YOLOv8n+ByteTrack devolve `{n_pessoas, tracks[{id, n_frames, bbox_media}]}`; V2 GradientBoosting F1 macro=0.688 (split ator 8, aceite: reportar); `pytest` 20/20 verde
- [x] T114 P1: plugar `infer.py` reais mantendo contrato
  - **Verificado:** `src/resolve.py` decide real×stub por módulo (pacote/artefato/credencial/
    re-export ausentes → stub **com motivo logado**); `TC4_FORCE_STUBS=1` e `TC4_REQUIRE_REAL`
    suportados; runner sem nenhum import de stub, imprime o mapa `resolved` e o tempo (ms) por
    módulo; `pytest` 48/48 verde (43 + 5 skips simulando clone sem pesos).
  - **V2 regenerado:** `keypoints.csv` 11504 linhas (atores 01–08, 0 erros de detecção) →
    **F1 macro 0,6868** (split por ator 8, seed 42) — reproduz o 0,688 do T113.
  - **Decisão `models/`:** `.pkl` com 723 KB (≤ 5 MB) ⇒ **versionado** (`models/v2_posture_head.pkl`
    + `v2_posture_metrics.json`), com o comando de regeneração no README; demais pesos seguem
    ignorados. Bug corrigido para plugar: `MODEL_PATH` do V2 apontava para fora do repo
    (`parents[4]` → `parents[3]`).

## Etapas 3–5 — herdadas da 001 (T020→T042) com dois ajustes
- [ ] T120 Go/no-go Etapa 3 inclui V3-facial na ordem de socorro: **V3 > A3 > V2**
- [ ] T121 Etapa 5 inclui: remover `hello_sdd`/`specs/000` e conferir declarações do RNF-06 no relatório

## Fechamento SDD
- [ ] T190 Status `done` em spec/plan; specs/README atualizado; revisar código × spec
