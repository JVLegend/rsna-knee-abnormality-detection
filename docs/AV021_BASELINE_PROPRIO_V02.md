# AV-021 — baseline próprio V02 com protocolo prospectivo

#RSNA #Kaggle #Pesquisa

17/09/2026. Fonte de verdade: 07_Estrategia_AVANCE do vault;
[espelho](ESTRATEGIA_AVANCE.md). Continua [AV-020](AV020_CORRECAO_PERCENTIS_V02.md).

## Protocolo antes da primeira avaliação

- V01 congelada:299treino/250desenvolvimento; confirmação150fechada.
  Exclusões gold/grupos do manifesto preservadas. Labels suaves do professor
  original sem alterações, inclusive0,5; não convertê-los em positivos.
- Encoder genérico DINOv2-S/14 congelado, FP32 CLS384, normalização de pixels
  corrigida e testada no piloto v3.3planos/3quartis por série,224x224.
  Não é fine-tuning do encoder nem representação de cortes adjacentes.
- Cabeça de atenção compartilhada384→64→1,classificador384→12. AdamW,
  learning_rate0,001,weight_decay0,0001,batch4,20épocas.
- Duas sementes predefinidas2026/42 para cabeça e embaralhamento. Encoder
  fixo seed2026; features compartilhadas entre os dois treinos.
- Métrica primária: média da BCE suave nos250estudos dev e12alvos; menor
  é melhor. Melhor época por mínimo dessa métrica, primeira em empate exato.
  Não buscar taxas de aprendizado, pesos de ensemble ou outras épocas pelo LB.
- Comparador: constante por alvo igual à média dos rótulos suaves dos299
  estudos de treino, sem ajustar valores com dados dev.
- Auditoria independente: refazer BCE em NumPy float64, conferir IDs/hash/
  seeds/seleção; recalcular atenção e logits a partir dos checkpoints.
  Tolerância de replay numérico de logits1e−4 absoluto/1e−5relativo e de
  métricas2e−6; pixels continuam exigindo igualdade exata por hash.

Esses rótulos são derivados dos laudos, não anotações clínicas independentes.
Não reportar AUC clínica ou confundir redução de BCE com melhora no leaderboard.
Dev seleciona épocas, logo não é teste final intocado. Grupos de laudos e
hashes de pixels não garantem ausência de repetição de paciente/scanner.

## Implementação e gates

1. scripts/prepare_v02_baseline.py:reverifica hashes V01/professor/metadata,
   separação por ID/laudo, tamanho/CRC/shape/dtype/índices/pixels de897arrays
   treino e750dev, e bloqueia arrays idênticos entre esses splits. Sem ler
   os450arrays da confirmação. Builder exige a versão do piloto auditada.
2. scripts/v02_baseline_runtime.py:reconstrói4.941DICOMs selecionados e exige
   os1.647hashes originais antes de treinar qualquer cabeça. Extrai features
   com6séries/lote, salva features/IDs/contrato e checkpoints a cada época.
3. Checkpoint inclui modelo/otimizador/RNG/época/histórico/melhor estado.
   --resume-directory pode recuperar features e checkpoints, exigindo contrato
   e fingerprint por semente; sem reutilização silenciosa de receita diferente.
4. scripts/assess_v02_baseline.py:audita artefatos, comparação com prior,
   métricas e replay NumPy dos checkpoints melhor/último. Não gera submissão.

Orçamento:1job privado/offline/T4x2, usa cuda:0, teto1.800s para as duas
sementes. Guarda interna aos1.650s mantém margem; se falhar, preservar os
checkpoints completos e diagnosticar antes de repetir. Cota disponível na
consulta inicial6,496078h, compartilhada com outras atividades Kaggle.

Não reexecutar o piloto aprovado. Não anexar pesos ajustados à competição;
apenas competição oficial e metaresearch/dinov2/PyTorch/small/1. Checkpoints
e IDs/rótulos gerados ficam privados no HD/Kaggle, fora do GitHub.

## Comandos de retomada

Verificação local: **141 testes e44subtestes passaram**, incluindo
interrupção/reinício CPU com histórico e logits idênticos à execução contínua,
bloqueio de contrato diferente, replay independente NumPy e gates de dados.
Isso não substitui a auditoria da futura execução GPU completa.

```sh
uv run --no-project --with numpy python -m scripts.prepare_v02_baseline audit \
  --output reports/avance_av021_v02/cache_train_dev_v1.json
uv run --no-project --with numpy python -m scripts.prepare_v02_baseline build \
  --cache reports/avance_av021_v02/cache_train_dev_v1.json \
  --output reports/avance_av021_v02/baseline_v1.py
uv run --no-project --with numpy --with torch python -m scripts.assess_v02_baseline \
  --directory reports/avance_av021_baseline_v1 \
  --cache reports/avance_av021_v02/cache_train_dev_v1.json \
  --build reports/avance_av021_v02/baseline_v1.py \
  --output reports/avance_av021_v02/baseline_audit_v1.json
```

Auditoria/build congelados recusam sobrescrita divergente. Não repetir a
auditoria longa se já houver resultado aprovado e sem mudança dos arquivos.

## Critério de decisão

Relatar resultados das duas sementes e do comparador, sem escolher só a melhor
semente. Se ambas perderem para o prior, bloquear promoção e diagnosticar
generalização antes de aumentar o modelo. Um resultado misto é inconclusivo.
Se ambas ganharem, isso habilita ablações visuais comparáveis na fila, não
submissão automática nem abertura imediata da confirmação.

Melhor público preservado0,941. Nenhuma nova submissão autorizada por este
baseline sem pipeline de inferência e gates próprios de elegibilidade.
