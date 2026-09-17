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

## Execução identificada

Auditoria local **PASSED_BASELINE_CACHE**:549estudos/1.647séries; sem
hashes de imagens exatamente iguais entre treino e desenvolvimento.
Manifesto V01/professor continuam com os hashes originais. Sem pixels da
confirmação. Cache congelado SHA129389226d6a270ca86690bfca509c7da4a8c31cca1b5d05896378da55112211.

Código e protocolo publicados no commit69876d6 antes da avaliação.
Build privado de879.561bytes, SHA0049a9b0dd59285a621eab8fd54af776a85b68a5d4220bbbf6d4a7a4bf19ef85.
Kernel **jvlegend/rsna-knee-v02-frozen-dino-baseline**, ID134788472 v1,
**COMPLETE e PASSED_V02_BASELINE_AUDIT**. Cota antes6,491713h. Não repetir.
Saídas privadas em reports/avance_av021_baseline_v1/; auditoria independente
em reports/avance_av021_v02/baseline_audit_v1.json.

## Resultado e decisão

| Comparador/modelo | Época selecionada | SoftBCE dev (menor melhor) | Delta vs prior |
|---|---:|---:|---:|
| Prior só do treino | — | 0,65707840 | — |
| Atenção compartilhada,seed2026 | 10 | **0,62607019** | −0,03100821 |
| Atenção compartilhada,seed42 | 6 | **0,63189521** | −0,02518320 |

Ambas as sementes melhoram o comparador; redução relativa aproximada4,7%/3,8%.
Não é ensemble das sementes nem ganho de leaderboard. O protocolo foi
mantido:20épocas em ambos os treinos, seleção prospectiva pelo mínimo dev.
Na época20, dev piorou para0,64938283/0,66013783; perdas de treino finais
0,51534088/0,51295165. Isso sinaliza risco de sobreajuste; não ampliar épocas
automaticamente com base na queda da perda de treino.

Contra o prior,11/12alvos melhoram na seed2026 e7/12na seed42. Baker's piora
em ambas (+0,014923/+0,018990); maiores reduções repetidas ocorrem em Effusion,
Lateral OA e PF OA. Esses achados são diagnósticos dos rótulos fracos, não
prova de erro clínico do professor ou causa anatômica da diferença.

Todos os1.647hashes de pixels passaram no worker antes do treino. Features
549x3x384, IDs e checkpoints íntegros; replay independente NumPy dos logits
melhor/último divergiu no máximo1,04e−6, bem abaixo da tolerância predefinida.
Features164,57s,total medido175,04s (~2,9min, fora imports), pico alocado153MiB.
Modelos/otimizadores/RNG/histórico foram preservados para retomada.

**Decisão:** aceitar como referência de desenvolvimento para ablações, não
como nova candidata de submissão. Próximo M01: controle com média dos3planos
versus atenção por alvo, mesmos embeddings/splits/labels/seeds/orçamento,
sem reextrair DICOMs. Depois G01 para amostragem adjacente. Não ajustar pesos
por alvo pelo leaderboard; confirmação150segue fechada. Melhor público0,941.
