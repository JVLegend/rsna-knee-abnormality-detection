# AV-027 — consistência aprovada no desenvolvimento; confirmar antes de enviar

#RSNA #Kaggle #Pesquisa

21/09/2026. Fonte canônica:07_Estrategia_AVANCE do vault.
Continuação de [AV-026](AV026_TREINO_ROBUSTO_E_CONSISTENCIA.md).

## Resultado real e auditado

Kernel jvlegend/rsna-knee-r02-robust-consistency, ID135135626 v1 COMPLETE.
Protocolo/código a6fc5dd anterior ao treino; build recomposto sem alteração:
17ad28e1a5882f76b1a342a768279ccd525b9a7eae5831e052f12db0903b4887.
Quota pré-envio84.547,447s. Seis heads sobre features1000treino/dev250,
sem novo DICOM extraído. Não é fine-tuning do encoder.

Menor softBCE é melhor. Médias de ausências incluem os três planos
removidos separadamente no checkpoint selecionado pela BCE INTACTA:

| Receita | Intacto2026 | Intacto42 | Ausências2026 | Ausências42 |
|---|---:|---:|---:|---:|
| Controle V03 | 0,611192285 | 0,613102226 | 0,636736768 | 0,640284785 |
| Dropout25 | 0,610715930 | 0,610301140 | 0,623104157 | 0,626013554 |
| Paired consistency | 0,607180631 | 0,609534184 | 0,618644572 | 0,623095017 |

Média intacta:0,612147256→0,608357407. Média ausências:
0,638510777→0,620869795. Ambos candidatos passam os gates nas duas sementes;
paired_consistency vence pelo desempate prévio de menor média intacta.
Promovido como referência PRÓPRIA DE DESENVOLVIMENTO, não como submissão.
Todos selecionaram épocas8/14 nas seeds2026/42.

Não houve melhora universal: MedialOA na seed42 piora0,014052832,
LateralOA piora0,004914453. Na seed2026 esses alvos melhoram; registrar
dispersão, não selecionar modelo por coluna depois de observar resultados.
Dropout25 fica em reserva, não se torna ingrediente automático de ensemble.
Receita paired muda supervisão/exposição além da consistência; não isolar
causalmente o ganho no termo MSE. Ausência sintética não valida protocolo
incompleto real. Rótulos fracos não medem acurácia clínica ou AUC Kaggle.

## Auditoria e artefatos

reports/avance_av026_r02/audit_v1.json: PASSED_R02_TRAINING_AUDIT,
SHA256e6208422e628ccc13c6164c4cd3fb6c6f3e690db408e0ff52549c2051fb7d172.
Saídas completas no HD: reports/avance_av026_r02_v1/.
ReceiptSHA2561b863e5ab81a28ff8b8c84f0b15c8e71b5336e3ac0b88c7fb3bdc7ecf1adb7a8.

- Curva, épocas e logits do controle reproduzem V03.
- Hashes/IDs/checkpoints best e last, seleção de época e perdas aprovados.
- RNG, contagens e cadeia SHA das máscaras reconstruídos independentemente.
- Replay NumPy dos heads intactos e mascarados; maior diferença1,281e−6.
- Tempo123,7558s fora imports; pico alocado75.997.184bytes, só cuda:0.
- 197testes+44subtestes passaram novamente. Sem relaxar tolerâncias.

Sem avaliação dos150reservados, CSV, nova submissão ou mudança de finalistas.
Melhor público confirmado0,941 continua pertencendo ao ensemble anterior.

## Protocolo V04 — próximo AVANCE, antes de abrir confirmação

Ainda NÃO implementado/executado. Objetivo: confirmar uma única comparação
previamente escolhida, sem continuar selecionando receitas nos150reservados.
O conjunto está fechado para esta linha de treino; não alegar virgindade
histórica de todo o projeto ou anotação clínica independente.

### Entradas e quatro checkpoints fixos

Usar somente control e paired_consistency do R02, seeds2026/42. Nenhum novo
treino, média de pesos, calibração, seleção de época, probabilidade ou alvo.
Checkpoint da seed2026 é época8; da seed42, época14, já escolhido no dev.
Congelar manifeste/contrato da inferência ANTES de observar previsões.

| Receita / seed | SHA256 do best.pt |
|---|---|
| control2026 | e2875906f644c76ee1a6c5b8b24af08050a3e4a59b448a647e29daaece7a654d |
| control42 | 5a099c18e7248ab498f8d135dc08c9962ca2f7043062598467045751d552f8d8 |
| paired2026 | 92aa32ae8ab84214a7dea9850cad03e378b78842c51614659522f3665b5c408b |
| paired42 | 70c41c23451945638eefc7d75aea4667c4137b73bca8e71f1315e9b003fa8972 |

IDs/labels/grupos dos150: data/processed/validation_weak_v1/manifest.json
SHA365566b0f830e398e78bd36bd9118a7365c3af340f588e774c8bcc351398b6df.
Preservar0,5 incerto. Metadados de cobertura já existem; nenhuma classe
sem positivos/negativos pelo corte0,5, mas isso é referência fraca.
Fracture tem14positivos/69negativos/67incertos; Baker's41/48/61.
Não confundir esses contadores com casos clinicamente adjudicados.

### Execução e validação

1. Checar nenhum ID/grupo reservado no treino1000; confirmar todos hashes.
2. Mesma escolha de séries/planos e exatamente mesmo decoder, percentis,
   adjacência física,224px e DINOv2-S oficial congelado do G01/V03.
3. Reusar referência de geometria para treino299+701novos/dev250; bloquear
   colisões EXATAS de pixels entre confirmação e treino/dev antes de medir.
   Não é garantia contra duplicatas aproximadas ou mesmo paciente.
4. Se houver metadado inconsistente, caso ausente ou colisão: registrar
   falha e parar. Não remover/substituir estudos para obter resultado.
5. Inferir quatro checkpoints e três máscaras fixas por checkpoint.
   Auditor independente de logits/BCE/IDs/modelos/features antes de concluir.
6. T4/offline, teto previsto1200s/guard1050s e quota>=2400s; inferência
   reservada, sem CSV da competição. Confirmar custo antes de lançar.

### Critério prospectivo de confirmação

Primário: mean softBCE intacta contra labels V01, sem threshold de predição.
Exigir melhora >2e−6 do paired sobre control nas DUAS sementes, além de
melhora >2e−6 da média dos três cenários de ausência nas duas sementes.
Margem é operacional/numerical, não prova de significância clínica.

Quantificar incerteza: calcular delta intacto por estudo (média dos12alvos e
das2sementes); bootstrap pareado de grupos de laudo,5000réplicas,
seed20260921. Sortear grupos com reposição, manter todos estudos de cada
grupo e ponderar pelo número de estudos resultante; intervalo percentil95%.
Para chamar o resultado de confirmado, exigir limite superior <0.
Além disso, nenhuma regressão média por alvo entre sementes >0,01softBCE.
Esse limite por alvo é uma proteção operacional definida agora, não limiar
clínico validado. Registrar Medial/LateralOA explicitamente.

AUC pode ser apenas diagnóstico secundário da referência fraca: excluir
labels exatamente0,5 e binarizar os demais por >0,5, publicar contagens e
limitações, não usar para escolher pesos. Se AUC indefinida, registrarNA.
Não transformar isso em métrica clínica ou misturá-la com softBCE.

Se gates falharem ou intervalo cruzar zero: resultado inconclusivo/negativo;
manter referência de desenvolvimento com essa ressalva, sem promovê-la
automaticamente a submissão. Não ajustar modelo e reconsultar os mesmos150.
Qualquer próxima seleção exige outra validação/OOF previamente definida.
Após consulta, registrar a exposição do conjunto, seja resultado bom ou ruim.

Mesmo se aprovado, ainda faltam inferência oficial, smoke/IDs/colunas,
benchmark de custo e verificação das regras/cota antes de submissão.
