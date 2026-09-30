# H-39 — DINOv3 + head V16 + CoAtNet

## Objetivo

Testar uma extensão controlada do H-38 (`0,929` público): o braço CoAtNet
H-36 permanece como base dominante; o DINOv3 continua com o residual de 20%;
e o head V16 acrescenta uma contribuição pequena apenas nos alvos aprovados
por validação OOF pública.

## Nova evidência incorporada

O head V16 foi recuperado do output público de
`romantamrazov/rsna-knee-dinosaur-v4-train`. A implementação usa cinco heads
de atenção, features DINOv3 por slot com 1152 dimensões e seis slots
`plano x Fat_Suppression`. O manifesto público promoveu somente:

- `Medial Meniscus` e `Lateral Meniscus`;
- `Lateral OA`, `PF OA` e `Baker's`.

O código repete o gate de deployment: ganho OOF, suporte bootstrap, estabilidade
entre folds e diversidade em relação ao anchor. Se o output do kernel fonte não
for montado ou o contrato falhar, o arquivo final permanece exatamente igual ao
H-38.

## Fontes e licenças auditadas

- `mattiaangeli/knee-mri-fold-weights`: CC0-1.0.
- `dreaddevelopment/raptor-knee-maxspan`: CC0-1.0.
- `romantamrazov/rsna-knee-dinosaur-v4-train`: kernel público usado somente
  como fonte do output V16, sem fontes privadas anexadas.

## Critério de decisão

1. Executar offline em T4.
2. Exigir schema exato, IDs únicos, valores finitos e cobertura integral.
3. Conferir o log para confirmar `[H-39/V16] promoted`.
4. Submeter somente após autorização explícita; promover apenas se superar
   H-38 `0,929`.

## Resultado da execução

A versão 4 concluiu no T4, encontrou o bundle público e promoveu os cinco
alvos previstos pelo manifesto. O CSV final passou schema, cobertura, IDs,
finitude e faixa, mas ficou byte a byte idêntico ao H-38 porque o teste público
tem apenas três estudos e os ranks não mudaram. Portanto o H-39 foi mantido
como implementação auditada, sem nova submissão.

#RSNA #Kaggle #Tecnologia #Academia
