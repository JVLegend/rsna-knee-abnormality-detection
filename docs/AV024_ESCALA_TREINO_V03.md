# AV-024 — escalar o treino próprio com a geometria vencedora

#RSNA #Kaggle #Pesquisa

18/09/2026. Fonte de verdade:07_Estrategia_AVANCE do vault.
Protocolo registrado ANTES da primeira avaliação V03.

## Resultado auditado da AV-023

Kernel134798342 v1 COMPLETE; reports/avance_av023_g01/g01_audit_v1.json
PASSED_G01_AUDIT. SoftBCE dev das sementes2026/42:

| Entrada | 2026 | 42 |
|---|---:|---:|
| Controle V02 | 0,62607019 | 0,63189521 |
| Quartis físicos | 0,62624314 | 0,63235150 |
| Adjacentes físicos | 0,62426164 | 0,62611569 |

Adjacentes vencem nas duas sementes: referência própria promovida conforme
regra prévia. Menor perda significa maior acordo com professor fraco, não
prova de acurácia clínica ou AUC Kaggle. Confirmação150não avaliada.
667/1.647séries mudam com a ordem;1.647mudam com adjacência.
Gap mediano26,4004mm nos quartis versus3,5000mm nos adjacentes.
597,66s totais,549,01s no pré-processamento; replay NumPy aprovado.

## Hipótese e protocolo V03

Pergunta: ampliar a base de treinamento melhora a receita própria?
Comparar299treino (controle G01 adjacente) com aproximadamente1.000estudos.
Não alterar teacher, decoder, amostragem adjacente, resolução224,3planos,
DINOv2-S congelado ou sharedattention nesta rodada.

- Preservar todos os299estudos originais. Acrescentar grupos completos de
  laudos elegíveis até atingir1.000; admitir pequeno excesso pelo último grupo.
- Excluir gold, laudos iguais aos gold, todos IDs/grupos de dev250 e
  confirmação150, rótulos inválidos, laudos vazios e cobertura insuficiente.
- Seleção dos novos grupos: hash SHA256 com seed20260918. Completar primeiro
  grupos de treino já existentes. Nunca escolher por loss/score de dev/LB.
- Série por plano: maior Fluid_Sensitive/Fat_Suppression, desempate UID,
  mesma política histórica. Ordem física estrita e triplet centro±1.
- Reusar features G01 antigas exatamente; extrair só novos pixels/features.
  Bloquear pixels novos idênticos a qualquer triplet de desenvolvimento.
  Hash exato não exclui duplicatas aproximadas/pacientes; grupos de laudos
  também não constituem identificação completa de paciente.
- Duas sementes2026/42,20épocas,batch4,AdamWlr0,001/decay0,0001. Selecionar
  menor softBCE dev, primeira época em empate. Checkpoint por época.
- Promover expandido somente se melhorar controle nas DUAS sementes por
  >2e−6. Auditor NumPy de logits/BCE/IDs/hashes/seleção e reprodução do controle.
- Mais dados a20épocas também aumenta número de passos e custo: é comparação
  de receitas, não isolamento causal de quantidade de dados versus computação.

Sem avaliar confirmação, AUC clínica, tuning por alvo, novo ensemble ou CSV.
Nenhum novo envio à competição nesta execução de treino.

## Orçamento e arquivos

Quota observada0,78639hGPU. Teto1.200s, T4x2/cuda:0, offline, sem serviço pago.
Usar a competição anexada no Kaggle; não baixar dezenas de GB extras só
para este teste. Manifesto, features e checkpoints permanecem privados no
HD, sob data/processed e reports. O código/protocolo podem ir ao GitHub.
Se faltar quota ou dados, guardar preparação verificável e bloqueio concreto;
não alterar validação nem excluir estudos problemáticos silenciosamente.

## Estado

G01 auditado e adjacentes promovidos. V03 implementado;168testes+44subtestes
passaram. Código/protocolo commit549b33e antes de observar resultado V03.
Manifesto congelado:
data/processed/validation_weak_v3_scale1000/manifest.json, SHA256
cba46e58d8665ae37e1949d2eef68a34a60f299c1ffda75e890d26a6857bc715.
1.000treino/954grupos,299originais+701novos,3.890elegíveis. Excluídos455
ligados à validação/confirmação e62ligados ao gold (inclui laudos repetidos).
Sem arrays adjacentes idênticos por hash entre treino/dev originais.

Build reports/avance_av024_v03/v03_v1.py,773.265bytes,SHA256
3ae4654579db73f31779962b8ea25a620be67ec257522973ce0c36947e57088b.
Kernel `jvlegend/rsna-knee-v03-scale1000`,ID134854744,versão1,RUNNING.
Quota imediatamente antes do envio:2.745,679s; gate conservador exigiu2×1.200s.
Anexos competição + DINOv2-S oficial + saída privadaG01. Sem outros pesos.
Último log extra_features2/701 em32,60s de log; não estimar desempenho a partir
disso. Nenhuma perda V03 observada ainda, nenhuma nova submissão.

## Retomada e auditoria

Não duplicar execução. Consultar status e baixar versão1 em
reports/avance_av024_v03_v1/ quando COMPLETE (ou logs em ERROR).
Auditor independente implementado, mas ainda não executado sobre saída V03:

```sh
python -m scripts.assess_v03_scale \
  --directory reports/avance_av024_v03_v1 \
  --build reports/avance_av024_v03/v03_v1.py \
  --manifest data/processed/validation_weak_v3_scale1000/manifest.json \
  --baseline reports/avance_av023_g01_v1 \
  --output reports/avance_av024_v03/v03_audit_v1.json
```

Usar PYTHONPATH=src:. e dependências NumPy/Pandas/PyTorch. O auditor recusa
sobrescrever relatórios, recompõe seleção por grupos e confere fontes.
Exige features antigas idênticas, reprodução do controle G01 adjacente,
geometria/amostragem/IDs e ausência de colisão exata de pixels novos com dev.
Refaz logits/BCE em NumPy e verifica checkpoints/histórico/época selecionada.

Implementação: scripts/prepare_v03_scale.py, scripts/v03_scale_runtime.py,
scripts/assess_v03_scale.py e tests/test_v03_scale.py. Laudos não entram no
código enviado; apenas IDs/hashes/labels necessários e metadados de série.
Protocolo e retomada registrados primeiro no vault, conforme skill Obsidian.
