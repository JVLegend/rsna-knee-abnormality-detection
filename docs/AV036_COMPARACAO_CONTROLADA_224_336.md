# Comparação controlada 224 versus 336 — G05

#JoaoVictor #Kaggle #Tecnologia #Academia

30/09/2026. Fonte de verdade do protocolo; contexto operacional em
[[07_Estrategia_AVANCE]]. Implementação e evidências no HD externo, repositório
`/Volumes/Karine HD Externo/Dados_JV/Projetos_GitHub/rsna-knee-abnormality-detection`.
Espelho detalhado: `docs/AV036_COMPARACAO_CONTROLADA_224_336.md`.

## Retomada AV-037 — 30/09

Extrator pareado CPU/GPU implementado em `scripts/g05_pair_runtime.py`, builder
`prepare_g05_extraction.py` e auditores independentes de headers/pixels. Gate
GPU mantém exigência de fonte verificável de PatientID, replay dos shards e
duplicatas; não confiar apenas nos booleanos de um recibo. Código testado não
significa qualidade medida. Protocolo científico e split continuam inalterados.

Auditoria exaustiva de todas as séries/headers dos1.600exames concluída,
kernel136572534v1 COMPLETE, CPU privado/offline; não duplicar.8.813séries,
296.241headers, zero inconsistências/SOPs repetidos entre exames;2.648,918s.
Auditor independente aprovou cobertura técnica, **não independência de paciente**.
1.600PatientIDs distintos, zero repetições; Issuer/StudyDate e declaração de
desidentificação ausentes.73testes aprovados; extrator técnicov2 empacotado
emLZMA, sem alterar receita científica. Sem GPU/treino/score da confirmação. A pesquisa
não encontrou garantia específica de estabilidade longitudinal/namespace
do PatientID. Documentação de ferramentas de anonimização não certifica dados.
Custos, recibos e decisão consolidada: [[RELATORIO_PARA_JIRAYA]].

Os blocos seguintes registram a AV-036; o estado atual de execução e os
bloqueios evoluídos estão no relatório AV-037, sem apagar o histórico.

## Estado e referência

**H46 ref56696639 COMPLETE, público0,943**, verificado pela API, novo melhor.
Preservar fonte SHA7dc49666…, CSV SHA7c6dfe8…, pesos e seleção; não duplicar.
G04v2 COMPLETE/auditado: 20 estudos, 60 séries, paridade224 exata; viabilidade
técnica336, **nenhuma evidência de melhora de qualidade**.

Consulta em30/09 às17:07:57BRT: prazo22/10/2026 às20:59BRT; GPU19,03h
restantes, reserva0; reset02/10 às21hBRT. H46/G04/V03 concluídos. Vaga de
execução não é comprovada pela quota: reconsultar antes de dispatch. Limite
5envios/dia, 1observado na dataUTC; HD ~222GiB livres. Não criar gastos novos.
Recibo: `reports/avance_av036_g05/resources_v1.json`.

## Receita pré-especificada

- Mesmos1.000treinos/300dev/300confirmação da V05 SHA9bb462ae…; não redividir.
- Mesmas séries/fatias: três planos, adjacentes centrais por posição física,
  FOV nativo e três canais grayscale. Mesmo decoder/percentis1–99/uint8/resize
  bilinear. Única variável experimental: tamanho224 ou336.
- DINOv2-S oficial congelado, CLS384, fp32/TF32off, mesmas versões e hashes;
  StudyAttention384→64→Tanh→1/softmax, classificador384→12.
- BCEWithLogits média não ponderada, mesmos rótulos fracos e incerteza0,5;
  AdamWlr0,001/wd0,0001, batch4, semaugmentation, seeds2026/42.
- Mesma inicialização e ordem de treino por seed, **20épocas fixas** para ambos;
  sem escolher época pelo dev, seed ou condição. Retreinar224 e336 em par.

Protocolo executável final: `reports/avance_av036_g05/protocol_v2.json`.
v1 foi rascunho prévio da mesma receita, sem resultado/treino; v2 acrescenta
pins de código. Fontes: `scripts/resolution_comparison.py`,
`scripts/run_resolution_heads.py`, `scripts/audit_resolution_patients.py`.

## Gates definidos antes de ver resultados

δ=336−224; menorBCE é melhor. Exigir no dev e depois, sem ajustes, na confirmação:

1. Primário: médiaBCE suave sobre estudos/12condições/2seeds; δ≤−0,001,
   IC95%pareado com limite superior<0 e melhora nas duas seeds.
2. Nenhuma condição com regressãoBCE>0,01; limite superior do IC simultâneo
   Bonferroni das12diferençasBCE≤0,02. Reportar todas as regressões menores.
3. AUC por condição/macro em rótulos exatos0/1, excluindo incertos/suaves;
   nenhuma queda pontual>0,02, pelo menos10positivos/10negativos por condição,
   ≥95%réplicas com AUC definida. Alertar baixa potência abaixo20porclasse.
4. Reportar IC95% dos braços e diferençasBCE/AUC por condição:5.000bootstraps,
   seed20260930, **estudo como unidade**, agrupando pacientes/laudos ligados.
   Manter pareamento; não contar fatias/seeds como pacientes independentes.
5. Job completo≤7.200s, picoGPUalocado≤4GiB, forward336/224≤3; quota≥14.400s
   antes de iniciar. Registrar memória alocada/reservada e tempos de decoding,
   cache, forward e heads, com custo reutilizado/incremental separados.

São métricas contra teacher fraco, não validação clínica nem garantia de score
Kaggle. Independência do nosso treino não resolve exposição histórica do teacher.
Resultado inconclusivo: não procurar split/seed/blend melhor na confirmação.

## Bloqueio de identidade e cobertura

V05 separa IDs de estudos e hashes de laudos; **hash de laudo não é paciente**.
Inventário local parcial de headers:299/1.600estudos com três séries, todos
treino original;3.903séries ausentes. PatientID existe, mas não há fonte
confirmando estabilidade da anonimização entre exames/sites. Zero colisões
observadas não certifica dev/confirm sem cobertura.

Exigir todos os headers das séries selecionadas, identidade consistente no
exame e fonte/mapping verificável da chave de paciente. Unir componentes por
paciente OU laudo. Qualquer ligação entre splits bloqueia os dois braços;
não mover casos silenciosamente. Nova divisão exigirá protocolo novo.
Inventário não leu pixels nem pontuou rótulos de confirmação; manifestos
contêm os rótulos fracos previamente congelados, não resultados de modelos.

## Execução e confirmação

- Reusar V03/G04 com SHA verificado. G04 tem apenas20features336; não basta
  para1.300estudos. Na AV-036 o extrator ainda não estava implementado;
  na AV-037 CPU/GPU dev foram implementados, sem extração real completa.
- Antes de fit: seleção física/hashes iguais nos braços, paridade224,
  auditoria de duplicatas exatas/aproximadas e headers completos. Treinador
  preparado falha antes do treino se gates/recibos/caches faltarem.
- Dev positivo: congelar os mesmos quatroheads; inferir uma vez nos300casos
  reservados, sem retreinar/ajustar. Mesmos gates. Marcador persistente de
  exposição impede segunda avaliação. Falha/interrupção exige reconciliação.
- Só depois de confirmação: preparar pacote offline, smoke e orçamento real
  de inferência para decisão pessoal doJV. **Sem submissão automática**,
  mesmo após aprovação. H46 não será substituído por este módulo sem evidência.

## Entrega e custo atual

G04:31,533s; forward336/224=2,219962; pico336≈204MiB. Extrapolação anterior
~93min para1.300casos/duasresoluções, não medição nem garantiaG05. Reuso pode
reduzir extração; auditoria completa acrescenta custo ainda não medido.
Nesta rodada: headersCPU90,078s;35testes locais passaram;0GPU,0jobs novos,
0envios e0serviços pagos novos. Métricas reais336/condições/IC: **não medidas**.

**Recomendação: manter H46 0,943; não enviar G05 agora.** Candidato entregue
é um protocolo/treinador de experimento, não checkpoint/Notebook apto ao envio.
Próximo avanço deve completar fonte de pacientes/cobertura e extrator pareado,
reconsultando recursos antes de executar. Confirmação permanece sem avaliação.
