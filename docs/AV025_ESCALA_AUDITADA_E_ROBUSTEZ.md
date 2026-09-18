# AV-025 — escala aprovada e diagnóstico de planos ausentes

#RSNA #Kaggle #Pesquisa

18/09/2026. Fonte de verdade: 07_Estrategia_AVANCE no vault.

## Resultado V03

Kernel jvlegend/rsna-knee-v03-scale1000, ID134854744 v1 COMPLETE.
Auditoria independente reports/avance_av024_v03/v03_audit_v1.json:
PASSED_V03_AUDIT. Artefatos privados em reports/avance_av024_v03_v1/.

| Treino / geometria adjacente | Seed2026 | Seed42 | Média |
|---|---:|---:|---:|
| Controle299 | 0,624261641 | 0,626115689 | 0,625188665 |
| Expandido1000 | 0,611192285 | 0,613102226 | 0,612147256 |

SoftBCE dev menor nas duas sementes: expanded promovido pelo critério
prospectivo da AV-024 (commit549b33e). Redução relativa média de cerca de2,09%;
não é AUC, melhora clínica ou score Kaggle. Épocas selecionadas8/14.
Fracture piora nas duas sementes; sem escolher checkpoint por alvo.
Última época20=0,637039960/0,637137413: preservar seleção/early stopping.
Mais dados também aumentam passos de otimização; não é ablação causal pura.

As features dos299originais e250dev são idênticas à referência G01.
Controle reproduzido; IDs, geometria, ausência de colisão exata com dev,
checkpoints e seleção auditados. Maior delta do replay NumPy <1e-6.
Total962,71s, features916,98s (fora imports). Confirmação150não avaliada.
Quota consultada nesta rodada:0,4849917hGPU, aproximadamente29,10min;
compartilhada, verificar novamente antes de qualquer novo job.

## Protocolo R02 — registrado antes do estresse

Diagnóstico local CPU, sem treinar. Usar somente best expanded seeds2026/42,
dev250 e labels/ordem de planos congelados do V03. Reproduzir logits e BCE
intactos antes; bloquear deriva de hashes, identidade ou resultados.

- Para cada plano Sagittal/Coronal/Axial: remover slot da atenção, equivalente
  à máscara booleana do head compartilhado sem posição.
- Controle separado: zerar feature e manter slot presente. Não representa
  ruído real de aquisição e não é tratamento correto de plano ausente.
- Reportar todas as12combinações (2sementes×3planos×2cenários), incluindo
  delta por alvo. Não selecionar novo modelo/época/labels com o estresse.
- Priorizar futuro ensaio de dropout se algum plano removido aumentar BCE
  >0,01 nas DUAS sementes. Limiar operacional, não significância estatística.
- Não tocar confirmação ou gold; não reextrair pixels; não submeter.

Implementação: scripts/diagnose_r02_slots.py; testes sintéticos em
tests/test_r02_slots.py com paridade contra a classe real StudyAttention.
O script recusa sobrescrever saída e registra hashes das entradas.

Com PYTHONPATH=src:. e NumPy/Pandas/PyTorch, limitar threads locais a1:

```sh
python -m scripts.diagnose_r02_slots \
  --directory reports/avance_av024_v03_v1 \
  --manifest data/processed/validation_weak_v3_scale1000/manifest.json \
  --audit reports/avance_av024_v03/v03_audit_v1.json \
  --build reports/avance_av024_v03/v03_v1.py \
  --geometry reports/avance_av023_g01_v1/g01_geometry.json \
  --output reports/avance_av025_r02/diagnostic_v1.json
```

## Histórico de implementação

V03 promovido. R02 registrado antes de executar o diagnóstico real.
Protocolo207544e. Primeira tentativa parou antes de qualquer métrica de
estresse: desenvolvimento V03 não contém séries, pois herda features G01.
Correção: exigir geometria G01 original por hash e validar todos IDs/planos
na ordem dos299treino+250dev; teste de regressão inclui esse esquema.
Sem alterar dados, modelo, regra de decisão ou relaxar a verificação.
Dropout de slots e ruído de aquisição ainda não testados. Estresse sintético
em exames completos não demonstra qualidade em pacientes com protocolo
incompleto. Nenhum CSV/novo envio; melhor público confirmado0,941 preservado.

## Resultado R02 — concluído

Protocolo207544e, correção de esquema fce369b, ambos antes das métricas.
reports/avance_av025_r02/diagnostic_v1.json:
COMPLETE_R02_SYNTHETIC_DIAGNOSTIC_NOT_TRAINING. Maior delta do replay
intacto9,664e−7; métricas intactas reproduzem V03. Sem reextrair DICOMs.

Delta de softBCE versus exame completo (positivo=piora):

| Cenário | Plano | Seed2026 | Seed42 |
|---|---|---:|---:|
| Removido da atenção | Sagittal | +0,021992 | +0,027067 |
| Removido da atenção | Coronal | +0,016185 | +0,006770 |
| Removido da atenção | Axial | +0,038457 | +0,047711 |
| Feature zerada, presente | Sagittal | +0,011438 | +0,008778 |
| Feature zerada, presente | Coronal | +0,008938 | +0,001686 |
| Feature zerada, presente | Axial | +0,021752 | +0,018713 |

Axial e sagital ultrapassam o limiar operacional0,01 nas duas sementes.
Coronal não ultrapassa nas duas. Menor queda com vetor zero NÃO demonstra
correção: é outra perturbação, altera o pooling/calibração e não valida
pacientes incompletos. Não trocar máscara por zeros nem excluir planos
com base neste resultado. Deltas por alvo preservados no relatório privado.

Fontes do diagnóstico: scriptSHA712ffb9e…; auditoriaV03SHA5260e807…;
featuresV03SHAbbc34ca6…; geometriaG01SHA349b57f4…; os hashes completos
constam do JSON. Checkpoints, features, rótulos e IDs não enviados ao GitHub.

## Próximo ensaio R02b — ainda não implementado/lançado

Controle V03 versus dropout de UM plano escolhido uniformemente em25%dos
estudos de treino. Não escolher planos pela queda observada. Duas sementes,
mesmos1000treino/dev250/features/labels/20épocas/batch4/AdamW; RNG de dropout
separado, mantendo shuffle do controle. Nunca remover todos os planos.
Escolher época somente por menor BCE intacta, sem seletor por alvo/cenário.

Promover apenas se BCE intacta melhorar >2e−6 nas duas sementes E a média
dos três cenários mascarados melhorar nas duas sementes. Se houver somente
ganho de robustez, registrar trade-off e manter V03 para submissão futura.
Registrar fonte antes do teste, reproduzir controle e auditar tudo.
Reusar features; teto previsto600sT4/offline, checar quota livre >=1200s
imediatamente antes do lançamento. Não migrar treino pesado para o Mac.
Confirmação150permanece fechada. R02b não é promessa de ganho no leaderboard.

## Verificação final

181testes e44subtestes passaram, incluindo13testes R02 (máscara real,
preservação dos inputs, rejeição de entradas inválidas, matriz completa,
limiar estrito nas duas sementes e esquema herdado de geometria).
Relatórios/features/checkpoints continuam ignorados pelo Git no HD externo.
Vault canônico e espelho conferidos; nenhum novo job desta rodada pendente.
