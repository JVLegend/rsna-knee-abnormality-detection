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
  --output reports/avance_av025_r02/diagnostic_v1.json
```

## Estado antes de avaliar

V03 promovido. R02 implementado, diagnóstico real ainda não executado.
Dropout de slots e ruído de aquisição ainda não testados. Estresse sintético
em exames completos não demonstra qualidade em pacientes com protocolo
incompleto. Nenhum CSV/novo envio; melhor público confirmado0,941 preservado.
