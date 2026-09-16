# AV-009 — agregação DINO estável e benchmark pareado

#RSNA #Kaggle #Pesquisa

15/09/2026 à noite São Paulo / 16/09 UTC. Fonte de verdade: nota
07_Estrategia_AVANCE do vault, com [espelho operacional](ESTRATEGIA_AVANCE.md).
Continua o [diagnóstico AV-008](AV008_FULLSTACK_E_EMPATES_DINO.md).

## Decisão e estado

A regra exata deixou de ser apenas protótipo do replay: está implementada em
dois notebooks experimentais completos. Ambos usam a mesma agregação DINO.
Somente a preparação Raptor muda entre serial e prefetch.
Execuções remotas iniciadas, sem paridade ou ganho de tempo concluídos ainda.
Não alterar H43 parent 0,939; probe22 56263721 permanece PENDING, sem reenvio.

## Código e teste local

- `scripts/h43_stable_rank.py`: média de ranks duplicados acumulada como
  int64, com rank global final pelo consumidor original. Exige 20 membros
  públicos fixados, 12 alvos, pesos unitários, IDs completos e únicos,
  previsões finitas e shapes corretos. Retorna somas de ranks, não probabilidades.
- `scripts/prepare_h43_stable_fullstack.py`: builder ancorado no SHA do
  benchmark36 aprovado. Muda somente células 33,55,57. O par difere apenas
  nas declarações de modo serial/prefetch; todos os demais modelos e a
  função _combine original ficam intactos.
- `scripts/assess_h43_stable_fullstack.py`: auditor preparado para composição,
  replay DINO independente, probabilidades raw alinhadas por membro/estudo,
  imagens/máscaras Raptor, raw Raptor, componentes e CSV final.
- Testes do helper sobre captura real 20×36×12 conferem exatamente com o
  algoritmo independente da AV-008 em 20 permutações. Há casos negativos de
  pesos não uniformes, NaN, IDs/shape incorretos, hash, modo e CSV adulterado.

**59 testes passaram** na suíte direcionada, executada com numpy/pandas em
ambiente efêmero uv; nenhum arquivo de dependências ou configuração alterado.

A mudança corrige empates artificiais; **não é paridade com o parent legado**.
Na captura AV-008 alterava 14 valores. Não houve avaliação AUC, ajuste de
pesos/epsilon/ordem pelo leaderboard nem consulta de dev/confirmation.

## Experimentos remotos

| Modo | Kernel privado | Versão / ID | Estado no despacho |
|---|---|---|---|
| Serial | jvlegend/rsna-knee-stable36-serial | v1 / 134546480 | RUNNING |
| Prefetch | jvlegend/rsna-knee-stable36-prefetch | v1 / 134546487 | RUNNING |

Cada execução: 36 estudos/205 séries do treino V01, cinco ramos, artefatos
fixados por 56 hashes, offline, T4, teto 1.800 segundos. Mesmo conjunto de
datasets/kernels/modelo do parent. **Nunca submeter estes notebooks de treino.**
Paridade pode falhar se outro acumulador não determinístico permanecer.

Cota: primeira resposta da API foi inconsistente com as seguintes (6h total,
10,31h usadas). Duas leituras subsequentes informaram 30h, 10,31h usadas,
19,6916h livres. Antes de criar os dois kernels, o gate exigiu saldo ≥3.600s.
Reset informado: 19/09/2026 00:00 UTC (18/09 21:00 São Paulo).
Nenhum recurso pago ou agendamento habilitado.

Builds no HD:

- `reports/avance_av009_build/h43_stable36_serial_v1.ipynb`:
  `b1e409de5e4192f61bc4d41c91a8d21c8f404bddeda605188affa6d82c5a6f41`.
- `reports/avance_av009_build/h43_stable36_prefetch_v1.ipynb`:
  `eb473bedf9c105247ef1fe4a12c2d34df53cbfd60d80ee739cc058bb39871093`.
- Helper exato:
  `0b660440c08ac21979b8e805f86d1ab3453ecfc5033f327e289a7edb13565a71`.
- Fonte benchmark:
  `835083352f0c0b7d8785794bb6846b6b2bd74a13846d5c68fbab7da713c4d614`.
- Manifesto V01 intacto:
  `365566b0f830e398e78bd36bd9118a7365c3af340f588e774c8bcc351398b6df`.

## Retomada

1. Consultar status dos dois kernels, sem criar nova versão. Se ERROR, baixar
   o log antes de corrigir. Se RUNNING, manter a identidade e não duplicar.
2. Quando COMPLETE, baixar somente os artefatos de auditoria no HD.
3. Rodar auditor abaixo. Exigir paridade final, raw e componentes; ausência
   de arquivo é bloqueio de auditoria, não aprovação. Um ganho de tempo em
   workers distintos não identifica sozinho o efeito causal do prefetch.
4. Se passar e houver ganho, preparar smoke no teste real em próxima etapa,
   claramente identificado como candidato de empates estáveis. Não reenviar
   probe22 pendente nem alterar seleção final.

Comandos a executar na raiz do repositório, após COMPLETE:

```sh
uvx --from kaggle kaggle kernels output jvlegend/rsna-knee-stable36-serial \
  -p reports/avance_av009_serial_v1 \
  --file-pattern 'h43_.*json|dino_stable_.*|e03_fullstack_inputs.json|e03_run0_raw.npz|coat_resgated_ep10_top3_submission_receipt.json|benchmark_predictions.csv|_raptor.csv|_coat_arm.csv|submission_public_0899.csv|submission_native_v38.csv|submission_legacy_fold_blend.csv|submission_e10_v2.csv|submission_parent_exact.csv'
uvx --from kaggle kaggle kernels output jvlegend/rsna-knee-stable36-prefetch \
  -p reports/avance_av009_prefetch_v1 \
  --file-pattern 'h43_.*json|dino_stable_.*|e03_fullstack_inputs.json|e03_run0_raw.npz|coat_resgated_ep10_top3_submission_receipt.json|benchmark_predictions.csv|_raptor.csv|_coat_arm.csv|submission_public_0899.csv|submission_native_v38.csv|submission_legacy_fold_blend.csv|submission_e10_v2.csv|submission_parent_exact.csv'
uv run --no-project --with numpy --with pandas python -m scripts.assess_h43_stable_fullstack \
  --reference reports/avance_av009_serial_v1 \
  --candidate reports/avance_av009_prefetch_v1 \
  --output reports/avance_av009_audit/assessment_v1.json
```

O output da auditoria é exclusivo: usar nome novo ao repetir. Artefatos,
previsões e pesos ficam no HD externo/área privada Kaggle, fora do GitHub.
