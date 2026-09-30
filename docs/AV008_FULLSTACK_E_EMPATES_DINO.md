# AV-008 — ganho de execução e empates DINO

#RSNA #Kaggle #Pesquisa

Evidência de 15/09/2026 à noite em São Paulo, 16/09 UTC. Fonte de verdade:
nota 07_Estrategia_AVANCE no vault, espelhada em [ESTRATEGIA_AVANCE.md](ESTRATEGIA_AVANCE.md).

## Resultado e decisão

O stack36 com prefetch terminou com os cinco ramos íntegros e 10,35% menos
tempo de etapas. **Não passou no gate de CSV idêntico**. A causa da divergência
DINO foi reproduzida: ordem de soma float64 varia com a conclusão dos workers.
Uma correção exata foi testada em replay, mas não aplicada ao serving.
H43 parent 0,939 continua preservado; probe22 56263721 PENDING, sem novo envio.

## Benchmark completo

Kernel jvlegend/rsna-knee-e03-fullstack36-prefetch v1, ID 134543253 COMPLETE.
36 estudos/205 séries de treino V01, 56 hashes verificados, T4x2, cinco ramos,
108 pares receita-estudo Raptor; CoAt três modelos, zero falhas/fallback.
Build SHA:
`7d2dc6c5f3538c451ae332c22ae92949b29370c1f02a82b3f86af6123285cb5c`.

| Etapa | Serial anterior (s) | Prefetch (s) |
|---|---:|---:|
| DINO | 201,3149 | 187,0247 |
| A5 | 29,4803 | 32,7226 |
| Rad | 49,4276 | 49,9913 |
| Raptor/CoAt/fusão | 387,4955 | 328,8618 |
| Soma das etapas | 667,7184 | 598,6004 |

Redução total 10,351%; subtotal Raptor/CoAt/fusão 15,131%. São workers de
execuções distintas; não atribuir toda variação ao prefetch. O teste isolado
ABBA anterior é a evidência controlada do Raptor. Não medimos o teste oculto.
Projeção com as mesmas hipóteses/margens: 7,69 h para 1.322 casos hipotéticos,
11,60 h para 2.000; N oculto desconhecido. Não é garantia de caber no limite.

CSV de referência SHA:
`d774e25c03db117dc708851d8fb9eb588c722f92935377a117825bac674832ae`.
Novo SHA:
`bb2dd2a4f2204c83a13182fbba8a6270b61c3d397bde640a49d7cf918220cdff`.
Quatro valores Lateral OA (delta máx 0,0277778) e dois Effusion (0,0138889);
dez alvos idênticos. Gate FAILED_EXACT_CSV_PARITY, promoção suspensa.

## Localização e prova por replay

Os componentes `_raptor.csv`, `_coat_arm.csv`, native DINO e legacy-fold
são idênticos byte a byte. O primeiro CSV divergente é
`submission_public_0899.csv`: dois LOA e dois Effusion, antes de executar
Raptor. A divergência propaga-se na fusão. Não é evidência de alteração das
previsões Raptor causada por prefetch.

A função DINO `bank()` adiciona membros conforme as GPUs terminam.
`_combine()` soma seus ranks percentuais em float64 nessa ordem, e
`write_submission()` faz um segundo rank. Arredondamento da soma rompe empates
matemáticos; o segundo rank amplifica diferenças minúsculas.
Os logs históricos trocam somente 8476b29285/72081758ce nas posições 15/16.

Captura diagnóstica:

- Kernel jvlegend/rsna-knee-dino36-rank-order-capture v1, ID 134544795 COMPLETE;
  offline/T4x2, teto 1.800 s, mesmos 36 casos, execução somente até DINO.
- Build SHA `59cfe0ec8896bfd54feaa63b02d154eb5a8ed2951175484e8db30da3c07e30d7`.
- NPZ com 20 membros × 36 estudos × 12 alvos, IDs e hash conferidos.
  SHA `0b4d07ece336ca5b16132bda9fb5f9d5d9276353c34cb666ee08a2a79fc991f0`.
- Replay local primeiro reproduz o CSV da captura; depois aplica cada ordem
  histórica às mesmas previsões. **Ambos os CSVs DINO históricos reproduzidos
  exatamente**, com zero divergência em todos os alvos. Causa reproduzida.

Nunca submeter benchmark/captura: são casos de treino e a captura omite ramos.
Não houve consulta de labels para selecionar ordem, nem avaliação AUC.

## Correção candidata e limites

Para os 20 membros públicos de pesos uniformes e cobertura completa:
rank médio ×2 é inteiro, inclusive empates. Somar esses inteiros e só então
fazer rank final elimina denominadores comuns e arredondamento cumulativo.
O replay exato não mudou entre as duas ordens nem em 20 permutações aleatórias.

**Isso muda a receita numérica**: comparado ao legado da captura, altera
14 valores (Medial Meniscus2, Lateral OA4, PF OA2, Baker's6), corrigindo empates
que antes eram rompidos numericamente. Não implica aumento de AUC/leaderboard.
Não aplicar genericamente à função ponderada, que também atende outros ramos.
Não escolher ordem ou epsilon conforme score. Nenhum notebook aprovado mudou.

Próximo AVANCE:

1. Consultar probe22 56263721 antes de qualquer envio; não duplicar.
2. Implementar a regra exata apenas no ramo público uniforme, com verificações
   de pesos, IDs, cobertura e empates. Manter receita H43 histórica separada.
3. Medir baseline serial e prefetch do stack36 com a MESMA regra exata.
   Exigir paridade dos componentes e do CSV final; investigar outros
   acumuladores concorrentes se ainda houver diferença.
4. Só depois preparar smoke no teste real; qualquer correção de empates será
   identificada como candidato distinto. Sem alterar seleção final.

## Código, testes e artefatos no HD

- `scripts/assess_h43_fullstack.py`: composição, hashes, tensor shapes,
  falhas CoAt, tempo e comparação final/por componente.
- `scripts/prepare_h43_dino_capture.py`: captura ancorada no SHA do benchmark,
  uma inserção de dump, encerramento após DINO, sem alterar predições.
- `scripts/dino_rank_replay.py`: contraexemplo sintético, replay real e
  alternativa inteira. Diagnóstico; nunca autoriza submissão automaticamente.
- 47 testes passaram. Manifesto V01 segue
  `365566b0f830e398e78bd36bd9118a7365c3af340f588e774c8bcc351398b6df`.
- Outputs: `reports/avance_av008_fullstack_v1/`,
  `reports/avance_av008_reference_components/`,
  `reports/avance_av008_candidate_components/`,
  `reports/avance_av008_dino_capture_v1/`.
- Auditorias: `reports/avance_av008_diagnosis/components.json`,
  `synthetic_order.json` e `actual_replay_v2.json`.
  São artefatos locais ignorados pelo git; nenhum dado/checkpoint enviado ao GitHub.

Reproduzir a partir da raiz do repositório:

```sh
uv run --no-project --with numpy python -m scripts.dino_rank_replay \
  --capture-directory reports/avance_av008_dino_capture_v1 \
  --output reports/avance_av008_diagnosis/actual_replay_next.json
```

Saída é exclusiva: usar nome novo para preservar as evidências anteriores.
