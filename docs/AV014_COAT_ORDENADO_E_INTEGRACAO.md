# AV-014 — CoAt ordenado aprovado; integração completa

#RSNA #Kaggle #Pesquisa

16/09/2026. Fonte de verdade: 07_Estrategia_AVANCE do vault,
[espelhada aqui](ESTRATEGIA_AVANCE.md). Continua [AV-013](AV013_COAT_LOTES_E_GATE_COMPLETO.md).

## Resultado auditado

CoAt36 Order ABBA v1 **ID134608117 COMPLETE**. Auditoria independente em
`reports/avance_av014_audit/coat_order_abba_v1.json`: **PASSED_ORDERED_REPEAT**.
Outputs completos no HD: `reports/avance_av013_coat_order_v1/`.

- Mesmos36estudos/205séries,3checkpoints, dois shards/T4, fp16/micro8/batch2.
  Preflight56hashes, código derivado, raw, shards, processos, CSV e replay
  independente de ranks verificados; nenhum fallback de microbatch.
- Inputs (imagens/slots/metadados) e ambiente iguais nas quatro passagens.
- Ordered B1/B2: probabilidades, ranks e ensemble **exatamente iguais**.
- Completion A1/A2: **248/1.296 valores raw diferentes**, máximo0,0001143664,
  em7estudos. Ranks e CSV iguais nesta amostra. Lotes registrados variam.
- Completion A1 versus ordered B1:320 diferenças raw; ranks/CSV iguais.
  Não exigir reprodução de uma ordem de lote variável como contrato da correção.

| Passagem | Ordem | Tempo (s) |
|---|---|---:|
| A1 | Conclusão do preparo | 116,5058 |
| B1 | Índice fixo | 113,9788 |
| B2 | Índice fixo | 115,8829 |
| A2 | Conclusão do preparo | 115,7087 |

Média ordered114,9309s. Não declarar speedup robusto ou ganho de AUC.
É evidência de estabilidade em duas repetições na mesma sessão, não prova
universal. Os lotes da antiga divergência ACL não foram capturados; não há
prova retrospectiva de sua causa. Não usamos novos labels/dev/confirmation.

Raw NPZ ordered B1/B2 SHA:
`b5b1b1f51e9bd9cf245bb7bd6c7f52987330b23fe269d8e07d6841e7f4988d6e`.
CSV comum às4passagens SHA:
`d379ede8268fa25f72e985b4c890fb3386ae1a88988b2e7aa9e84931616d04aa`.

## Integração implementada

`prepare_h43_ordered_fullstack.py` reaudita os arquivos ABBA antes de construir.
Só célula57 difere dos builds determinísticos AV-012 correspondentes. DINO,
Raptor, pesos, resolução, backend, fusão e amostra permanecem iguais.
O processo CoAt carrega uma cópia derivada no working; arquivos montados
permanecem intactos. Os workers executam essa mesma cópia pelo __file__.
Preserva recibo de receita, código e log do filho. PACKED FALLBACK aborta.

- Helper CoAt SHA:
  `8f77911f466fabe64678a6e05c24be02d3cb47dc0bb45ee0cf6e9aee8273384c`.
- Runtime ordered SHA:
  `31c4acb53b666374b02ead1dc1d2554cbb80c467ede780b3377ef0ace8952f3a`.

`assess_h43_ordered_fullstack.py` conserva o gate completo anterior e acrescenta
raw/ranks/CSV, imagens/slots/metadados, lotes e ambiente CoAt iguais à âncora
run1 ordered do ABBA. Comparação sem tolerância e sem excluir componentes.
Uma falha impede promoção mesmo que o CSV final coincida.

**87 testes passaram**, incluindo parse do código completo do processo filho,
rejeição de helper/build/receita diferentes e modificação restrita à célula57.
Manifesto V01 intacto:
`365566b0f830e398e78bd36bd9118a7365c3af340f588e774c8bcc351398b6df`.

## Par completo iniciado

Privados/offline/T4/teto1.800s cada,36estudos/205séries do treino V01.
Cota antes do despacho18,3958h. São benchmarks, nunca submeter estes outputs.

| Modo Raptor | Kernel | ID / versão |
|---|---|---|
| Serial | jvlegend/rsna-knee-ordered36-serial | 134609177 / v1 |
| Prefetch | jvlegend/rsna-knee-ordered36-prefetch | 134609181 / v1 |

CoAt usa a mesma ordem fixa nos dois. Builds em `reports/avance_av014_build/`:

- `h43_ordered36_serial_v1.ipynb`:
  `7ec87b02730c848097f841c366d2783b62d32878a5f15a794a33de5af7e3728a`.
- `h43_ordered36_prefetch_v1.ipynb`:
  `11ddd8e5ad7123ae7f6d322f77b42f37c6ccaf0dd0ca139d59454161bcb58927`.

## Próxima retomada

Consultar os dois kernels sem duplicar. Após COMPLETE:

```sh
uvx --from kaggle kaggle kernels output jvlegend/rsna-knee-ordered36-serial \
  -p reports/avance_av014_serial_v1 \
  --file-pattern '.*csv|.*json|.*npz|h43_coat_ordered_runtime.py|h43_coat_worker.log'
uvx --from kaggle kaggle kernels output jvlegend/rsna-knee-ordered36-prefetch \
  -p reports/avance_av014_prefetch_v1 \
  --file-pattern '.*csv|.*json|.*npz|h43_coat_ordered_runtime.py|h43_coat_worker.log'
uv run --no-project --with numpy python -m scripts.assess_h43_ordered_fullstack \
  --reference reports/avance_av014_serial_v1 --candidate reports/avance_av014_prefetch_v1 \
  --output reports/avance_av014_audit/fullstack_v1.json
```

Se passar com ganho de tempo, adaptar smoke para a mesma receita; o builder
de smoke antigo não contém todas as correções. Se voltar a divergir, registrar
E03 inconclusiva e seguir V02, sem nova rodada cega de flags/precisão.
Melhor público **0,941** preservado. Não houve nova submissão, alteração da
seleção final, serviço pago ou automação. Estabilidade não é novo score.
