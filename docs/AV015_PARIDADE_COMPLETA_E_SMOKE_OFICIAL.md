# AV-015 — paridade completa aprovada e smoke oficial

#RSNA #Kaggle #Pesquisa

16/09/2026. Fonte de verdade: 07_Estrategia_AVANCE do vault,
[espelhada aqui](ESTRATEGIA_AVANCE.md). Continua [AV-014](AV014_COAT_ORDENADO_E_INTEGRACAO.md).

## Resultado confirmado

Ordered36 serial134609177/prefetch134609181 v1 COMPLETE.
Auditoria: `reports/avance_av015_audit/fullstack_v1.json`,
**PASSED_ORDERED_FULLSTACK_PARITY**, elegível para smoke oficial.

- Mesmos36estudos/205séries do treino V01,5ramos,56hashes,T4x2.
- Todos os CSVs e diagnósticos byte a byte idênticos, sem excluir native.
- DINO raw canônico e replay inteiro exatos.
- Raptor imagens/máscaras/raw/ranks exatos entre modos e à âncora isolada.
- CoAt raw/ranks/ensemble, imagens/slots/metadados, lotes e ambiente exatos
  entre modos e ao run1 ordered do ABBA. Fonte derivada verificada por SHA.
- Nenhum fallback de microbatch, ausência de membro ou tolerância ajustada.

CSV final SHA:
`87ee1ac74c3e58d7dadfef705634eb23745f2322ad002e7772b6d03bad2ee777`.
Outputs em `reports/avance_av014_serial_v1/` e `reports/avance_av014_prefetch_v1/`.

| Etapa | Serial (s) | Prefetch (s) |
|---|---:|---:|
| DINO | 185,3382 | 190,6885 |
| A5 | 30,5360 | 30,4530 |
| Rad | 49,5616 | 49,3479 |
| Raptor/CoAt/fusão | 358,8682 | 310,1589 |
| Total | **624,3039** | **580,6483** |

Redução total **6,9927%**; subtotal Raptor/CoAt/fusão **13,5730%**.
Workers distintos: ganho observado, não atribuição causal integral ao prefetch.
Não é intervalo de confiança nem ganho de AUC. Projeções com margem:
7,4614h para1.322estudos hipotéticos,11,2585h para2.000; não confirmam
tamanho do teste oculto ou garantia de respeitar9h. Revalidar regras no envio.

## Smoke implementado e iniciado

`prepare_h43_ordered_smoke.py` reaudita o par e deriva diretamente do build
prefetch aprovado SHA11ddd8e5…, em vez de reutilizar o smoke antigo incompleto.
Só células4/24/43/55/57/61 mudam: retira seleção de treino, restaura descoberta
dos inputs oficiais DINO/A5/Rad/Raptor/CoAt e altera publicação final.
Modelos, pesos, flags, soma inteira, prefetch, ordem CoAt e fusão preservados.

Saída **smoke_predictions.csv**, sem publicação de submission.csv. A receita
continua outer parent0,60: não é o preset probe22 histórico0,941. Nenhum novo
score deve ser atribuído à correção de estabilidade antes de avaliar uma
candidata de competição separada.

Recibos de receita preservam seus nomes de protocolo de origem, incluindo
nomes históricos com36. Isso não é contagem da execução: ordered_smoke_receipt
e h43_parent_integrity registram3estudos reais e root oficial. Metadata ativa
do notebook não contém seleção ou caminhos do benchmark.

`assess_h43_ordered_smoke.py` exige3IDs do sample oficial,5ramos/56hashes,
raízes oficiais, CSVs finitos e completos, fonte/flags ordenadas, zero fallback,
replay independente de DINO e CoAt, cobertura raw/inputs Raptor e tempos válidos.
Leitor CoAt aceita contagem/series explícitas para smoke; testes36/205 seguem
com a mesma exigência. Teste adicional cobre fixture3casos e rejeita205séries
quando o contrato explicitado é18. Fixture não é contagem declarada do teste real.

**91 testes passaram**, V01 intacta:
`365566b0f830e398e78bd36bd9118a7365c3af340f588e774c8bcc351398b6df`.
Dev/confirmation intocados; sem seleção por labels ou novo treino.

Kernel privado [Ordered Official Smoke](https://www.kaggle.com/code/jvlegend/rsna-knee-ordered-official-smoke),
v1 **ID134610738**, iniciado offline/T4/teto1.800s. Cota antes18,0020h.
Build: `reports/avance_av015_build/h43_ordered_official_smoke_v1.ipynb`.
SHA `9ca08d0ab5465e7da77dd3459201901e7f2ea9ceb8ab64b70def386a428470e7`.

## Retomada

Consultar ID134610738, sem duplicar. Após COMPLETE:

```sh
uvx --from kaggle kaggle kernels output jvlegend/rsna-knee-ordered-official-smoke \
  -p reports/avance_av015_smoke_v1 \
  --file-pattern '.*csv|.*json|.*npz|h43_coat_ordered_runtime.py|h43_coat_worker.log'
uv run --no-project --with numpy python -m scripts.assess_h43_ordered_smoke \
  --directory reports/avance_av015_smoke_v1 \
  --output reports/avance_av015_audit/smoke_v1.json
```

Se passar, revisar preset/candidata de envio, regras vigentes, orçamento e
duplicidade. Não submeter este smoke automaticamente;3exemplos verificam
integração, não AUC ou runtime oculto. Melhor público **0,941** preservado.
Nenhuma nova submissão, seleção final, automação ou serviço pago nesta rodada.
