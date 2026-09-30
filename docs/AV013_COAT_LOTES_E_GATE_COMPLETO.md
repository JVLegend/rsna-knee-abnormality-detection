# AV-013 — Raptor integrado, divergência CoAt e ordem dos lotes

#RSNA #Kaggle #Pesquisa

16/09/2026. Fonte de verdade: 07_Estrategia_AVANCE do vault,
[espelhada aqui](ESTRATEGIA_AVANCE.md). Continua [AV-012](AV012_PROBE22_0941_E_STACK_DETERMINISTICO.md).

## Resultado do par completo

Serial134588338/prefetch134588340, v1, COMPLETE. Ambos:36 estudos/205 séries,
T4x2, 56 hashes e todos os ramos presentes. Auditoria em
`reports/avance_av013_audit/fullstack_v1.json`:
**FAILED_DETERMINISTIC_FULLSTACK_PARITY**, não promover nem executar smoke.

- CSV final idêntico: `87ee1ac74c3e58d7dadfef705634eb23745f2322ad002e7772b6d03bad2ee777`.
- DINO público, raw canônico, replay inteiro e diagnóstico native idênticos.
- Raptor inputs/raw/ranks idênticos, inclusive à repetição isolada anterior.
- Único CSV divergente: `_coat_arm.csv`, dois valores ACL, delta1/108.
  Demais arquivos de previsão idênticos. Isso não permite ignorar a divergência.

| Etapa | Serial (s) | Prefetch (s) |
|---|---:|---:|
| DINO | 184,0139 | 202,3010 |
| A5 | 29,0438 | 31,5751 |
| Rad | 49,1475 | 48,0100 |
| Raptor/CoAt/fusão | 351,3457 | 320,2062 |
| Total | **613,5509** | **602,0923** |

Redução total1,8676%; subtotal Raptor/CoAt/fusão8,8629%. Workers distintos:
não atribuir toda a diferença ao prefetch, nem declarar ganho robusto.
Projeção com margem7,7366h para1.322 estudos hipotéticos;11,6739h para2.000.
Tamanho do teste oculto desconhecido, não afirmar que o limite está garantido.

O download completo trouxe também `legacy_fold_diagnostics.csv`. O auditor
anterior tentou tratá-lo como previsões e parou com KeyError StudyInstanceUID.
Corrigido o contrato: esse arquivo deve conter exatamente5folds com4membros;
continua sujeito à igualdade byte a byte. Inventários de CSV também precisam
ser iguais (antes era interseção). Nenhuma previsão excluída, nenhuma tolerância.

## Diagnóstico CoAt com artefatos reais

`reports/avance_av013_audit/coat_raw_v1.json`: hash, IDs, checkpoints,
shards e replay independente de ranks/CSV validados.

- Raw3×36×12: **71/1.296 valores diferentes**, máximo0,00003410875797.
- Somente estudos de índices11 e13, base zero, no shard0. Shard1 idêntico.
- No checkpoint e06, ACL do estudo11 muda0,12671905756→0,12672750652;
  cruza o estudo26, que fica em0,12672515213. Dois ranks mudam1/36;
  a média dos3checkpoints muda dois valores1/108.
- IDs e receita iguais. O lote histórico não foi registrado, logo não é
  possível atribuir retrospectivamente essa diferença à ordem com certeza.

Fonte original baixada no HD (49.679bytes), sem pesos/dados novos:
`reports/avance_av013_sources/coat/coatnet_resgated_ep10_top3_inference.py`.
SHA `b11e58f8d7cabe9e264ac70b01e811dc6a846aa01248ace1f0e6536d0d4094d9`,
igual ao runtime fixado no preflight. Origem:
[dataset CoAt](https://www.kaggle.com/datasets/mattiaangeli/rsna-knee-coat-resgated-ep10-top3).

Código usa FIRST_COMPLETED, ordena somente os futures que terminam naquele
instante e concatena em ready antes de formar batch2. Assim, pares e ordem
podem depender do tempo de preparo. `_infer_one_model` empacota imagens,
microblocos8 e padding: mudar o agrupamento pode mudar resultados fp16.
É hipótese específica, não prova causal. Teste CPU do loop real reproduz
pares variáveis no legado e pares fixos na variante.

## Experimento isolado iniciado

Kernel privado [CoAt36 Order ABBA](https://www.kaggle.com/code/jvlegend/rsna-knee-coat36-order-abba),
v1 **ID134608117**, RUNNING. Offline/T4/teto1.800s, cota antes18,5429h.
Preflight T4x2/56 hashes aprovado e primeira passagem concluída em116,51s.

Sequência: completion → ordered → ordered → completion.
Mesmos36/205, três checkpoints, fp16, micro8 e batch2. Ordered aguarda o menor
índice pendente; mantém até16preparos pendentes/2processos. Não muda modelo,
pesos, decode, resolução, backend ou referência dos labels. Nenhum novo treino.
Arquivos montados são preservados; cópias derivadas ficam apenas em working.

Instrumentação comum registra hashes de imagens/slots/metadados por estudo,
ordem real dos lotes e flags/versões em cada shard. Novo auditor verifica
recibos, código derivado, processos, cobertura, fallback, raw e replay de ranks.
Exige igualdade nas duas passagens ordered e inputs/ambiente iguais nas quatro.
Não exige que uma nova receita replique o erro numérico de um lote legado.
Resultado positivo habilita **teste completo**, não submissão automática.

Build: `reports/avance_av013_build/coat36_order_abba_v1.ipynb`, SHA:
`a530fafe06977ec9152d41d229f1596cab374d1396588d55a22f3b983133c419`.

Código: `h43_coat_order.py`, `prepare_h43_coat_order.py`,
`assess_h43_coat_order.py`. **83 testes passaram**; manifesto V01 intacto:
`365566b0f830e398e78bd36bd9118a7365c3af340f588e774c8bcc351398b6df`.
Dev/confirmation intocados, seleção final preservada, melhor público0,941.
Nenhuma nova submissão, automação ou recurso pago.

## Retomada

Após COMPLETE, sem duplicar:

```sh
uvx --from kaggle kaggle kernels output jvlegend/rsna-knee-coat36-order-abba \
  -p reports/avance_av013_coat_order_v1 \
  --file-pattern 'coat_order_run[0-3]/.*|h43_.*json|coat_order_abba_receipt.json'
uv run --no-project --with numpy python -m scripts.assess_h43_coat_order \
  --directory reports/avance_av013_coat_order_v1 \
  --output reports/avance_av013_audit/coat_order_abba_v1.json
```

Se ordered repetir exatamente, integrar o runtime derivado e testar o par
completo antes de smoke; preservar a baseline0,941. Se continuar divergindo,
registrar E03 inconclusiva e seguir V02, sem insistir em flags/precisão sem
evidência. Não relaxar o gate para esconder diferenças.
