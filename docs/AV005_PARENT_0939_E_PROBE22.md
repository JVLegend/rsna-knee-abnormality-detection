# AV-005 — parent confirmado em 0,939; A02 probe22

#RSNA #Kaggle #Pesquisa #Tecnologia

15/09/2026. [Plano AVANCE](ESTRATEGIA_AVANCE.md), continuação da
[submissão AV-004](AV004_BENCHMARK_E_SUBMISSAO.md).

## Resultado observado — A01 concluída

API Kaggle consultada em 15/09/2026, aproximadamente 19:43 São Paulo:

- **H43 parent**, ref **56253529**, scriptVersionId **350055640**:
  **COMPLETE**, público **0,939**, sem erro.
- H38, ref 55916072: 0,929. Delta público exibido **+0,010**.
- H42, ref 56217840: 0,881, permanece rejeitada.

H43 passa a ser a melhor referência pública confirmada. H38 permanece
disponível e nenhuma seleção final oficial foi alterada. A01 reproduziu o
score público 0,939 da fonte; não há OOF independente nem garantia de ganho
no privado. O Kaggle concluiu o teste oculto sem timeout, mas não obtivemos
um tempo de execução oculto nem o número real de estudos por esse endpoint.

## A02 — hipótese e escolha antes da execução

Escolha única: **probe22**, já publicado na
[fonte auditada](https://www.kaggle.com/code/maverickss26/rsna-knee-0941-restructured).
Não testar também halfway nesta rodada e não buscar uma grade própria.
A fonte relata 0,941; esse número ainda não é nosso resultado.

Única mudança numérica: peso externo do ramo híbrido Raptor/CoAt contra
Transformer/Rad. A mistura interna Raptor 60% / CoAt residual 40% e o
reranking global com empates médios permanecem iguais.

| Alvo | Parent | Probe22 |
|---|---:|---:|
| ACL | 0,60 | 0,75 |
| Medial Meniscus | 0,60 | 0,80 |
| Lateral Meniscus | 0,60 | 1,00 |
| Lateral OA | 0,60 | 0,75 |
| Fracture | 0,60 | 0,75 |
| Outros sete | 0,60 | 0,60 |

Risco: Lateral Meniscus deixa de receber Transformer/Rad. Pesos publicados
foram explorados no leaderboard público pelo autor; isso não é seleção
independente. Não ajustar nos 58 gold, não consultar desenvolvimento ou
confirmação V01. Depois desta confirmação, sair da exploração de pesos
externos para a próxima família elegível, sem grid no leaderboard.

## Implementação e gates

- `scripts/prepare_h43_probe22.py` aceita somente o build parent confirmado,
  SHA `a103e27720ba52f584c377ac6ffa470aad062c0a4e8558df22da06535a6b0306`.
- Apenas células 3, 4 e 52 mudam: gate/recibo, seletor fixo e mensagem final.
  Todas as células de inferência/fusão são idênticas; não remove membros,
  não muda crops, fatias, TTA, checkpoints ou precisão. Fonte/credits mantidos.
- Gate novo verifica o dicionário RUN inteiro contra a receita publicada;
  preserva validações de schema, IDs, valores, cinco ramos e 56 hashes.
  O gate e o builder parent originais não foram alterados.
- Recibo próprio `h43_probe22_integrity.json`, status esperado
  `PASSED_PROBE22_INTEGRITY`, preset/run e ref parent explícitos.
- `scripts/assess_h43_probe22.py`: valida os recibos, composição e IDs;
  compara parent diagnóstico com CSV da execução anterior e exige paridade
  exata dos sete alvos inalterados. Não calcula AUC nos três exemplos.
- **28 testes passaram**. Cobrem drift de fonte, preservação das demais células, receita
  incorreta, membro ausente, recibo, IDs, NaN e deriva de alvo não alterado.

Build no HD: `reports/avance_av005_build/h43_probe22_strict_v1.ipynb`, SHA
`beb65225a4ed75d1ce2d60f0afdd5ee46c39ed843c151e465ff07e6eb80d572c`.
Manifesto V01 permanece congelado; grandes artefatos ignorados pelo Git.

## Execução e orçamento

[Kernel privado A02](https://www.kaggle.com/code/jvlegend/rsna-knee-h43-probe22-strict-submission),
v1, ID **134536133**, offline, T4x2 exigida, limite 32.400 s. Anexos iguais
ao parent, publicados via API tipada, sem editar configurações locais.
Execução **COMPLETE**, último evento aos **248,28 s**; CoAt 30,33 s,
zero fallback/falhas. `PASSED_PROBE22_INTEGRITY` e comparação pareada local
`PASSED_PAIRED_INTEGRITY`. CSV **3×13**, IDs oficiais, ordem/schema/finitude,
56 hashes/T4x2 e composição dos cinco ramos conferidos.

O parent diagnóstico coincide exatamente com a execução anterior, SHA
`7d3b8bd4e76b309171e2c44a58301e71da26104d1049cc9f8c445c17eee3c770`.
Os sete alvos inalterados coincidem; nos três exemplos, apenas duas linhas
de Medial Meniscus mudam numericamente após o ranking. Não interpretar isso
como ausência de efeito nos demais alvos no oculto.
SHA do CSV probe22:
`ff848c73ba6f31e487d175161304d26df307e6e777189c01dfb508a76532ddfc`.

[Regras oficiais](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation)
revalidadas nesta rodada: Notebook-only, GPU até 9h, internet desligada,
`submission.csv`. Benchmark anterior de 36 estudos: etapas 667,72 s.
Como a inferência permanece igual e o parent completou no oculto, usamos
o mesmo teto, sem alegar nova medição de runtime oculto. Snapshot CLI: 20,31h
GPU disponíveis; quatro submissões disponíveis, uma já usada hoje pela AV-004.

## Retomada

**Enviado à competição:** ref **56263721**, scriptVersionId **350168027**,
em **15/09/2026 22:51:17 UTC / 19:51:17 São Paulo**. Status **PENDING**,
sem score ou erro informado. API confirmou 25 envios históricos, dois hoje
e três restantes. Único envio da AV-005; o outro foi na AV-004.

Consultar a submissão **56263721**, sem duplicar enquanto aguarda avaliação.
Outputs e `paired_integrity.json` em `reports/avance_av005_probe22_v1/`.
Comparar com H43 parent 0,939, sem mudar seleção final automaticamente.
Depois desta confirmação, priorizar outra família (V02/E03 conforme cota),
não outra grade de pesos externos. Nenhuma automação nova foi criada.

```sh
python3 -m scripts.assess_h43_probe22 --directory reports/avance_av005_probe22_v1 --output reports/avance_av005_probe22_v1/paired_integrity.json
```
