# AV-002 — parent estrito e execução piloto

#RSNA #Kaggle #Tecnologia #Pesquisa

14/09/2026. Continuação de [AV-001](AV001_AUDITORIA_E_VALIDACAO.md) e do
[plano AVANCE](ESTRATEGIA_AVANCE.md). Experiência A01 em execução.

## Resultado observado

- Checagem remota em CPU **COMPLETE**, `PASSED_ARTIFACT_PREFLIGHT`:
  56 arquivos, 4.483.039.152 bytes lidos para hash, 48,39 s na função de auditoria.
  Esse tempo não mede inicialização do worker nem inferência.
- Fonte da checagem: [RSNA Knee H43 Artifact Preflight](https://www.kaggle.com/code/jvlegend/rsna-knee-h43-artifact-preflight),
  versão 1, kernel ID `134328050`, privado, internet desligada, limite 1.200 s.
- Piloto criado e consultado **RUNNING**:
  [RSNA Knee H43 Parent Strict Pilot](https://www.kaggle.com/code/jvlegend/rsna-knee-h43-parent-strict-pilot),
  versão 1, kernel ID `134328295`, privado, T4 solicitada, internet desligada,
  limite **1.800 s**. O código exige duas T4 reais antes do restante.
- **16 testes passaram**: 10 novos de integridade/builder e 6 da partição V01.
- Nenhuma submissão à competição. H-38 segue `0,929`; nenhum score novo.

## O que foi implementado

Código original nosso, no Git:

- [h43_integrity.py](../scripts/h43_integrity.py): preflight de arquivos, hashes,
  manifesto, folds, epochs, identidade de membros, schema, IDs/ordem, cobertura,
  finitude e publicação somente depois do gate final.
- [prepare_h43_parent.py](../scripts/prepare_h43_parent.py): builder ancorado no
  SHA da fonte auditada; recusa versão diferente ou patch ambíguo. Não executa
  fonte pública localmente, não publica kernel e não envia submissão.
- [test_h43_integrity.py](../tests/test_h43_integrity.py) e
  [test_prepare_h43_parent.py](../tests/test_prepare_h43_parent.py).

Mudanças no notebook gerado:

1. Fixa `parent`, peso externo CoAt 0,60 para todos os alvos; preserva pesos
   internos, pooling, ranks e demais receitas. Sem ajuste nos 58 gold.
2. Exige todos os IDs DINO, incluindo votos public-frontier com janelas
   completas. Falhas de worker que reduzam o ensemble impedem publicação.
3. Remove a chamada de benchmark constante e o fallback global que gravava
   0,5 após falha. Falha na promoção public-frontier agora interrompe.
4. Rejeita keys de estado ausentes em A5, exige cinco modelos e predições
   finitas para todos os estudos. Isso pode revelar incompatibilidade que
   antes ficava oculta; não enfraquecer o gate sem auditar as keys.
5. Interrompe em falha de construção/inferência Raptor ou estudo sem slot
   decodificado; não converte ranks não finitos para 0,5.
6. Falha do ramo CoAt é fatal. Exige também que o calibrador Rad tenha sido
   aplicado. Os cinco estágios devem estar registrados no recibo.
7. Intermediários usam `h43_candidate.partial.csv`. Somente o gate final pode
   publicar `submission.csv`; `sample_submission.csv` é preservado. O recibo
   próprio é `h43_parent_integrity.json`, não os recibos legados do autor que
   ainda se intitulam probe22 ou dizem “reproduced”.

Os 56 hashes observados na CPU foram incorporados como lock na candidata GPU.
Para pesos sem hash publicado no manifesto, este lock congela o arquivo
observado; não prova autoria nem equivalência numérica. Fingerprints,
state_dict, contratos de pixels e execução continuam sendo gates separados.
Não declaramos correção de todo fallback por fatia: a receita original ainda
tem tratamento de decode; sua cobertura precisa ser examinada no piloto.

## Calibrador e interpretação

O payload comprimido é JSON de coeficientes, não código: 12 saídas × 88
features, com normalização, interceptos, 12 contagens de protocolo e 4 grupos
de alvos. Atua em 7 alvos. SHA do JSON descomprimido:
`8b616cd8462ef0808702000c4e9d9e29d58be1a9939b5787c246dc7bff26eab2`.
Ele foi inspecionado estruturalmente, sem executar inferência local.
Sua origem de treino/OOF independente não foi estabelecida; tratá-lo como
parte da reprodução pública, não como validação limpa própria.

O manifesto CoAt confirma seleção e4/e6/e8 pelo Gold58. Portanto, não usar
aquele gold para alegar generalização independente deste stack.

## Artefatos e reprodução

Todos os downloads/builds/outputs estão no HD externo e ignorados pelo Git:

- `reports/avance_av002_preflight/h43_preflight.json`, SHA
  `29202e4949db3de2613bfef3ea97128f66c40f4a4d35ccb168502a58fce6032d`.
- `reports/avance_av002_build/h43_parent_strict_locked_v2.ipynb`, SHA
  `a103e27720ba52f584c377ac6ffa470aad062c0a4e8558df22da06535a6b0306`.
  `v2` é a revisão local do build; a versão remota do piloto é **1**.
- Runtime incorporado, SHA
  `76b8d01de1dfaef97cab6af8974c31ebae98af362ab2222092fdd630c4c1510a`.
- Outputs a recuperar em `reports/avance_av002_pilot_v1/`.

```sh
python3 -m unittest tests/test_h43_integrity.py tests/test_prepare_h43_parent.py tests/test_freeze_weak_validation.py -v
python3 scripts/prepare_h43_parent.py --artifact-receipt reports/avance_av002_preflight/h43_preflight.json --output reports/avance_av002_build/h43_parent_strict_locked_v2.ipynb
uvx --from kaggle kaggle kernels status jvlegend/rsna-knee-h43-parent-strict-pilot
```

Cota CLI antes do piloto: 24,38 h de GPU restantes, 30 h totais. Refresh
informado: 19/09/2026 00:00 UTC (18/09 às 21h de São Paulo). Snapshot, não
garantia futura de disponibilidade. Kernels criados pela API com parâmetros
explícitos, sem editar nenhum `kernel-metadata.json` existente.

## Retomada obrigatória

1. Consultar **o piloto v1 existente** e recuperar logs/recibos. Não duplicar.
2. Se ERROR, localizar o primeiro erro e distinguir incompatibilidade de
   checkpoint, ausência de fonte ou bug de integração. Registrar antes de repetir.
3. Se COMPLETE, conferir `h43_parent_integrity.json`, todos os ramos, preflight
   GPU, cobertura e CSV. Um arquivo sozinho não prova conclusão.
4. **Não submeter esta versão de 30 minutos**. O piloto nos três exemplos não
   dimensiona o teste oculto. Ainda é necessário lote representativo, orçamento
   completo e revalidação das regras/cota antes de uma versão de submissão.
5. Se o stack integral se mostrar inviável, seguir A03 Native384Dense com
   receita nativa. A partição V01 está intacta para o treino próprio V02.
