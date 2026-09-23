# AV-016 — candidata estável com preset probe22

#RSNA #Kaggle #Pesquisa

16/09/2026. Fonte de verdade: 07_Estrategia_AVANCE do vault,
[espelho operacional](ESTRATEGIA_AVANCE.md). Continua [AV-015](AV015_PARIDADE_COMPLETA_E_SMOKE_OFICIAL.md).

## Hipótese fixada

Verificar no teste oculto se a receita estável/mais rápida preserva o desempenho
do probe22 histórico0,941 (ref56263721). Não é novo treino, novo grid de pesos
ou promessa de aumento de AUC. Diferenças frente ao histórico: soma DINO exata,
ordem native fixa, backend Raptor fixado, prefetch e lotes CoAt ordenados.
Ganhos de velocidade anteriores foram medidos em36estudos, não no teste oculto.

## Implementação

- Builder scripts/prepare_h43_ordered_probe22.py reaudita o smoke real e exige
  a fonte SHA9ca08d0a… aprovada. Somente células3/4/5/61 mudam: gate de receita
  completa, finalidade, seletor probe22 e publicação final com captura da fusão.
  As células de inferência e mistura permanecem idênticas ao smoke aprovado.
- Pesos externos fixos: ACL0,75, Medial Meniscus0,80, Lateral Meniscus1,00,
  Lateral OA0,75, Fracture0,75; sete demais0,60. Sem mudanças internas/TTA.
- Auditor scripts/assess_h43_ordered_probe22.py confere56hashes/T4x2,
  cinco ramos, IDs/schema/finitude, receita/backend, CoAt sem fallback,
  componentes CSV e raw DINO/Raptor/CoAt contra o smoke, DINO/CoAt e mistura
  externa com replay independente. Sete alvos inalterados devem coincidir.
- Extração de validate_components no auditor smoke compartilha o mesmo gate,
  sem retirar verificações do smoke anterior. Build final gera submission.csv;
  o smoke e os benchmarks anteriores continuam não submetíveis.
- CLI/API; sem treinar no Mac, sem editar configurações ou consultar labels
  dev/confirmation. A partição V01 permanece congelada.
- **97 testes passaram**, incluindo fonte/gates inválidos, herança das células,
  gate de publicação, comparação raw com canonização DINO e replay de ranks
  com empates contra pandas. SHA V01 intacto365566b0… .

Build: reports/avance_av016_build/h43_ordered_probe22_v1.ipynb.
SHA256: fcf4ec54f77845837df3898ca29a47b18e0c8043b28c36326587ce61238501f3.

## Execução e regras

[Kernel privado](https://www.kaggle.com/code/jvlegend/rsna-knee-ordered-probe22-submission),
v1 ID134618609 COMPLETE, offline/T4/32.400s. Cota antes17,92378254h;
cinco envios disponíveis antes desta rodada. Submissão realizada após auditoria.
Consulta por slug sem correspondência antes do despacho; resposta sem erros
ou anexos inválidos. Listagem ampla não terminou em10páginas; busca específica
retornou13resultados, nenhum slug exato, segunda página vazia.

[Code Requirements oficiais](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/code-requirements)
revalidados via API list_competition_pages em16/09: notebook, GPU/CPU até9h,
internet desligada, dados externos publicamente acessíveis, submission.csv.
Navegação web retornou páginas vazias; a evidência de conteúdo veio da API.
Parent e probe22 históricos completaram no oculto; não conhecemos seu tempo
real nem o número de estudos ocultos. Projeção local não é garantia de9h.

## Retomada

Auditoria **PASSED_ORDERED_PROBE22_CANDIDATE**,3estudos/15séries,56hashes,
cinco ramos completos. CSVs intermediários e raw DINO/Raptor/CoAt iguais ao
smoke; replay DINO/CoAt e fusão externa aprovados. Zero fallback.

CSV SHA ff848c73ba6f31e487d175161304d26df307e6e777189c01dfb508a76532ddfc,
igual ao probe22 histórico nos exemplos visíveis. Contra o parent0,60, apenas
duas linhas Medial Meniscus mudaram após ranking; sete alvos inalterados exatos.
Isso não prova paridade ou ganho no oculto. Etapas114,5048s: DINO31,9789s,
A59,6561s, Rad9,3230s, Raptor/CoAt/fusão63,5468s. Não extrapolar3casos.
Outputs baixados no HD em reports/avance_av016_candidate_v1/.
Comando executado (não sobrescrever auditoria existente):

```sh
uv run --no-project --with numpy python -m scripts.assess_h43_ordered_probe22 \
  --directory reports/avance_av016_candidate_v1 \
  --build reports/avance_av016_build/h43_ordered_probe22_v1.ipynb \
  --output reports/avance_av016_audit/candidate_v1.json
```

**Submissão56281610**, scriptVersionId350342219, enviada em
**16/09/2026 15:00:59 UTC / 12:00:59 São Paulo**. API confirmou PENDING,
sem score ou erro. Cota antes17,86903667h; histórico antes25envios sem
duplicata desta candidata; depois26total/1hoje/4restantes. Único envio AV-016.
Recibo local: reports/avance_av016_audit/submission_v1.json.

Próximo: consultar esse ID, sem reenviar. Comparar com56263721 (0,941) ao
concluir; não mudar seleção final automaticamente. Enquanto aguarda, seguir
V02/L01 (validação própria), não uma nova grade de pesos no leaderboard.
Nenhuma automação ou serviço pago. Melhor confirmado0,941, sem score
atribuível à nova receita ainda.
