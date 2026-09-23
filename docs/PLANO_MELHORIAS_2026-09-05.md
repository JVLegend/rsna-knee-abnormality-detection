# Plano de melhorias — RSNA Knee Abnormality Detection

#RSNA #Kaggle #Pesquisa #Tecnologia

Documento operacional consolidado a partir da revisão estratégica, do fórum,
das pesquisas em soluções públicas e dos experimentos H-27–H-42. A nota
detalhada no vault é a fonte de decisão; este arquivo acompanha o código e os
artefatos reproduzíveis.

## Estado factual

Atualização de execução: [ESTRATEGIA_AVANCE.md](ESTRATEGIA_AVANCE.md) é a fila
vigente com 27 alternativas. Seus critérios e correções de interpretação
prevalecem sobre a ordem preliminar de experiências abaixo.

- H-38 é a referência do projeto: public score `0,929`; H-36 (`0,928`) é o
  fallback visual mais próximo. No snapshot de `14/09/2026 03:23 UTC`, a
  equipe DataRockstar estava em `1.385/3.706`; o topo marcava `0,956` e
  `0,941` já alcançava aproximadamente a faixa 250–500.
- O H-42 DINOv2 members tem duas famílias públicas CC0, cinco folds cada,
  e o blend externo `20% champ + 80% llm199e30`. O gold local marcou
  `0,995527`, mas isso é diagnóstico contaminado: os pesos foram treinados no
  treino da competição e os 58 estudos gold fazem parte dele.
- Os `557` DICOMs baixados são somente os três estudos visíveis do teste, não
  uma amostra suficiente para inferir o comportamento privado.
- A H-42 foi executada e submetida pelo fluxo Notebook-only. O resultado final
  foi `0,881`, queda de `-0,048` contra H-38; a família foi rejeitada como
  candidata principal e o gold local `0,995527` ficou confirmado como leaky.
- Dados DICOM, CSVs, pesos, cache, relatórios e submissões permanecem fora do
  Git. Licenças devem ser verificadas antes de anexar qualquer fonte a um
  kernel público.

## Correções de interpretação

- Igualdade entre CSVs em três linhas não prova igualdade no teste oculto.
- Ranks em três estudos têm pouca resolução e não servem para calibrar blends.
- AUC do gold deve ser tratada como diagnóstico; promoção exige OOF limpo,
  estabilidade entre folds e, quando houver cota, confirmação Kaggle.
- Hash de laudo é um controle de duplicação, não uma prova de identidade de
  paciente. Laterality não é confiável nos headers auditados; flip medial /
  lateral permanece bloqueado.

## Implementação iniciada nesta rodada

| Frente | Implementação | Estado / gate |
|---|---|---|
| Memória de inferência | H-42 agora decodifica cada estudo uma vez em cache `.npz` versionado e faz forward somente em batches | Implementado; batch `2` reproduziu byte a byte o CSV anterior nos três estudos |
| Diagnóstico DICOM | Cada slot registra ausência de série, poucos cortes, arquivo ausente, shape/render ou exceção DICOM; estudos sem slot falham explicitamente | Implementado; cobertura visível `2/3/3/2/1/1` |
| Invariância a batch | `RSNA_BATCH_SIZE` configurável; rank só após concatenar todos os estudos | Implementado e validado com batch `1` e streaming batch `2` |
| Alinhamento | Comparador DINOv2 exige `study_ids`, rejeita duplicatas/conjuntos divergentes e reindexa por UID | Implementado; relatórios antigos só entram com `--allow-legacy-report-order`, marcado no JSON |
| OOF | Manifesto 5-fold por grupos de laudo e prevalência por alvo | Implementado em `reports/grouped_oof_manifest_gold_20260905.json`; zero overlap, 58 estudos |
| Auditoria histórica | Comparação H-42 reexecutada com o guard de UID e exceção legada explícita | Implementado; resultado continua diagnóstico, não promoção |

## Backlog de modelagem e como testar

### 1. OOF e contaminação — prioridade máxima

- [x] Congelar splits agrupados e sementes.
- [ ] Reconstituir, para cada família, a origem dos pesos e os estudos usados
  em treino.
- [ ] Treinar heads/encoders próprios por fold, excluindo o fold de validação
  de teachers visuais e textuais.
- [ ] Salvar `study_ids`, previsões brutas por fold, máscara de cobertura,
  protocolo, scanner quando disponível e tempo/RSS.
- [ ] Escolher blends somente nas previsões OOF; repetir em pelo menos duas
  sementes antes de promover.

Critério: nenhuma célula de validação foi vista pelos pesos, teachers,
calibradores ou escolha de hiperparâmetros do próprio fold.

### 2. Supervisão fraca mascarada

- [ ] Converter laudos em estados `positivo`, `negativo`, `incerto` e `não
  mencionado`, mantendo a evidência textual.
- [ ] Começar com BCE mascarada; ausência de menção nunca vira negativo.
- [ ] Peso gold maior e limitado; testar a exceção de Synovitis somente dentro
  do OOF.
- [ ] Registrar perda e cobertura por estudo/alvo/fonte, bloqueando hashes
  compartilhados com o fold de validação.

### 3. Visual 2.5D com contexto e detalhe

- [ ] Comparar, mantendo o mesmo split: média atual vs atenção por alvo sobre
  grupos; janela central vs cobertura ampla; global vs global+local.
- [ ] Preservar posição física e máscara de slot; não trocar lateralidade sem
  metadado confiável.
- [ ] Testar crop físico `130–140 mm` em resolução maior apenas como ablação
  declarada, sem confundir interpolação com informação nova.

### 4. Robustez e custo

- [x] Streaming/cache para não manter `(N,6,3,3,224,224)` inteiro em RAM.
- [ ] Teste de carga com estudos locais: pico de RSS, VRAM, tempo por estudo e
  projeção com margem para o limite oficial.
- [ ] Testar slots ausentes, DICOM corrompido, batch final incompleto e ordem
  reembaralhada.
- [ ] Separar módulos de decoder, modelo, inferência e fusão; depois gerar o
  entrypoint Kaggle standalone a partir deles.

### 5. Ensemble e destilação

- [ ] Medir complementaridade por erro em OOF, não por score de três linhas ou
  quantidade de folds.
- [ ] Comparar logits, probabilidades e rank como hipóteses independentes;
  começar com dois ou três braços e pesos globais regularizados.
- [ ] Destilar o ensemble aprovado em um aluno eficiente e comparar AUC/custo,
  considerando o prêmio de eficiência.

## Hipóteses a priorizar

| Hipótese | Experimento mínimo | Não fazer |
|---|---|---|
| Atenção por alvo evita diluir lesões focais | trocar somente pooling de grupos no mesmo OOF | substituir os pesos H-42 sem treino compatível |
| Global + local ajuda menisco/ligamento sem perder derrame/Baker | dois ramos com ablação de resolução e crop físico | chamar resize de nova informação |
| Labels weak com incerteza ajudam | BCE mascarada com peso por fonte e hash blocking | tratar silêncio textual como normal |
| Blend melhora generalização | rank/probabilidade em previsões OOF cross-fitted | calibrar nos 58 gold completos |
| Destilação reduz custo | professor congelado sem exposição ao fold e aluno menor | otimizar eficiência sem medir AUC e runtime juntos |

## Gate de submissão

1. `submission.csv` tem exatamente `StudyInstanceUID + 12 targets`, IDs únicos,
   finitos e no intervalo `[0,1]`.
2. O notebook roda sem internet, com GPU T4 quando usar CUDA, dentro do limite
   medido e sem depender de outputs privados não montados.
3. Pesos e datasets anexados têm licença compatível com a competição e o
   projeto.
4. A variante supera ou complementa H-38 em OOF independente; um ganho local
   no gold ou em três linhas públicas não basta.
5. Código, fingerprint dos pesos, SHA-256 do CSV e decisão ficam registrados
   no log e no vault antes do envio.

## Próximas execuções

1. Recuperar/produzir previsões OOF para uma família visual própria usando o
   manifesto já congelado.
2. Medir as ablações de pooling e global+local em um único fold barato antes
   de treinar cinco folds completos.
3. Só após um ganho consistente, consumir nova cota T4 para uma submissão
   notebook-only e comparar com H-38.

Artefatos centrais:

- `scripts/build_grouped_oof_manifest.py`
- `scripts/compare_dinov2_members_gold.py`
- `kaggle/rsna_knee_dinov2_members_exp056/rsna_knee_dinov2_members_exp056.py`
- `reports/grouped_oof_manifest_gold_20260905.json` (ignorado pelo Git)
- [[06_Revisao_Estrategica_2026-09-05]] no vault

## Atualização operacional — 13/09/2026

- A autenticação Kaggle voltou a funcionar: a competição está aceita pelo
  usuário, com prazo informado pela API para `22/10/2026` e execução T4
  disponível.
- A primeira publicação do H-42 (`jvlegend/rsna-knee-dinov2-members-exp056`,
  versão 1) falhou antes da inferência porque a montagem dos datasets anexados
  tinha mais níveis de diretório do que o buscador de checkpoints percorria.
  Isso foi classificado como falha de integração, não como falha de modelo.
- A versão 2 corrigiu a descoberta limitada por slug, sem fazer uma busca
  recursiva na árvore DICOM. O kernel terminou `COMPLETE` no T4, carregou os
  dez checkpoints públicos CC0 e gerou `submission.csv` em `48,7 s`, com
  `3×13`, IDs únicos, valores finitos em `[0,1]` e cobertura de slots
  `2/3/3/2/1/1`.
- A saída Kaggle e a reprodução local são byte-idênticas:
  `SHA-256=2b9a159784efeeb45d50e209b5bdbc58317fa9ec512e7d2dad0974338ba77e22`.
  O artefato foi salvo em
  `submissions/submission_h42_dinov2_members_exp056_kaggle_v2.csv`, com o
  diagnóstico em `reports/h42_kaggle_v2_diagnostics.json`.
- O primeiro baseline OOF efetivamente cross-fitted usando o manifesto agrupado
  foi executado com `C=32`, lexicon e seed `2026`: macro-AUC `0,626918` em 58
  estudos, sem usar relatórios não rotulados para construir o vocabulário.
  Esse resultado é o gate de texto/metadados; ainda não é OOF visual dos
  checkpoints públicos.

Decisão: H-38 (`0,929`) continua sendo o baseline oficial. A submissão
Notebook-only da H-42, criada em `2026-09-14 00:49:02 UTC`, terminou
`COMPLETE` com public score `0,881`; não promover nem ajustar pesos dessa
família usando o gold leaky ou as três linhas visíveis.

## Pesquisa de fórum e notebooks — 14/09/2026

### Fatos novos que mudam a ordem do backlog

1. **O salto público reproduzível mais próximo é `0,941`.** Os notebooks
   [0.941 reestruturado](https://www.kaggle.com/code/maverickss26/rsna-knee-0941-restructured),
   [source checklist](https://www.kaggle.com/code/starkhushi/rsna-knee-0-940-source-checklist-4-diffs)
   e [fast 2xT4](https://www.kaggle.com/code/jiweiliu/rsna-knee-fast-2xt4-inference)
   descrevem o mesmo stack de DINOv2/v3, A5, RadImageNet e quatro braços
   CoAtNet. O `probe22` usa pesos CoAtNet por alvo e marcou `0,941`; o parent
   com peso global `0,60` marcou `0,939`. Os pesos por alvo foram ajustados no
   leaderboard público, logo o parent/halfway é a opção mais conservadora para
   uma segunda submissão final.
2. **Cobertura de fatias é efeito real, mas o H-38 já incorpora o principal
   ganho publicado.** No tópico
   [corpus geometry](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737696),
   os mesmos pesos passaram de `0,924` com 42 janelas para `0,927` com 62.
   H-36/H-38 já usam 64 fatias, span `0,02–0,98`, crop `140 mm` e 62 janelas;
   não há ganho novo em simplesmente repetir essa troca. O próximo teste é
   80–96 fatias ou 384 px, alterando uma variável por vez.
3. **Adjacência vale mais que espalhar cortes.** Um teste controlado reportou
   `+0,018` no fold 0 para nove cortes adjacentes centrais contra nove cortes
   equiespaçados; outro notebook reporta `+0,0060` em 10/10 seeds ao passar o
   cache de 16 para 32 fatias, evitando triplets 2.5D espacialmente
   subamostrados. Isso atinge diretamente H-42, que usa uma banda central
   estreita; exige treinar pesos compatíveis, não apenas mudar inferência.
4. **Os 58 gold não podem selecionar microganhos.** O notebook
   [58-study validation set](https://www.kaggle.com/code/starkhushi/58-study-validation-set-is-lying-to-you)
   mostra inversão entre a ordem dos modelos no gold e no leaderboard e propõe
   um holdout derivado de 250 estudos, agrupando 46 conjuntos de laudos
   byte-idênticos. Nosso protocolo usa hashes normalizados e encontrou 54
   grupos/204 linhas; manter grupos inteiros e reservar os 58 como auditoria,
   não como seletor de checkpoint.
5. **Modelo maior não resolve o gargalo atual.** DINOv2-Base contra Small, no
   mesmo OOF de 2.652 estudos, mudou `0,7931→0,7942`, abaixo do noise floor
   `0,0020`, com aproximadamente 4× o custo. Base convergiu mais rápido em
   treino curto, mas Small chegou ao mesmo endpoint. Antes de trocar encoder,
   conferir o `preprocessor_config.json`: normalização errada no RAD-DINO
   produziu uma curva plausível e falsa.
6. **Labels continuam sendo o maior espaço de ganho próprio.** Evidências
   novas: negação turca pode vir depois do achado; intensidade leve pode ser
   anotada como negativa; osteoartrite costuma aparecer como consequência
   (`osteófito`, estreitamento, perda/defeito condral, condrose), não como
   `OA`. O notebook
   [Osteoarthritis is almost never written as OA](https://www.kaggle.com/code/busyaprime/osteoarthritis-is-almost-never-written-as-oa)
   não reivindica score de leaderboard, mas oferece um léxico auditável e
   `silver_labels.csv` com abstention. Ausência de menção continua proibida de
   virar negativo.
7. **Especialistas só devem entrar onde acrescentam.** O residual público de
   [Medial Meniscus](https://www.kaggle.com/code/renta0426/rsna-knee-0-937-weak-label-dinov2-meniscus-resid)
   altera apenas esse alvo (`0,30` transformer + `0,60` Raptor + `0,10`
   especialista) e preserva os outros 11. Isso é mais defensável que misturar
   um modelo fraco em todas as colunas. A evidência de quatro braços também
   mostra que um componente com correlação de ranking próxima a `0,83` pode
   adicionar valor; medir correlação e delta OOF por alvo é obrigatório.
8. **SWA e blends fracos não merecem novas tentativas.** Seis pares SWA/não-SWA
   tiveram efeito médio nulo e um par público marcou `0,937` contra `0,938`.
   O histórico de 22 submissões em
   [6 lessons](https://www.kaggle.com/code/yosukeinada/rsna-knee-0-937-22-submissions-6-lessons)
   mostra que braços `0,874–0,910` não melhoraram um parent `0,936`, mesmo
   quando pareciam diversos.

### Auditoria de licenças das fontes do stack público

Consulta feita com `kaggle datasets metadata`; licença declarada pelo autor,
não parecer jurídico:

| Grupo | Licença declarada | Decisão |
|---|---|---|
| Raptor MaxSpan/Native384/Native384Dense, folds Mattia, CoAt residual, Pilkwang labels/weights | `CC0-1.0` | compatível com nossa linha equity-free |
| Especialista de menisco Renta | `Apache-2.0` | compatível, mantendo avisos |
| Band32 adjacency leg | `CC0-1.0` | compatível como artefato experimental |
| Encoder/heads RadImageNet de Marwan e Antoine | `CC-BY-NC-SA-4.0` | pode servir à competição, mas **não** é base segura para produto comercial no Brasil |
| Heads Prvsiyan e bundle Tonylica | `other` | bloquear na linha equity-free até leitura da licença completa |

### Fila de experimentos revisada

| Ordem | Candidata | Comparação controlada | Gate |
|---:|---|---|---|
| 1 | `H-43A` — reprodução exata do parent público `0,939/0,941` | `probe22`, `halfway` e parent, sem braço próprio | contar todos os membros, hashes/fingerprints, zero `dropped`, T4×2 e runtime < 9 h |
| 2 | `H-43B` — âncora H-38 + braço público realmente decorrelacionado | peso global pequeno em rank; nunca 12 pesos escolhidos no gold | correlação por alvo + delta em OOF de produção; preservar H-38 se faltar fonte |
| 3 | `H-44` — professor textual v2 | léxico atual vs negação bilateral por idioma + severidade + vocabulário de consequências OA + abstention | holdout derivado ≥250, grupos de laudo inteiros, gold só como auditoria |
| 4 | `H-45` — geometria 2.5D própria | 16 vs 32 fatias; equiespaçado vs adjacente; depois 64/80–96 | mesmos folds/seeds/backbone; ganho > noise floor em ≥2 seeds |
| 5 | `H-46` — especialista de estrutura fina | ROI/atenção espacial para menisco/MCL, alterando somente alvos aprovados | especialista melhora o alvo e o macro OOF sem degradar os outros 11 |

### Decisão imediata

- Preparar `H-43A` como controle de reprodução é a ação com maior chance de
  salto rápido de `0,929` para a faixa pública `0,939–0,941`.
- Não substituir H-38 como escolha privada apenas por esse score: `probe22`
  tem tuning target-wise no public LB. Manter H-38 e, se executado, guardar o
  parent/halfway como alternativas finais.
- A linha própria deve concentrar compute em labels, validação agrupada e
  geometria/adjacência. Backbones maiores, SWA e grids de blend no gold saem
  do backlog ativo.
