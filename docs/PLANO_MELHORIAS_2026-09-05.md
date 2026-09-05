# Plano de melhorias — RSNA Knee Abnormality Detection

#RSNA #Kaggle #Pesquisa #Tecnologia

Documento operacional consolidado a partir da revisão estratégica, do fórum,
das pesquisas em soluções públicas e dos experimentos H-27–H-42. A nota
detalhada no vault é a fonte de decisão; este arquivo acompanha o código e os
artefatos reproduzíveis.

## Estado factual

- H-38 é a referência do projeto: public score `0,929`; H-36 (`0,928`) é o
  fallback visual mais próximo.
- O H-42 DINOv2 members tem duas famílias públicas CC0, cinco folds cada,
  e o blend externo `20% champ + 80% llm199e30`. O gold local marcou
  `0,995527`, mas isso é diagnóstico contaminado: os pesos foram treinados no
  treino da competição e os 58 estudos gold fazem parte dele.
- Os `557` DICOMs baixados são somente os três estudos visíveis do teste, não
  uma amostra suficiente para inferir o comportamento privado.
- A cota semanal de GPU Kaggle estava esgotada no último push do H-42; não há
  novo score remoto desta rodada.
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
