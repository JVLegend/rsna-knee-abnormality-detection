---
titulo: Estratégia, evidências e hipóteses — RSNA Knee Abnormality Detection
projeto: RSNA Knee Abnormality Detection
updated: 2026-08-18
status: living-document
tags: [Kaggle, RSNA, Medicina, Tecnologia]
---

# Estratégia, evidências e hipóteses

Documento vivo para transformar a pesquisa no Kaggle em experimentos controlados. A ideia é registrar o que foi observado, o que é apenas uma afirmação de notebook, o que vamos testar e qual evidência fará uma hipótese avançar ou ser descartada.

> [!warning]
> Os notebooks e datasets públicos do Kaggle podem mudar, depender de bundles anexados ou conter resultados não reproduzidos por nós. Um score citado abaixo deve ser tratado como evidência de direção, não como garantia de que o mesmo resultado aparecerá em uma execução nova.

## Como usar este arquivo

Para cada mudança relevante:

1. Criar um ID de experimento (`E-###`) e, quando aplicável, uma hipótese (`H-##`).
2. Alterar uma família principal por vez: rótulo, representação, geometria, pooling ou ensemble.
3. Registrar o commit, o dataset/artefato usado, o seed, o fold, a métrica local e a submissão.
4. Guardar sempre uma submissão de fallback e o CSV que a gerou.
5. Não considerar uma diferença menor que `0,02` conclusiva quando ela vier de apenas 58 estudos rotulados ou de uma única submissão.

### Template de novo experimento

```text
E-### — nome curto
Data:
Hipótese relacionada:
Família: labels | representação | geometria | pooling | ensemble | dados externos | engenharia
Baseline comparável:
Uma única mudança:
Artefatos e fingerprint:
Métrica local: macro AUC, AUC por alvo, cobertura, calibração
Resultado Kaggle: submission_id / public score / status
Decisão: manter | descartar | combinar | repetir
Aprendizado:
```

## Snapshot do ponto de partida

- Melhor submissão confirmada até este registro: H-23, ref `55582655`, public score `0,718`; ganho de `+0,006` sobre H-22 (`55551332`, `0,712`). H-23 é a nova referência; manter H-22 como fallback reproduzível.
- H-26 está em avaliação no Kaggle: mesma H-23, trocando somente o pooling de probabilidades para `mean` nos 12 alvos. O gate local dense6/janela por série marcou `0,643152` contra `0,635104` da política top-k/mean atual. A v1 caiu por alocação automática de P100 `sm_60`; a v2 foi relançada fixando `NvidiaTeslaT4` e está aguardando saída/submissão. Não há score H-26 ainda.
- DINOv2-S oficial MetaResearch foi baixado para auditoria no HD com licença Apache 2.0. No holdout dos 58, embedding médio marcou `0,568709` e MIL top-k `0,584075`, abaixo do B0 congelado (`0,636834`); não consumir T4 com a versão congelada.
- Auditoria H-24: o mapa target-wise atual marcou macro-AUC `0,899120` nos 58 estudos; o melhor blend simples testado (`Steven v4` mascarado + `Pilkwang` + `Lixin`) marcou `0,895808` (`-0,003311`). O relatório reprodutível está em `reports/teacher_blend_audit_20260816.json` e o código em `scripts/audit_teacher_blends.py`.
- Submissões anteriores registradas: aproximadamente `0,505`, `0,607`, `0,582`, `0,605` e `0,635`; a v10 melhorou `+0,020` sobre a referência multi-view.
- Treino: `4.407` estudos; somente `58` possuem os 12 rótulos oficiais.
- Alvos: ACL, MCL, meniscos medial/lateral, OA medial/lateral/patelar, derrame, sinovite, cisto de Baker, contusão e fratura.
- Dados locais no HD: `80.278` DICOMs, aproximadamente `50 GB`, cobrindo 758 estudos e 2.159 séries; `test_series/` continua ausente localmente. O CV visual local é útil para ablações de treino, mas não representa sozinho o exame de teste.
- Baseline atual: uma série por plano, três fatias quantílicas, média das representações, EfficientNet-B0 congelada e peso visual global. Isso é um baseline útil, mas não é próximo do padrão dos notebooks públicos mais fortes.
- A auditoria local de labels encontrou macro-AUC nos 58 de `0,892707` para Steven v4, `0,887349` para Steven v2, `0,870040` para Pilkwang, `0,835194` para Lixin e `0,892826` para consenso por ranking. Os quatro datasets baixados pelo Kaggle CLI declararam `CC0-1.0`.
- Os CSVs estão no HD, fora do Git: `/Volumes/Karine HD Externo/Dados_JV/Datasets/rsna-knee-abnormality-detection/labels/`. A variante `kaggle/rsna_knee_v3_external_labels.py` neutraliza o `0,5` do Steven v2 antes de usar o v4 e aplica confiança mínima `0,85`.
- O bundle `ericwang03/rsna-knee-dinov2-mil-bundle` foi baixado para `/Volumes/Karine HD Externo/Dados_JV/Datasets/rsna-knee-abnormality-detection/bundles/ericwang-dinov2-mil`. O backbone local tem `88.283.115` bytes, SHA-256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`, 175 tensores e embedding 384. O Kaggle CLI reportou licença `other`; o bundle fica em auditoria e não deve ser usado no leaderboard até a licença dos heads e dos pesos ser esclarecida.
- O `predict.py` original tentava carregar o backbone via URL. `kaggle/rsna_knee_dinov2_offline.py` corrige o caminho com `weights=<arquivo local>` e uma trava `RSNA_DINOV2_LICENSE_ACK=1`. A construção do modelo e uma inferência `224×224 → (1,384)` passaram localmente; o smoke DICOM completo não foi possível porque o checkout incremental mantém CSVs, não `test_series/` bruto.
- A v4 agressiva está em `kaggle/rsna_knee_v4_dense_target_pool.py`: `dense6/dense9` são janelas adjacentes de três canais, `view_pooling=target` treina em views repetidas e agrega `top-k`/`mean` por alvo, e `teacher_profile=targetwise` usa a fonte mais forte por alvo com cobertura controlada. A v6 concluiu no T4 com submissão `55413852` e público `0,706`; H-25 marcou `0,708`; H-22 elevou a referência para `55551332` e público `0,712`; mudanças novas precisam superar esse fallback.
- Limitação de inferência: os relatórios existem no treino, mas não no teste. O relatório deve servir para gerar supervisão auxiliar; o modelo final precisa inferir somente de imagem e metadados disponíveis no teste.
- Runtime alternativo confirmado: o Mac local tem PyTorch `2.11.0` com MPS funcional. H-22 processou um estudo real com `dense6`/6 views em `2,96 s`, gerando embedding `(6, 1280)` finito. O HD, porém, não contém DICOM de `test_series`; esse ambiente é adequado para smoke/CV e não para uma submissão oficial.
- O código aceita `--device mps` e `auto` seleciona CUDA, MPS ou CPU nessa ordem. Não há host DGX/túnel SSH configurado nesta máquina; Docker/Colima não fornece uma GPU alternativa no Mac.
- Ponto crítico do fórum: o host informou que os rótulos oficiais são derivados das imagens e que os relatórios podem ser ambíguos ou inconsistentes. O texto não é ground truth.

### Insight novo: a melhor fonte varia por alvo

O audit dos 58 estudos mostra que “Steven v4 é melhor” é uma conclusão média,
não uma regra para cada saída. Em AUC bruto por alvo, os líderes foram:

| Alvo | Fonte líder no audit | AUC | Observação de cobertura |
|---|---|---:|---|
| ACL | Pilkwang | `0,997` | cobertura `1,00` |
| MCL | Pilkwang | `0,976` | cobertura `1,00` |
| Medial Meniscus | Steven v2 | `0,954` | cobertura `0,95` |
| Lateral Meniscus | Steven v4 | `0,879` | cobertura `1,00` |
| Medial OA | Steven v4 | `0,932` | cobertura `1,00` |
| Lateral OA | Steven v4 | `0,833` | cobertura `1,00` |
| PF OA | Steven v4 | `0,902` | cobertura `1,00` |
| Effusion | Steven v4 | `0,877` | cobertura `1,00` |
| Synovitis | Steven v2/v4 | `0,790` | empate; v4 tem cobertura maior |
| Baker's | Steven v2 | `0,946` | cobertura somente `0,43`; v4 `0,944` com `1,00` |
| Contusion | Steven v4 | `0,860` | cobertura `1,00` |
| Fracture | Pilkwang | `0,871` | cobertura `1,00` |

Isso sugere uma ablação target-wise, especialmente para ACL/MCL/Fracture, mas
selecionar a fonte diretamente nos mesmos 58 é suscetível a overfit. A
variante atual usa Steven v4 em todos os alvos por ser a política simples e
de cobertura completa. Primeiro vamos medir seu efeito no visual; depois
testaremos um mapa por alvo com validação leave-one-source-out e pesos suaves.

### Conclusão operacional

O gargalo provável não é ajustar o `visual_weight` do baseline. A maior oportunidade é trocar a representação: rótulos fracos melhores, DICOM em ordem física, seis slots de aquisição, grupos de fatias adjacentes, DINOv2/MIL ou backbone parcialmente ajustado, e ensemble por ranking. A primeira meta é reproduzir uma melhoria robusta local; a segunda é aproximar a faixa pública alegada de `0,89` sem transformar um único leaderboard em critério de verdade.

## Evidência nova — ordenação anatômica e ensemble — 15/08/2026

O teste controlado no gold oficial comparou a mesma EfficientNet-B0 e os mesmos estudos, mudando somente a ordem dos cortes 2.5D. A ordenação por `InstanceNumber`/header marcou macro-AUC médio `0,578718` em duas seeds, contra `0,543691` da ordem lexicográfica. O ganho não foi uniforme por alvo e o gold tem cobertura insuficiente para validar três planos.

Ao escalar a hipótese para os 700 estudos weak (2.100 séries), o holdout independente dos 58 estudos marcou `0,585771` com mean pooling e `0,591101` agregando as previsões por série. Uma mistura fixa 50/50 dos ramos filename + header chegou a `0,614597` com mean pooling e `0,607885` por série. Esses valores são evidência de complementaridade, não calibração de leaderboard: o ramo filename pode estar oferecendo views diferentes, não uma ordem anatômica melhor.

Decisão operacional: preservar `55551332` (`0,712`) como fallback e preparar uma única variante Kaggle de uma nova família. A fusão contínua foi auditada, mas não passou o gate local; a próxima rodada deve testar fine-tuning leve, sem selecionar pesos diretamente nos 58 estudos.

### Auditoria H-24 — fusão contínua de teachers — 16/08/2026

Para evitar uma submissão às cegas, foi criado `scripts/audit_teacher_blends.py`. O
teste combina, por alvo, os scores públicos de Steven v2, Steven v4 mascarado
por estados não abordados do v2, Pilkwang e Lixin. Foram comparados média e
rank-mean, sempre no mesmo holdout de 58 estudos e contra o mapa target-wise
usado pelo H-22.

| Candidato | Macro-AUC | Delta vs target-wise |
|---|---:|---:|
| `targetwise_current` | `0,899120` | `0,000000` |
| `mean_v4_masked_pilkwang_lixin` | `0,895808` | `-0,003311` |
| `mean_v4_masked_v2_pilkwang_lixin` | `0,893615` | `-0,005504` |
| `rankmean_v4_masked_v2_pilkwang_lixin` | `0,892811` | `-0,006309` |
| `rankmean_v4_masked_v2_pilkwang` | `0,892092` | `-0,007028` |
| `mean_v4_masked_v2_pilkwang` | `0,891664` | `-0,007456` |

O melhor blend simples ficou abaixo do fallback target-wise. No bootstrap
pareado de 1.000 reamostragens, o delta médio do blend de três fontes contra o
target-wise foi `-0,007790`, com intervalo `[-0,018389, +0,002357]`; o intervalo
amplo confirma a instabilidade dos 58, não uma vitória oculta. Portanto H-24 é
**refutada para promoção/submissão nesta rodada**. O resultado não prova que
nenhum ensemble possa funcionar, mas não justifica trocar a política atual.

## O que a pesquisa do Kaggle mostrou

### Padrões que aparecem repetidamente

1. **Weak labels multilíngues, graduais e com estado de afirmação.** Os melhores notebooks não usam simplesmente “apareceu a palavra = 1”. Eles separam positivo, negativo, incerto e não abordado; usam contexto anatômico, termos de grau, seções do laudo, sinônimos em vários idiomas e, em alguns casos, um LLM para arbitrar os casos difíceis.
2. **Texto somente no treino.** Como o teste não tem `Report`, o texto é destilado em pseudo-rótulos, pesos e prioridades de amostragem; a entrada do modelo final continua sendo a imagem.
3. **DINOv2-small ViT-S/14.** O backbone mais recorrente nas soluções públicas recentes é DINOv2 ViT-S/14, geralmente com 384 dimensões, cabeça MIL/attention e algum fine-tuning dos últimos blocos.
4. **Seis slots clínicos de MRI.** A organização mais comum é `SAG_FLUID_FS`, `COR_FLUID_FS`, `AX_FLUID_FS`, `SAG_FLUID_NOFS`, `COR_T1` e `SAG_T1`, com máscara de presença. Os nomes exatos devem ser confirmados pelos headers DICOM, não apenas pelas flags do CSV.
5. **Ordem física dos cortes.** `IOP`/`IPP` e geometria do paciente são preferidos a filename. `SliceLocation` e `InstanceNumber` entram como fallback auditado.
6. **2.5D e sequência.** Três imagens adjacentes em um input, várias janelas ao longo da série e pooling ao nível do estudo são um compromisso prático entre 2D e 3D.
7. **Resolução física e crop.** Menisco e pequenas lesões podem desaparecer em 126/224 px. A hipótese pública mais promissora usa campo físico de cerca de 150 mm e 336 px; esse ganho precisa ser reproduzido sob o nosso orçamento.
8. **Pooling por alvo.** Média é razoável para achados difusos; máximo/top-k/attention é mais plausível para fratura, contusão, ruptura focal e cisto. O notebook público reporta ganho especial no alvo Fracture com max pooling.
9. **Laterality e orientação.** Correção por geometria do paciente, inversão horizontal quando necessário e nenhuma inversão vertical indiscriminada. Flips podem trocar o significado de alvos laterais.
10. **CV agrupado e ensemble por ranking.** Duplicatas de relatório devem ficar no mesmo fold. Diferenças pequenas de score são instáveis; rank averaging, folds balanceados, pesos encolhidos e TTA parecem mais seguros que uma grade grande de pesos ajustada em um único split.
11. **Pré-processamento cacheado.** Arrays `.npz` ou pacotes MIL podem acelerar iterações, mas só devem entrar no pipeline depois de conferir licença, regras de dados externos, fingerprint do pré-processamento e espaço no HD.
12. **Atalhos de scanner/site.** O fórum mostrou diferença grande entre folds aleatórios e folds agrupados por scanner/site. Metadata pode subir o CV e falhar no teste; deve ser tratado como auditoria e ramo auxiliar de baixo peso, não como solução principal.

## Inventário de notebooks e datasets encontrados

“Evidência” indica o quanto podemos confiar na fonte para decidir o próximo teste: **alta** = código/artefato público ou padrão repetido; **média** = experimento bem descrito, mas score/execução ainda não reproduzido; **baixa** = título, descrição ou claim sem validação independente.

### Notebooks e análises

| Fonte | O que foi usado | Insight acionável | Evidência |
|---|---|---|---|
| [Pilkwang — RSNA Knee Baseline V1](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1) | Rótulos multilíngues por regras, seções/cláusulas, assertion, seis slots, DICOM físico, DINOv2, attention/MIL, rank ensemble | É o melhor blueprint de ponta a ponta; inclui auditoria de cobertura/silêncio, fingerprint e fallback de submissão | Alta para implementação; score a reproduzir |
| [Prvsiyan — Read the report then the knee](https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee) | Labeler multilíngue graduado, validação nos 58 dourados, EfficientNet-B3, seis fatias de treino, fine-tuning parcial, rank blend | Reportar rótulos para treino e imagem para inferência; combinar fontes de labels por alvo; não tratar `0,5` como negativo | Média; score `0,891–0,894` é claim do notebook |
| [Karnak — EDA to 2.5D](https://www.kaggle.com/code/karnakbaevarthur/rsna-knee-eda-to-2-5d) | Auditoria dos 58 estudos, teacher de relatório, TF-IDF, 2.5D, TTA horizontal, ramo de metadata | Bom caminho de baixo custo para ablação de labels e 2.5D antes de DINOv2 | Média |
| [W. Guesdon — DINOv2 at meniscus resolution](https://www.kaggle.com/code/wguesdon/rsna-knee-dinov2-at-meniscus-resolution) | DINOv2 ViT-S/14, 336 px, FOV físico de 150 mm, tokens CLS/patch mean/patch max, probe de lesão | Reproduzir primeiro um probe 224 vs 336; o argumento anatômico para menisco é forte, mas o ganho não é fato estabelecido | Média |
| [Roman Tamrazov — DINOsaur V2](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v2) | Slots por contraste, ordem física, correção de laterality, qualidade da série, calibrador de labels, pooling por alvo, TTA gamma, soups | Reúne as melhores práticas; útil como checklist de reprodução e não como pacote para copiar sem auditoria | Média/alta |
| [Hida — Public 4-fold DINOv2 V4](https://www.kaggle.com/code/hida1211/rsna-knee-public-4-fold-dinov2-v4) | Ensemble cross-validated, no vertical flip, EMA, pairwise ranking pequeno, fallback de laterality | Testar folds/seeds e política de augment antes de aumentar arquitetura | Média |
| [Tonylica — RSNA Knee Infer](https://www.kaggle.com/code/tonylica/rsna-knee-infer) | Bundle DINOv2, 3 fatias adjacentes, attention por diagnóstico, prior de slots, top-k para achados focais, rank average | Top-k/max deve ser target-specific; manter caminho de inferência rápido e verificável | Média |
| [Sakhawat — Enhanced Ensemble](https://www.kaggle.com/code/sakhawathossen/rsna-knee-enhanced-ensemble) | Rank baseline igualitário, pesos de qualidade encolhidos, TTA windows e AUC OOF por alvo | Comparar rank-mean puro contra qualquer peso aprendido; nunca perder o baseline | Média |
| [Roman Rozen — Data Structure EDA Baseline](https://www.kaggle.com/code/romanrozen/rsna-knee-data-structure-eda-baseline) | EDA, 58/4407, duplicatas de relatório, flags idênticas, slots fixos | Serve como checklist para evitar leakage e interpretar corretamente o CSV | Média |
| [Ryan Holbrook — Efficiency LB](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb) | Leitura de leaderboard publicado | Não é modelo; útil apenas para medir custo/ganho da rotina de submissões | Baixa |

### Datasets e bundles de labels/modelos

| Fonte | Conteúdo | Como pode ajudar | Risco/decisão |
|---|---|---|---|
| [Steven Lee Hans — LLM report labels](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels) | CSVs para 4.407 estudos; v1/v2/v4 blend; scores, veredictos e confiança | Principal candidato para comparar contra nosso léxico. O claim da descrição dá macro AUC `0,8927` para v4 | Licença não clara na API; confirmar antes de uso final |
| [Pilkwang — LLM labels](https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels) | 12 scores, confiança e veredito YES/NO/UNK por relatório original | Fonte independente para consenso e ablação de pseudo-rótulos | `licenseName` não informado; verificar |
| [Lixin73 — LLM labels SOL56](https://www.kaggle.com/datasets/lixin73/rsna-knee-llm-report-labels-sol56) | Rótulos gerados por uma segunda fonte de LLM | Útil para consenso/discordância, não para escolher uma fonte por autoridade | Descrição/licença ausente; verificar |
| [Barun — stratified folds and soft labels](https://www.kaggle.com/datasets/barun2104/rsna-knee-stratified-folds-and-llm-soft-labels) | Folds estratificados, flags de manual/soft e pseudo-rótulos | Acelera CV reprodutível e permite peso suave dos exemplos | Descrição declara CC BY-NC-SA 4.0; confirmar compatibilidade com regras da competição |
| [Eric Wang — DINOv2-MIL bundle](https://www.kaggle.com/datasets/ericwang03/rsna-knee-dinov2-mil-bundle) | Pesos DINOv2 ViT-S/14, quatro checkpoints, `model.py` e `labels.py` | Bundle pequeno e muito acionável para reproduzir arquitetura pública sem treinar tudo do zero | Verificar licença, preprocessing, slots e compatibilidade dos checkpoints antes de anexar |
| [Barun — processed 3D volumes](https://www.kaggle.com/datasets/barun2104/rsna-knee-mri-processed-3d-volumes) | 24.371 arrays `.npz` normalizados, um por série, cerca de 7,36 GB | Pode eliminar custo de indexar DICOM repetidamente e acelerar 2.5D/3D | Dataset derivado da competição; confirmar regras/licença; download não prioritário agora |
| [Namra — Knee MILpack](https://www.kaggle.com/datasets/namra001/knee-milpack) | Pacote MIL com slices 384 e metadados, cerca de 11,7 GB só no array principal | Acelera MIL se o fingerprint bater com nossa convenção | Grande, descrição/licença insuficiente; deixar para P4 |

### Datasets externos de MRI ou segmentação

| Fonte | Conteúdo | Uso permitido como hipótese | Limite |
|---|---|---|---|
| [KneeMRI dataset](https://www.kaggle.com/datasets/sohaibanwaar1203/kneemridataset) | Volumes 3D PD fat-sat, ACL saudável/parcial/completa, scanner e ROI próprios | Pré-treino de leitura de sequência, teste de arquitetura e probe de ACL | Domínio diferente e licença não clara; não misturar labels diretamente sem verificar regras |
| [PCIR Knee MRI](https://www.kaggle.com/datasets/abbymorgan/pcir-knee-mri) | Uma sequência DICOM de demonstração | Teste unitário de leitura DICOM, ordenação e visualização; CC0 declarado | Não fornece volume suficiente para treinar |
| [OAI tissue segmentations](https://www.kaggle.com/datasets/kgaooo/oai-tissue-segmentations) | Máscaras de tecidos, menisco e cartilagens de MRI DESS | Inspirar crop/segmentation, ou pretraining separado se regras/licença permitirem | Outro protocolo/domínio; não é fonte direta dos 12 labels |
| [OAI MRI 3D DESS](https://www.kaggle.com/datasets/mohamedberrimi/oaimri3ddess) | Volumes 3D de OA normal/anormal | Probe de representação e robustez de OA | Muito grande, outro domínio e sem ganho imediato no leaderboard |

### Modelos

- [MetaResearch DINOv2 no Kaggle Models](https://www.kaggle.com/models/metaresearch/dinov2): candidato preferencial para fonte de backbone, sujeito à licença do modelo original.
- `keras/dinov2` e variantes de terceiros aparecem na busca, mas não devem ser escolhidos apenas por popularidade. Preferir origem oficial ou bundle cuja licença e fingerprint estejam claros.

## Convergência com as competições anteriores

Os resultados anteriores da RSNA reforçam a mesma direção:

- No RSNA Lumbar 2024, a solução pública de segundo lugar usou localização de coordenadas, crop anatômico, sequência de 24 embeddings, LSTM/attention, pseudo-rótulos com filtragem out-of-fold, augment pesado e TTA. [Código público](https://github.com/brendanartley/RSNA-2024-Competition).
- No RSNA Cervical Spine 2022, a solução de terceiro lugar separou detecção/localização e classificação de vértebras/fatias. [Código público](https://github.com/darraghdog/RSNA22).
- No RSNA Intracranial Hemorrhage 2019, a combinação CNN por fatia + GRU/LSTM apareceu como padrão vencedor. [Referência](https://pubmed.ncbi.nlm.nih.gov/34411910/).
- No CheXpert, a própria organização separa extração de menção, polaridade e incerteza; ausência de menção não deve ser convertida automaticamente em negativo. [Descrição oficial](https://stanfordmlgroup.github.io/competitions/chexpert/).
- Em NegBio, negação e incerteza dependem do escopo sintático. Uma janela fixa de caracteres não é robusta. [Artigo](https://pmc.ncbi.nlm.nih.gov/articles/PMC5961822/).

Isso não prova que precisamos copiar uma solução de coluna lombar. Prova que “uma imagem independente por estudo” é uma hipótese fraca para uma MRI seriada com alvos anatômicos diferentes.

## Gap entre o nosso código e a fronteira pública

| Componente | Hoje | Fronteira observada | Próxima ação |
|---|---|---|---|
| Supervisão | Léxico local com precisão abaixo de `0,70` em 9/12 alvos | Labels multilíngues graduados, várias fontes, confidence e OOF | Fazer ablação de labels antes de mudar tudo |
| Backbone | EfficientNet-B0 congelada | DINOv2 ViT-S/14, às vezes últimos 4–6 blocos ajustados | Reproduzir o bundle pequeno e comparar sob o mesmo split |
| Aquisição | Uma série por plano e flags simplificadas | Seis slots por contraste/plano e máscara de presença | Construir inventário por headers DICOM |
| Ordem | Precisa ser auditada em cada loader | IOP/IPP, depois SliceLocation/InstanceNumber | Teste determinístico de ordenação |
| Contexto | 3 fatias quantílicas e média | 2.5D adjacente, várias janelas, MIL/attention | Implementar cache e janela 3/6/12 |
| Resolução | Baseline pequeno | FOV físico de ~150 mm e 336 px em alguns notebooks | Probe 224 vs 336 com mesmo encoder |
| Pooling | Média global | Pooling por alvo: média, attention, max ou top-k | Começar por Fracture/Contusion/Baker |
| Orientação | Não comprovada em todos os ramos | Laterality por geometria; sem vertical flip indiscriminado | Auditoria visual e teste com PCIR |
| Ensemble | Peso visual global | Rank mean, folds, seeds, TTA e pesos encolhidos | Salvar sempre rank-mean como controle |
| CV | 58 dourados limitam inferência | Grupo por relatório, folds balanceados e OOF | Fixar protocolo antes de comparar |

## Hipóteses testáveis

Status: `nova`, `em teste`, `apoiada`, `descartada`, `bloqueada` ou `engenharia`.

| ID | Hipótese | Teste mínimo | Critério de avanço | Status |
|---|---|---|---|---|
| H-01 | Consenso de labels públicos é melhor que o léxico local bruto | Avaliar léxico, Pilkwang, Steven v2/v4 e consenso em CV agrupado nos 58 | Macro OOF subir pelo menos `0,02` ou ganho consistente em ≥8/12 alvos | apoiada nos 58; falta testar visual |
| H-02 | DINOv2-S/14 + cabeça MIL supera EfficientNet-B0 congelada | Mesmos slots, slices, labels, folds e seed; mudar só encoder/head | Macro OOF `+0,02` sem queda grave nos alvos raros | nova |
| H-03 | Descongelar os últimos 4–6 blocos melhora sobre DINOv2 congelado | Comparar frozen, last-4 e last-6 com LR pequeno | Ganho OOF e estabilidade entre folds; não aceitar apenas um fold | nova |
| H-04 | 336 px com crop físico preserva melhor menisco e lesões focais que 224 | Probe 224/336, mesmo número de cortes e mesmo backbone | Ganho em meniscos/fratura e macro OOF; medir custo por estudo | nova |
| H-05 | Ordem física supera ordem por filename | Auditoria de correlação/visualização + treino controlado | Menos inversões e ganho OOF; se não houver ganho, manter por correção física | parcial — `InstanceNumber` superou filename, mas `IPP/IOP` caiu `0,005838` contra o header no gold 3-view |
| H-06 | Pooling por alvo supera média global | Mean vs attention vs max/top-k nos mesmos embeddings | Ganho em Fracture/Contusion/Baker sem prejudicar OA/derrame | nova |
| H-07 | Seis slots clínicos + presence mask superam apenas três planos | Treinar com slots fixos, faltantes mascarados e diagnóstico de cobertura | Macro OOF subir e nenhuma aquisição dominar artificialmente | parcial — ensemble por plano ganhou no holdout, mas o gold não tem Axial |
| H-08 | Laterality por geometria e ausência de vertical flip reduzem erro sistemático | Auditar 20–50 estudos e comparar com/sem normalização | Menos inversão lateral; ganho ou neutralidade OOF | nova |
| H-09 | Folds agrupados por relatório duplicado dão estimativa mais honesta | Agrupar fingerprints, balancear os 12 alvos e comparar com split ingênuo | Reduzir gap CV/LB; usar apenas o protocolo agrupado para decisão | apoiada pelo fórum |
| H-10 | Rank averaging é mais robusto que média de probabilidades | Comparar rank mean, prob mean, quality-weight e target-weight | Rank mean não pode perder o baseline em nenhuma submissão de controle | apoiada parcialmente |
| H-11 | Labeler por seção + escopo + grau supera busca de palavras | Impression/Findings, unwrap, negação, hedge, anatomia e backoff sinovite/derrame | Melhorar precisão/cobertura nos 58; usar somente casos confiáveis | apoiada por notebook e audit local |
| H-12 | Pseudo-rótulo só deve entrar com confiança e peso suaves | Variar `confidence = 2*abs(p-.5)`, limiar e peso dos 58 dourados | Melhorar OOF; retirar a fonte que piora um alvo | nova |
| H-13 | Metadata de scanner/site é atalho, não sinal clínico generalizável | Comparar random folds e folds por scanner/site | Se só funcionar em random, não usar no ensemble principal | apoiada pelo fórum |
| H-14 | TTA de janela, gamma e flip controlado ajuda depois da representação | Aplicar TTA a um modelo já estável, mantendo fallback sem TTA | Ganho pequeno e repetível, sem escolher por leaderboard isolado | nova |
| H-15 | Soups de checkpoints/folds melhoram a variância | Média/rank de checkpoints de sementes/folds já bons | Ganho sobre melhor membro sem aumentar muito o custo | nova |
| H-16 | Arrays processados aceleram iteração sem mudar score | Comparar DICOM→array e `.npz` no mesmo subset/fingerprint | Mesma predição dentro de tolerância e economia de tempo clara | engenharia |
| H-17 | Pretraining/crop com OAI ou KneeMRI pode ajudar, mas não deve misturar labels diretamente | Primeiro probe de transferência e verificação de licença/regras | Só avançar com autorização documental e ganho em CV | bloqueada |
| H-18 | Um modelo especialista por família de alvo pode superar um único head | Ramo ligamentos/meniscos, OA, fluido e focal; ensemble por rank | Ganho em pelo menos duas famílias sem overfit dos 58 | nova |
| H-19 | Fonte de weak label por alvo supera Steven v4 uniforme | Comparar v4 uniforme, mapa target-wise e consenso com fonte neutra quando a cobertura cair | Ganho OOF agrupado em pelo menos 8/12 alvos, sem escolher pelo leaderboard isolado | nova |

## Plano de execução por fases

### P0 — Reprodutibilidade e segurança

- [x] Confirmar o status da `55392604`: `COMPLETE`, public score `0,655`.
- [ ] Fixar um split agrupado por `Report`/fingerprint e preservar os 58 dourados fora do treino quando forem validação.
- [ ] Medir baseline atual por alvo, fold e seed; salvar logits e máscara de presença.
- [ ] Confirmar as regras de dados externos e as licenças de todos os datasets/bundles. `licenseName` ausente no Kaggle não é autorização.
- [x] Baixar e auditar somente o bundle DINOv2 pequeno, registrando hashes, arquitetura e metadata dos quatro folds.
- [ ] Liberar o bundle DINOv2 para submissão somente após confirmar a licença `other`, a origem dos heads e a compatibilidade com as regras.

### P1 — Labels, sem trocar o visual

- [x] Colocar os rótulos Pilkwang/Steven/Lixin em uma pasta de artefatos com checksum e versão.
- [x] Comparar `raw lexicon`, `Pilkwang`, `Steven v2`, `Steven v4 blend` e consenso.
- [x] Calcular por alvo: AUC nos 58, precisão, cobertura e confiança mínima.
- [ ] Treinar o baseline atual com exatamente a mesma imagem para isolar o efeito da supervisão.
- [ ] Excluir fontes/targets cuja confiança não bata com o gold ou cuja melhoria apareça em um único fold.

### P2 — Representação visual mínima forte

- [ ] Implementar loader dos seis slots com presença e fallback explícito.
- [ ] Ordenar cortes por projeção de `IPP` na normal de `IOP`; registrar fallback usado.
- [ ] Corrigir `MONOCHROME1`, intensidade, crop físico e laterality.
- [ ] Testar 2.5D com janelas de 3, 6 e 12 cortes adjacentes.
- [x] Corrigir e validar o carregamento offline do frozen DINOv2-S/14 + MIL.
- [ ] Reproduzir a inferência DINOv2-S/14 + MIL em DICOM real no Kaggle, com licença liberada.
- [ ] Fazer o probe 224 vs 336 antes de pagar o custo de cinco folds.

### P3 — Treino e ensemble

- [ ] Fine-tuning dos últimos 4–6 blocos com LR discriminativo e `GOLD_WEIGHT` explícito.
- [ ] Attention por alvo; mean para achados difusos e max/top-k para focais.
- [ ] Folds balanceados, seed adicional, EMA/checkpoint soup e TTA controlada.
- [ ] Comparar rank-mean, rank-weighted e prob-mean; salvar sempre o membro e o rank baseline.
- [ ] Fazer uma submissão apenas quando houver uma mudança controlada e uma hipótese clara.

### P4 — Escala e dados opcionais

- [ ] Considerar `.npz` processado ou MILpack só depois de validar licença, fingerprint e espaço.
- [ ] Considerar localização/crop treinado ou um ramo de segmentação somente após o baseline visual forte.
- [ ] Avaliar dados externos para pretraining/transferência, nunca para fabricar rótulos do desafio sem autorização.

## Matriz de execução

| Experimento | Família | Artefato principal | Custo esperado | Métrica principal | Submission | Score | Decisão |
|---|---|---|---|---|---|---|---|
| E-000 | baseline | EfficientNet-B0 atual | baixo | macro AUC / alvo | `55365537` | `0,635` | controle |
| E-001 | weak labels | v10 com teacher textual + léxico explícito | baixo/médio | macro OOF + LB | `55392604` | `0,655` | nova referência |
| E-002 | labels | Steven v2/v4 mascarado vs léxico | baixo | AUC/precisão nos 58 | local | `0,892707` | fonte escolhida; testar no visual |
| E-003 | labels | consenso Steven/Pilkwang/Lixin | baixo | AUC por alvo | local | `0,892826` | não supera materialmente Steven v4; manter como ablação |
| E-002a | labels | mapa target-wise de fontes, com cobertura mínima | baixo | AUC/coverage por alvo | — | — | preparar somente depois do score de E-002; evitar seleção circular |
| E-004 | representação | DINOv2 frozen + MIL offline | médio | OOF agrupado | — | — | código/auditoria prontos; bloqueado por licença e execução DICOM |
| E-004a | engenharia | Loader local de pesos + inferência unitária | baixo | shape, finitude, hash | local | passou | caminho offline validado; não é score de competição |
| E-009 | representação/pooling/labels | v4 `dense6 + target pooling + teacher targetwise` | alto | macro OOF + LB | v4 v2 em execução | — | v1 falhou só no mount do peso; v2 corrigida está sem logs ainda |
| E-010 | representação | 2.5D filename vs `InstanceNumber` no gold e nos 700 estudos weak | médio/alto | holdout oficial por alvo | — | `0,591` header por série | Header supera filename no holdout, mas ainda fica abaixo de `0,706`; manter como ramo complementar |
| E-011 | ensemble | Mistura fixa 50/50 filename + header, sem tuning targetwise | médio | holdout oficial + LB | — | `0,615` mean / `0,608` série | Hipótese aprovada para uma execução Kaggle; grade de pesos é apenas diagnóstico |
| E-005 | geometria | ordem física + seis slots | médio | OOF + auditoria | — | — | planejado |
| E-006 | resolução | 224 vs 336 | médio/alto | menisco/focal + macro | — | — | planejado |
| E-007 | pooling | target-specific max/top-k | médio | focal vs difuso | — | — | planejado |
| E-008 | ensemble | rank mean + TTA + fold | alto | macro + estabilidade | — | — | planejado |

## Protocolo de avaliação

1. **Métrica:** macro AUC não ponderada dos 12 alvos. Um alvo raro ainda importa tanto quanto um alvo frequente; por isso não otimizar somente a média dos casos positivos.
2. **Validação:** usar folds agrupados por estudo e por fingerprint de relatório. A validação dos 58 deve ser OOF, nunca uma justificativa para ajustar manualmente dezenas de parâmetros.
3. **Rótulos:** `0,5` em um dataset de relatório normalmente significa “não abordado/indeterminado”, não “negativo”. Usar máscara de observação e confiança.
4. **Comparabilidade:** mudar um componente por vez e manter o mesmo conjunto de estudos, preprocessamento, seed e número de épocas quando a pergunta for uma ablação.
5. **Leaderboard:** usar como confirmação tardia. Um ganho isolado menor que `0,02` não vence a evidência de um OOF mais honesto.
6. **Artefatos:** cada corrida precisa salvar config, commit, checksum dos labels, lista de séries, transformação de orientação, logits OOF, score por alvo e CSV de submissão.
7. **Segurança:** não subir DICOM, relatórios, pesos privados ou datasets derivados grandes para o GitHub. O Git deve conter código, metadados e instruções; dados ficam no HD/Kaggle.

## Registro de resultados

| Data | Variante | Score público | Status | Observação |
|---|---|---:|---|---|
| anterior | baseline inicial | `~0,505` | confirmado no histórico | ponto de partida |
| anterior | baseline ajustado | `~0,607` | confirmado no histórico | ainda representação fraca |
| anterior | targetwise | `~0,582` | confirmado no histórico | não escalar pesos por um split |
| anterior | shrink/blend | `~0,605` | confirmado no histórico | ganho insuficiente |
| anterior | melhor confirmada `55365537` | `0,635` | confirmado | controle anterior |
| 10/08/2026 | v10 `55392604` | `0,655` | confirmado | nova referência |
| 10/08/2026 | E-002 Steven v4 mascarado | — | smoke local aprovado | aguardando T4 |
| futuro | E-004 em diante | — | não iniciado | preencher pelo template |

## Registro de famílias de pesquisa

| Família | Estado atual | Evidência | Lacuna | Próxima ação |
|---|---|---|---|---|
| Qualidade dos labels | mais importante que o léxico atual | audit local + Steven v4 `0,892707` nos 58 | efeito no visual completo | P1 |
| Representação | baseline está muito atrás | convergência DINOv2/2.5D/MIL | custo e compatibilidade | P2 |
| Geometria DICOM | provavelmente subexplorada | notebooks e engenharia física | auditoria do loader atual | P2 |
| Pooling | média global é limitada | Fracture max/top-k e attention pública | validação por alvo | P3 |
| Ensemble | rank aparece repetidamente | notebooks públicos | evitar overfit dos 58 | P3 |
| Dados externos | potencial, mas não urgente | OAI/KneeMRI/arrays | licença, regras e domain shift | P4/bloqueada |
| Metadata | risco de shortcut | fórum: random vs scanner/site | generalização privada | auditoria, não core |

## Riscos e decisões que não devemos tomar ainda

- Não baixar o corpus de NIfTI/volumes inteiro antes de termos um loader e uma pergunta experimental que o justifique.
- Não misturar os 12 alvos de um dataset ACL externo com os rótulos RSNA; os protocolos, scanners, definições e distribuição são diferentes.
- Não confiar em `licenseName: None`, descrição vazia ou dataset derivado da competição sem confirmar regras e licença.
- Não usar o score alegado de `0,894` como ground truth. É um relato de notebook e o acesso programático ao leaderboard completo estava bloqueado por `403`.
- Não usar os três placeholders do CSV público de teste para estimar distribuição. O teste real pontuado é oculto.
- Não escolher o melhor peso de ensemble olhando muitas combinações no mesmo leaderboard. Primeiro congelar o modelo, depois submeter uma combinação justificada.
- Não transformar relatório em entrada de inferência se ele não existir no teste.

## Perguntas abertas

- A submissão `55392604` processou e qual foi o score?
- O bundle `ericwang03/rsna-knee-dinov2-mil-bundle` foi treinado com qual exatamente preprocessing, slot manifest e versão de labels?
- O `Steven v4 blend` pode ser usado segundo a licença do dataset e as regras atuais da competição?
- O nosso loader já ordena por geometria física ou ainda há alguma série ordenada por filename?
- Qual é o ganho real de 336 px depois de limitar o número de cortes e o tempo de GPU?
- A fonte de labels melhora todos os targets ou somente ACL/MCL/alguns achados? O consenso deve ser target-specific.
- O DICOM disponível no HD é suficiente para um smoke test ou precisamos baixar somente as séries faltantes dos 58 estudos?

## Fontes principais

### Competição e fórum

- [RSNA Knee MRI AI Challenge](https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge)
- [Fórum — incerteza com 58 estudos](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733876)
- [Fórum — shortcut de DICOM/scanner/site](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733517)
- [Fórum — labels derivados da imagem](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826)
- [Fórum — ambiguidade dos relatórios](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733491)
- [Fórum — weak supervision com 4.407 relatórios](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733836)
- [Fórum — teste sem Report](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733592)
- [Fórum — PatientSex no DICOM](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733423)
- [Busca Kaggle usada](https://www.kaggle.com/search?q=knee+in%3Acompetitions+in%3Amodels+in%3Adatasets)

### Competições anteriores e labelers

- [RSNA Lumbar 2024 — solução pública de segundo lugar](https://github.com/brendanartley/RSNA-2024-Competition)
- [RSNA Cervical Spine 2022 — solução pública de terceiro lugar](https://github.com/darraghdog/RSNA22)
- [RSNA Lumbar 2024 — resultado oficial](https://www.rsna.org/media/press/i/2554)
- [RSNA ICH — artigo sobre modelos sequenciais](https://pubmed.ncbi.nlm.nih.gov/34411910/)
- [CheXpert — labeler e incerteza](https://stanfordmlgroup.github.io/competitions/chexpert/)
- [NegBio — escopo de negação](https://pmc.ncbi.nlm.nih.gov/articles/PMC5961822/)

## Histórico de atualização

### 2026-08-10 — ablação rápida da v4

A v4 `dense6` foi cancelada pelo timeout depois de chegar a aproximadamente
1.425/4.410 estudos, porque três planos × seis centros geraram 18 views por
estudo. A nova ablação `adjacent3` conserva os três planos e troca os seis
centros por três slabs 2.5D adjacentes, reduzindo o volume para 9 views por
estudo. O modo `fast_preprocess` restringe os tags de header, estima os
percentis em subamostragem 4× e reutiliza fatias repetidas. No smoke local de
10 estudos, a diferença média foi `0,34` níveis de uint8 e a máxima `5`,
enquanto o tempo de uma série caiu de `1,173 s` para `0,012 s`. A decisão é
executar essa variante no T4 antes de alterar o teacher, o pooling ou os pesos
de fusão.

Na execução Kaggle da ablação, o worker terminou a parte visual em `4.897 s`
com 4.410 estudos válidos e 39.672 views, mas falhou ao procurar
`llm_labels_v2.csv`: os datasets estavam montados sob
`/kaggle/input/datasets`, um nível abaixo do buscador original. A correção
passa a procurar o filename recursivamente e valida os três arquivos externos
antes de iniciar o loop DICOM.

Na versão seguinte, a recursão foi limitada às pastas de inputs externos,
excluindo `competition(s)`, porque uma busca global também atravessava o
volume DICOM e atrasava o primeiro log. O gate dos labels agora roda antes do
TF-IDF e da leitura visual.

O kernel v6 completou a execução após essa correção e gerou `submission.csv`.
Ele processou `4.410/4.410` estudos válidos e `79.380` views em `6.779,9 s`,
usando CUDA e a configuração real `dense6`, `view_pooling=target`,
`teacher_profile=targetwise` e `fast_preprocess=False`. A submissão de origem
Notebook `55413852` marcou público `0,706`, ganho de `+0,051` sobre a
referência `0,655`. Isso confirma a família densa, mas ainda não mede a
ablação `adjacent3_fast` com max pooling.

### 2026-08-10 — auditoria do pipeline Yash Bishnoi

O artigo [Inside the pipeline that placed 18th in a $77,000 Kaggle knee MRI
competition](https://huggingface.co/blog/bishnoiyash/rsna-competetion) descreve
um score público `0,903`, posição 18 de 792 no snapshot de 09/08/2026. A
consulta atual do leaderboard já deslocou esse score para aproximadamente a
posição 55; portanto, a colocação é temporal e não uma propriedade fixa do
modelo. O notebook associado é
[rsna-knee-infer-v1](https://www.kaggle.com/code/yashbishnoi98/rsna-knee-infer-v1).

Os mecanismos transferíveis, separados do claim de score, são:

- **Labels:** extrator local Qwen, regras clínicas por alvo, fusão suave e
  pesos de confiabilidade por alvo; a execução publicada usou também
  `Pilkwang` como terceira opinião.
- **Representação:** até três séries fluido-sensíveis, uma por plano, janela
  de intensidade `1–99%` por série, cache de volumes e mais fatias adjacentes.
- **Modelo:** EfficientNet-B3 fine-tunada, MIL por slice, `max pooling` e
  ensemble de folds; não é apenas uma troca de backbone.
- **Robustez operacional:** não aplicar flip horizontal por causa de
  medial/lateral, fallback de decodificação, checkpoint periódico e redução
  adaptativa de views/modelos sob limite de tempo.

Diferenças críticas para o nosso código: a v4 atual usa B0 congelada,
normalização por slice, regressão logística sobre embeddings e um único ajuste;
ela ainda não reproduz fine-tuning, janela por série ou ensemble. A hipótese de
labels suaves foi implementada no wrapper `v4_dense6_soft_labels`, mas ainda
não tem score. O artigo também informa que o próprio `0,9357` interno estava
vazado nos 58 estudos e caiu para `0,8568` após cross-fitting. O `0,903` público
é evidência de direção, não uma validação local limpa.

Hipóteses abertas a partir dessa auditoria:

| ID | Hipótese | Teste decisivo | Estado |
|---|---|---|---|
| H-20 | Max pooling de probabilidades por view supera `top-k/mean` na nossa v4 | Dense6 integral, mesmo teacher/blend, trocar somente a agregação para `max` | refutada — ref `55442653`, público `0,663` |
| H-21 | `adjacent3 + fast_preprocess` mantém sinal suficiente com menor custo | Kernel T4 completo e comparação de score/tempo contra v6 | enfraquecida |
| H-22 | Janela de intensidade por série supera normalização independente por slice | Dense6 integral, mesma configuração da H-25, labels weak suaves e janela comum `1–99%` nas fatias usadas por série | confirmada — v1 falhou no P100 por `sm_60`; v2 T4; ref `55551332`, público `0,712` |
| H-23 | Fine-tuning leve do encoder supera B0 congelada | Uma época no último bloco, head auxiliar multilabel, mesma imagem/teacher/pooling da H-22 e sem flip horizontal | confirmada — ref `55582655`, público `0,718`; ganho de `+0,006` sobre H-22 |
| H-24 | Fusão contínua e simétrica de teachers supera seleção de uma fonte por alvo | Auditoria fixa nos 58 com média/rank-mean de Steven, Pilkwang e Lixin | refutada para submissão — melhor blend `0,895808` vs target-wise `0,899120`; não gastar kernel |
| H-25 | Preservar a probabilidade dos weak labels supera a conversão hard 0/1 | Dense6 integral, mesmo teacher/pooling/blend da v6, duplicar cada weak como pesos `p`/`1-p` | confirmada — ref `55446808`, público `0,708` |
| H-26 | Média de probabilidades por view supera a política top-k/mean por alvo | Preservar H-23 integralmente e trocar somente `TARGET_VIEW_POOLING` para `mean` nos 12 alvos | ativa — gate local `0,643152` vs `0,635104`; kernel Kaggle publicado, aguardando score |
| H-27 | DINOv2-S oficial congelado supera EfficientNet-B0 | Mesmo dense6/janela/teacher e pooling MIL, trocar somente o backbone para DINOv2-S Apache 2.0 | refutada no gate local — embedding mean `0,568709`, MIL top-k `0,584075` vs B0 `0,636834`; fine-tuning DINO permanece hipótese distante |

Decisão imediata: H-20 foi isolada na v9 e refutada (`0,663` contra `0,706`).
H-25 foi executada na v10 e marcou público `0,708`. H-22 está empacotada como
script standalone em
`kaggle/rsna_knee_v4_series_window_soft_kernel/`. O kernel Kaggle v1 falhou
no P100 por incompatibilidade `sm_60`; a v2 concluiu no T4, gerou CSV íntegro
e foi submetida pelo Notebook como ref `55551332`, marcando público `0,712` e
ganho de `+0,004`. A auditoria H-24 não encontrou ganho direcional para a fusão
simétrica. H-23 superou H-22 e passa a ser a nova referência; H-22 continua
como fallback reproduzível. A auditoria seguinte refutou DINOv2 congelado no
holdout local, enquanto H-26 foi publicada como uma ablação barata de pooling
e aguarda o leaderboard antes de qualquer promoção.

### Resultado H-20/H-21 — v8

O kernel standalone executou a ablação sem alterar teacher, B0 ou blend:
`adjacent3`, `fast_preprocess=True`, `target_pooling=max`, 9 views por estudo,
`39.690` views totais e `4.416,4 s`. O CSV de 3 linhas foi validado e enviado
via Notebook com ref `55418681`, que marcou público `0,673`. O custo caiu de
`6.779,9 s`/`79.380 views` na v6 para `4.416,4 s`/`39.690 views` na v8, mas o
score ficou `-0,033` abaixo de `0,706` e `+0,018` acima da v3 (`0,655`). Como
H-20 e H-21 foram confundidas nessa rodada, a próxima execução isola max
pooling com `dense6` integral antes de descartar a hipótese.

### Implementação H-25 — labels weak suaves

O wrapper `kaggle/rsna_knee_v4_dense6_soft_labels.py` mantém `dense6`,
preprocessamento integral, teacher target-wise, pooling original por alvo,
B0 e blend global. A única mudança é no treino visual: cada estudo weak com
probabilidade `p` gera duas cópias das mesmas views, com pesos `p` para a classe
1 e `1-p` para a classe 0; os 58 estudos gold continuam com rótulo binário e
peso `1,0`. A implementação passou os 24 testes do projeto, completou no T4
em `6.882,9 s` e gerou CSV válido; a submissão `55446808` marcou público `0,708`,
ficando como referência anterior.

### Implementação H-22 — janela de intensidade por série

O script standalone `kaggle/rsna_knee_v4_series_window_soft_kernel/` mantém
teacher, pooling, labels weak suaves, B0, blend e `dense6` da H-25. A única
mudança é calcular uma janela `1–99%` comum às fatias que entram nos slabs da
série, preservando contraste relativo entre cortes. JSON, `py_compile`, smoke
dos helpers e `27` testes passaram; o commit `29e030a` foi publicado. O push
T4 foi bloqueado pelo Kaggle com `Maximum batch GPU session count of 2
reached`; o v1 chegou a iniciar no P100 e falhou por `sm_60`; a v2 foi enviada
com `NvidiaTeslaT4`, concluiu em `8.228,2 s` e gerou `submission.csv` com 3
linhas. A submissão via Notebook recebeu a ref `55551332` e marcou público
`0,712`; não será forçada em CPU
sem medir antes a viabilidade do tempo.

### Implementação H-23 — fine-tuning leve do encoder

O kernel privado `jvlegend/rsna-knee-v4-fine-tune-light` parte do standalone
H-22. Antes da extração dos embeddings, ele congela o prefixo do
EfficientNet-B0 e ajusta somente o último bloco por uma época, com um head
auxiliar multilabel de 12 saídas. Cada slab recebe o label do estudo; os
labels weak usam a probabilidade contínua e o mesmo peso `0,10` do H-22, e
alvos não abordados são mascarados. O head é descartado e o ajuste
target-wise original continua sendo feito sobre os embeddings finais. Não há
flip horizontal ou texto no teste.

O smoke real no MPS processou 1 estudo e 18 views, com loss observado
`0,699462`, sem NaN. O worker T4 completou `4.407/4.407` estudos de treino,
`79.326` views no fine-tuning e `79.380` views na extração, com loss final
`0,625231`; o tempo total foi `15.404,7 s`. O CSV de 3 linhas passou a
validação de colunas, UIDs, finitude e faixa `[0,1]`; SHA-256
`47a783fadca59d69cba08cb6ef90b2053623cc0b3a005a2c3b277d4bfb776ffc`.

A saída foi submetida pelo fluxo Notebook-only como ref `55582655` em
17/08/2026. A submissão foi processada com status `COMPLETE` e public score
`0,718`, superando H-22 (`0,712`) em `+0,006`. H-23 está promovida como nova
referência; o CSV local é apenas cópia de auditoria/backup e não precisa ser
enviado manualmente.

### Auditoria H-27 — DINOv2-S oficial — 17/08/2026

O modelo oficial `metaresearch/dinov2/pytorch/small/1` foi consultado pelo
Kaggle CLI e identificado como Apache 2.0. A cópia local foi baixada para
`/Volumes/Karine HD Externo/Dados_JV/Datasets/rsna-knee-abnormality-detection/models/dinov2_official_small/`;
os pesos ficam fora do Git. O smoke real no MPS gerou embeddings finitos
`(6,384)` a partir de slabs dense6.

No holdout dos 58 estudos, com a mesma ordenação e janela por série, o DINOv2
congelado marcou macro-AUC `0,568709` usando média dos embeddings. Treinando o
classificador nas views e agregando por estudo, os melhores modos foram
`topk=0,584075`, `max=0,581314` e `mean=0,576975`. O B0 congelado na mesma
geometria marcou `0,636834`; portanto H-27 foi refutada para submissão nesta
forma. O código oficial permanece em
`kaggle/rsna_knee_dinov2_official_kernel/` para uma futura hipótese de
fine-tuning, sem usar o bundle de licença `other`.

### H-26 — pooling mean — 17/08/2026

O ensaio local usou a geometria H-23 (`dense6`, janela `1–99%` por série,
EfficientNet-B0, 58 estudos gold disponíveis localmente) e comparou apenas a
agregação final das probabilidades por view. A política atual top-k/mean marcou
`0,635104`; `mean` em todos os 12 alvos marcou `0,643152`, ganho local de
`+0,008048`. Esse resultado é um gate de direção, não um score Kaggle, e pode
ser afetado pelo checkout local parcial.

O kernel privado foi publicado como
[`jvlegend/rsna-knee-v4-mean-pool-h-26`](https://www.kaggle.com/code/jvlegend/rsna-knee-v4-mean-pool-h-26),
mantendo fine-tuning H-23, teachers, blend e janela. A primeira versão foi
alocada em P100 `sm_60` e falhou antes da inferência porque o PyTorch do Kaggle
aceita `sm_70+`. A segunda versão foi relançada com `NvidiaTeslaT4`, concluiu e
gerou um `submission.csv` íntegro. A submissão Notebook-only `55610358` está
`PENDING`; o score ainda não foi processado. H-23 (`0,718`) permanece como
referência.

### Testes adicionais — resolução e slots — 18/08/2026

- O probe de resolução simples comparou 224 px e 336 px nos mesmos 59 conjuntos
  locais, com três views por série, ordenação por header e EfficientNet-B0. O
  macro-AUC médio caiu de `0,578718` para `0,558250` (`-0,020469`). Isso
  **refuta o resize simples para 336 px** neste protocolo; não refuta um crop
  físico de aproximadamente 150 mm.
- A auditoria do lote visual confirmou cobertura dos três planos: `333 + 367 =
  700` estudos, todos com Sagittal, Coronal e Axial. Porém cada estudo tem
  somente uma série por plano, não seis slots de contraste. A próxima
  implementação deve preservar uma máscara de presença e não inventar slots
  ausentes.
- O probe de ordenação física implementou a projeção de `ImagePositionPatient`
  na normal de `ImageOrientationPatient`. Embora tenha alterado 27/59 séries,
  marcou `0,572880` contra `0,578718` da ordenação por `InstanceNumber`.
  Mantemos o loader atual; isso não invalida testar seis slots ou crop físico,
  que são hipóteses diferentes.
- A submissão H-26 foi criada como `55610358` somente depois da validação do
  CSV. Enquanto o Kaggle não retornar score, nenhuma decisão de promoção será
  tomada.

### Probe de representação por plano — 18/08/2026

Para transformar a auditoria de cobertura em um teste de modelagem, foi criado
`scripts/evaluate_plane_presence_holdout.py`. O treino usa os 700 estudos do
lote weak (`2.100` embeddings B0, exatamente uma série por plano) e o holdout
usa os `58` estudos com os 12 rótulos oficiais. A comparação mantém o encoder,
o limiar `0,5` e a regressão logística fixos:

| Representação | Steven v4 | Pilkwang v2 | Delta vs média global (Pilkwang) |
|---|---:|---:|---:|
| Média de todos os planos disponíveis | `0,585771` | `0,587916` | — |
| Concatenação Sagittal/Coronal/Axial + máscara | `0,600978` | `0,606066` | `+0,018150` |
| Ensemble de modelos separados por plano | `0,619018` | `0,634186` | `+0,046270` |

O sinal é consistente em duas fontes weak independentes e favorece separar a
distribuição visual de cada plano, em vez de misturá-la antes do classificador.
Ainda não é uma validação de seis slots: o gold local tem `56` Sagittal, `3`
Coronal e `0` Axial. Assim, a decisão é **aprovar a implementação como
variante Kaggle de baixo risco**, mantendo H-23 como fallback, mas não declarar
ganho de leaderboard até o primeiro score. O próximo kernel deve preservar a
seleção H-23, a janela por série, fine-tuning leve e teacher target-wise, trocando
somente a cabeça visual por modelos por plano e máscara explícita de presença.

### 2026-08-10 — primeiro ganho confirmado e labels públicos

- A submissão `55392604` saiu de `PENDING` para `COMPLETE` com public score `0,655`, superando `55365537` (`0,635`) em `+0,020`.
- Os CSVs pequenos de Steven, Lixin e Pilkwang foram baixados para o HD externo; o Kaggle CLI reportou `CC0-1.0` para os três datasets.
- O novo auditor `scripts/audit_external_labels.py` comparou as fontes contra os 58 estudos oficiais, sem alterar o modelo visual.
- Steven v4, mascarado pelo `v2` quando o caso não é abordado, foi escolhido para a próxima variante visual. O smoke local passou, mas o push T4 foi temporariamente bloqueado pelo limite de duas sessões GPU do Kaggle.
- O bundle DINOv2-MIL foi auditado: backbone local carregou sem rede e gerou embedding finito `(1,384)`. A variante offline corrigida está no repositório, mas não foi submetida porque o Kaggle reportou licença `other` e o checkout local não contém DICOM de teste completo.
- A v4 agressiva foi implementada e auditada contra erros de canais, views vazias, pooling e `UNK`. Ela combina `dense6`, MIL aproximado por views e teacher target-wise; ainda precisa de execução T4 para qualquer conclusão de score.

### 2026-08-09 — pesquisa Kaggle e consolidação

- Inventariados notebooks de baseline, EDA, 2.5D, DINOv2, MIL, pooling, TTA e ensemble.
- Inventariados bundles de labels, folds, volumes processados e pesos DINOv2.
- Separados claims de score de notebooks de evidência reproduzível.
- Incorporadas as conclusões anteriores do fórum, das competições RSNA passadas e do audit local de labels.
- Definida a ordem P0→P4: reprodutibilidade, labels, representação, ensemble e somente depois dados externos/escala.

### Próxima atualização

Adicionar o resultado T4 de `E-009`, a decisão de licença do bundle DINOv2 e a primeira comparação DINOv2 vs EfficientNet-B0 em DICOM real.
