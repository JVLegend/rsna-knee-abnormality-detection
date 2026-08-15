# Dados e regras da competição

#RSNA #Kaggle #Medicina #Tecnologia

## Snapshot consultado

Página consultada em 07/08/2026 (horário de São Paulo): [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).

O desafio é organizado pela Radiological Society of North America. A página oficial da RSNA descreve o primeiro desafio da entidade a combinar imagens de MRI do joelho com o texto original dos laudos.

## Tarefa e métrica

Para cada estudo, prever probabilidades para 12 achados binários. A pontuação é a média macro dos 12 ROC-AUCs:

```text
score = (AUC_ACL + AUC_MCL + ... + AUC_Fracture) / 12
```

Alvos, na ordem do `submission.csv`:

1. `ACL`
2. `MCL`
3. `Medial Meniscus`
4. `Lateral Meniscus`
5. `Medial OA`
6. `Lateral OA`
7. `PF OA`
8. `Effusion`
9. `Synovitis`
10. `Baker's`
11. `Contusion`
12. `Fracture`

## Arquivos

- `train.csv`: uma linha por estudo, com `StudyInstanceUID`, sexo, laudo e os 12 rótulos quando disponíveis.
- `train_series.csv`: uma linha por série, com plano anatômico, sensibilidade a fluido e supressão de gordura.
- `train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm`: fatias DICOM.
- `test.csv` e `test_series.csv`: esquema de teste; os DICOMs de exemplo são substituídos durante a avaliação.
- `sample_submission.csv`: formato válido com probabilidades iniciais 0,5.

O snapshot da Kaggle informa 819.640 arquivos, 569,76 GB e aproximadamente 1.300 estudos de teste. As intensidades, orientações, resoluções e transfer syntaxes DICOM variam. Os metadados foram reduzidos pela organização a uma lista permitida de 86 tags.

Um ponto central do desafio é que somente uma pequena parte dos estudos de treino tem rótulos por condição. Os laudos estão disponíveis para todos os estudos e podem ser usados para extrair sinal supervisionado adicional, desde que façamos isso sem transformar ausência de rótulo em negativo.

## Estado da aquisição local — 07/08/2026

- Metadados baixados em `data/raw/`: `train.csv`, `test.csv`, `train_series.csv`, `test_series.csv` e `sample_submission.csv`.
- `train.csv`: 4.407 estudos, todos com relatório preenchido; 58 estudos rotulados por alvo.
- `train_series.csv`: 24.371 séries, média de 5,53 séries por estudo.
- Os CSVs passaram por validação de colunas, duplicidades e contrato da submissão.
- Nenhum DICOM foi versionado. A estratégia é adquirir primeiro um subconjunto estratificado para validar o pipeline visual e só então avaliar a necessidade de baixar o conjunto integral.

## Estado da aquisição local — 08/08/2026

- Uma única série técnica de treino foi baixada para o HD externo, sem UID publicado neste repositório.
- A série contém 30 fatias, cerca de 6,19 MB, resolução `320×320`, `uint16`, `MONOCHROME2` e `PixelSpacing 0,5×0,5`.
- A leitura com `pydicom 3.0.2` funcionou para todas as fatias; a sequência de instâncias é `1–30` e há um único `SeriesInstanceUID`.
- O download confirma que a API permite aquisição por arquivo individual. O conjunto integral de aproximadamente 569 GB continua pendente; não iniciar esse download antes de validar seleção de séries, normalização e custo do pipeline.

## Estado da aquisição local — lote estratificado — 08/08/2026

- O manifesto `data/processed/dicom_subset_manifest.json` seleciona 10 estudos únicos e uma série fluido-sensível/sagital por estudo, cobrindo os valores 0 e 1 dos 12 alvos.
- O lote selecionado tem 289 fatias e `193.721.314` bytes; somado à série técnica anterior, há 319 DICOMs locais no HD externo.
- A leitura completa dos 289 arquivos passou com `pydicom`: `MONOCHROME2`, transfer syntax explícita little endian, dimensões `256×256`, `320×320`, `512×512`, `640×640` e `800×800`, com pixels `uint16` ou `int16`.
- O downloader usa a listagem em árvore do Kaggle CLI `>=2.2.2`, consulta somente os diretórios do manifesto e é incremental. O conjunto integral continua fora de escopo até validar o pipeline visual.

## Estado da aquisição local — segundo lote — 08/08/2026

- O segundo manifesto excluiu os 10 estudos anteriores e acrescentou 8 estudos, 8 séries e 232 fatias, totalizando `169.086.016` bytes.
- Os dois manifestos são disjuntos. Juntos, somam 18 estudos, 521 fatias e `362.807.330` bytes; com o smoke técnico, há 551 DICOMs locais no HD externo.
- A leitura completa dos dois lotes passou com `pydicom`; todas as 521 fatias têm `MONOCHROME2` e pixels presentes. As dimensões observadas variam de `256×256` a `800×800`.

## Estado da aquisição local — orçamento de aproximadamente 50 GB — 14/08/2026

- O manifesto `data/processed/dicom_budget_50gb_manifest.json` selecionou 333 estudos, 999 séries e 37.363 DICOMs, totalizando `23.762.891.238` bytes.
- O manifesto `data/processed/dicom_budget_50gb_manifest_v2.json` selecionou mais 367 estudos, 1.101 séries e 40.936 DICOMs, totalizando `28.220.610.242` bytes.
- Os lotes são disjuntos e excluem os DICOMs já presentes. Juntos, adicionam 700 estudos, 2.100 séries, 78.299 DICOMs e `51.983.501.480` bytes; o HD ficou com 758 estudos, 2.159 séries, 80.278 DICOMs e aproximadamente `50G` ocupados em `data/raw/train_series/`.
- A seleção mantém uma melhor série `Fluid_Sensitive/Fat_Suppression` por plano `Sagittal`, `Coronal` e `Axial` em cada estudo, com estratificação pelos weak labels públicos e quotas para não concentrar apenas casos fáceis ou positivos.
- A transferência foi feita por `scripts/download_dicom_zip_subset.py`, lendo o ZIP Kaggle por HTTP Range e validando CRC por arquivo, com retomada incremental. Isso contornou os `HTTP 429` do download por arquivo individual. O seletor reproduzível está em `scripts/select_dicom_budget.py`.
- Não ficaram arquivos `.part`; todos os diretórios das 2.100 séries selecionadas existem e não estão vazios. Uma amostra de 100 cabeçalhos DICOM passou com `pydicom`. Os DICOMs continuam ignorados pelo Git e `test_series/` permanece sem imagens locais.
- O perfil exploratório `--fast-file-order` gerou 2.100 arrays `(3, 224, 224)` e embeddings `(2.100, 1.280)` em MPS. Contra os weak labels, a concatenação dos três planos marcou macro-AUC diagnóstico `0,655891`; no holdout independente dos 58 estudos oficiais, mean pooling marcou `0,577565` e treino por série `0,575001`. Como os estudos de treino foram selecionados pelos weak labels, esse perfil não será submetido; a ordenação anatômica por header continua necessária.
- A ablação gold `scripts/evaluate_gold_visual_ordering.py` comparou as duas ordens nos mesmos 58 estudos com os 12 rótulos oficiais, usando duas seeds e CV agrupada por estudo. A ordem por `InstanceNumber` marcou macro-AUC médio `0,578718`, contra `0,543691` da ordem lexicográfica, ganho pareado de `+0,035027`. O sinal foi forte em `Effusion` (`+0,2547`) e `Medial Meniscus` (`+0,1538`), mas houve perdas em `Medial OA`, `Fracture` e `MCL`; não é prova de leaderboard.
- O gold contém 59 séries para 58 estudos: 56 `Sagittal`, 3 `Coronal` e nenhuma `Axial`. Portanto, a ordenação anatômica foi aprovada como hipótese para escalar, mas pooling de três planos ainda está sem validação local independente.
- A escala anatômica processou as 2.100 séries dos 700 estudos adicionais: 2.100 arrays íntegros `(3,224,224) uint8`, 700 séries por plano (`Sagittal`, `Coronal`, `Axial`) e embeddings EfficientNet-B0 `(2.100,1.280)` finitos em MPS. No holdout oficial treinado com Steven v4, mean pooling marcou `0,585771` e média das previsões por série `0,591101`; a ordem melhora o smoke anterior, mas ainda não substitui a referência Kaggle `0,706`.
- Uma mistura pré-especificada 50/50 entre os ramos filename e header foi a hipótese mais promissora desta rodada: `0,614597` com mean pooling e `0,607885` por série. A grade de pesos é apenas diagnóstico no gold; não será usada para calibrar targetwise nem para declarar vitória sem submissão independente.

## Artefato visual local — 2.5D — 08/08/2026

- `scripts/build_dicom_25d_features.py` leu as 18 séries dos dois manifestos e criou 18 arrays compactos em `data/processed/dicom_25d_v0/`.
- Cada array tem três canais grayscale correspondentes às fatias nos quantis `0,25`, `0,50` e `0,75`, redimensionados para `(224, 224)` e armazenados como `uint8`.
- O `index.json` preserva os UIDs locais, índices, nomes das fatias, dimensões de entrada e limites de intensidade. Nenhum DICOM ou array foi versionado.

## Ampliação para o universo rotulado — 08/08/2026

- O manifesto adicional contém 40 estudos, 1.428 fatias e `945.572.590` bytes segundo a listagem em árvore do Kaggle CLI.
- Uma auditoria encontrou 34 séries completas e 6 séries incompletas, com 132 arquivos ainda faltantes. A aquisição foi reduzida a esse residual para não repetir os 1.296 arquivos já presentes.
- Os 34 estudos completos foram processados junto dos 18 estudos anteriores: `data/processed/dicom_25d_labeled_v0/` contém 52 arrays `(3, 224, 224) uint8`, e `data/processed/dicom_embeddings_efficientnet_b0_labeled/` contém embeddings `(52, 1280)` `float32`.
- O baseline visual linear, avaliado por estudo em 5 folds e duas seeds, alcançou macro-AUC `0,593222`/`0,584142`, média `0,588682`. É um gate preliminar, pois seis estudos ainda não entraram e não há comparação pareada definitiva com o texto.

## Cronograma

Todos os horários abaixo são 23:59 UTC, salvo atualização da organização:

- Início: 30/07/2026.
- Entrada e fusão de equipes: 15/10/2026.
- Submissão final: 22/10/2026.
- Obrigações dos vencedores: 05/11/2026.

Há limite de 5 integrantes por equipe, até 5 submissões por dia e até 2 submissões finais selecionadas para julgamento.

## Regras operacionais importantes

- A submissão é feita por notebook Kaggle, não por upload direto de um CSV local.
- O arquivo final precisa se chamar `submission.csv`.
- Notebook CPU ou GPU deve terminar em até 9 horas.
- A internet fica desativada durante a execução.
- Dados e modelos externos são permitidos quando públicos, igualmente acessíveis e de custo razoável; registrar a licença antes de usar.
- Os dados da competição não podem ser transmitidos, duplicados, publicados ou redistribuídos para pessoas que não aceitaram as regras. Por isso, nenhum DICOM, CSV, relatório ou peso treinado com os dados será colocado no GitHub.
- O vencedor deve entregar código, pesos e documentação reproduzíveis; o tipo de licença para a submissão vencedora é CC-BY-NC 4.0.

## Uso no Brasil e limites da licença

- As regras abrem a competição a residentes do mundo todo, com as exceções listadas; o Brasil não aparece entre as jurisdições excluídas. Devemos ainda observar LGPD, políticas institucionais e requisitos regulatórios locais.
- A entrada não prevê cessão de equity ou participação societária. Para vencedores, porém, existem obrigações de entrega de código, pesos e documentação e de licenciamento da submissão vencedora.
- A competição seleciona **Commercial and Academic Research — MIRA license**. A licença MIRA permite pesquisa sem custo, inclusive pesquisa para produzir ou fabricar produtos destinados à venda, mas proíbe vender, publicar, redistribuir ou monetizar o dataset. O uso é **Research Use Only** e não pode sustentar diagnóstico ou atendimento de pacientes.
- Não compartilhar DICOM, relatórios, links de download ou derivados com pessoas que não aceitaram as regras; não tentar reidentificar indivíduos. Para qualquer produto/serviço comercial em produção, obter confirmação escrita da RSNA antes de usar este dataset ou pesos derivados.

Fontes: [regras da competição](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules) e [licença MIRA (PDF)](https://www.rsna.org/-/media/files/rsna/RSNAI/RSNA-MIRA-Dataset-License-2025.pdf). Esta é uma leitura operacional, não parecer jurídico.

## Fonte complementar

- [RSNA Knee MRI AI Challenge (2026)](https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge)
