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
