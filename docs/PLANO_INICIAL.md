# Plano inicial de modelagem

#RSNA #Kaggle #Pesquisa #Tecnologia

## Objetivo da primeira rodada

Produzir uma submissão válida, barata e auditável para estabelecer o primeiro ponto de comparação. O v0 não tentará resolver toda a complexidade visual do desafio: ele usará o laudo e metadados de séries para medir quanto sinal existe antes de carregar centenas de gigabytes em um modelo 3D.

## Fases

### Fase 0 — infraestrutura e auditoria

- [x] Criar repositório no HD externo, separado do vault.
- [x] Criar `.gitignore` que bloqueia DICOM, CSVs baixados, pesos, credenciais e saídas.
- [x] Adicionar smoke test sintético.
- [x] Baixar os metadados pela Kaggle para `data/raw/`.
- [x] Registrar tamanho, cabeçalhos, número de estudos e cobertura por alvo.
- [x] Baixar uma série DICOM técnica (30 fatias, ~6,19 MB) e confirmar leitura com `pydicom`.
- [x] Baixar primeiro lote estratificado de 10 estudos (289 fatias, ~193,7 MB) e confirmar leitura/pixels com `pydicom`.
- [x] Ampliar para 18 estudos em dois lotes disjuntos (521 fatias, ~362,8 MB) e confirmar leitura/pixels com `pydicom`.

### Fase 1 — v0 report + metadata

1. Treinar um modelo independente para cada um dos 12 alvos.
2. Usar somente linhas em que o alvo está anotado; `NaN` não significa negativo.
3. Representar o laudo com TF-IDF de palavras e caracteres, preservando acentos e idioma.
4. Acrescentar sexo, contagem de séries, plano anatômico, séries sensíveis a fluido e séries com supressão de gordura.
5. Avaliar por validação estratificada dentro dos estudos rotulados, reportando AUC por alvo e média macro.
6. Treinar em todos os rótulos disponíveis e gerar `submission.csv`.

O entrypoint está em `scripts/run_baseline.py` e a cópia para a esteira Kaggle em `kaggle/rsna_knee_v0.py`. A candidata v0.1 usa `C=32`; a v0.2 acrescenta o léxico auditável como feature.

### Fase 2 — weak supervision controlada

Antes de usar o restante dos laudos como rótulo, medir cobertura e precisão de um léxico multilíngue por alvo. Só promover regras que tenham validação contra os estudos anotados e registrar os falsos positivos mais perigosos. O primeiro uso será como feature ou peso de confiança, não como verdade binária automática.

**Gate de 08/08/2026:** a auditoria encontrou precisão entre `0,43` e `0,88` nos 58 estudos anotados. Baker's, MCL, Fracture e menisco lateral têm sinal inicial; ACL e menisco medial exigem regras melhores. Pseudo-rótulos binários estão proibidos nesta rodada.

**Resultado do teste de feature de 08/08/2026:** com `C=32` e duas seeds, a v0.2 atingiu macro-AUC `0,628815` (seed 42) e `0,630918` (seed 2026), média `0,629867`. O ganho sobre a média da v0.1 foi `0,064212`. O léxico permanece apenas como feature derivada do texto; a decisão não promove regras a rótulos.

### Fase 3 — imagem por série

O smoke DICOM e o primeiro lote estratificado já foram validados com
`scripts/inspect_dicom_series.py` e `pydicom`. Os dois lotes contêm 18 estudos,
uma série fluido-sensível por estudo e 521 fatias; a soma local inclui ainda os
30 slices técnicos baixados anteriormente. A aquisição integral continua fora
de escopo até medirmos ganho visual e custo de armazenamento.

O manifesto `data/processed/dicom_subset_manifest.json` foi gerado com 10
estudos únicos, cobrindo os dois valores de cada alvo. A listagem plana sofreu
`429 Too Many Requests`; o downloader foi atualizado para usar a listagem em
árvore do Kaggle CLI `>=2.2.2`, consultando somente os diretórios selecionados.
Os lotes foram baixados incrementalmente para o HD externo e validados:
`289/289 + 232/232` arquivos esperados, `362.807.330` bytes, `MONOCHROME2`,
dimensões entre `256×256` e `800×800`, e `uint16`/`int16`.

1. [x] Decodificar DICOM com `pydicom`, normalizar intensidade e tratar `MONOCHROME1` sem depender de tags ausentes.
2. [x] Selecionar séries por plano e por `Fluid_Sensitive`/`Fat_Suppression`.
3. [x] Amostrar três fatias por série (25%, 50%, 75%) e materializar entrada 2.5D `224×224` em `uint8`.
4. [ ] Extrair embedding por fatia com encoder pré-treinado disponível no ambiente Kaggle.
5. [ ] Fazer pooling por série e depois por estudo.
6. [ ] Treinar 12 cabeças de classificação com máscara de perda para rótulos ausentes.

### Fase 4 — fusão e eficiência

Concatenar embedding visual com sinal textual/metadados, calibrar probabilidades por alvo e testar um ensemble pequeno. A competição tem um trilho de eficiência; portanto, tempo de inferência e tamanho de pesos serão métricas de projeto desde o início.

## Validação e gates

- Fixar seed, split e lista de estudos por rodada.
- Nunca usar o teste para estimar prevalência, vocabulário supervisionado, limiar ou pseudo-rótulo.
- Medir ROC-AUC por alvo, macro-AUC, cobertura de rótulo, dispersão entre folds e custo computacional.
- Só enviar uma hipótese se ela superar o baseline em validação independente e produzir um CSV que passe o protocolo de formato.
- Não retunar um modelo com base em uma única pontuação pública.

## Próxima ação concreta

Corrigir primeiro o vínculo da fonte de dados no kernel privado v0.2. A versão
2 foi publicada, mas terminou sem montar `train.csv` em
`/kaggle/input/rsna-knee-abnormality-detection/`; não houve submissão ao
leaderboard. Em paralelo, usar o lote local para implementar e validar a leitura
visual 2.5D antes de ampliar a aquisição. Depois de a conta aceitar as regras e
o caminho existir, executar a candidata v0.2:

```bash
python scripts/run_baseline.py --data-dir data/raw --c 32 --use-lexicon --output submissions/submission_v0_2_report_metadata_lexicon.csv
python scripts/validate_submission.py --test data/raw/test.csv --submission submissions/submission_v0_2_report_metadata_lexicon.csv
```

O primeiro registro de leaderboard será usado apenas para confirmar a esteira e a direção da validação, não como substituto da avaliação local. O envio continua manual e depende de confirmação explícita.
