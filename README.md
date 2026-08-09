# RSNA Knee Abnormality Detection

Baseline reprodutível para a competição [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection), mantido no HD externo e versionado sem dados da competição.

## Objetivo

Prever, para cada estudo de MRI do joelho, a probabilidade de 12 anormalidades. A métrica é a média macro de ROC-AUC nos 12 alvos.

O primeiro marco foi uma submissão textual; a referência atual é a candidata multi-view texto–imagem, com score público `0,635`. A próxima hipótese amplia o treino visual com supervisão fraca controlada dos 4.407 laudos, sem depender de `Report` no teste.

## Estrutura

```text
src/rsna_knee_baseline/   carregamento, atributos e modelo v0
scripts/                   comandos locais de inspeção e treino
kaggle/                   cópia do entrypoint para notebook Kaggle
docs/                     plano, regras, dados e protocolo de submissão
tests/                    smoke tests sem dados reais
data/raw/                 dados Kaggle locais; ignorados pelo Git
data/processed/           derivados locais; ignorados pelo Git
models/                   pesos locais; ignorados pelo Git
submissions/              CSVs de submissão; ignorados pelo Git
reports/                  relatórios locais; ignorados pelo Git
```

## Preparação local

```bash
cd "/Volumes/Karine HD Externo/Dados_JV/Projetos_GitHub/rsna-knee-abnormality-detection"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Baixe os arquivos pela competição e extraia-os dentro de `data/raw/`, sem adicionar os dados ao Git. A estrutura esperada é:

```text
data/raw/
├── train.csv
├── train_series.csv
├── train_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
├── test.csv
├── test_series.csv
├── test_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
└── sample_submission.csv
```

Para apenas inspecionar os arquivos:

```bash
python scripts/inspect_data.py --data-dir data/raw
```

Para gerar a v0:

```bash
python scripts/run_baseline.py \
  --data-dir data/raw \
  --output submissions/submission_v0_report_metadata.csv
```

Para a candidata v0.1, selecionada por validação cruzada com `C=32`:

```bash
python scripts/run_baseline.py \
  --data-dir data/raw \
  --c 32 \
  --output submissions/submission_v0_1_report_metadata.csv
```

Para a candidata v0.2, que adiciona o léxico auditável do laudo:

```bash
python scripts/run_baseline.py \
  --data-dir data/raw \
  --c 32 \
  --use-lexicon \
  --output submissions/submission_v0_2_report_metadata_lexicon.csv
```

A v0.2 atingiu macro-AUC média `0.629867` em dois seeds (`42` e `2026`), contra `0.565655` da v0.1. O léxico é usado como feature, sem criar pseudo-rótulos.

## Candidata atual: v2 multi-view e v3 weak visual

O kernel `jvlegend/rsna-knee-abnormality-detection-v2-multiview` seleciona uma série fluido-sensível por plano anatômico, agrega as vistas com EfficientNet-B0 e mistura o visual com o texto. A submissão `55365537` marcou `0,635` no público e é a referência congelada.

A v3 mantém essa arquitetura e treina o componente visual nos 4.407 estudos disponíveis. Os 58 rótulos oficiais recebem peso `1,0`; linhas sem rótulo só entram quando há menção lexical explícita e o professor textual está em `p≥0,85` ou `p≤0,15`, com peso máximo `0,10`. Não mencionar um achado nunca é tratado como normalidade. O worker privado está em execução; ainda não há score da v3.

Para medir a v0 sem usar rótulos do fold de validação:

```bash
python scripts/evaluate_baseline.py \
  --data-dir data/raw \
  --folds 5 \
  --seed 42 \
  --output reports/v0_report_metadata_cv.json
```

Para auditar o léxico multilíngue sem criar pseudo-rótulos:

```bash
python scripts/audit_weak_lexicon.py \
  --data-dir data/raw \
  --output reports/weak_lexicon_audit.json
```

Para testar a escala da feature lexical sem usar o leaderboard:

```bash
python scripts/evaluate_lexicon_weight_grid.py \
  --data-dir data/raw \
  --weights 0.5,1,2,4 \
  --seeds 42,2026
```

Para inspecionar uma série DICOM já baixada, sem enviar imagens ao Git:

```bash
python scripts/inspect_dicom_series.py \
  data/raw/train_series/<StudyInstanceUID>/<SeriesInstanceUID>
```

Para materializar o primeiro artefato visual 2.5D, usando três fatias por
série e sem alterar os DICOMs brutos:

```bash
python scripts/build_dicom_25d_features.py \
  --manifest data/processed/dicom_subset_manifest.json \
  --manifest data/processed/dicom_subset_manifest_v2.json \
  --data-dir data/raw \
  --output-dir data/processed/dicom_25d_v0
```

O resultado local é um `index.json` com 18 estudos e arrays compactos
`(3, 224, 224) uint8`; `data/processed/` permanece ignorado pelo Git.

Para extrair um embedding visual compacto, usando os pesos EfficientNet-B0 já
disponíveis no cache local e sem rede:

```bash
python scripts/extract_efficientnet_embeddings.py \
  --index data/processed/dicom_25d_v0/index.json \
  --output-dir data/processed/dicom_embeddings_efficientnet_b0 \
  --device cpu
```

O primeiro lote validado produziu uma matriz `(18, 1280)` `float32`, finita e
com 18 linhas distintas. O hash SHA-256 dos pesos usado nessa rodada fica
registrado no `index.json` local.

Na ampliação para os estudos rotulados, 34 das 40 séries adicionais já estão
completas. Somadas aos dois lotes anteriores, elas permitiram processar 52
estudos em 2.5D e gerar embeddings `(52, 1280)`. O baseline visual linear teve
macro-AUC `0,593222` (seed 42) e `0,584142` (seed 2026), média `0,588682`.
Esse resultado é preliminar: seis séries ainda têm 132 arquivos faltantes e a
seleção visual não deve ser tratada como comparação final com a v0.2 textual.

Como primeira fusão, combinamos as probabilidades do texto v0.2 com as do
classificador visual, usando peso visual `0,25` e textual `0,75`. No mesmo
subconjunto de 52 estudos, a fusão marcou macro-AUC `0,643647` (seed 42) e
`0,660061` (seed 2026), média `0,651854`, contra `0,638499` do texto sozinho.
Esse é o candidato multimodal atual; o peso só será congelado após completar
os 58 estudos.

Depois de materializar o índice completo dos estudos rotulados, o baseline
visual pode ser avaliado por alvo com regressão logística e validação
estratificada:

```bash
python scripts/evaluate_visual_embeddings.py \
  --data-dir data/raw \
  --index data/processed/dicom_embeddings_efficientnet_b0_labeled/index.json \
  --folds 5 \
  --output reports/visual_embeddings_cv.json
```

O relatório só deve ser interpretado quando houver variação suficiente de
positivos e negativos em cada alvo; o script reduz o número de folds quando
necessário e falha de forma explícita se um alvo não permitir AUC.

Para selecionar e baixar um lote visual pequeno, sem iniciar os ~569 GB:

```bash
python scripts/select_dicom_subset.py \
  --data-dir data/raw \
  --per-class 1 \
  --max-studies 24 \
  --output data/processed/dicom_subset_manifest.json

python scripts/download_dicom_subset.py \
  data/processed/dicom_subset_manifest.json \
  --data-dir data/raw \
  --dry-run
```

Remova `--dry-run` somente depois de conferir a estimativa; o downloader é
incremental e pula arquivos já existentes. Com Kaggle CLI `>=2.2.2`, ele usa a
listagem em árvore para consultar diretamente cada diretório de série, sem
varrer todos os arquivos da competição; em versões anteriores há um fallback
para a listagem plana, que pode sofrer rate limit nesse desafio.

Para lotes maiores, `--workers 4` permite downloads paralelos controlados, mas
a API da competição pode responder `429`; o comportamento padrão é serial e
qualquer reexecução pula arquivos já completos. Para uma aquisição mais
conservadora, use `--workers 1 --request-delay 1.5 --retry-attempts 6`.

Para ampliar a amostra sem repetir estudos de um lote anterior, gere outro
manifesto excluindo o primeiro:

```bash
python scripts/select_dicom_subset.py \
  --data-dir data/raw \
  --per-class 2 \
  --max-studies 24 \
  --exclude-manifest data/processed/dicom_subset_manifest.json \
  --output data/processed/dicom_subset_manifest_v2.json
```

Antes de subir o notebook, valide o contrato do CSV:

```bash
python scripts/validate_submission.py \
  --test data/raw/test.csv \
  --submission submissions/submission_v0_report_metadata.csv
```

O mesmo entrypoint pode ser usado dentro do notebook Kaggle, apontando `--data-dir` para `/kaggle/input/rsna-knee-abnormality-detection` e gravando `submission.csv` em `/kaggle/working/`. Para o smoke local da v3: `python kaggle/rsna_knee_v2_multiview.py --data-dir data/raw --weak-visual --device cpu --output /tmp/submission_v3_weak_local.csv`.

O kernel privado v3 foi publicado por um pacote temporário autocontido, com
GPU solicitada, pesos EfficientNet-B0 públicos e internet desligada. Como o
Kaggle Script executa apenas o arquivo indicado em `code_file`, a versão enviada
contém toda a implementação em um único entrypoint. Hoje o limite de cinco
submissões já foi atingido; o worker serve primeiro para validar tempo, dados e
CSV, não para enviar automaticamente ao leaderboard.

## Regras de publicação

- O repositório não contém DICOM, CSV da competição, pesos ou credenciais.
- Não enviar dados da competição para APIs externas.
- Qualquer peso ou dado externo usado deve ter licença e proveniência registradas.
- A submissão oficial desta competição é um notebook, com arquivo final chamado `submission.csv` e execução sem internet em até 9 horas.

## Referências

- [Página da competição](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
- [RSNA Knee MRI AI Challenge](https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge)
