# RSNA Knee Abnormality Detection

Baseline reprodutível para a competição [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection), mantido no HD externo e versionado sem dados da competição.

## Objetivo

Prever, para cada estudo de MRI do joelho, a probabilidade de 12 anormalidades. A métrica é a média macro de ROC-AUC nos 12 alvos.

O primeiro marco é uma submissão v0 baseada em laudo radiológico e metadados de séries. Depois acrescentaremos a informação visual dos DICOM em uma arquitetura 2.5D por estudo.

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

Para medir a v0 sem usar rótulos do fold de validação:

```bash
python scripts/evaluate_baseline.py \
  --data-dir data/raw \
  --folds 5 \
  --seed 42 \
  --output reports/v0_report_metadata_cv.json
```

Antes de subir o notebook, valide o contrato do CSV:

```bash
python scripts/validate_submission.py \
  --test data/raw/test.csv \
  --submission submissions/submission_v0_report_metadata.csv
```

O mesmo entrypoint pode ser usado dentro do notebook Kaggle, apontando `--data-dir` para `/kaggle/input/rsna-knee-abnormality-detection` e gravando `submission.csv` em `/kaggle/working/`.

## Regras de publicação

- O repositório não contém DICOM, CSV da competição, pesos ou credenciais.
- Não enviar dados da competição para APIs externas.
- Qualquer peso ou dado externo usado deve ter licença e proveniência registradas.
- A submissão oficial desta competição é um notebook, com arquivo final chamado `submission.csv` e execução sem internet em até 9 horas.

## Referências

- [Página da competição](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
- [RSNA Knee MRI AI Challenge](https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge)
