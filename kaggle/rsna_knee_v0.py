"""Entry point para copiar para um notebook Kaggle.

No notebook final, execute este arquivo ou copie suas células e mantenha o
arquivo produzido com o nome /kaggle/working/submission.csv.
"""

from pathlib import Path
import sys

DATA_DIR = Path("/kaggle/input/rsna-knee-abnormality-detection")
OUTPUT = Path("/kaggle/working/submission.csv")

PROJECT_SRC = Path("/kaggle/working/rsna-knee-abnormality-detection/src")
if PROJECT_SRC.exists():
    sys.path.insert(0, str(PROJECT_SRC))
else:
    sys.path.insert(0, "/kaggle/input/rsna-knee-abnormality-detection-code/src")

from rsna_knee_baseline.data import load_competition_tables
from rsna_knee_baseline.model import KneeReportBaseline


tables = load_competition_tables(DATA_DIR)
model = KneeReportBaseline()
model.fit(tables["train"], tables["train_series"])
submission = model.predict(tables["test"], tables["test_series"])
submission.to_csv(OUTPUT, index=False)
print(f"submission gravada em {OUTPUT} com {len(submission)} linhas")
