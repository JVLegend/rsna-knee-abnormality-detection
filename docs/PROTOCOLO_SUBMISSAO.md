# Protocolo de submissão

#RSNA #Kaggle #Importante #Tecnologia

## Antes de publicar o notebook

- [ ] A conta aceitou as regras e está inscrita na competição.
- [ ] O notebook roda do início ao fim com internet desligada.
- [ ] O tempo total fica abaixo de 9 horas no hardware escolhido.
- [ ] Não há caminho local do HD externo, credencial, segredo ou arquivo fora de `/kaggle/input` no notebook.
- [ ] Não há DICOM, CSV da competição ou peso treinado incluído no repositório público.
- [ ] Licenças de bibliotecas, modelos e dados externos estão registradas.

## Validação do arquivo

```python
import numpy as np
import pandas as pd

target_columns = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
    "Baker's", "Contusion", "Fracture",
]

submission = pd.read_csv("/kaggle/working/submission.csv")
test = pd.read_csv("/kaggle/input/rsna-knee-abnormality-detection/test.csv")

assert list(submission.columns) == ["StudyInstanceUID", *target_columns]
assert submission["StudyInstanceUID"].astype(str).tolist() == test["StudyInstanceUID"].astype(str).tolist()
values = submission[target_columns].to_numpy(dtype=float)
assert np.isfinite(values).all()
assert ((values >= 0) & (values <= 1)).all()
assert not submission["StudyInstanceUID"].duplicated().any()
```

## Envio e registro

1. Fazer commit do notebook no Kaggle.
2. Confirmar que `submission.csv` está no diretório de saída.
3. Enviar uma única hipótese identificada por versão (`v0_report_metadata`, `v1_weak_supervision`, etc.).
4. Registrar data, commit, hash do notebook, tempo, memória, score público e observações em `docs/LOG_EXPERIMENTOS.md`.
5. Preservar o melhor modelo por validação local e o melhor resultado público como referências separadas.

## Limites

- Máximo de 5 submissões por dia.
- Até 2 submissões finais selecionadas.
- Entrada e fusão de equipes até 15/10/2026.
- Submissão final até 22/10/2026.
