# Entry point Kaggle

`rsna_knee_v0.py` é autossuficiente e pode ser colado como células de um notebook de competição. No notebook publicado, o código deve produzir exatamente `/kaggle/working/submission.csv`; não dependa de clone ou `pip install` com internet.

Para o primeiro envio, usar a candidata `v0.1` com `C=32`:

```python
submission = run(
    Path("/kaggle/input/rsna-knee-abnormality-detection"),
    Path("/kaggle/working/submission.csv"),
    c=32,
)
```

O entrypoint também aceita `--c 32` quando executado como script. A checagem de contrato roda antes de gravar o CSV.

Antes de enviar:

1. confirmar o caminho do dataset montado no notebook;
2. rodar o notebook do início ao fim sem internet;
3. validar número de linhas, UIDs, colunas, ausência de NaN e valores no intervalo `[0, 1]`;
4. fazer o commit do notebook e somente então usar `Submit`.

Para smoke local com dados em outra pasta, use `RSNA_DATA_DIR=/caminho/dos/dados python kaggle/rsna_knee_v0.py --output /tmp/submission.csv`.
