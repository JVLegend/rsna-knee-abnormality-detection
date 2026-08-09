# Entry point Kaggle

`rsna_knee_v0.py` é autossuficiente e pode ser colado como células de um notebook de competição. No notebook publicado, o código deve produzir exatamente `/kaggle/working/submission.csv`; não dependa de clone ou `pip install` com internet.

Para o primeiro envio, usar a candidata `v0.2`, que combina `C=32` com
features lexicais auditáveis do laudo:

```python
submission = run(
    Path("/kaggle/input/rsna-knee-abnormality-detection"),
    Path("/kaggle/working/submission.csv"),
    c=32,
    use_lexicon=True,
)
```

O entrypoint também aceita `--c 32 --use-lexicon` quando executado como
script. Sem argumentos, `rsna_knee_v0.py` já executa a v0.2 (`C=32` e léxico
ligado); use `--no-use-lexicon` para reproduzir a variante sem léxico. A
checagem de contrato roda antes de gravar o CSV. O caminho solicitado é
`/kaggle/input/rsna-knee-abnormality-detection`; se o Kaggle montar a fonte
com outro nome, o script procura `train.csv` e `test.csv` nas primeiras
camadas de `/kaggle/input` e imprime o caminho efetivamente usado.

O candidato anterior `v0.1` continua reproduzível com `use_lexicon=False`.

## Candidata visual v1

`rsna_knee_v1_visual.py` preserva o baseline textual e acrescenta uma fusão
com embeddings EfficientNet-B0 ImageNet extraídos de uma representação 2.5D
(fatias em 25%, 50% e 75% da série DICOM preferida). A validação local com 58
estudos rotulados foi preliminar; o melhor ponto explorado até aqui usa
`visual_weight=0.4` e regularização visual `C=0.1`.

O código não baixa pesos nem instala dependências durante a execução. No
Kaggle, os pesos devem ser montados pelo dataset público
`jvlegend/efficientnet-b0-imagenet-weights-public`, criado a partir do arquivo
oficial/cache local do torchvision. A fonte é mantida separada do repositório
para que o GitHub não receba um binário de modelo.

Smoke local:

```bash
python kaggle/rsna_knee_v1_visual.py --data-dir data/raw --output /tmp/submission_visual.csv --device cpu
```

O kernel `jvlegend/rsna-knee-abnormality-detection-v1-visual`, versão 2,
foi executado com sucesso e a submissão `55364637` marcou score público
`0,607`, acima do baseline `55364182` (`0,505`).

## Candidata visual v1.1 targetwise

`rsna_knee_v1_targetwise.py` mantém o mesmo modelo visual, mas calibra a
fração visual por alvo. Os pesos foram congelados antes do próximo envio a
partir da média de dois seeds no CV local: mais imagem para ACL, MCL e
Effusion; menos imagem para OA medial, PF OA e Synovitis. No subconjunto de
52 estudos, a média foi `0,680149` contra `0,657018` do blend global `0,4`.
Como a seleção foi feita no mesmo conjunto pequeno, a próxima submissão será
tratada como experimento controlado, não como solução final.

Após o score targetwise completo (`0,582`), a candidata `v1.2` aplica apenas
`λ=0,25` da correção por alvo sobre o blend global `0,4`. No CV local, essa
versão marcou `0,654588`/`0,669523` nos seeds 42/2026, média `0,662055`.

## Candidata visual v2 multi-view

`rsna_knee_v2_multiview.py` abandona a hipótese de que uma única série
representa o estudo. Para cada estudo, seleciona até uma série fluido-sensível
por plano (sagital, coronal e axial), extrai o embedding 2.5D de cada uma e
faz a média dos vetores antes da regressão logística e do blend global `0,4`.
O `train_series.csv` tem média de `5,8` séries por estudo e os três planos
estão presentes nos 58 estudos rotulados; portanto, este experimento só pode
ser validado adequadamente no worker Kaggle, que possui o conjunto completo.
O smoke local confirmou o fallback para os DICOMs já baixados, mas não é uma
validação de ganho porque o HD ainda contém essencialmente uma série por
estudo.

Antes de enviar:

1. confirmar o caminho do dataset montado no notebook;
2. rodar o notebook do início ao fim sem internet;
3. validar número de linhas, UIDs, colunas, ausência de NaN e valores no intervalo `[0, 1]`;
4. fazer o commit do notebook e somente então usar `Submit`.

Para smoke local com dados em outra pasta, use `RSNA_DATA_DIR=/caminho/dos/dados python kaggle/rsna_knee_v0.py --output /tmp/submission.csv`.

## Kernel privado

`kernel-metadata.json` aponta para `rsna_knee_v0.py`, mantém a internet
desligada e associa a competição como fonte de dados. Para publicar uma nova
versão e iniciar a execução:

```bash
kaggle kernels push -p kaggle -t 3600
kaggle kernels status jvlegend/rsna-knee-abnormality-detection-v0-2
```

Se o editor mostrar a competição no painel `Input`, mas o worker ainda falhar
sem `train.csv`, abrir `Edit` → `Add Input`, pesquisar a competição, remover e
adicionar novamente a fonte e usar `Save Version` com `Save & Run All (Commit)`.
O código agora reporta as entradas disponíveis em `/kaggle/input` quando a
autodetecção também falha. Não enviar nada ao leaderboard enquanto esse gate
não passar.
