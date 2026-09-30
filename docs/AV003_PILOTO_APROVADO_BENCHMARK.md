# AV-003 — piloto íntegro; benchmark ampliado

#RSNA #Kaggle #Tecnologia #Pesquisa

14/09/2026, noite de São Paulo (consultas em 15/09 UTC).
Continuação de [AV-002](AV002_PARENT_ESTRITO.md), experiência A01 do
[plano AVANCE](ESTRATEGIA_AVANCE.md).

## Resultado efetivamente observado

O piloto `jvlegend/rsna-knee-h43-parent-strict-pilot`, versão 1, ID 134328295,
terminou **COMPLETE**, com `PASSED_PARENT_INTEGRITY`.

- Duas Tesla T4 confirmadas; 56 hashes fixados e conferidos no worker.
- 20 fingerprints DINO passaram e 20 membros foram contabilizados, com
  10 janelas por membro no voto public-frontier.
- A5 carregou cinco folds DINOv3 sem keys ausentes; predições completas.
- RadImageNet executou os dois layouts E13 e o calibrador.
- Quatro views Raptor executadas: v5, v5 reverse, v10 Native384Dense e v8.
- CoAt residual: três checkpoints e4/e6/e8, zero `fallback_studies`, lista
  de falhas vazia, dois subprocessos com returncode 0.
- CSV **3 × 13**, IDs/ordem/colunas/finitude verificados novamente localmente,
  SHA igual ao recibo: `7d3b8bd4e76b309171e2c44a58301e71da26104d1049cc9f8c445c17eee3c770`.
- Gate final apareceu aos **289,22 s**; último evento do log aos **301,06 s**.
  CoAt levou 32,92 s no recibo; Raptor registrou cerca de 55 s. Não extrapolar
  esses três exemplos para todo o teste oculto sem benchmark maior.

Isso valida integração e execução nos três exemplos, **não AUC, score 0,939,
generalização independente ou ausência de todo fallback por fatia**. As
receitas de decode herdadas ainda precisam ser avaliadas por cobertura.
Artefatos locais: `reports/avance_av003_pilot_v1/` no HD externo.

## Próximo teste iniciado — 36 estudos / 205 séries

Notebook privado: [RSNA Knee H43 Runtime Benchmark](https://www.kaggle.com/code/jvlegend/rsna-knee-h43-runtime-benchmark).
Versão remota **1**, ID **134423935**; última consulta **RUNNING**.
Início registrado pela API: `2026-09-15 02:30:47.867 UTC`.
T4 solicitada, código exige duas T4 reais; internet desligada, limite 3.600 s.

Amostra determinística por quantis da quantidade de séries dentro dos **299
estudos de treino da V01**, cobrindo 4–11 séries por estudo. São 36 estudos,
205 séries completas do mount Kaggle. Não usa labels/desempenho para escolher
os casos e não consulta os 250 de desenvolvimento nem os 150 de confirmação.
Não é uma amostra aleatória representativa de scanner/site/patologia nem
garante os casos extremos de resolução/número de fatias do teste oculto.

O adaptador cria um root de benchmark em `/kaggle/working/` com links apenas
para os estudos selecionados. Não modifica o dataset original e redireciona
explicitamente os quatro leitores: DINO, A5, Raptor e subprocesso CoAt.
Os rótulos não são usados para medir AUC. A EDA é desativada, as receitas dos
modelos e os pesos do parent permanecem iguais. Cronômetros separados medem
DINO, A5, Rad e Raptor/CoAt/fusão.

**Este kernel nunca deve ser submetido à competição**: suas linhas são casos
de treino. O resultado se chama `benchmark_predictions.csv`, não
`submission.csv`. O recibo marca `runtime_benchmark_only_no_auc_no_submission`.

## Código e testes

- [prepare_h43_benchmark.py](../scripts/prepare_h43_benchmark.py): seleção,
  adaptação dos leitores, instrumentação e build sem publicar remotamente.
- [assess_h43_runtime.py](../scripts/assess_h43_runtime.py): lê recibos completos,
  valida a saída e projeta cenários explicitamente hipotéticos de 1.322/2.000
  estudos. Esses números **não são tamanho confirmado do teste oculto**.
- Projeção usa tempo por estudo incluindo carregamentos, acréscimo fixo de
  120 s além do preflight e fator 1,25. A estimativa de cache DINO exclui
  decoder, workers, modelos e demais arrays: não é pico de RAM medido.
- Nenhum cenário autoriza envio automaticamente. O limite histórico de 9h
  deve ser revalidado. Nesta rodada, as páginas públicas de rules/code-requirements
  abriram sem conteúdo textual na ferramenta web; não tratamos isso como
  confirmação nova das regras.
- **22 testes passaram**: 16 anteriores + 3 de seleção/roteamento do benchmark
  + 3 de cálculo/rejeição de projeções inválidas.

Build no HD: `reports/avance_av003_build/h43_benchmark36_timed_v2.ipynb`, SHA:
`835083352f0c0b7d8785794bb6846b6b2bd74a13846d5c68fbab7da713c4d614`.
`v2` é revisão local, não a versão remota. Manifesto V01 permanece com SHA
`365566b0f830e398e78bd36bd9118a7365c3af340f588e774c8bcc351398b6df`.

## Retomada

1. Recuperar o benchmark v1 existente em `reports/avance_av003_benchmark_v1/`.
   Não duplicar enquanto estiver em execução.
2. Se COMPLETE, exigir `h43_parent_integrity.json`, `h43_benchmark_selection.json`,
   `h43_benchmark_timings.json` e `benchmark_predictions.csv` consistentes.
3. Rodar a análise de runtime; se a projeção não couber com margem, priorizar
   E03 (eficiência) ou A03 (ramo Native384Dense), sem descartar membros silenciosamente.
4. Só depois preparar uma versão dedicada ao teste real, com orçamento e
   regras verificados. Não enviar o piloto de 30 minutos nem o benchmark.

```sh
uvx --from kaggle kaggle kernels status jvlegend/rsna-knee-h43-runtime-benchmark
python3 scripts/assess_h43_runtime.py --directory reports/avance_av003_benchmark_v1 --output reports/avance_av003_benchmark_v1/runtime_assessment.json
```

Snapshot de cota antes do benchmark: 20,96 h GPU disponíveis. Nenhuma
submissão nova e nenhum score novo; H-38 permanece referência pública 0,929.
