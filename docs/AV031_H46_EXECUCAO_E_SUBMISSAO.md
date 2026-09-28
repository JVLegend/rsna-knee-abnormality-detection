# AV-031 — execução e submissão H46

#JoaoVictor #Kaggle #Tecnologia #Academia

Data: 28/09/2026.

## Resultado

A receita pública H46 `speedy` foi executada no Kaggle com T4×2, auditada e
submetida como notebook-only. A primeira submissão, ref **56640374**, notebook
`jvlegend/rsna-knee-h46-speedy-fixed-v1`, versão 2, terminou com
`Notebook Threw Exception` na reexecução oculta e **não recebeu pontuação**.
O Kaggle não expôs o traceback oculto; a versão pública continua `COMPLETE`
em 281,6 s. A versão 3 corrigida passou na execução e auditoria públicas e foi
submetida como ref **56652369**; neste registro, a reexecução oculta está
`PENDING`/`Notebook Running`. A melhor pontuação nossa confirmada continua
**0,941**; o 0,943 é a referência declarada pela fonte pública e não foi
atribuído à nossa execução.

## Identidade da candidata

- Notebook construído: `reports/avance_av031_h46/candidate_v2.ipynb`.
- SHA-256: `71e22cce751a5fc1e379180a117aa73daa7933ac2f238f23d31bfa332f9f51ef`.
- Kernel: 136214178, versão 2, `COMPLETE`.
- Runtime visível: 281,6 s; pipeline final: 198,612 s no teste público de três estudos.
- Ambiente: Python 3.12.13, PyTorch 2.10.0+cu128, CUDA 12.8, duas Tesla T4.
- Quota antes do lançamento: limite 108.000 s, reserva zero, uso restante
  informado pelo launcher 104.029,052149 s; reset 03/10/2026 00:00 UTC.

## Auditoria de execução

O recibo `h46_release_gate.json` registrou
`PASSED_H46_RELEASE_GATE_NOT_SCORE_VALIDATION` com:

- 20 membros DINOv2, todos com fingerprint aprovado;
- 5 folds A5 estritamente carregados;
- 4 vistas Raptor;
- família CoAt completa: `resgated_top3`, `global96_top3` e `d4_swa3`;
- pesos iguais de 1/3 dentro da família;
- redução `rank_of_member_probability_mean`;
- `score_reproduced=false` e paridade privada não verificada.

O `submission.csv` tem SHA-256
`7c6dfe8ba6c71d557d2a6b21af8a96bddfeb4b75626ee38a7a8cae44ec80c23d`,
igual ao recibo final `btkd_v559_complete.json`. A validação independente
confirmou três linhas, 12 alvos, IDs/ordem/colunas idênticos ao
`sample_submission.csv`, IDs únicos, valores finitos e intervalo [0,1].

## Incidente controlado da versão 1

A versão 1 completou DINO, A5, RadImageNet, Raptor e os três ramos CoAt, mas
nosso gate final consultou `len(models)` depois que o notebook restaurou uma
variável homônima anterior. O gate rejeitou incorretamente a contagem A5.

A versão 2 passou a contar `len(_A5_LOAD_RECEIPT)` e exige explicitamente os
folds 0–4. Essa correção não alterou pesos, imagens, preset, blending ou
aritmética. A repetição foi permitida apenas para essa candidata auditada.

## Submissão

- Referência: **56640374**.
- Descrição: `AV031 H46 speedy fixed v2: 20 DINO + 5 A5 + 4 Raptor + CoAt family (resgated/global96/d4); strict 109-asset and release gates; rank of member probability mean.`
- Status antes da reconciliação: `PENDING`.
- Resultado final: `ERROR — Notebook Threw Exception`; score público nulo.
- Submissões restantes no dia após o envio: 4.
- A ref foi reconciliada e não deve ser reenviada como se estivesse pendente.

## Diagnóstico e candidata v3

A diferença executável entre a fonte pública H46 e a nossa v2 foi auditada.
Além dos pins/recibos, a v2 transformava qualquer evento de reparo por estudo
(`fallback`, `neutral`, `partial`, `unreadable` etc.) em exceção fatal no gate
final. Isso é incompatível com o runtime original, que registra esses eventos,
preenche a linha de forma finita e continua. Com três estudos públicos não
houve reparos; no conjunto oculto de 1.322 estudos basta um DICOM atípico para
explicar o padrão observado. Como o traceback oculto não é fornecido, esta é a
causa mais provável, não uma exceção textualmente confirmada pelo Kaggle.

A v3 foi construída em
`reports/avance_av032_h46/candidate_v3.ipynb`, SHA-256
`9c9924f9ec55ebf193661e5b96e7f072b988e285f5a1d2a1cfc91e52d7cf58eb`.
Ela mantém pesos, imagens, preset e aritmética; bloqueia perda de membros,
schema/composição incorretos e valores inválidos, mas registra reparos por
linha em `warning_event_counts` sem abortar a submissão inteira.

Após a liberação de uma vaga GPU, o kernel 136214178, versão 3, terminou
`COMPLETE` em T4×2. O pipeline registrou 194,159 s e o gate
`PASSED_H46_SOURCE_COMPATIBLE_GATE_NOT_SCORE_VALIDATION`: 20 DINO, 5 A5,
4 vistas Raptor, os três membros CoAt esperados, pesos internos de 1/3 e
redução `rank_of_member_probability_mean`. Os três recibos CoAt tiveram
`fallback_studies=0`; o gate registrou `warning_event_counts={}` no teste
visível. O CSV 3×13 permaneceu finito, completo e com SHA-256
`7c6dfe8ba6c71d557d2a6b21af8a96bddfeb4b75626ee38a7a8cae44ec80c23d`.

Com confirmação do JV no ato, a versão 3 foi submetida à competição:

- Referência: **56652369**.
- Script version: **353696869**.
- Arquivo: `submission.csv`.
- Estado inicial confirmado pela UI e API: `Notebook Running` / `PENDING`.
- Submissões restantes informadas antes do envio: 4; o envio consumiu uma.

## Verificação local

`PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m pytest -q tests`

Resultado após a correção: **266 testes + 44 subtestes passaram**.

## Próxima decisão

Aguardar a reexecução oculta da ref **56652369** sem reenviar ou duplicar o
candidato. Quando o Kaggle concluir, registrar status, score público e eventuais
erros, comparando com H43A 0,941. Não usar o teste público de três estudos, o
score publicado de terceiros ou os recibos de integridade como substitutos da
avaliação no leaderboard.
