# AV-004 — benchmark ampliado e confirmação H43

#RSNA #Kaggle #Pesquisa #Tecnologia

15/09/2026. Continuação de [AV-003](AV003_PILOTO_APROVADO_BENCHMARK.md),
A01 do [plano AVANCE](ESTRATEGIA_AVANCE.md).

## Benchmark observado

Kernel `jvlegend/rsna-knee-h43-runtime-benchmark`, v1, ID 134423935:
COMPLETE. Saída 36×13, 205 séries, IDs/ordem/schema/finitude/hash conferidos
por `scripts/assess_h43_runtime.py`. Todos os cinco ramos presentes,
lock de 56 artefatos, duas Tesla T4. CoAt: três checkpoints, zero fallback,
zero falhas, dois workers com returncode 0. Não medimos AUC neste lote.

| Etapa | Segundos medidos |
|---|---:|
| DINO | 201,31 |
| A5 | 29,48 |
| Rad | 49,43 |
| Raptor + CoAt + fusão | 387,50 |
| Total das etapas | 667,72 |

CoAt isolado: 102,29 s; pico reservado CUDA por worker aproximadamente
2,50 GB decimais. O conjunto Raptor/CoAt/fusão responde por 58% do tempo
medido e é o primeiro alvo de E03. Ainda não isolamos decode de inferência
nesse subtotal; não concluir que o gargalo é só GPU ou só DICOM.

## Orçamento e decisão

[Requisitos oficiais](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation)
revalidados em 15/09: Notebook-only, internet desligada, GPU/CPU até 9 horas,
arquivo `submission.csv`, fontes externas públicas permitidas.

Projeção com fator 1,25, preflight e mais 120 s fixos:

- 1.322 estudos hipotéticos: **8,57 h** (cabe, mas com pouca folga restante).
- 2.000 estudos hipotéticos: **12,94 h** (não cabe).

Esses tamanhos não foram confirmados como o teste oculto. A amostra de 36
estudos é sistemática por número de séries, não um limite estatístico de
runtime. Cache DINO bruto projetado em 10,01/15,14 GiB exclui outros arrays,
workers e modelos; RAM total não foi medida. Existe risco real de timeout.

Decisão: preparar uma única confirmação pública do parent fixo, com teto de
9h, sem remover membros ou retunar pesos para caber. Se o teste oculto falhar
por tempo/memória, seguir E03/A03 antes de repetir. O benchmark em treino
jamais deve ser submetido. Reprodução pública; OOF independente indisponível.
Componentes NC continuam limitando reutilização comercial; esta execução
não declara liberação comercial ou clínica.

## Versão dedicada

[RSNA Knee H43 Parent Strict Submission](https://www.kaggle.com/code/jvlegend/rsna-knee-h43-parent-strict-submission),
v1, ID **134484564**, privado, offline, duas T4 exigidas, timeout 32.400 s.
Mesma fonte e anexos auditados, código exatamente igual ao piloto aprovado:
`reports/avance_av002_build/h43_parent_strict_locked_v2.ipynb`, SHA-256
`a103e27720ba52f584c377ac6ffa470aad062c0a4e8558df22da06535a6b0306`.
Criado via API tipada; nenhum `kernel-metadata.json` foi editado.

Execução dedicada **COMPLETE**, último evento aos 281,49 s. Recibo
PASSED_PARENT_INTEGRITY, cinco ramos, lock 56, T4x2; CoAt 30,85 s, três
checkpoints, zero fallback/falhas, dois subprocessos com returncode 0.
CSV 3×13 conferido contra `data/raw/sample_submission.csv`: IDs/ordem,
schema, valores finitos em [0,1] e SHA exato do piloto aprovado:
`7d3b8bd4e76b309171e2c44a58301e71da26104d1049cc9f8c445c17eee3c770`.
Isso é paridade nos três exemplos, não desempenho no teste oculto.

**Submissão efetivamente enviada:** ref **56253529**, scriptVersionId
**350055640**, em **15/09/2026 11:47:46 UTC (08:47:46 São Paulo)**.
Último status **PENDING**, sem erro informado e sem score. A API confirmou
24 envios históricos, um hoje e quatro restantes. Único envio desta AVANCE.

CLI de cota antes do envio: 20,56 h GPU restantes; o retorno bruto da API
usa campos de duração inconsistentes com essa apresentação, portanto não
usamos esse snapshot como garantia de disponibilidade futura.

Artefatos do benchmark em `reports/avance_av004_benchmark_v1/`, incluindo
`runtime_assessment.json`. SHA das predições:
`d774e25c03db117dc708851d8fb9eb588c722f92935377a117825bac674832ae`.
Saída dedicada esperada em `reports/avance_av004_submission_v1/`.

## Retomada

Consultar a submissão **56253529**, sem reenviar enquanto PENDING/RUNNING.
O kernel privado executado nos exemplos já está COMPLETE; não confundir seu status com
a avaliação oculta. Se COMPLETE, comparar score com H38 0,929 e registrar
delta sem alterar a seleção final automaticamente. Se erro de tempo/memória,
usar a mensagem real para orientar E03/A03; não remover membros silenciosamente.
Score H43 ainda não observado. Os 22 testes locais passaram novamente e o
manifesto V01 manteve SHA
`365566b0f830e398e78bd36bd9118a7365c3af340f588e774c8bcc351398b6df`.
