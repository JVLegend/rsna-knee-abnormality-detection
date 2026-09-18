# AV-010 — paridade final e variação intermediária Raptor

#RSNA #Kaggle #Pesquisa

Evidência: 15/09/2026 à noite São Paulo / 16/09 UTC.
Fonte de verdade: 07_Estrategia_AVANCE no vault, [espelhada aqui](ESTRATEGIA_AVANCE.md).
Continua [AV-009](AV009_DINO_ESTAVEL_STACK_PAREADO.md).

## Resultado do par completo

Serial v1ID134546480 e prefetch v1ID134546487 COMPLETE; 36 estudos/205 séries,
cinco ramos,56 hashes/T4x2, CoAt sem falhas/fallback. Ambos os CSVs finais:
`87ee1ac74c3e58d7dadfef705634eb23745f2322ad002e7772b6d03bad2ee777`.
Nenhuma diferença final nos 12 alvos. Isso não mede AUC nem garante paridade oculta.

| Etapa | Serial (s) | Prefetch (s) |
|---|---:|---:|
| DINO | 198,5515 | 189,6312 |
| A5 | 32,3989 | 29,8285 |
| Rad | 57,4264 | 49,2260 |
| Raptor/CoAt/fusão | 421,6460 | 337,2915 |
| Soma das etapas | 710,0229 | 605,9772 |

Redução total14,654%; subtotal Raptor/CoAt/fusão20,006%. Workers diferentes,
não atribuir todas as diferenças ao prefetch. Cenários com margem: 7,78h para
1.322 casos hipotéticos e 11,75h para2.000; tamanho oculto não confirmado.

## O que passou e o que falhou

- DINO público: CSV idêntico e reprodução independente da regra inteira nos
  dois workers. Raw canônico também idêntico:
  `fba28692e7d32288f5257870fdbcfa3d17f60fa68e1821ee212fd15055251107`.
- Capturas DINO têm hashes de arquivo diferentes pela ordem dos membros:
  serial `107beae1ee2f566c7a75581661274a0c42edc9a312efbadc073d1f4addcca19f`;
  prefetch `4b8f0f9f524a8160ef6252a30e52c84415ca5cf5b42b734fd76b060cf03bf3c2`.
  Auditor alinha por membro/estudo antes de comparar.
- Raptor: todos os108 contratos de imagem/máscara idênticos; raw diverge
  em1727/1728 probabilidades, delta máximo0,00040036439895629883.
  Dois ranks Baker's trocam (delta1/35 no raw). O CSV híbrido _raptor.csv
  também muda dois valores Baker's, delta1/36.
- Native DINO: dois valores Medial OA mudam, delta1/72. Ainda exige
  classificação do fluxo de consumo/causa, sem alterar automaticamente a
  função ponderada. Não confundir esse diagnóstico com o DINO público corrigido.
- CoAt, legacy-fold, E10, parent e CSV final idênticos.

**Gate FAILED_STABLE_FULLSTACK_PARITY.** A fusão final absorveu as diferenças
nesta amostra; isso não demonstra que seriam inofensivas no oculto.
Não flexibilizamos o gate depois de observar o resultado.
Auditoria no HD: `reports/avance_av010_audit/assessment_v2.json`.
Outputs: `reports/avance_av009_serial_v1/` e `reports/avance_av009_prefetch_v1/`.

## Diagnóstico em andamento

O código Raptor fixa `torch.backends.cudnn.benchmark=True`, habilita
`allow_tf32` e usa autocast fp16. Backend/autotuning é hipótese para a
diferença entre workers, não causa demonstrada. Inputs iguais isolam a
preparação das imagens, mas não provam que todos os detalhes do forward sejam
determinísticos entre execuções.

Novo kernel privado:
`jvlegend/rsna-knee-raptor36-deterministic-abba`, v1 **ID134547620**, RUNNING.
Offline/T4/teto1.800s; só os quatro ramos Raptor, mesmos36 estudos/205 séries.
Sequência serial–prefetch–prefetch–serial na mesma sessão GPU, com:

- cudnn benchmark=False, deterministic=True;
- algoritmos determinísticos obrigatórios (erro explícito se incompatível);
- matmul/cudnn TF32 desligados;
- CUBLAS_WORKSPACE_CONFIG=:4096:8 antes do preflight;
- seed2026 e recibo das versões torch/CUDA/cuDNN e flags.

Pesos, funções de modelo e decode preservados. Flags alteram a receita
numérica; não é promoção nem reprodução byte a byte do parent anterior.
Mesmo passando, este teste só demonstra paridade dentro da sessão; não
identifica isoladamente o autotuner nem comprova repetibilidade entre workers.

Build: `reports/avance_av010_build/raptor36_deterministic_abba_v1.ipynb`,
SHA `a8fa2aad9a4b436b72fa06b5e777701dc04729a252b2c2d58c8f9efcfb75350a`.
Cota consultada antes do despacho:19,2727h disponíveis; exigido saldo≥1.800s.
Nenhum recurso pago, nova automação ou nova submissão.

## Código e retomada

66 testes passaram. Suporte explícito36 foi acrescentado ao builder/auditor
ABBA, mantendo12 como padrão. Auditor stable36 agora registra deltas raw.
Novos builder/auditor determinísticos e testes de flags/cobertura.
Manifesto V01 intacto:
`365566b0f830e398e78bd36bd9118a7365c3af340f588e774c8bcc351398b6df`.

`prepare_h43_stable_smoke.py` foi implementado, mas **nenhum smoke foi
gerado ou executado**: o gate foi testado com a auditoria real e recusou a
promoção. Quando elegível, usará somente roots oficiais e saída
smoke_predictions.csv; não produz submission.csv. Não contornar essa proteção.

Próximo AVANCE: consultar o kernel existente, baixar após COMPLETE:

```sh
uvx --from kaggle kaggle kernels output jvlegend/rsna-knee-raptor36-deterministic-abba \
  -p reports/avance_av010_raptor_determinism_v1 \
  --file-pattern 'e03_.*|h43_.*json|raptor_determinism_environment.json'
uv run --no-project --with numpy python -m scripts.assess_h43_raptor_determinism \
  --directory reports/avance_av010_raptor_determinism_v1 \
  --output reports/avance_av010_audit/determinism_v1.json
```

Se ERROR, ler log antes de alterar/repetir. Se paridade passar, medir custo
e preparar repetição entre workers antes de integração completa. Resolver/
classificar também o native DINO; não fingir que a correção pública o alterou.
Se falhar, localizar raw/flags/ops e não relaxar tolerâncias pelo resultado.
Consultar probe22 ref56263721 sem duplicar; H43 parent0,939 segue melhor
confirmado. Nenhuma avaliação nova de dev/confirmation ou seleção final.
