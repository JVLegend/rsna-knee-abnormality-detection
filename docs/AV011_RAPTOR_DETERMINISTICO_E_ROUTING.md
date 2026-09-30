# AV-011 — Raptor determinístico e encaminhamento DINO

#RSNA #Kaggle #Pesquisa

15/09/2026 à noite São Paulo / 16/09 UTC. Fonte de verdade:
07_Estrategia_AVANCE do vault, [espelhada aqui](ESTRATEGIA_AVANCE.md).
Continua [AV-010](AV010_PARIDADE_FINAL_E_VARIACAO_RAPTOR.md).

## Resultado confirmado

Raptor36 ABBA determinístico v1, ID **134547620**, COMPLETE. A auditoria local
validou ambiente, versões, T4x2/56 hashes, 36 estudos/205 séries do treino V01,
108 pares receita-estudo por passagem e os quatro arquivos de previsões.
**IDs, imagens/máscaras, probabilidades dos quatro ramos e ranks idênticos**,
sem tolerância: delta máximo zero.

| Passagem | Modo | Tempo (s) |
|---|---|---:|
| A1 | Serial inicial | 287,4890 |
| B1 | Prefetch | 177,1051 |
| B2 | Prefetch | 175,4631 |
| A2 | Serial aquecido | 240,7954 |

Prefetch médio: 176,2841 s. Speedup ABBA 1,4984×; contra serial aquecido
1,3660×, ou **26,7909% menos tempo**. Ganho do Raptor, não do stack completo.
O primeiro serial inclui caches mais frios; ABBA não elimina toda variação.

Os quatro NPZs têm SHA:
`1f42760aec1b82bb875e0c456f9dd5b11a653546640f1daa29a91d7977e4c1f5`.
GPU reserved chegou a 7.639.924.736 bytes tanto em prefetch quanto no serial
final; não atribuir economia de memória ao prefetch. RSS é máximo acumulado.

Ambiente: torch2.10.0+cu128, CUDA12.8, cuDNN91002, T4, seed2026,
cudnn benchmark=False/deterministic=True, algoritmos determinísticos obrigatórios,
TF32 desligado e CUBLAS_WORKSPACE_CONFIG=:4096:8.
Isso demonstra paridade na sessão. Não prova que o autotuner causou sozinho
a divergência anterior, nem estabilidade entre sessões ou melhora de AUC.

Outputs: `reports/avance_av010_raptor_determinism_v1/`.
Auditoria: `reports/avance_av011_audit/determinism_v1.json`.

## Repetição em nova sessão

Kernel privado `jvlegend/rsna-knee-raptor36-deterministic-repeat`, v1,
**ID134548619**, RUNNING. Offline/T4/teto1.800s, mesmos 36 estudos/205 séries
e quatro ramos Raptor. Uma passagem prefetch, com os mesmos pesos, funções,
preparo e flags do ABBA. Cota antes do despacho: 19,0094 h disponíveis.
Não é submissão. Uma sessão nova não garante GPU física/host distinto.

Builder `prepare_h43_raptor_repeat.py` exige auditoria ABBA aprovada e eficiente,
verifica SHA do build anterior e preserva todas as células de modelo/preparo.
Muda somente descrição e execução final para uma passagem. Auditor
`assess_h43_raptor_repeat.py` compara ambiente/artefatos/IDs, imagens e
probabilidades/ranks com B2 predefinido; não usa allclose ou tolerância ajustada.

Build `reports/avance_av011_build/raptor36_deterministic_repeat_v1.ipynb`,
SHA `a8973f90f58870632ad7b7a5c9706e80412db4460d6b3459d24faceb947ba5fb`.
Fonte ABBA SHA:
`a8fa2aad9a4b436b72fa06b5e777701dc04729a252b2c2d58c8f9efcfb75350a`.

## Native DINO: teste de encaminhamento

O notebook stable36 tem uma única ocorrência literal de
submission_native_v38.csv: escrita do diagnóstico, célula37/linha43.
No trecho main de promoção, o arquivo principal é explicitamente substituído
pelo DINO público antes de retornar às etapas seguintes.

`audit_h43_native_routing.py` extrai somente esse trecho e seu validador do
build fixado por SHA. Inferência simulada, arquivos sintéticos em diretório
temporário e sem fallback de treino compilado. Testamos:

- native constante 0; 0,25; 0,75; 1: mesmo arquivo principal nos quatro casos;
- diagnóstico native preservado com os valores diferentes;
- público ausente ou com IDs inválidos: erro explícito, sem fallback native.

SHA comum do arquivo principal sintético:
`53b0d3c4a4759ceac3205868b3bb7a1877b261be763a666e1ab1f721b0090282`.
Recibo: `reports/avance_av011_audit/native_routing_v1.json`.
Classificação: diagnóstico no trecho auditado. Busca literal não é prova geral
de dependências dinâmicas. Não altera modelos, não mede AUC e não resolve Raptor.
**O gate do stack completo permanece inalterado.**

## Próxima retomada

Consultar ID134548619 sem duplicar. Após COMPLETE:

```sh
uvx --from kaggle kaggle kernels output jvlegend/rsna-knee-raptor36-deterministic-repeat \
  -p reports/avance_av011_raptor_repeat_v1 \
  --file-pattern 'e03_run0_raw.npz|h43_.*json|raptor_repeat_receipt.json|raptor_determinism_environment.json'
uv run --no-project --with numpy python -m scripts.assess_h43_raptor_repeat \
  --reference reports/avance_av010_raptor_determinism_v1 \
  --candidate reports/avance_av011_raptor_repeat_v1 \
  --output reports/avance_av011_audit/repeat_v1.json
```

Se houver paridade, preparar o par completo com a mesma receita determinística
e documentar o tratamento do diagnóstico native, antes de qualquer smoke.
Não reutilizar um smoke com flags diferentes das efetivamente validadas.
Se falhar, comparar ambiente/raw/inputs; não relaxar tolerância pelo resultado.

71 testes passaram. Manifesto V01 intacto:
`365566b0f830e398e78bd36bd9118a7365c3af340f588e774c8bcc351398b6df`.
Nenhuma nova avaliação de dev/confirmation, alteração de seleção final,
automação, recurso pago ou submissão. Melhor público H43 **0,939**;
probe22 ref56263721 segue PENDING, sem reenvio.
