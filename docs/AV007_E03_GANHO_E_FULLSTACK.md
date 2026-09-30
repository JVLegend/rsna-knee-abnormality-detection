# AV-007 — ganho E03 isolado confirmado; teste completo iniciado

#RSNA #Kaggle #Pesquisa #Tecnologia

15/09/2026 à noite em São Paulo; evidências de 16/09 UTC.
Continuação de [AV-006](AV006_E03_PREFETCH.md), frente E03 do
[plano AVANCE](ESTRATEGIA_AVANCE.md).

## Resultado efetivamente observado

Kernel `jvlegend/rsna-knee-e03-prefetch-abba` v1, ID 134542682:
**COMPLETE**, `PASSED_EXACT_PARITY`. Download completo no HD em
`reports/avance_av007_e03_v1/`; os arquivos não estavam disponíveis nas
consultas iniciais da AV-006, portanto esta é a pasta efetiva da evidência.

| Passagem | Modo | Tempo do Raptor |
|---|---|---:|
| A1 | Serial, caches inicialmente mais frios | 123,55 s |
| B1 | Preparo antecipado de um estudo | 64,08 s |
| B2 | Preparo antecipado de um estudo | 63,37 s |
| A2 | Serial com caches aquecidos | 86,11 s |

Média B **63,73 s**. Speedup ABBA **1,645×**, mas parte dessa comparação
inclui aquecimento. Contra o serial final aquecido, **1,351×**, ou **25,99%
menos tempo**. O critério previamente fixado (≥1,05× em ambos) foi superado.
Só 12 estudos/70 séries, em uma sessão; não é ganho assegurado no ensemble
completo, no teste oculto ou em outros tamanhos de lote.

## Auditoria independente dos outputs

Implementado `scripts/assess_h43_e03.py`: lê quatro NPZs com pickle desativado,
confere SHA, shape 4×12×12 por ramo, 12×12 ranks, IDs oficiais do benchmark,
faixa/finitude, preflight/56 hashes/T4x2 e 36 pares receita-estudo por passagem.
Recalcula paridade, ordem ABBA, tempos e elegibilidade sem confiar apenas no
booleano produzido pelo notebook. `local_assessment.json` terminou
**VERIFIED_LOCAL_AUDIT**, paridade verdadeira.

- Volumes, máscaras, IDs, probabilidades dos quatro ramos e ranks exatamente
  iguais nas quatro passagens. Delta máximo de probabilidade **0,0**.
- Os quatro arquivos NPZ têm o mesmo SHA:
  `498c9ea93f3544d8babe47bd45ac7c4d0647ad0fe8f8879c38a4485737b61278`.
- Auditoria local SHA:
  `2af01d4529bfc8ccbb77dc64725163702170bd5e08180d44c871d589bc86566e`.
- CUDA reservado: A1 12,53 GB decimais; demais passagens 8,02 GB. Isso **não**
  demonstra redução de memória pelo prefetch: o serial final também usa 8,02 GB.
- RSS máximo cumulativo Linux chegou a 2.866.908 KiB (~2,73 GiB); não é pico
  individual de cada modo nem estimativa do ensemble completo.
- Duas threads Torch; Raptor usou uma GPU como na fonte. Nenhuma AUC medida.

O NumPy da auditoria foi carregado em ambiente efêmero do `uv`, sem editar
dependências ou configuração do projeto. Os arquivos grandes permanecem no HD.

## Próximo experimento iniciado — stack completo / 36 estudos

`scripts/prepare_h43_e03_fullstack.py` parte do benchmark de 36 estudos
fixado por SHA, não de uma versão remota atualizada arbitrariamente.
Apenas células **55 e 57** mudam: entrega dos estudos Raptor e conexão do
helper. DINO/A5/Rad, CoAt, pesos, geometria, IDs, cronômetros e gate final
permanecem iguais. Mesmo conjunto de treino V01: **36 estudos/205 séries**.

Critério antes do resultado:

1. Cinco ramos e 56 hashes íntegros, T4x2, CoAt sem fallback/falhas.
2. IDs, colunas, finitude e hash do CSV iguais ao benchmark anterior:
   `d774e25c03db117dc708851d8fb9eb588c722f92935377a117825bac674832ae`.
   Se houver diferença, investigar por alvo/componente antes de promover.
3. Comparar os tempos por etapa com DINO 201,31 s; A5 29,48 s; Rad 49,43 s;
   Raptor/CoAt/fusão 387,50 s, total 667,72 s. É uma comparação entre workers,
   não um ABBA do stack completo; diferenças de carga/hardware/cache limitam
   atribuição causal. Não afirmar ganho total só extrapolando o Raptor isolado.
4. Se íntegro e promissor, confirmar execução na versão dedicada ao teste real
   antes de eventual substituição operacional. Não criar submissão só para
   medir tempo nem mudar seleção final por esse benchmark.

Kernel privado:
[RSNA Knee E03 Fullstack36 Prefetch](https://www.kaggle.com/code/jvlegend/rsna-knee-e03-fullstack36-prefetch),
v1, ID **134543253**, último estado **RUNNING**, offline/T4x2, teto **3.600 s**.
Nunca submeter este notebook: usa casos de treino e publica apenas
`benchmark_predictions.csv`, além dos diagnósticos.

Build HD: `reports/avance_av007_build/h43_e03_fullstack36_v1.ipynb`, SHA
`7d2dc6c5f3538c451ae332c22ae92949b29370c1f02a82b3f86af6123285cb5c`.
Referência base SHA:
`835083352f0c0b7d8785794bb6846b6b2bd74a13846d5c68fbab7da713c4d614`.
Outputs esperados: `reports/avance_av007_fullstack_v1/`.

**39 testes passaram**, incluindo testes que recusam promoção por efeito
apenas de cache frio, falta de paridade, tempos/ordem/cobertura inválidos e
deriva do notebook. Manifesto V01 manteve SHA
`365566b0f830e398e78bd36bd9118a7365c3af340f588e774c8bcc351398b6df`.
Sem consulta à validação de desenvolvimento/confirmação.

## Submissões e retomada

Probe22 **56263721** permanece PENDING, sem score/erro informado. Não foi
reenviado. H43 parent **56253529 COMPLETE, 0,939** permanece melhor público;
H38 e seleção final preservadas. Nenhuma submissão ou automação nova.

No próximo AVANCE, consultar o submission_id e o kernel fullstack existentes,
sem duplicar. Depois de baixar o benchmark completo, rodar
`scripts/assess_h43_runtime.py`, comparar o CSV com a referência e conferir os
108 pares receita-estudo em `e03_fullstack_inputs.json`. O resultado isolado
não autoriza habilitar prefetch no candidato de produção por si só.
