# H-37 — DINOv3 + CoAtNet em rank por alvo

## Objetivo

Testar se um braço DINOv3 independente melhora o novo baseline H-36
(CoAtNet Max-Span, `0,928`) quando os dois são combinados em rank por alvo.
O candidato é uma fusão de famílias visuais, não uma reutilização de output
de kernel privado.

## Contratos

- DINOv3: cinco folds públicos `m_f0.pt`–`m_f4.pt`, `vit_small_patch16_dinov3.lvd1689m`,
  336 px, crop físico de 130 mm, seis slots, 16 fatias por slot e pooling
  `xcodex`.
- CoAtNet: checkpoint público H-36, cinco slots, 64 fatias, span de 2–98%,
  crop físico de 140 mm, janelas sobrepostas e rank-percentile.
- Fusão final: ranks por coluna, com o arm DINOv3 como pai e o braço CoAtNet
  como residual de 50% por alvo (a mesma política pública do DINOsaur V4.5).

## Fontes e licenças

- `mattiaangeli/knee-mri-fold-weights`: CC0-1.0.
- `dreaddevelopment/raptor-knee-maxspan`: CC0-1.0.
- A lógica de inferência foi adaptada de notebooks públicos Apache-2.0,
  preservando atribuição e sem anexar fontes privadas.
- A versão H-37 não usa RadImageNet, DINOv3 hospedado de origem incerta,
  outputs de kernels, texto do teste ou dados derivados da competição.

## Critério de decisão

1. Rodar sem internet em T4.
2. Exigir cobertura integral, schema exato, IDs únicos, valores finitos e
   nenhum estudo oculto descartado silenciosamente.
3. Submeter somente depois de conferir o CSV e confirmar a ação final.
4. Promover apenas se superar H-36 `0,928`; se não superar, manter H-36 como
   produção e guardar o H-37 apenas como diagnóstico de diversidade.

#Kaggle #Tecnologia #Academia #JoaoVictor
