# H-34 — DINOv2 20-member rank ensemble

## Objetivo

Medir, em uma ablação limpa sobre o H-29 (`0,759`), se um ensemble público de 20
checkpoints DINOv2-S supera o único fine-tune DINOv2 do H-29. A variante usa
ranks por alvo, janelas sobrepostas, pooling focal por diagnóstico e TTA rígido
sem flip horizontal.

## Fontes e licenças

- Código-base de inferência: notebook público Apache 2.0, adaptado para retirar
  os braços RadImageNet/DINOv3 e preservar apenas a família DINOv2.
- Checkpoints: `pilkwang/rsna-knee-weights`, declarado `CC0-1.0` na API do
  Kaggle.
- Backbone: `metaresearch/dinov2/PyTorch/small/1`, modelo oficial MetaResearch
  Apache 2.0.
- O bundle privado `tonylica/rsna-knee-bend-dinov3-0917-repro-assets`, com
  licença `other` e aviso de não redistribuição, não é usado.
- RadImageNet `CC-BY-NC-SA-4.0` e checkpoints DINOv3 não são usados nesta
  variante.

## Critérios de aceitação

1. Kernel privado concluído em T4 sem internet.
2. Os 20 membros carregados com o fingerprint compatível.
3. Todos os estudos ocultos cobertos, sem NaN/duplicatas e com CSV 3×13 válido.
4. Submeter somente uma vez; promover apenas se o score público superar o H-29
   (`0,759`).

## Proveniência

Estratégia derivada da leitura de notebooks públicos da competição, especialmente
o ensemble DINOv2 rank-based de `ieshanmeghani/rsna-knee-v39-public-0-916-reproduction`
e suas fontes públicas. O score desses notebooks não é tratado como resultado
nosso até a submissão ser processada.
