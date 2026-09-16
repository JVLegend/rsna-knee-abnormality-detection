# H-40 — WideDense residual sobre H-38

## Objetivo

Testar a complementaridade do checkpoint CoAtNet WideDense v4, usado como
braço auxiliar no DINOsaur V10 público, sobre a referência H-38 (`0,929`). O
H-38 permanece como âncora; o WideDense só recebe pesos pequenos e
target-wise, com redução por correlação.

## Fontes auditadas

- `dreaddevelopment/raptor-knee-widedense`: checkpoint CoAtNet v4, licença
  `CC0-1.0`, SHA-256 local `d8bb0f8751b4bb65750257869ddc4c7a3c0cdfc6e62596fc68193919406c53eb`.
- `dreaddevelopment/raptor-knee-maxspan`: H-38/H-36, `CC0-1.0`.
- `mattiaangeli/knee-mri-fold-weights`: DINOv3 residual da H-38, `CC0-1.0`.

## Contrato de execução

O kernel roda sem internet, usa T4 e reproduz a etapa H-38 antes do residual.
O WideDense usa o mesmo protocolo CoAtNet MaxSpan (cinco slots, crop físico,
62 janelas). Se o checkpoint não montar, a cobertura cair ou qualquer erro
ocorrer, `/kaggle/working/submission.csv` volta ao H-38.

O artefato `submission_widedense.csv` só é candidato. Não submeter sem
verificar schema, IDs, finitude, faixa e mudança efetiva em relação ao H-38.

## Resultado esperado

Como o teste público tem somente três estudos, muitos blends por rank podem
ser byte a byte idênticos. O critério é observar se a ordem realmente muda e
preservar H-38 caso não mude.

## Resultado da execução

A versão 3 concluiu com cobertura `1.000`, mas o residual WideDense sobre a
âncora H-38 produziu exatamente o CSV do H-36 (`SHA-256 fc8b32d5964f54619ef358b8cf97291012806c16aea6fe8db799c1d3279829b7`). A saída difere do H-38 em 24 células e foi descartada; não houve submissão.

#RSNA #Kaggle #Tecnologia #Academia
