# H-35 — rank stack CPU sobre saídas validadas

## Objetivo

Testar uma combinação de baixo risco entre submissões já executadas, sem decodificar
DICOM e sem usar quota GPU. O H-29 recebe peso `0,80`; H-27, `0,15`; H-33, `0,05`.
As predições são convertidas em percentis por alvo antes da média, pois o critério da
competição é ROC AUC.

## Controle de promoção

- H-29 continua sendo o fallback oficial e precisa estar montado como kernel source.
- O script aceita somente CSVs com o schema completo, os mesmos UIDs do `test.csv`,
  valores finitos e faixa `[0,1]`.
- A execução é CPU-only e deve produzir `submission.csv` sem consumir a quota GPU.
- O resultado só substitui H-29 se o score público superar `0,759`; caso contrário,
  permanece uma ablação registrada.

## Proveniência e privacidade

Os três kernel sources são nossos outputs privados já concluídos. Nenhum DICOM, CSV de
competição ou predição é armazenado neste repositório. O H-35 é uma ponte operacional
enquanto o H-34 DINOv2 CC0 aguarda a renovação da quota.

#RSNA #Kaggle #Tecnologia #Pesquisa
