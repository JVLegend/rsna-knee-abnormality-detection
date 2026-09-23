# H-36 — CoAtNet Max-Span standalone

## Objetivo

Testar uma família visual independente do H-34: CoAtNet RMLP 2 RW em 384 px,
treinado no corpus público Max-Span. O checkpoint documenta gold_auc =
0,921448 e a hipótese pública é que a amostragem densa das fatias periféricas
recupere sinal de menisco lateral, ligamentos e fratura.

## Contrato congelado

- cinco slots: Sagittal fluid-sensitive (18), Sagittal estrutural (14),
  Coronal fluid-sensitive (12), Coronal estrutural (8) e Axial (12);
- 64 fatias por estudo, span de 2–98% da série;
- crop central físico de 140 mm e resize de 336 para 384 px;
- 62 janelas sobrepostas de três fatias;
- atenção específica para cada um dos 12 alvos;
- saída em rank-percentile por alvo.

O código faz inferência em blocos de janelas para reduzir risco de OOM no T4,
mas concatena todas as features antes do pooling; portanto, a alteração é de
memória, não da arquitetura. Em caso de falha em estudo oculto, um kernel com
mais de 4.000 estudos recusa gerar uma saída parcial.

## Fontes e licenças

- Checkpoint: dreaddevelopment/raptor-knee-maxspan, CC0-1.0.
- A implementação foi escrita no projeto a partir do contrato público do
  checkpoint; a referência conceitual é o notebook público de inferência do
  Dread Development.
- Não usa RadImageNet (CC-BY-NC-SA-4.0), DINOv3, bundle privado, labels de
  texto como entrada de teste ou outputs de kernels privados.

## Critério de decisão

1. Executar sem internet em T4.
2. Cobrir 3/3 estudos de teste, sem NaN, duplicatas ou drift de schema.
3. Submeter pelo Notebook-only apenas após a confirmação do usuário no botão
   final.
4. Promover somente se superar o H-34, 0,899; esse gate foi cumprido com
   public score `0,928`. H-36 é a referência atual; H-34 e H-29 ficam como
   fallbacks auditáveis.

## Smoke local

O checkpoint fica no HD externo, fora do Git. Com dados locais compatíveis:

~~~bash
RSNA_KNEE_ROOT=/caminho/para/competicao \
RSNA_KNEE_WEIGHT=/caminho/para/raptor_ft_coatnet_v5_full_swa.pt \
RSNA_KNEE_LIMIT=1 \
python3 rsna_knee_h36_coatnet_maxspan.py
~~~

#Kaggle #Tecnologia #Academia #JoaoVictor
