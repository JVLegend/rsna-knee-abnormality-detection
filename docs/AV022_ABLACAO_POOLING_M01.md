# AV-022 — M01: média versus atenção por alvo

#RSNA #Kaggle #Pesquisa

17/09/2026. Protocolo registrado antes de executar as novas cabeças.
Fonte de verdade: nota07_Estrategia_AVANCE do vault; [espelho](ESTRATEGIA_AVANCE.md).
Continua [baseline AV-021](AV021_BASELINE_PROPRIO_V02.md).

## Hipótese e desenho

Comparar três cabeças nos MESMOS embeddings congelados549x3x384:

1. `shared`: atenção compartilhada384→64→1, linear384→12. Controle AV-021.
2. `mean`: média dos planos presentes, linear384→12, sem atenção aprendida.
3. `target`: scores384→64→12; softmax entre planos para cada alvo;
   pool separado[B,12,384], um vetor classificador384+bias para cada alvo.

Máscaras booleanas e bags não vazios obrigatórios. Dados reais têm3planos
presentes; não simular ausências, alterar geometria ou acrescentar coordenadas.
Capacidade e inicialização diferem entre arquiteturas; comparação de receitas,
não prova causal isolada da atenção. Duas sementes não estimam incerteza clínica.

Split/teacher fixos AV-021:299treino/250dev. Confirmação150não avaliada.
Seeds2026/42,20épocas,batch4,AdamWlr0,001,decay0,0001; nenhuma grade.
Selecionar menor softBCE média dev, primeira época em empate. Reportar ambas
sementes, última época, perdas por alvo e prior0,657078403. Não medir AUC.

Gate: controle shared reproduz épocas AV-021 e logits melhor/último
(atol1e−4,rtol1e−5), métricas até2e−6. Caso contrário, suspender atribuição.
Replay independente NumPy nas3cabeças/checkpoints e BCE float64.
Promover referência apenas se nova receita superar shared em AMBAS sementes
por >2e−6. Se duas forem elegíveis, menor média entre sementes; empate exato
favorece mean. Não selecionar ensemble/pesos por alvo com este experimento.

## Proveniência e custo

Somente saída privada do kernel `jvlegend/rsna-knee-v02-frozen-dino-baseline`,
v1 ID134788472. Arquivo features SHA
`d5ad2044ce6991959183a00c24fc3a0168ca98356df0222438f6023610d63d87`;
contrato original `1167612d930940ee3e2c1d3ec795620de3e0fde89cad82ea0dd4514f8fb07234`.
Builder exige build AV-021 SHA0049a9b0… e auditoria aprovada.
Hashes/IDs/pixels do archive são conferidos antes de qualquer treino.
Sem anexar competição, encoder ou pesos de participantes. Código privado,
offline,T4,teto1.800s, somente quota existente. Não gerar CSV de submissão.
Arquivos privados de dados/labels/checkpoints permanecem ignorados pelo Git.

## Estado

Builder, runtime e auditor implementados em scripts/*m01_ablation*.py.
Cinco testes sintéticos CPU passaram (máscaras, gradientes, replay NumPy,
gates de features/proveniência, seleção prospectiva). Build privado
reports/avance_av022_m01/m01_v1.py SHA
`5a93e1ebc90df370514b74a9c5225166f9ad3baf5dc6d6af3fbcd845bbb62a6c`.
Quota consultada antes do lançamento:5,2438horas GPU disponíveis.
Suíte completa:146testes+44subtestes passaram. Código/protocolo commit538a0e9
antes de avaliar. Kernel `jvlegend/rsna-knee-m01-pooling-ablation`,v1
ID134795832 COMPLETE com anexo privado aceito. Auditoria PASSED_M01_AUDIT.
O uso da skill Obsidian mantém o protocolo/cursor canônico no vault antes
do espelho operacional e dos resultados.

## Resultados auditados

SoftBCE nos250estudos dev, menor melhor; prior de treino0,657078403.
Entre parênteses, época selecionada. Não é AUC nem score de competição.

| Cabeça | Seed2026 | Seed42 | Média das sementes |
|---|---:|---:|---:|
| Shared (controle) | 0,62607019 (10) | 0,63189521 (6) | 0,62898270 |
| Mean (média simples) | 0,62733778 (7) | 0,62448577 (7) | 0,62591178 |
| Target (atenção por alvo) | 0,63315980 (6) | 0,62473064 (9) | 0,62894522 |

Delta versus shared por semente2026/42:
mean +0,00126759/−0,00740944;target +0,00708961/−0,00716457.
Mean melhora a média cerca de0,49%, mas piora2026. Nenhuma alternativa
atinge a regra prospectiva de melhorar AMBAS sementes por >2e−6.
Decisão: manter shared como referência; mean em reserva, não descartar
o sinal favorável nem promovê-lo sem replicação. Não ajustar pesos por alvo.

Última época20:shared0,64938284/0,66013784;
mean0,64053301/0,65124329;target0,65783177/0,66395749.
Todos pioram comparados à melhor época, sem justificar estender treino.

Controle shared reproduziu as mesmas épocas, perdas e logits da AV-021.
Auditor recalculou logits melhor/último de todos os12checkpoints em NumPy,
BCE/perdas por alvo, identidade de estudos, hashes e estado do otimizador.
Maior diferença numérica1,20515e−6, abaixo da tolerância fixada.
Tempo36,0247s fora imports; pico alocado76,74MiB, não reserva total da GPU.
Sem reextração de imagens. HardwareT4x2, apenas cuda:0 utilizado.

Dados e resultados completos permanecem privados:

- reports/avance_av022_m01_v1/:recibo,6NPZs,12checkpoints e log.
- reports/avance_av022_m01/m01_audit_v1.json:auditoria independente e
  métricas por alvo, épocas, regra de seleção e limitações.
- reports/avance_av022_m01/m01_v1.py:fonte congelada enviada ao Kaggle.

Nenhuma nova submissão; melhor público conhecido0,941 preservado.
Não abrir confirmação nem alegar ganho clínico com rótulos do professor.
Próximo G01: comparar cortes adjacentes com espaçados, mantendo centros,
views, encoder e shared. Registrar geometria antes de extrair novo cache;
preservar V02 original e não mudar teacher/cabeça junto com os cortes.

## Reproduzir auditoria

Com dependências NumPy/Pandas/PyTorch e PYTHONPATH=src:.:

```sh
python -m scripts.assess_m01_ablation \
  --directory reports/avance_av022_m01_v1 \
  --build reports/avance_av022_m01/m01_v1.py \
  --baseline reports/avance_av021_baseline_v1 \
  --output reports/avance_av022_m01/m01_audit_recheck.json
```

O auditor recusa sobrescrever saída. Conferir resultado existente antes
de repetir qualquer treino; este kernel não produz CSV elegível.
