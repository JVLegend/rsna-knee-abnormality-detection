# AV-026 — treino robusto e consistência entre planos

#RSNA #Kaggle #Pesquisa

20/09/2026. Fonte canônica:07_Estrategia_AVANCE do vault.
Protocolo definido antes de treinar ou observar novas métricas.

## Hipótese e alternativas

A AV-025 mostrou perda de qualidade quando removemos planos. Em vez de
somente corrigir a inferência, ensinar o head a lidar com informação parcial.
Não é garantido: certos achados podem depender de um plano específico.

| Receita | Entrada de treino | Objetivo |
|---|---|---|
| control | Sempre3planos | BCE original V03 |
| dropout25 | Em25%dos estudos, retirar UM plano uniforme | BCE com máscara correta |
| paired_consistency | Par completo + UM plano ausente uniforme em cada estudo | 0,5BCEcompleto +0,5BCEmascarado +0,1MSE entre probabilidades, alvo completo sem gradiente |

Todas usam as mesmas features1000treino/dev250, rótulos fracos, head
sharedattention384→64→1, DINOv2-S congelado,20épocas, batch4,
AdamWlr0,001/weight_decay0,0001 e seeds2026/42. RNG separado seed+10000
para máscaras: não alterar inicialização ou shuffle global do controle.
Treino paired muda exposição e objetivo; não atribuir causalmente o efeito
apenas à consistência. Não explorar grid/novas seeds/seleção por alvo.

Selecionar primeira época de menor softBCE dev INTACTA. Só depois avaliar
ausências Sagittal/Coronal/Axial no best. Candidato promovido apenas se
melhorar intacta >2e−6 E média das três ausências >2e−6 nas DUAS sementes.
Se ambos elegíveis: menor média intacta entre sementes; empate favorece
dropout25. Se nenhum elegível: manter V03. Robustez isolada não promove.
Sem confirmação150/gold; nenhum CSV, seleção de finalistas ou submissão.

## Proveniência e proteção contra regressões

V03 kernel jvlegend/rsna-knee-v03-scale1000 v1 COMPLETE, já auditado.
BuildSHA3ae46545…; featuresSHAbbc34ca6…; contrato31930692….
Os hashes completos estão no builder e no build privado. Só anexar saída
privada V03; não reextrair pixels nem baixar mais dados para este ensaio.

Controle usa função train_seed ORIGINAL. Candidatos derivam a mesma função
com substituições verificadas: objetivo, RNG/trace das máscaras e estado
de retomada. Checkpoint por época; preservar optimizer e RNGs. Replay
independente das máscaras, contagens, cadeia SHA e RNG final no auditor.
train_soft_bce dos candidatos contém o objetivo combinado (nome herdado);
somente development.mean_soft_bce é a métrica comparável de seleção.

Auditoria exige controle reproduzindo curva/época/logits V03; hashes,
IDs,20épocas, seleção intacta, replay NumPy dos checkpoints best/last e
predições com ausência. Paridade1e−4absoluta/1e−5relativa nos logits e
2e−6 na BCE, iguais às auditorias anteriores.
Retomada exata validada em dados SINTÉTICOS CPU; o kernel ainda não tem
CLI de retomada entre jobs. Não afirmar que basta relançar após uma falha.

## Execução e limites

Quota inicial107.543,516sGPU (~29,87h), compartilhada. Consultar novamente
antes de enviar. Teto600sT4x2/offline, apenas cuda:0, guard450s;
exigir pelo menos1200s livres. Sem treino pesado no Mac.
Seis heads curtos; não é fine-tuning do encoder. Artefatos ficam privados
sob reports/avance_av026_r02/ e reports/avance_av026_r02_v1/ no HD.

Implementação: scripts/prepare_r02_training.py,
scripts/r02_training_runtime.py, scripts/assess_r02_training.py,
tests/test_r02_training.py.191testes+44subtestes passaram, incluindo
retomada exata dos dois candidatos em dados sintéticos, loss/detach,
RNG independente, máscara >=2planos, replay de trace e critérios de promoção.
Uma falha inicial do harness sintético (Path não injetado no escopo) foi
corrigida no teste, sem mudar runtime ou receita.
Build reports/avance_av026_r02/r02_v1.py,239.496bytes,SHA256
17ad28e1a5882f76b1a342a768279ccd525b9a7eae5831e052f12db0903b4887.
Ainda sem resultados reais de treino nesta versão do protocolo.

```sh
python -m scripts.prepare_r02_training --output reports/avance_av026_r02/r02_v1.py
python -m scripts.assess_r02_training \
  --directory reports/avance_av026_r02_v1 \
  --build reports/avance_av026_r02/r02_v1.py \
  --baseline reports/avance_av024_v03_v1 \
  --output reports/avance_av026_r02/audit_v1.json
```

Usar PYTHONPATH=src:. e NumPy/Pandas/PyTorch; testes também requerem pytest.
Rótulos fracos e dev que escolhe épocas não são teste clínico independente
nem score Kaggle. Melhor público confirmado continua0,941.
