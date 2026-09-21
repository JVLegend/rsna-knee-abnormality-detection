# AV-029 — nova validação e piloto de resolução

#RSNA #Kaggle #Pesquisa

21/09/2026. Fonte canônica:07_Estrategia_AVANCE do vault.
Continua [AV-028](AV028_CONFIRMACAO_RESERVADA.md), cuja confirmação foi
negativa. Não alterar a loss e consultar novamente os mesmos150estudos.

## Estado final desta rodada

V05 congelado e aprovado por auditoria local de metadados;
**G04 preparado, mas NÃO EXECUTADO por falta de vagaGPU simultânea.**
SaveKernel: Maximum batch GPU session count of 2 reached., kernelId0,
versionNumbernull. Status do slug404; listagem só placeholder refvazia/id0.
Não existe versão para acompanhar. Outros jobs não foram interrompidos.

Código/protocolo f7d0b53 commitado/push ANTES da tentativa. Quota no envio
35.804,072601s, suficiente para480s: bloqueio de concorrência, não de horas.
Não houve feature336 produzida, treinamento, nova avaliação ou submissão.
Na retomada, reconciliar o slug antes de lançar; não criar uma cópia com
outro nome para tentar contornar o limite. Nenhuma automação foi criada.

Auditor local adicional scripts/assess_v05_validation.py:
PASSED_V05_METADATA_AUDIT_NOT_PIXEL_VALIDATION. Verificou SHA/fontes,
IDs/grupos/labels congelados, cobertura, exclusões, folds e ausência de
alegações de inferência no manifesto. Não extraiu imagens ou predições.
Arquivo privado reports/avance_av029_v05/audit_v1.json,
SHA2565ce4982165cfc726780506d6b2ae1519c9ab1fc395d89f31ff4b75cefb3dda41.
233testes+44subtestes passaram, incluindo22novos nesta rodada e5testes
do auditor adicional. Nenhuma alteração do build piloto congelado.

## V05 — partições congeladas, sem avaliações novas

Manifesto privado no HD: data/processed/validation_weak_v5/manifest.json.
SHA2569bb462aee209c132af51073851525e885545c76627fe542c0248361673e6de84.
Implementação scripts/prepare_v05_validation.py;9testes novos passaram.
V01/V03/V04 e CSVs de origem fixados por SHA. Não alterar esses arquivos.

| Destino | Estudos | Grupos de laudo |
|---|---:|---:|
| Treino V03 preservado | 1000 | 954 |
| Novo desenvolvimento | 300 | 297 |
| Nova confirmação | 300 | 299 |
| Restantes não usados | 2290 | 2253 |

2.890elegíveis após1.455exclusões por IDs/grupos da linha própria e62por
gold/grupo. Grupos são laudos normalizados, não identidade comprovada de
paciente. Seleção por hash namespaceV05/seed20260921, grupos inteiros,
confirmação primeiro e depois desenvolvimento, sem procura de outra seed.
Rótulos e prioridade de séries permanecem os mesmos. Cobertura dos12alvos
passou (ao menos um positivo/negativo), mas isso não garante poder estatístico.
Fracture reservado:18positivos/110negativos/172incertos; dev24/108/168.
Os valores0,5 permanecem incertos. Nenhuma inferência nesses600novos estudos.

Planejados5folds de260estudos sobre1000treino+300novo dev. Grupos inteiros
alocados por tamanho e hash, sem otimizar labels. OOF NÃO executado.
Na execução futura, não selecionar época pelo fold externo: fixar época
antes ou usar validação interna restrita aos outros folds. O histórico de
seleção do treino impede chamar OOF retroativo de teste externo imparcial.

Não há alegação de dados historicamente virgens no projeto inteiro;
exclusões valem para os conjuntos conhecidos desta linha própria e gold.
Metadados não certificam presença/decodificação/geometria nem duplicatas
aproximadas. Verificar pixels/identidades antes de treino/avaliação futura.
O novo reservado fica fechado até receita/checkpoints/gates definidos.

## G04 — piloto técnico antes do treino224vs336

20exames SÓ DO TREINO,60séries, escolhidos por hash G04:20260921:UID.
Kernel não contém labels nem IDs dos novos dev/reservados. Decoder, percentis,
triplas fisicamente adjacentes e campo de visão nativo iguais; resize224/336
como única diferença de imagem. Não é novo crop calibrado em milímetros.
DINOv2-S oficial congelado. Aumento de resolução ainda NÃO provou ganho.

Warm-up separado para ambos tamanhos;10batches de2exames com ordem224/336
alternada. Sincronizar CUDA ao cronometrar; registrar decode+doisresizes,
transferência/normalização/forward por resolução e pico de memória alocada.
Inferência somente em cuda:0 da T4x2. Nenhuma cabeça de224aplicada a336.

Gates prévios:

- Pixel224 idêntico ao hash histórico, geometria/séries sem alteração.
- Features224 reproduzem cache V03, atol1e−4/rtol1e−5; sem relaxar após teste.
- Features224/336 com forma20×3×384,float32 e finitas;336não idêntico a224.
- Todos20casos presentes,10batches ordenados e tempos positivos/finito.
- Pico alocado<14GiB em cada tamanho (margem operacional, não memória total).
- Auditor local revalida fonte/receipt/artefatos/IDs/paridade/geometria.
  Extração336 não tem uma segunda implementação independente.

Estimativa de planejamento para1.300estudos:1,5×1.300×maior tempo observado
por estudo para decode+ambosforwards+420s para inicialização/treino/auditoria.
É extrapolação de20casos, não limite garantido; I/O frio pode superar.
Para treino real, orçamento e checkpoint/retomada serão definidos antes.

Build privado reports/avance_av029_g04/preflight_v1.py,66.245bytes,
SHA256ca3dd37aea22cdacfc7899938d2a68f6475fc68bf55b90aa645546a857dfeb21.
228testes+44subtestes passaram antes do envio, incluindo17novos.
Código/protocolo commitado/push f7d0b53 antes da tentativa recusada.

Slug previsto:jvlegend/rsna-knee-g04-resolution-preflight,sem execução
existente na busca inicial. Anexar V03scale1000, competição e DINOv2-Ssmall/1;
Docker fixado igual ao V03. Privado/offline/T4,teto480s/guard360s.
Quota inicial79.735,289898s; exigir>=960s e reconciliar antes do envio.
Não interromper jobs de outros projetos. Não usar serviço pago novo.

Saídas futuras: reports/avance_av029_g04_v1/; auditoria independente
scripts.assess_g04_preflight → reports/avance_av029_g04/audit_v1.json.
Não relançar cegamente depois de queda de rede: verificar ID/versão/outputs.

## Próximo experimento após o piloto

Se compatível, implementar treino224vs336 no MESMO treino1000/novo dev300,
cabeças treinadas separadamente, seeds2026/42, mesma receita/épocas máximas.
Congelar critérios de seleção e promoção antes de observar o dev novo;
não comparar os scores absolutos com o dev250antigo. OOF é uma etapa distinta,
não chamar este holdout de OOF. Usar confirmação300só uma vez por lote fixado.
Se custo/compatibilidade falhar, corrigir infraestrutura ou seguir outra
hipótese elegível, sem usar labels reservados para orientar a correção.

Sem nova submissão ou resultado de leaderboard. Melhor público confirmado
0,941 preservado, pertencente ao ensemble anterior.
