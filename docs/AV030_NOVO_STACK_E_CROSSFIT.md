# AV-030 — novo stack público, auditoria OOF e contratos de treino

#RSNA #Kaggle #Pesquisa

23/09/2026. Fonte canônica: `07_Estrategia_AVANCE` no vault. Continuação da
[pesquisa de 23/09](PESQUISA_KAGGLE_2026-09-23.md). As quatro frentes avançaram,
mas nenhuma nova performance de imagem ou nota Kaggle foi medida.

## P1 — auditoria CPU concluída; candidata H46 construída

Kernel privado `jvlegend/rsna-knee-h46-cpu-assets-v1`, ID135536476, V1,
**COMPLETE**. CPU/offline/teto600s. Somente código próprio de inventário e
hash; não importou fontes públicas, não carregou pickle/pesos e não rodou
inferência. Código da auditoria SHA
`e02828855c82198bd7d1aa162d1fc64b337d8f8b3934c865164fb285e21b4f4a`.

Resultado: **PASSED_H46_ASSET_AUDIT_NOT_INFERENCE**,63,3255s:
56registros de arquivos compartilhados,20membrosDINO;45arquivos do pacoteD4
e17do Global96. São registros de inventário, não contagem de modelos ou
arquivos globalmente únicos. D4/Global96 declarados CC0-1.0 no catálogo;
demais licenças herdadas não mudaram (inclusive restrições RadImageNet).

Manifestos fixados:

- D4: `7ada0605bca6b0530569c6454e988ace479606a3328ed591d090e5764fea661d`.
- Global96: `015f09030e76f86274cadae40777b3c3cf5c19f8c835d5ad88e15db41903779e`.

Recibo privado: `reports/avance_av030_h46_v1/h46_preflight.json`, SHA
`aec26af61c4485359ff40431cc381ad38ecf8fcc454697402f43d3b79a377b76`.
Lançamento/quota: `reports/avance_av030_h46/launch_v1.json`.

Builder `scripts/prepare_h46_candidate.py` preserva a receita **speedy** da
fonte MaverickV3, fixada por SHA; não tenta outros pesos. Acrescenta lock de
**109pares nome/hash**, contagem real deA5, verificação de preparação Raptor
e gate antes da publicação de `submission.csv`. Exige20DINO,5A5,4viewsRaptor,
três leitoresCoAt e rank da média de probabilidades. Corrige o recibo final
para incluir Global96 e pesos1/3. Rejeita membro ausente, redução alternativa,
predições neutralizadas, descartes ou reparos registrados. Relocação de
scratch e retryFP32 finito continuam permitidos, preservando o caminho autoral.

Candidata privada: `reports/avance_av030_h46/candidate_v1.ipynb`,301.010bytes,
SHA `d3029315a5f7beb419aeba943f3d4d0b68603078e59777c68b56c2acea196b53`.
**Construída/analisada sintaticamente, NÃO executada em GPU e NÃO enviada.**
A receita é do notebook público0,943; nossa nota permanece0,941. Gates mais
estritos podem abortar casos antes tolerados pelo autor: isso exige smoke
real antes de elegibilidade. Não há prova de paridade numérica/tempo oculto.

## P2 — OOF público não aprovado para selecionar o blend de fratura

Baixados recibos pequenos DINO/Rad, sem novos DICOM/pesos. Auditor próprio
`scripts/audit_av030_oof.py` verificou IDs,12alvos, máscaras, finitude,
probabilidades e SHA do CSV Rad contra PROVENANCE.md. Excluiu58gold da análise
de folds. Não calculou AUC nem consultou a confirmação V05.

| Origem | Estudos não-gold | Grupos de laudo | Grupos cruzando folds | Estudos nesses grupos |
|---|---:|---:|---:|---:|
| DINO public20 | 4.349 | 4.200 | **11** | **35** |
| Rad V52 | 4.349 | 4.200 | 0 | 0 |

Grupos usam a normalização conservadora já adotada pelo projeto. Não são
identidade comprovada de paciente: o diagnóstico demonstra violação do nosso
critério de agrupamento, não prova vazamento de imagens. Além disso, falta
auditar exposição de treino/seleção/pretreino de cada membro e o reciboE13.
Logo **não promover a remoção de Fracture da exclusão Rad com essa evidência**.

Resultado `STRUCTURE_AUDITED_PROVENANCE_NOT_CONFIRMED`, em
`reports/avance_av030_oof/audit_v1.json`, SHA
`f250883743d970ba362fe4b27ac24e80ca6a13ef3113ae9c54ed0bade37bd77a`.
Não existe novo resultado de ablação nem ganho público nesta frente.

## P3 — infraestrutura cross-fit e fine-tuning parcial implementada

`crossfit_contracts.py` exige identidades de treino, seleção, pretreino
específico e predição de cada professor, hash de checkpoint, cobertura,
alinhamento de alvos e probabilidades válidas. Impede usar duas cópias do mesmo
checkpoint como diversidade. O chamador deve fornecer explicitamente os
grupos proibidos, incluindo confirmação e o fold externo do aluno quando
produz pseudo-rótulos para treino desse aluno. Cross-fit global reutilizado
sem essa exclusão pode vazar informação entre níveis.

`configure_dino_tail` libera somente os últimos dois blocos+layernorm do
contrato HF DINO, com controle totalmente congelado. Teste sintético fez
backprop/SGD e confirmou atualização do final e preservação do prefixo.
É teste de implementação, não treino em MRI nem medição de melhoria.

Derivado novo, sem alterar V05: `reports/avance_av030_crossfit/plan_v1.json`,
SHA `df8258e1cb94e67b47dacd137945e40c64be0a1c5d1b019361b339b3430f68b0`.
Mantém5folds externos de260estudos; seleção interna por hash de grupos:

| Fold | Treino interno | Seleção interna | Predição externa |
|---:|---:|---:|---:|
| 0 | 887 | 153 | 260 |
| 1 | 887 | 153 | 260 |
| 2 | 871 | 169 | 260 |
| 3 | 877 | 163 | 260 |
| 4 | 889 | 151 | 260 |

Sem professores treinados, pseudo-rótulos gerados, OOF executado ou nova
confirmação consultada. Integração ao treinador/DICOM, auditoria dos pixels,
loss/endpoints/épocas/custo pré-registrados ainda necessários antes do treino.
O manifesto não é autorização para escolher época pelo fold externo.

## P4 — contrato de canais implementado, ablação clínica pendente

`input_contracts.py` implementa adjacentes versus central repetido emNCHW,
antes da normalização, sem mutar o array original. Checkpoint e inferência
devem concordar em canais, resolução, normalização e hash de geometria.
Testes cobrem equivalência do canal central, finitude, shape e rejeição de
contrato incompatível. Nenhum checkpoint RadImageNet foi alterado; faltam
treinar separadamente as duas receitas, com mesmos dados/seeds e validação.

## Bloqueio e retomada

API de quota retornou `timeUsed="46725.135.0s"`, `totalTimeAllowed="21600s"`,
`timeReserved="0s"`; a string de duração tem formatação anômala e foi
preservada bruta no recibo. Uso informado supera o limite: não presumir GPU
disponível. Reset informado **2026-09-26T00:00:00Z**, ou **25/09 às21hBRT**.
Nenhum GPU lançado, job de outro projeto cancelado, gasto ou automação criado.
Não houve necessidade de editar configuração do projeto/usuário.

Próximo AVANCE: consultar quota/estado novamente; reconciliar H46CPU V1 sem
duplicar; preparar smoke delimitadoT4×2 da candidata fixa e auditoria de
paridade/cobertura/tempo antes de qualquer submissão. G04 continua preparado
e dependente deGPU; não relançarV04. Melhor H43A0,941 e finalistas preservados.

## Verificação

Suíte completa delimitada a `tests/`: **263testes+44subtestes passaram**,
incluindo30testes novos. Comando: `PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m pytest -q tests`.
A primeira descoberta genérica sem diretório foi interrompida antes de rodar
testes porque percorria dados noHD; a execução delimitada terminou em63,62s.
Arquivos
volumosos/recibos/IDs permanecem emreports/ no HD e fora doGit. Código, testes
e documentos podem ser versionados; não publicar fontes de terceiros como
se fossem nossas nem dados privados de treino/validação.
