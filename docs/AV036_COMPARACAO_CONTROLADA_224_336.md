# AV-036 — comparação controlada 224 × 336

#JoaoVictor #Kaggle #Tecnologia #Academia

30/09/2026. Estado: **protocolo/código preparados; treino bloqueado por
identidade de paciente e cobertura de pixels**. Não existe nova candidata
qualificada para submissão. Não interpretar G04 como melhora de qualidade.

## Referência preservada e recursos

- H46 original ref **56696639 COMPLETE, público 0,943**, novo melhor nosso;
  kernel136476642v1 COMPLETE. Fonte SHA7dc49666… e CSV SHA7c6dfe8… preservados.
  H43A0,941 é histórico. Não reexecutar, editar, enviar novamente nem alterar
  seleção final automaticamente.
- Consulta API em **30/09 às17:07:57 BRT**: prazo **22/10/2026 às20:59 BRT**
  (23:59UTC); entrada/fusão15/10 às20:59BRT. Fonte:
  [competição oficial](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).
- GPU:108.000s totais,39.483,928s usadas,0s reservadas, **68.516,072s/19,03h
  restantes**. Reset03/10 00hUTC = **02/10 às21hBRT**. Usar timedelta completo;
  a serialização antiga pode perder o componente de dias e sugerir só6h.
- H46/G04/V03 COMPLETE, sem falha. Zero reserva não comprova vaga de execução:
  reconsultar cotas/concorrrência imediatamente antes de qualquer dispatch.
  Não duplicar esses jobs nem lançar a experiência se houver recibo de tentativa
  não reconciliado. Nenhum job novo neste avanço.
- Limite5submissões/dia;1observada na dataUTC, não promessa de4slots no futuro.
  HD externo238.567.555.072bytes livres (~222GiB). Zero serviço pago novo.
- Recibo privado: `reports/avance_av036_g05/resources_v1.json`.
  Python local3.14.6/Node22.22.2/zsh; sem venv ativo/pwsh.
  Kaggle2.1 do sistema não expõe quota_view; consulta utilizou o runtime2.2 já
  existente no cacheuv/Python3.13.13, sem instalar/atualizar pacotes.

## Pergunta e receita fixadas antes dos resultados

G05 testa **somente tamanho224 versus336**, não densidade de fatias, FOV,
backbone maior ou loss nova. V05 congelada SHA9bb462ae…: mesmos1.000estudos
de treino,300dev e300confirmação, sem redividir ou escolher seed por prevalência.

| Componente | Mantido nos dois braços |
| --- | --- |
| Séries/fatias | Mesmas3séries Sagittal/Coronal/Axial; adjacentes centrais por posição física; canaisgrayscale3; FOVnativo |
| Pré-processamento | Mesmo decoder, percentis1–99, quantizaçãouint8 e resizebilinear; só224→336 muda |
| Encoder | DINOv2-S oficial congelado, CLS384, float32, TF32off; SHA1051e25b…; config1809f83e… |
| Pooling/head | StudyAttention384→64→Tanh→1/softmax; classificação384→12 |
| Loss/treino | BCEWithLogits médio não ponderado; mesmo teacher congelado, incerto0,5 preservado; AdamWlr0,001/wd0,0001; batch4; semaugmentation |
| Aleatoriedade/épocas | Seeds2026e42, mesma inicialização e mesma ordem por seed; **20épocas fixas**, sem early stopping/dev para escolhercheckpoint |
| Comparador | Retreinar224 e336 em par; não comparar um novo336 com head224 previamente selecionado no dev antigo |

Contrato executável: `scripts/resolution_comparison.py`. O protocolo final é
`reports/avance_av036_g05/protocol_v2.json`; v1 foi o registro preliminar da
mesma receita, sem treino, anterior aos pins de código. Nenhum resultado real
de desenvolvimento foi consultado para escolher estes critérios.

## Critérios pré-especificados

Definir δ =336−224; menorBCE é melhor. Nos **dois conjuntos, separadamente**:

1. Primário: médiaBCE suave sobre estudos/12condições/2seeds; δ≤−0,001,
   limite superiorIC95%pareado<0 e melhora em cada seed.
2. Segurança: nenhuma condição com regressãoBCE>0,01; limite superior do
   ICsimultâneoBonferroni das12diferençasBCE≤0,02. Não chamar IC95%pontual de
   simultâneo. Listar inclusive regressões menores, não só gates reprovados.
3. AUC por condição e macro: somente rótulos exatos0/1; excluir0,5/outros
   suaves e informar cobertura. Nenhuma queda pontualAUC>0,02; pelo menos
   10positivos/10negativos por condição e≥95%réplicas com AUCdefinida.
   Menos20porclasse recebe alerta de baixa potência (Fractureconfirm tem18
   positivos conhecidos). Resultado inconclusivo não autoriza ressplit/tuning.
4. IC95% paraBCE/AUC dos braços e suasdiferenças por condição;5.000bootstraps,
   seed20260930. Unidade é **estudo**; estudos ligados por paciente OU laudo
   são reamostrados juntos, mantendo pares/seeds. Nunca bootstrap por fatia.
5. Custo: job completo≤7.200s, picoGPUalocado≤4GiB, forward336/224≤3,
   quota disponível≥14.400s antes do início; T4existente, internetoff.
   Reportar memóriaalocada/reservada e tempos de decoding/cache/forward/head,
   separando custo reutilizado de custo incremental. Semcriarinfra paga.

Referência fraca derivada de laudos/teacher, **não adjudicação clínica, nem
estimativa garantida do leaderboard**. Independência em relação a nosso treino
não prova independência de exposição histórica do professor. Sem laudos para
APIs novas e sem selecionar blend/H46 pela confirmação.

## Gate de vazamento: motivo do bloqueio

V05 tinha disjunção de StudyInstanceUID e hashnormalizado de laudo, não prova
de pacientes. Cabeçalhos locais inventariados sem pixels: **299/1.600estudos
com as3séries disponíveis, todos do treino original**;3.903séries ausentes.
Os PatientID não coincidem com StudyUID nesses headers, mas não há fonte
atestando que a chave anonimizada é estável entre exames/sites. Zero colisão
entre splits é inconclusivo quando nenhumdev/confirm foi coberto.

O inventário foi parcial (primeiroDICOM de cada série disponível), não auditoria
completa; não afirmar pacientes independentes. MetadadosV05 contêm rótulos
fracos previamente congelados: foram carregados, não pontuados. Nenhum pixel
de confirmação/predição de modelo foi aberto. Recibo
`reports/avance_av036_g05/patient_inventory_v1.json`.

Antes de treino: obter cobertura deheaders de todasasséries selecionadas,
verificarPatientID consistente dentrodoestudo e obter documentação/mapping
de anonimização estável entreexames. Evidência deve ter fonte verificável,
não preencher flagTrue por conveniência. Unir componentes porpaciente/laudo;
se houver ligação entrepartições, **bloquear ambas resoluções**. Não mover
casos silenciosamente para manter a mesma divisão. Qualquer novo split exigirá
versão/protocolo novo antes de avaliar resultados.

## Caches, execução e confirmação independente

- V03NPZ SHAverificado contra constante fixada; G04NPZ SHA3175b1a0…verificado;
  auditoriaSHA0b67b935…verificada. G04224 delta0 contraV03;336 tem apenas20casos.
  Reusar treino224 e as20features336 já verificadas, sem reextração gratuita.
- **Falta extrator completo G05** para701treinos+dev/confirm336 e dev/confirm224:
  manter seleção/hashes de pixels e conferirparidade224. Não declarar que o
  NPZdoG04 basta para a comparação1.300estudos. Gate requer também auditoria
  de duplicatas exatas/aproximadas entrepartições; isso ainda não foi feito.
- Treinador preparado `scripts/run_resolution_heads.py`: exige contrato,
  identidade completa, recibo/cacheSHA/IDs exatos, paridadeeproveniência de
  features; falha antes de treino quandoausentes. Não contém APIdejob/envio.
  Implementação aguarda integração/testeT4 comextratorcompleto, nãofoiexecutada.
- Fit fixa épocas sem olhar métricasdev intermediárias; salva4heads/predições
  e relatório comgates. Seed não é observação independente.
- Se dev falhar, preservarH46 e encerrarreceita, sem abrirconfirmação.
  Se passar, congelar hashes dosmesmos4heads e inferir **uma vez** nos300casos
  reservados, semretreino/ajuste. Marcadorpersistente `confirmation_exposure.json`
  impede repetição inadvertida. Interrupção requer reconciliar artefatos,
  nunca apagar marcador para reconsultar. Exigir mesmosgates antesde promover.
- Mesmo confirmação positiva não gera submissão: exige empacotamento/teste
  completo nosrequisitosoffline/tempo e decisão pessoal doJV. Nunca substituir
  H46 automaticamente nem anexar336 aoensemble sem validação própria.

## Evidência/custo/recomendação nesta entrega

- G04:20estudos/60séries,31,533s; forward336/224=2,219962,
  picoalocado336=213.534.208bytes (~204MiB). Isso é viabilidadetécnica.
- Estimativa conservadora anterior para1.300casos/duasresoluções=5.570s/~93min;
  não é mediçãoG05 nem garante tempo/cota. Reuso reduzparte desse custo,
  auditoriafullheaders/duplicatas pode acrescentar custo. Não presupor4GiB
  para todaexecução com baseapenasnopiloto.
- Custo novo: inventárioheaderCPU90,078s; testeslocais35passaram;
  **0GPU/0jobs/0submissões/0novosserviçospagos** nesteavanço.
- Qualidade336 real/AUC/BCE/IC porcondição **não medidos**. Apenas testes
  sintéticos doavaliador; não inserir suasmétricas em logcomo resultadoMRI.
- Recomendação: **não enviar G05 agora**. Manter H46público0,943. Próximo
  passo seguro: completar evidência de paciente/coberturanoKaggle existente
  e extratorpareado, reconsultandorecursos antesde executar; confirmaçãofechada.

Fonte canônica no vault:
`01_Projects/Competicoes/RSNA_Knee_Abnormality_Detection/08_Comparacao_Controlada_224_336.md`.
