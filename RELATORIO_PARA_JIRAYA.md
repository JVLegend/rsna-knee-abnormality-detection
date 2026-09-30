# Relatório para Jiraya — G05, comparação 224 versus 336

#JoaoVictor #Kaggle #Tecnologia #Academia

30/09/2026, AV-037. Fonte de verdade: esta nota do vault; espelho com o mesmo
nome na raiz do repositório no HD externo. Protocolo científico:
[[08_Comparacao_Controlada_224_336]]; continuidade: [[07_Estrategia_AVANCE]].

## Decisão e limite da entrega

**Preservar H46, público 0,943, ref56696639 COMPLETE. Não enviar G05 agora.**
A extração pareada e os auditores foram implementados; ainda não existem
features G05 completas, heads treinados ou resultados de qualidade 224/336.
73 testes focais aprovados validam implementação e salvaguardas sintéticas,
não qualidade nas imagens reais nem independência de pacientes.

O único job lançado nesta rodada foi CPU privado/offline, exclusivamente
para headers. Não houve GPU, treino, submissão ou serviço pago novo.
A confirmação continua sem encoder, predições ou pontuação de rótulos.
Leitura de headers e futura triagem de pixels reservados para duplicatas são
auditorias de integridade, não avaliação do modelo. Nenhum caso será movido,
excluído ou escolhido pelo resultado da confirmação.

## Auditoria completa de headers — concluída e verificada

Job `jvlegend/rsna-knee-g05-full-headers-v1`, kernel136572534, versão1,
lançado 30/09/2026 às21:00:20UTC. Um único dispatch, CPU, offline, privado,
timeout3.600s/guard3.300s. **COMPLETE**, confirmado pela API30/09 às21:45UTC;
recibos baixados no HD e auditados independentemente por fonte/contrato/SHA.
Resultado: **1.600/1.600estudos, 8.813séries, 296.241headers, zero inconsistências,
zero SOPs duplicados entre exames**. CPU2.648,918s, aproximadamente44min09s.
Split1000/300/300 intacto; zero pixels/rótulos de confirmação avaliados.

Lê todos os arquivos DICOM de **todas** as séries oficiais dos estudos da
V05, inclusive séries não usadas pelo encoder. Compara inventário da pasta
com `train_series.csv`, Study/Series/SOPInstanceUID, PatientID consistente
entre fatias/séries, ausência de chave vazia/placeholder, SOP repetido intra
e entre exames, IssuerOfPatientID e indícios de desidentificação.
Não lê pixels, PatientName, laudos ou rótulos. Identificadores de paciente,
issuer, data e SOP são registrados apenas como hashes com domínio separado;
isso não os torna prova de identidade nem garantia criptográfica de anonimato.
Artefatos privados em `reports/`, excluídos do Git.

Recibos no HD: `reports/avance_av037_g05/headers_launch.json`,
`headers_v1.py`, `headers_output/g05_headers.json` e
`headers_output/g05_headers_receipt.json`. Auditoria independente:
`scripts/assess_g05_headers.py`, resultado em `headers_audit_v1.json`, status
`PASSED_FULL_HEADERS_IDENTITY_SEMANTICS_UNVERIFIED`. Não relançar o kernel.

O inventário local anterior de299 exames não era exaustivo: leu apenas o
primeiro header das três séries selecionadas, todos do treino original.
Não usar aquele inventário para certificar desenvolvimento/confirmação.

## Evidência de identidade — bloqueio científico

**Não foi localizada evidência específica desta competição que garanta
PatientID estável entre exames do mesmo paciente e namespace entre sites.**
A ausência de colisões não demonstra que exames com IDs diferentes pertencem
a pessoas diferentes. Hash de laudo também não é identificador de paciente.
Nesta auditoria:1.600chaves distintas, nenhuma repetição e nenhum cruzamento
observado. **IssuerOfPatientID e StudyDate ausentes em todos os1.600exames**;
DeidentificationMethod/CodeSequence ausentes; PatientIdentityRemoved vazio.
Não há sequer evidência positiva de repetição longitudinal neste subconjunto.
Isso não demonstra que a anonimização foi por estudo, nem que as pessoas são
distintas: ambas as situações continuam compatíveis com o resultado observado.
O gate executado nos dados reais rejeitou a construção GPU:
`Patient-key semantics unverified; report hashes are insufficient`.
Recibo `identity_block_v1.json`; nunca alterar a flag para “true” por ausência
de colisões ou por semelhança do formato do identificador.

Fontes primárias consultadas em30/09:

- [RSNA: Knee MRI AI Challenge](https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge): contexto multicêntrico; não especifica semântica longitudinal do PatientID.
- [RSNA: Imaging research tools](https://www.rsna.org/research/imaging-research-tools): ferramentas de anonimização são configuráveis; isso não prova a configuração aplicada a este dataset.
- [CTP DICOM Anonymizer](https://mircwiki.rsna.org/index.php?title=The_CTP_DICOM_Anonymizer): opções de hashing/lookup de identificadores; capacidade da ferramenta não equivale a proveniência do dataset.
- [Clinical Trial Administrator’s Manual](https://mircwiki.rsna.org/index.php?title=Clinical_Trial_Administrator%27s_Manual): identificadores e política dependem do ensaio/configuração.
- [Competição: descrição dos dados](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data): conteúdo dinâmico não forneceu declaração verificável sobre estabilidade; busca no fórum não encontrou confirmação específica dos organizadores.

Nenhuma mensagem foi publicada nem contato enviado. Pergunta preparada para
ação pessoal do JV: “Nesta competição, o PatientID desidentificado identifica
a mesma pessoa em todas as StudyInstanceUID, incluindo exames bilaterais,
datas distintas e sites? Há reanonimização por estudo/lote ou namespace por
site? Podem disponibilizar declaração ou mapping pseudonimizado de grupos?”

Gate de treino permanece fechado: fonte/mapping verificável + cobertura
integral + componentes ligados por paciente **OU** laudo sem cruzar splits.
Se surgir sobreposição, conservar o split e registrar bloqueio; qualquer
redivisão exigirá novo protocolo explícito, nunca correção silenciosa.

## Extrator pareado implementado

Fontes: `scripts/g05_pair_runtime.py`, `prepare_g05_extraction.py`,
`g05_duplicate_audit.py`, `assess_g05_pixels.py`, `launch_g05_pixels.py`.

1. CPU: uma decodificação por fatia, mesmos percentis1–99/uint8 e posições
   físicas; produzir224 e336 do mesmo pixel normalizado. Sem crop/FOV novo.
2. Oito dezenas de shards de20 estudos, com IDs, contrato e SHA256; recibos
   de seleção/fatias/pixels por série. Exigir os1.600 estudos/4.800 séries.
3. Paridade224 exata nos3.000 registros de treino G01/V03 com referência:
   arquivos, índices, posições/gaps e SHA dos três canais. Não inventar
   referência antiga para os600 exames reservados de desenvolvimento/confirmação.
4. Duplicatas exatas: SHA dos três canais224 entre splits, inclusive planos
   diferentes. Aproximadas: mesmo plano, thumbnail central32×32/PILbilinear,
   centragem/L2, coseno≥0,995 e pHashDCT63bits com Hamming≤4. Qualquer candidato
   bloqueia até revisão documentada; nenhum descarte/rebalanceamento automático.
5. Auditor local independente reabre cada shard por SHA/IDs/shape/dtype,
   recalcula SHA224/336/pHash e triagem, verifica ordem completa e referências.
   Booleano no recibo, sozinho, não autoriza reuso. Triagem negativa não prova
   independência de pacientes; possíveis duplicatas abaixo do limiar persistem.
6. GPU, somente após identidade e pixels aprovados: encoder oficial congelado,
   SHA de pesos/config, T4/fp32/TF32off; reusar1.000features224 V03 e20features336
   G04 somente com hashes/seleção/pixels compatíveis. Recalcular224 dos20 pilotos
   para paridade de features (atol1e-4/rtol1e-5). Só codificar1.000treino+300dev.
7. Custo de resolução medido em pares iguais,10repetições com ordem alternada;
   não calcular razão de latência com populações diferentes por reuso de cache.

Extração completa de pixels reais ainda não iniciada neste checkpoint;
auditoria de duplicatas real pendente. Um ensaio local adicional de paridade
com os20IDs originais G04 foi bloqueado antes da primeira série: DICOM ausente
no HD para o primeiro estudo. Não trocar casos para produzir um “pass”. Recibo
`reports/avance_av037_g05/local_pair_v1.json`; zero qualidade avaliada.
Builder GPU se recusa a preparar fonte sem evidência
de identidade e replay independente dos pixels. Ainda não há integração de
extração das300features de confirmação em job GPU; implementar o estágio
separado e protegido somente após dev elegível, sem abrir seus rótulos agora.

## Critérios mantidos antes dos resultados

Manifesto V05:1.000treino/300dev/300confirmação, sem alterações. Protocolo
`reports/avance_av036_g05/protocol_v2.json` inalterado. DINOv2-S CLS384
congelado; mesmas três séries/fatias, StudyAttention384→64→1 e classificador12;
BCEWithLogits suave não ponderada, AdamWlr0,001/wd0,0001, batch4,
20épocas fixas, seeds2026/42, mesma inicialização/ordem; única variável é224/336.

δ=336−224. Exigir no dev e depois uma única vez na confirmação:

- BCE média: δ≤−0,001, IC95% pareado superior<0, melhora nas duas seeds.
- Por condição: regressãoBCE≤0,01; IC simultâneo Bonferroni superior≤0,02.
- AUC0/1 exclui rótulos suaves/incertos; nenhuma queda pontual>0,02;
  ≥10positivos/10negativos por condição e ≥95%bootstraps válidos; alerta<20/classe.
- 5.000 bootstraps seed20260930, estudo como unidade e agrupamento por
  componentes paciente/laudo; IC dos braços e diferenças, todas regressões.
-GPU≤7.200s incluindo extração e heads, picoalocado≤4GiB, forward336/224≤3;
  quota≥14.400s antes do dispatch. Registrar alocado/reservado e tempos por fase.

Sem resultados reais neste relatório: BCE/AUC por condição, IC e regressões
G05 **não medidos**. Rótulos fracos derivados do teacher não são ground truth
clínico; confirmar sobre o mesmo teacher não elimina seus erros/exposição
histórica. Não promover G05 com base apenas em teste unitário ou piloto G04.
Dev positivo congela os mesmos quatro heads e receita; confirmação abre uma
vez com marcador persistente, sem retreino, troca de seeds, blend ou ajuste.

## Recursos, custo e preservação

API30/09/2026 às21:46:32UTC/18:46:32BRT:
GPU68.516,072s =19,03h livres; reserva0; reset03/10 às00hUTC =02/10 às21hBRT.
Deadline22/10/2026 às23:59UTC =20:59BRT; entrada/equipe15/10 no mesmo horário.
Limite5submissões/dia,1observada hojeUTC. HD~222GiB livres.
Recibo `reports/avance_av037_g05/resources_final.json`.
Quota não comprova vaga física; auditar jobs próprios e reconsultar no dispatch.
Consulta21:29:40UTC:100jobs recentes próprios cobrem24h; nenhum jobGPU recente
ativo/em fila observado. `slots_v1.json`. API não garanteT4 nem mostra todas
as sessões interativas não salvas; não iniciar probe para reservar vaga.

Custos atuais: um jobCPU de headers concluído,2.648,918s, zeroGPU novo, zero treino,
zero envios, zero serviços pagos. Ensaio local de paridadeCPU:5,092s,
0séries concluídas, bloqueado por DICOM ausente; não conta como validação.
Limite do futuro pixelCPU: guard6.900s,
timeout7.200s; cacheuint8 estimado~2,2GiB bruto, compressão ainda não medida.
G04 medido:31,533s, razão336/224=2,219962, pico336~204MiB; extrapolação
antiga~93min para1.300casos/duasresoluções não é medição/garantia G05.

Pins preservados:

| Artefato | SHA256 |
|---|---|
| H46 fonte | `7dc49666e01c46e5017d4975960b06b359e869d8fd916d1be41cb90561beb522` |
| H46 CSV público | `7c6dfe8ba6c71d557d2a6b21af8a96bddfeb4b75626ee38a7a8cae44ec80c23d` |
| V05 manifesto | `9bb462aee209c132af51073851525e885545c76627fe542c0248361673e6de84` |
| G05 protocolo | `90c68b302630749b833fc136947170989c47643ac3a74e73992f8ca7d7eaa863` |
| G05 contrato científico | `1f7bc36b4376263c9cf69e5eb0bdb53d13e8bbd33eaa77707f07a7b2d15c8448` |
| Job headers v1 | `7f77a7e52445f93fd1d0f62306c7784504cec50d057a23345c06efea9e3ce14a` |
| Headers completos | `21705a260bce91d5682c28c519a87ca1782d2c84dc3761b480c036010d95d8c8` |
| Recibo headers | `56fcba3d786f66f61d04ead539d91be806f3c7c7b4d72c07699e63fa96a44b6d` |
| ExtratorCPU técnico v2 | `b01b583221c392b4131dd6771c9fb03f56467dc2c7c6bd76dfb2335432980655` |
| V03 features224 | `bbc34ca678aed570c4325b283b7a33901134cd29b6b4314ad11c207c07012971` |
| G04 features224/336 | `3175b1a08d08a80fd371e8c9c10ed1dde2fd4d93b0ed27d1215764dfe45d9948` |

## Próxima ação elegível

- [x] Reconciliar conclusão do kernel136572534v1 e auditar recibo completo.
- [ ] Fechar semântica longitudinal/namespace do paciente com evidência específica.
- [ ] Executar e auditar extraçãoCPU pareada uma vez, sem avaliação da confirmação.
- [ ] Apenas com gates anteriores aprovados, reconsultar GPU/quota/vaga e
  extrair1.300features; comparar224/336 nas duas seeds pelo protocolo congelado.
- [ ] Dev aprovado: proteger e abrir confirmação uma única vez; reportar decisão.
- [ ] Entregar candidato real, custo e recomendação ao JV para ação pessoal.

**Nenhum upload/submissão automático à competição; H46 não será substituído
por um candidato sem evidências.** Upload de datasets/pesos, seleção final,
contato com organizadores e serviços pagos não foram realizados.

Código implementado e auditado no commit `0ff319e`; arquivos privados de
dados/caches/recibos não entram no Git. O envio de código ao GitHub não é
submissão à competição. Não alterar fonte/CSV H46 nem os pins científicos G05.

### Correção técnica de empacotamento, antes de qualquer qualidade

O teste adicional da fonteGPU integral detectou que a compressãozlib excederia
o limite de1MB ao incorporar toda a proveniência. Corrigido paraLZMA/base85
(biblioteca padrão), versão técnica do extratorv2. Não foram removidos hashes,
estudos ou evidências; receita científica e protocoloV2 permanecem idênticos.
A fonteCPU antiga `pixels_v1.py` nunca foi despachada; foi preservada como
histórico, sem caches reutilizáveis. Versão atual `pixels_v2.py`:729.154bytes,
contrato técnico `9a38c007466d8522a96b3eec0b367246c012721d8aea7dc92c59c3fee032137b`.
O teste de tamanho GPU usa evidência explicitamente sintética em diretório
temporário, não certifica pacientes reais nem produz fonteGPU utilizável.
Não houve upload de dataset/modelo nem dispatch de pixels/GPU; o candidato é
código de auditoria/extração, ainda não um modelo ou Notebook apto à submissão.
