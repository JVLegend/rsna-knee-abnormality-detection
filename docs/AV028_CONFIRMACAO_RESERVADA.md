# AV-028 — confirmação reservada V04

#RSNA #Kaggle #Pesquisa

21/09/2026. Fonte canônica:07_Estrategia_AVANCE do vault.
Implementação do [protocolo AV-027](AV027_RESULTADOS_R02_E_CONFIRMACAO.md),
fixado no commit bb7897a antes de consultar as previsões reservadas.

## Resultado final — não confirmado

Kernel135297768 v1 COMPLETE; auditoria PASSED_V04_CONFIRMATION_AUDIT.
Isso atesta integridade dos artefatos, não superioridade: decisão
**NOT_CONFIRMED**, com3dos4gates prospectivos reprovados.
Os150estudos/147grupos agora foram avaliados; não reutilizar para seleção.

SoftBCE menor é melhor; referência fraca congelada, não rótulo clínico:

| Receita | Intacto2026 | Intacto42 | Ausências2026 | Ausências42 |
|---|---:|---:|---:|---:|
| Controle | 0,605170901 | 0,598663756 | 0,631176056 | 0,626289777 |
| Paired | 0,603563784 | 0,602545337 | 0,614469876 | 0,614488246 |

Média intacta0,601917329→0,603054561: piora+0,001137232.
IC95% agrupado[−0,002663697;+0,005025211],5.000réplicas/seed20260921.
Média ausências0,628732916→0,614479061: melhora nas duas sementes.
Regressões médias MedialOA+0,017349355 e LateralOA+0,021597027
ultrapassam o teto prévio0,01. Gate de ausência passou; intacto nas duas
sementes, intervalo abaixo de zero e limite por alvo falharam.

AUC weak secundária: control0,754226/0,760428; paired0,758379/0,764292.
Essa melhora de ranking NÃO substitui os critérios primários já fixados
nem corresponde à AUC do Kaggle. Não mudar retrospectivamente a métrica.

Decisão: não promover paired nem submeter. Controle V03 permanece comparador,
não um novo vencedor escolhido neste teste. Não selecionar seed2026,
misturar colunas ou tentar outro peso de consistência nos mesmos150.
O sinal de robustez fica documentado como hipótese, não melhora universal.

Próxima rodada V05: inventariar grupos ainda não avaliados e congelar novo
desenho de validação/OOF e confirmação, com orçamento e exclusões auditáveis.
Depois, G04 resolução224vs336 da fila; não precisa insistir nesta loss.
Não implementado ainda; nenhum novo treino/manifesto/resultado prometido.

### Evidência e exposição

- Auditoria privada: reports/avance_av028_v04/audit_v1.json,
  SHAb821b2caafbdfe51f3c1be4961b8707548632a4f5bdb28a9f1912369326f5be1.
- ReceiptSHA1677cbc7fec84b8e57f20ff477ce67b79c0664a0b735e66e614f0171ed6455cc.
- ExposureSHAd597e24e33a78983f1b37961af1d2fdfca2b22474300ca573d6f746ee8c27ae9,
  stage predictions_complete. Auditoria confirma confirmation_evaluated=true.
- Todas450séries e150IDs preservados; sem colisão exata treino/dev encontrada.
- Replay NumPy independente: diferença máxima1,145590744e−6.
- Runtime151,0931s, features150,5285s; exclui inicialização/imports.
- Fontes/protocolo bcd2b76 publicados antes da execução;211testes+44subtestes.
- Outputs privados completos no HD; nenhum dado/label/peso incluído no Git.

Não alterar o manifesto histórico V01 para registrar exposição: seu hash
continua fixo. O estado atual está nesta nota/vault e nos receipts V04.
As seções seguintes preservam o protocolo congelado e o histórico do envio.

## Congelamento anterior à inferência

Build privado no HD: reports/avance_av028_v04/v04_v1.py,372.104bytes.
SHA256:1270ab3045f2294f47ffedb2e9d2fb7087a513bc9583efe0d0611b3eac68f50f.
O literal V04 embutido é o manifesto imutável:150IDs/labels/grupos/séries,
quatro checkpoints, proveniência e hashes de pixels das3.750séries de
treino/desenvolvimento. Esses dados não são commitados no Git.

Somente control/paired_consistency,seeds2026/42,épocas8/14; hashes AV-027.
Nenhum novo treino, ajuste de probabilidades, escolha de alvo ou ensemble.
Helpers de decoder/percentis/geometria/modelo copiados do build V03 congelado.
Não alterar builders antigos nem substituir casos que falhem nas verificações.

- Builder verifica IDs/grupos, seleção de séries, rótulos e fontes fixadas.
- Runtime T4x2/offline extrai450triplas físicas adjacentes,224px/DINOv2-S.
- Bloqueia colisão exata com treino/dev antes de qualquer previsão do head.
- Salva features, geometria/cabeçalhos, marcador de exposição e logits.
- Auditor local reconstitui a geometria e faz replay NumPy dos quatro heads,
  intactos e com cada plano ausente; só então calcula métricas.
- Features do encoder têm proveniência/hash/forma verificados; não há uma
  segunda extração independente de pixels/encoder. Não alegar essa garantia.

## Decisão fixa

Paired precisa melhorar softBCE intacta e média dos três planos ausentes
em AMBAS sementes, por mais de2e−6. Bootstrap agrupado por laudo,
5.000réplicas/seed20260921,delta intacto médio das sementes/alvos,
intervalo95% com limite superior<0. Nenhum alvo pode regredir>0,01
na média das duas sementes. Todos os gates são necessários.

AUC contra professor fraco é diagnóstico secundário: exclui0,5incerto,
binariza demais valores>0,5; não serve para promover ou calibrar modelos.
Auditoria aprovada não significa confirmação positiva. Se NOT_CONFIRMED,
registrar o resultado; não ajustar e reconsultar estes150 para seleção.
Após exposição, futuras alternativas exigem novo desenho de validação/OOF.

## Execução e retomada

Slug previsto:jvlegend/rsna-knee-v04-reserved-confirmation.
R02 confirmado COMPLETE; busca inicial não encontrou V04 existente.
GPU restante83.004,021396s no primeiro pré-check; consultar novamente antes
de lançar. Teto1.200s/guard1.050s; exigir>=2.400s livres. Sem serviço pago.
Anexos: competição RSNA Knee, modelo oficial DINOv2-S small/1 e output R02.
Docker fixado igual ao V03/R02. Kernel privado, sem internet ou submission.csv.
Não interromper jobs de outros projetos se as vagas estiverem ocupadas.

211testes+44subtestes passaram, incluindo14novos. Código/protocolo
commitado/push bcd2b76 antes da inferência. SaveKernel aceito:
ID135297768 v1, jvlegend/rsna-knee-v04-reserved-confirmation,
quota81.881,219227s no pré-envio. Não relançar: recuperar esta versão.
Primeira tentativa de pré-check falhou localmente por acessar a resposta
tipada como dicionário; nenhum SaveKernel foi chamado nessa tentativa.
Corrigido acesso status.status; uma única chamada SaveKernel aceita.
Na falha: baixar outputs e verificar v04_exposure.json/v04_failure.json
antes de decidir qualquer recuperação. Nunca relançar cegamente.

Inferência baixada para reports/avance_av028_v04_v1/ e auditada uma vez
com scripts.assess_v04_confirmation; resultado já salvo em
reports/avance_av028_v04/audit_v1.json. Não relançar o ensaio.

Sem submissão nova. Público confirmado0,941 continua do ensemble anterior;
este ensaio não mede leaderboard nem desempenho clínico independente.
