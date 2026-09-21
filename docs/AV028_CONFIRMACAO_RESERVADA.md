# AV-028 — confirmação reservada V04

#RSNA #Kaggle #Pesquisa

21/09/2026. Fonte canônica:07_Estrategia_AVANCE do vault.
Implementação do [protocolo AV-027](AV027_RESULTADOS_R02_E_CONFIRMACAO.md),
fixado no commit bb7897a antes de consultar as previsões reservadas.

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

211testes+44subtestes passaram, incluindo14novos. Não lançado ainda;
código e protocolo serão commitados antes da primeira inferência.
Na falha: baixar outputs e verificar v04_exposure.json/v04_failure.json
antes de decidir qualquer recuperação. Nunca relançar cegamente.

Inferência concluída deve ser baixada para reports/avance_av028_v04_v1/.
Auditar uma vez com scripts.assess_v04_confirmation e salvar
reports/avance_av028_v04/audit_v1.json. Arquivos privados no HD externo.
Atualizar esta nota e o vault com ID/versão, exposição e resultado real.

Sem submissão nova. Público confirmado0,941 continua do ensemble anterior;
este ensaio não mede leaderboard nem desempenho clínico independente.
