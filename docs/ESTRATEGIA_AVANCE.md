# Estratégia de execução — comando AVANCE

#JoaoVictor #Kaggle #Tecnologia #Academia

Criado em 14/09/2026. Plano operacional aprovado pelo pedido do JV para
transformar as pesquisas em alternativas e testá-las a cada comando AVANCE.
Atualizado na AV-022 (17/09): M01 pooling treinado e auditado, seis combinações.
146testes/44subtestes passaram; controle AV-021 reproduzido.
Média simples teve menor softBCE média0,625912, mas não venceu nas duas
sementes; atenção por alvo também inconsistente. Referência shared mantida.
Próximo G01: cortes adjacentes versus espaçados; sem nova submissão.
Melhor público **0,941** preservado; softBCE de rótulos fracos não é score Kaggle.


Espelho operacional da fonte de verdade no vault:
/Users/iaparamedicos/Documents/GitHub/SuperJV/01_Projects/Competicoes/RSNA_Knee_Abnormality_Detection/07_Estrategia_AVANCE.md.
Ao executar AVANCE, atualizar a nota canônica e este espelho. Evidências do
repositório: docs/PLANO_MELHORIAS_2026-09-05.md e docs/LOG_EXPERIMENTOS.md.

## Objetivo e ponto de partida

Melhorar o modelo de imagem para os 12 alvos, com experimentos comparáveis,
submissões rastreáveis e custo medido. Trabalhar em três frentes:
reprodução pública forte, treino próprio e combinação/eficiência.

Melhor referência pública: H43A probe22, público **0,941**, ref 56263721.
H43 parent 0,939, ref 56253529, permanece comparador histórico.
H-38 0,929 e H-36 0,928 permanecem disponíveis; seleção final não alterada.
H-42 terminou com 0,881 e não será promovida. Sua avaliação local de 0,995527
já tinha exposição dos pesos ao gold; a queda no leaderboard não identifica,
sozinha, qual componente causou a diferença.

O parent público 0,939 foi reproduzido pela H43 na AV-005, superando H38.
O preset probe22 reproduziu **0,941** na nossa submissão, confirmado na AV-012.
Meta pública de superar 0,94 alcançada; ganho público não garante o privado.

## O que significa AVANCE neste projeto

1. Ler este documento e o log antes de agir. Aceitar AVANCE/avance; quando o
   contexto for esta competição, pequenas variações de digitação têm o mesmo
   sentido. Pedidos de status apenas consultam e informam o estado.
2. Reconciliar a experiência em andamento com processos locais, kernel,
   versão e eventual submission_id. Se já estiver executando, recuperar o
   estado e continuar; não lançar uma cópia após queda de rede ou do Mac.
3. Escolher a próxima experiência elegível da fila. Implementar, verificar,
   executar o teste delimitado, interpretar e registrar. Planejar sozinho
   não conta como testar. Cada avanço deve produzir resultado, artefato
   executável ou progresso verificável de uma execução longa.
4. O comando autoriza as etapas rotineiras dessa fila: código, testes,
   inferência/treino nos recursos já disponíveis, downloads incrementais
   necessários no HD e submissão Notebook-only quando passar pelos critérios
   abaixo. Não pedir novamente uma aprovação genérica de continuidade.
5. No máximo uma nova submissão à competição por AVANCE, depois de consultar
   a cota disponível e confirmar que o mesmo candidato não foi enviado.
   Implementação e ablações locais podem avançar mais de uma linha da fila
   quando forem baratas e independentes.
6. Se GPU, dados ou uma dependência impedirem o teste, registrar o bloqueio
   específico e trabalhar no próximo item elegível. Perguntar ao JV somente
   por escolha necessária que extrapole o plano, como serviço pago novo ou
   uso de componente com restrição incompatível com o escopo acordado.
7. Encerrar cada rodada com: experiência, mudança, resultado e comparador,
   decisão, envio realizado ou não, e próximo item. Atualizar o cursor,
   tabela e registro antes da resposta.
8. O trabalho ocorre ao receber o comando. Este documento não cria agendamento
   nem execução autônoma entre mensagens. Se houver processo remoto iniciado,
   registrar sua identidade para retomá-lo.

“Testar todas” significa dar destino documentado a todas as alternativas:
testada, rejeitada, inconclusiva, bloqueada ou não aplicável com justificativa.
Uma dependência reprovada pode impedir um teste derivado; nunca chamar isso
de teste concluído nem treinar combinações sem base apenas para preencher a lista.

## Cursor de retomada

- AV-023 iniciada17/09: revisão Kaggle solicitada + implementação G01.
  Novos códigos consultados: Geometry to6Slots (Xiaolei Lian), The bee's knees
  (Prvsiyan, revisão17/09) e Labeling Deathmatch (Joshua Ziel, revisão17/09).
  Fontes/código preservados em reports/avance_av023_sources/. Resultados
  negativos recentes desaconselham prometer salto com blend/intensidade/LLM.
  Teste prospectivo: controle V02 congelado; quartis em ordem física;
  adjacentes físicos centro±1. Comparar ordem física vs controle, depois
  adjacência vs quartis físicos. Mesmos3planos/1view por plano/3canais/224px,
  mesma normalização por fatia, encoder congelado, shared,teacher299/250,
  seeds2026/42,20épocas,batch4,AdamW0,001/0,0001. Confirmação150fechada.
  Não mudar crop, contraste, número de slots ou labels junto com geometria.
  Gate geométrico: todos headers finitos/consistentes, projeções distintas;
  não preencher metadados ausentes silenciosamente. Verificar pixels V02
  dos mesmos arquivos por hash; não sobrescrever o cache original.
  Reproduzir controle e auditar logits/BCE. Promover apenas melhora >2e−6
  nas duas sementes versus V02; registrar também delta pareado entre braços.
  Orçamento1jobT4/offline/1.800s, sem CSV ou submissão automática.
  Protocolo antes de avaliar: docs/AV023_PESQUISA_E_GEOMETRIA_G01.md.
- AV-022 concluída (17/09): PASSED_M01_AUDIT; protocolo prévio commit538a0e9.
  Controle shared reproduz AV-021. SoftBCE seeds2026/42:
  shared0,626070/0,631895;mean0,627338/0,624486;target0,633160/0,624731.
  Mean tem menor média0,625912, mas ambas alternativas pioram seed2026.
  Critério pré-registrado não atingido; manter shared, mean em reserva.
  Não é evidência de ganho clínico/LB nem prova de inutilidade da atenção.
  Kernel jvlegend/rsna-knee-m01-pooling-ablation ID134795832 v1 COMPLETE;
  seis treinos em36,02s sem imports,76,74MiB pico alocado, zero DICOM reextraído.
  146testes+44subtestes passaram; replay NumPy até1,21e−6.
  Outputs reports/avance_av022_m01_v1/; auditoria
  reports/avance_av022_m01/m01_audit_v1.json; buildm01_v1.py SHA5a93e1eb… .
  Não repetir nem submeter este kernel. Sem CSV/avaliação confirmation150.
  Próximo G01: registrar adjacentes versus espaçados com mesmos centros,
  número de views, crop, encoder e referência shared. Medir índices/gaps/
  orientação antes; reconstruir cache alternativo separado, sem sobrescrever
  V02. Não misturar mudança de geometria com cabeça/teacher novos.
  Protocolo/resultados: docs/AV022_ABLACAO_POOLING_M01.md no HD externo.
- AV-021 concluída: V02 próprio treinado e PASSED_V02_BASELINE_AUDIT.
  Prior softBCE0,65707840; seed2026época10=0,62607019;
  seed42época6=0,63189521. Ambas melhoram; época20 piora dev. Baker's piora
  nos dois modelos. Não é AUC clínica nem melhoria confirmada de leaderboard.
  Features164,57s,total175,04s fora imports; pico153MiB; replay logits até1,04e−6.
  Próximo M01: média dos3planos como controle vs atenção por alvo, usando
  mesmas features/splits/teacher/seeds e20épocas; registrar antes de avaliar.
  Não reextrair imagens à toa nem abrir confirmação. G01 fica na sequência.
  Cache PASSED_BASELINE_CACHE, SHA12938922…; fonte SHA0049a9b0… em
  reports/avance_av021_v02/baseline_v1.py. Código/protocolo commit69876d6.
  Kernel jvlegend/rsna-knee-v02-frozen-dino-baseline ID134788472 v1
  COMPLETE, não repetir. Saídas reports/avance_av021_baseline_v1/;
  auditoria reports/avance_av021_v02/baseline_audit_v1.json.
  Protocolo prospectivo preservado; docs/AV021_BASELINE_PROPRIO_V02.md.
  Treino299/dev250 da V01; confirmação150fechada. DINOv2-S genérico congelado,
  entrada/cabeça do piloto, duas sementes2026/42,20épocas,batch4,AdamWlr0,001/
  decay0,0001. Escolher menor softBCE média dev, primeira época em empate.
  Comparador: constante por alvo igual à média dos softlabels do treino.
  Não binarizar0,5 nem alegar AUC clínica; dev seleciona época, não é teste final.
  Auditados1.647arrays no HD e igualdade exata na reconstrução Kaggle antes do
  treino. Um job privado/offline/T4/teto1.800s executou as duas sementes, checkpoint
  a cada época, features preservadas; nenhuma submissão automática do baseline.
  Código prepare_v02_baseline/v02_baseline_runtime/assess_v02_baseline.
- AV-020 concluída: piloto134631088 v3 COMPLETE, PASSED_V02_PILOT_AUDIT.
  Não repetir piloto nem submetê-lo. Build SHA212fb597… em
  reports/avance_av020_v02/v02_fixed_v3.py; outputs reports/avance_av020_pilot_v3/.
  Auditoria reports/avance_av020_v02/pilot_audit_v3.json.36séries/108DICOMs
  exatos, features12x3x384 finitas, cabeça treinada e checkpoint retomado
  exatamente na mesma sessão GPU. Sem AUC/dev/confirmation; não é baseline.
  Causa v2: NumPy2.0.2 percentis com posição float32, reproduzida localmente.
  Correção float64/limites float32 preserva36/36séries nos dois ambientes;
  receita antiga32/36 no NumPy2.0.2. Helper histórico/cache/teacher intactos.
  Busca de anexos sem percorrer DICOM:0,001893s; reconstrução3,2128s,
  encoder0,8088s,total medido6,8730s (fora imports). Pico alocado153MiB.
  Projeção inicial treino299+dev250/20épocas:377,48s com margem2x, não garantia.
  Próximo: congelar protocolo/custo do baseline; auditar cache completo treino/dev,
  implementar299treino/250dev com professor original, seeds2026/42.
  Confirmação150fechada; teto inicial1.800s por execução e sem grade pelo LB.
  136testes/44subtestes passaram; docs/AV020_CORRECAO_PERCENTIS_V02.md.
- Histórico AV-019: submissão56281610 COMPLETE0,941, empata com histórico; não reenviar.
  Piloto134631088 v1 ERROR por pixels diferentes do cache. Reconstrução local
  das36séries/108DICOMs passou, artefato reports/avance_av019_v02/local_rebuild_v1.json.
  Versão2 de diagnóstico no mesmo kernel, T4/offline/1.800s, salva versões e
  arrays/hashes intermediários antes de falhar. Build SHA2b174453… .
  Comparar saídas em reports/avance_av019_pilot_v2/ antes de alterar receita.
  Não afrouxar igualdade exata nem substituir cache esperado; sem dev/confirmation.
  134testes/44subtestes passaram. V2 RUNNING na última consulta.
  Retomada e evidências: docs/AV019_PARIDADE_PIXELS_V02.md.
- Histórico AV-018 (superado pela AV-019): jvlegend/rsna-knee-v02-generic-dino-pilot
  v1 ID134631088, RUNNING. T4x2/offline/1.800s, piloto usa cuda:0 somente.
  Não duplicar. Leitura de logs expirou; isso não prova falha do kernel.
  Build reports/avance_av018_v02/v02_generic_pilot_v1.py SHAe310c754… .
  Auditoria local cache PASSED_TRAIN_CACHE_PIXELS:299estudos/897séries/
  2.691canais/94.557.531bytes, sem constantes/hash de pixels duplicado no treino.
  Canais espaçados em quartis, gaps2a80, não adjacentes. Piloto seleciona12
  estudos só por custo de pixels; deverá reconstruir36séries/108DICOMs iguais.
  DINOv2 genérico oficial congelado, cabeça própria de atenção compartilhada,
  teste16passos+checkpoint/retomada exata. Sem dev/confirmation ou AUC.
  Após COMPLETE, baixar em reports/avance_av018_pilot_v1/ e rodar
  scripts.assess_v02_pilot; comandos em docs/AV018_PREFLIGHT_V02_GENERICO.md.
  Só após gate e custo planejar treino V02 completo299/dev250, seed2026/42.
  105 testes locais passaram; gates CUDA ainda não confirmados.
- AV-017 concluída: diagnóstico L01 em299estudos/296grupos de treino.
  3.588pares;169sinais mudam contra extrator legado,17/723discordam do professor.
  Apenas725pares recebem sinal definido;653sinais antigos passam à abstenção.
  Não substituir professor nem chamar isso ganho de AUC.102testes passaram.
  Outputs privados no HD: reports/avance_av017_l01_v2/. Fila cega60casos
  com contexto completo e chave separada; nenhum adjudicado. V01/professor
  intactos; dev/confirmation não analisados. L01 efeito visual pendente.
  Próximo trabalho: V02 com pretreino genérico e professor original. Verificar
  pixels/cache/proveniência/custo e checkpoint antes do treino; a revisão L01
  não bloqueia o baseline. Retomada: docs/AV017_AUDITORIA_L01_TREINO.md.
- Submissão AV-016 **56281610**, agora COMPLETE0,941 (AV-019), enviada
  16/09/2026 12:00:59 São Paulo; scriptVersionId350342219. Não reenviar.
  Comparar com probe22 histórico56263721 (0,941), sem mudar seleção final.
  Enquanto aguarda, seguir V02/L01 com validação própria, sem grade de pesos.
  Kernel jvlegend/rsna-knee-ordered-probe22-submission v1 ID134618609
  COMPLETE, T4/offline/32.400s. Build
  reports/avance_av016_build/h43_ordered_probe22_v1.ipynb, SHAfcf4ec54… .
  PASSED_ORDERED_PROBE22_CANDIDATE:3estudos/15séries, componentes/raw iguais
  ao smoke, replay da fusão aprovado. CSV SHAff848c73… igual ao probe22
  visível histórico; não prova paridade oculta ou AUC. Etapas114,5048s.
  Outputs reports/avance_av016_candidate_v1/; auditoria
  reports/avance_av016_audit/candidate_v1.json. 97 testes passaram.
  Regras9h revalidadas; cota antes do envio17,8690h; após envio1usado/
  4disponíveis hoje. Único envio desta rodada. Melhor0,941 inalterado.
  Retomada: docs/AV016_CANDIDATA_ESTAVEL_PROBE22.md.
- Smoke jvlegend/rsna-knee-ordered-official-smoke v1 ID134610738 COMPLETE,
  PASSED_ORDERED_OFFICIAL_SMOKE. Outputs em reports/avance_av015_smoke_v1/;
  auditoria em reports/avance_av015_audit/smoke_v1.json. Não repetir.
  3estudos/15séries,5ramos/56hashes, IDs oficiais exatos, replay DINO/CoAt
  aprovado e sem fallback. Saída smoke_predictions.csv; SHA7d3b8bd4…,
  igual ao piloto histórico visível. Etapas117,6857s, sem extrapolar de3casos.
  O smoke usa outerparent0,60; probe22 histórico0,941 continua separado.
  Retomada: docs/AV015_PARIDADE_COMPLETA_E_SMOKE_OFICIAL.md.
  Nunca submeter o piloto de 30 minutos nem o benchmark com casos de treino.
- Par ordered completo serial134609177/prefetch134609181 COMPLETE.
  PASSED_ORDERED_FULLSTACK_PARITY na AV-015, sem relaxamento do gate.
  Todos os CSVs/diagnósticos exatos, DINO raw/replay iguais; Raptor inputs/raw/
  ranks iguais à âncora; CoAt raw/ranks/inputs/lotes/ambiente iguais ao run1 ABBA.
  Etapas624,3039→580,6483s (−6,9927%); Raptor/CoAt/fusão358,8682→310,1589s
  (−13,573%). Workers distintos: ganho observado, não atribuição causal total.
  Projeção7,4614h para1.322estudos hipotéticos;11,2585h para2.000. Tamanho
  oculto desconhecido. Arquivos em reports/avance_av014_{serial,prefetch}_v1/;
  auditoria em reports/avance_av015_audit/fullstack_v1.json.
- CoAt36 ABBA v1 ID134608117 COMPLETE, PASSED_ORDERED_REPEAT na AV-014.
  Inputs/ambiente iguais nas4 passagens; ordered raw/ranks exatos, delta0.
  Completion A1/A2:248/1296 valores raw diferentes, máx0,0001143664,
  sete estudos; ranks/CSV iguais. Lotes variam no modo antigo e ficam fixos
  no ordered. Evidência a favor da correção, não determinismo universal ou AUC.
  Tempos A/B/B/A:116,5058/113,9788/115,8829/115,7087s, média ordered114,9309s.
  Outputs em reports/avance_av013_coat_order_v1/; auditoria em
  reports/avance_av014_audit/coat_order_abba_v1.json.
- Par determinístico completo serial134588338/prefetch134588340 COMPLETE.
  Auditoria AV-013: FAILED_DETERMINISTIC_FULLSTACK_PARITY. Total613,5509 →
  602,0923s (−1,868%), Raptor/CoAt/fusão351,3457→320,2062s (−8,863%).
  Raptor inputs/raw/ranks iguais entre os modos E à âncora isolada; DINO
  público/raw e native iguais. CoAt:71/1296 valores raw, máx3,41088e−5,
  só estudos de índices11/13 no shard0; inversão ACL no checkpoint e06,
  dois valores finais CoAt mudam1/108. Demais CSVs e final idênticos.
  Outputs em reports/avance_av012_serial_v1/ e avance_av012_prefetch_v1/;
  auditorias fullstack_v1.json e coat_raw_v1.json em reports/avance_av013_audit/.
  Não atribuir todo ganho ao prefetch nem declarar causalidade da ordem
  histórica (lotes não registrados). Não promover só porque final coincidiu.
- E03 isolado jvlegend/rsna-knee-e03-prefetch-abba v1, ID 134542682 COMPLETE:
  12 estudos/70 séries, paridade exata; prefetch 63,73 s vs serial aquecido
  86,11 s (25,99% menos tempo). Auditoria local independente aprovada.
- E03 stack36 v1 ID 134543253 COMPLETE: 598,60 vs 667,72 s (−10,35%),
  cinco ramos/36 estudos/205 séries íntegros; CSV diferente em 6 valores.
  Raptor/CoAt idênticos, divergência começa no DINO público, antes do prefetch.
  Outputs reais: reports/avance_av008_fullstack_v1/. Nunca submeter.
- Captura jvlegend/rsna-knee-dino36-rank-order-capture v1 ID 134544795 COMPLETE:
  20 membros × 36 estudos × 12 alvos, offline/T4x2, só DINO. Nunca submeter.
  Mesmas previsões + duas ordens históricas reproduzem exatamente ambos os
  CSVs DINO. Soma inteira de ranks duplicados é invariável em 20 permutações,
  mas altera 14 valores contra o legado: é candidata distinta, não paridade.
- E03 par estável v1 serial134546480/prefetch134546487 COMPLETE: etapas
  710,0229 → 605,9772s (−14,654%), CSV final idêntico; DINO público/raw exatos.
  Outputs no HD em reports/avance_av009_serial_v1/ e avance_av009_prefetch_v1/.
  Gate FAILED_STABLE_FULLSTACK_PARITY: Raptor raw1727/1728 valores diferentes
  (máx0,0004003644), 2 ranks Baker's; native DINO2 valores Medial OA.
  Imagens/máscaras Raptor idênticas. Não atribuir automaticamente ao prefetch.
- Raptor36 determinístico v1 ID 134547620 COMPLETE e auditado na AV-011.
  ABBA: serial 287,49/240,80s; prefetch 177,11/175,46s. Média B 176,28s,
  −26,79% contra serial aquecido; inputs/raw/ranks exatos nas quatro passagens.
  Outputs: reports/avance_av010_raptor_determinism_v1/. Só Raptor, não stack.
- Repetição jvlegend/rsna-knee-raptor36-deterministic-repeat v1 ID134548619
  COMPLETE e PASSED_CROSS_SESSION_PARITY na AV-012. Mesmos36/205,
  ambiente/modelos/IDs/inputs/raw/ranks exatos; 193,94s vs B2 175,46s.
  Sessão nova; host físico não identificável. Não demonstra novo ganho de AUC.
  Artefatos em reports/avance_av011_raptor_repeat_v1/; auditoria em
  reports/avance_av012_audit/repeat_v1.json. Sem reexecução duplicada.
- Native DINO classificado por teste isolado: diagnóstico preservado, enquanto
  o DINO público substitui o arquivo principal. Quatro variantes native não
  alteram a saída; público ausente/inválido falha. Não alteramos o gate completo.
  Na AV-012, soma weighted native ordenada por ID em ambos os modos para
  remover dependência da conclusão dos workers. Não muda pesos nem exclui
  diagnóstico do gate. Causa cuDNN não confirmada.
  Native ficou idêntico no par AV-012; agora a primeira divergência é CoAt.
  Retomada: docs/AV015_PARIDADE_COMPLETA_E_SMOKE_OFICIAL.md.
- Piloto jvlegend/rsna-knee-h43-parent-strict-pilot v1, ID 134328295:
  COMPLETE, PASSED_PARENT_INTEGRITY; 3/3 estudos, cinco ramos, gate em 289,22 s.
- Benchmark v1 ID 134423935 COMPLETE: 36 estudos / 205 séries, todos os ramos,
  zero fallback CoAt. Etapas 667,72 s. Projeção com margem: 8,57 h para 1.322
  ou 12,94 h para 2.000 estudos hipotéticos; tamanho oculto não confirmado.
- Kernel jvlegend/rsna-knee-h43-parent-strict-submission, v1,
  ID 134484564 COMPLETE; CSV idêntico ao piloto, cinco ramos íntegros.
  T4x2, offline, limite 9h; último evento aos 281,49 s.
- Preflight CPU: jvlegend/rsna-knee-h43-artifact-preflight, v1,
  ID 134328050, COMPLETE; 56 arquivos/4,48 GB em 48,39 s da função.
- Submissão H43 enviada na AV-004: ref 56253529, scriptVersionId 350055640,
  15/09/2026 08:47 São Paulo; COMPLETE, público 0,939, confirmado na AV-005.
- Kernel A02: jvlegend/rsna-knee-h43-probe22-strict-submission v1,
  ID 134536133 COMPLETE. T4x2/offline/9h, cinco pesos externos publicados,
  mesmas células de inferência; comparação pareada aprovada, último log 248,28 s.
- Submissão A02: ref 56263721, scriptVersionId 350168027, 15/09/2026 19:51
  São Paulo, COMPLETE **0,941**, confirmado via API na AV-012 (16/09).
  Um envio na AV-005; nenhum novo na AV-006 até AV-015, não reenviar.
- Melhor candidato confirmado: H43A probe22 0,941; parent0,939 e H380,929 preservados.
- Depois da confirmação A02, seguir outra família (V02 ou E03 conforme cota),
  sem grade de pesos no leaderboard; A03 permanece alternativa se houver bloqueio.
- A00: inventário estático concluído; não é reprodução do score público.
- V01: concluída; manifesto validation_weak_v1 congelado no HD, 299 treino,
  250 desenvolvimento e 150 confirmação; nenhum modelo avaliado nesta partição.

## Validação e regras de decisão

### Dois caminhos legítimos

**Reprodução pública (A):** pode chegar a uma submissão de confirmação sem OOF
independente dos pesos públicos, desde que código/versão/fontes e receita
sejam rastreáveis, a execução real esteja íntegra e o custo caiba na cota.
Registrar explicitamente “reprodução pública; OOF independente indisponível”.
Não fabricar validação independente nem ajustar os pesos nos 58.

**Pesquisa própria (V/L/G/M/E):** selecionar no conjunto de desenvolvimento
agrupado e confirmar em conjunto reservado, com pesos que não viram os casos
avaliados. Um encoder ajustado na competição não vira limpo apenas por trocar
a cabeça. Se sua exposição for desconhecida, iniciar do pretreino genérico.

### Protocolo inicial proposto

- Usar os estudos com dados completos disponíveis no HD. Congelar grupos por
  laudo normalizado; acrescentar paciente/duplicatas visuais quando houver
  evidência confiável. Laudos iguais não provam identidade do paciente.
- Formar desenvolvimento weak com pelo menos 250 estudos, se houver cobertura
  e classes suficientes, e confirmação separada com cerca de 150 estudos.
  Dimensionar depois do inventário. O restante serve ao treino. Se faltar
  alguma classe, registrar a limitação e ampliar dados antes da confirmação.
- O manifesto anterior cobria apenas os 58 gold. V01 criou
  data/processed/validation_weak_v1/manifest.json para 699 estudos weak;
  não confundir as duas partições nem usar o gold como teste virgem.
- Manter os 58 oficiais fora do treino dos novos modelos e como diagnóstico
  secundário. Já foram consultados repetidamente e não são um teste virgem.
- Labels de validação ficam congelados. Ao comparar professores, não medir
  cada modelo contra os rótulos que ele próprio gerou. Usar a mesma referência,
  análise de discordâncias e auditoria cega à variante; rótulos derivados ainda
  medem concordância com professor e podem compartilhar seus vieses.
- Triagem barata: uma dobra/partição e semente 2026. Repetir candidatos
  promissores com semente 42 e depois obter OOF agrupado mais amplo. Medir
  variação entre sementes no mesmo hardware; não importar o noise floor de
  0,002 de outro participante como nosso limiar.
- Registrar macro-AUC, AUC e número de positivos/negativos por alvo, cobertura,
  delta pareado e bootstrap por grupo, além de tempo, RAM e VRAM.
- Avançar quando houver melhora consistente nas repetições e sem regressão
  relevante nos alvos. Usar o conjunto reservado para confirmar a seleção,
  com uma consulta por lote de finalistas. Após consultar, ele deixa de ser
  virgem para pesquisas posteriores; registrar isso.
- Delta incerto: permitir uma repetição justificada. Se continuar inconclusivo,
  suspender e seguir a fila. Limite inicial: duas variantes por fator;
  resolução tem três níveis. Toda expansão exige hipótese registrada.
- Cada linha compara uma mudança principal. Só combinar vencedoras após os
  testes isolados. AUCs de texto, imagem, gold e weak têm referências distintas.

### Submissão e promoção

- Validar IDs, ordem, colunas, finitude, cobertura e composição realmente
  executada. Os três estudos visíveis só verificam execução; igualdade do CSV
  nesses exemplos não demonstra redundância no teste oculto.
- Fazer ranks após juntar todas as previsões, com tratamento explícito de
  empates. Rank por minibatch altera o modelo.
- Usar T4 quando CUDA for necessário, internet desligada na inferência final,
  e orçamento com margem para as 9 horas registradas nas regras. Revalidar
  limites vigentes no momento do envio.
- Estimar custo a partir de um lote representativo de dezenas de estudos,
  incluindo decode, forward e escrita; os três exemplos não dimensionam o teste
  oculto. Uma sessão de treino só começa com orçamento e checkpoint de retomada.
- Falta de membro, erro de decode relevante ou fallback global impede envio
  automático. Guardar o resultado diagnóstico e a âncora intacta.
- Confirmar resultado da submissão antes de promover. Um aumento no public LB
  cria um melhor público; não comprova superioridade no privado.
- Nenhuma mudança automática da seleção final oficial; manter H-38 disponível
  e documentar a recomendação de finalistas.

## Fila de experiências e alternativas

Todos os custos são classes relativas, a medir no preflight: baixo = CPU ou
inferência curta; médio = inferência de conjunto/treino de triagem;
alto = fine-tuning, múltiplas sementes ou folds. Não são horas prometidas.

| ID / família | Alternativa e comparação | Dependência / custo | Estado |
|---|---|---|---|
| A00 / H-43A | Inventário do parent público: fonte, versão, licença, hash, receita, número de membros e disponibilidade; confrontar com H-38 | início / baixo | REPRODUZIDA — auditoria estática AV-001; runtime e cadeia completa pendentes A01 |
| A01 / H-43A | Reproduzir primeiro um único parent fixado, sem pesos próprios; verificar predições e execução antes da confirmação Kaggle | A00 com fontes utilizáveis / médio | REPRODUZIDA — AV-005; ref 56253529 COMPLETE 0,939, +0,010 vs H38 |
| A02 / H-43A | Comparar parent com um único preset publicado escolhido previamente: halfway OU probe22; aproveitar previsões dos mesmos membros | A01 / baixo após inferência | CONFIRMADA — AV-012; submissão56263721 COMPLETE 0,941, +0,002 vs parent; OOF independente indisponível |
| A03 / H-43B | Se o stack integral bloquear, testar Native384Dense com sua receita nativa como alternativa ao MaxSpan H-36 | A00; pesos disponíveis e receita compatível / médio | PENDENTE |
| V01 / validação | Inventariar cobertura local e congelar treino/desenvolvimento/confirmacão por grupos, labels e hashes; avaliar ausência de classes | início / baixo | REPRODUZIDA — AV-001; 699 elegíveis, 692 grupos, 6 testes OK |
| V02 / validação | Treinar baseline próprio DINOv2-S 2.5D + atenção por estudo, partindo de pretreino genérico; repetir duas sementes para medir variação | V01 / alto | CONCLUÍDO — AV-021;duas sementes auditadas,softBCE dev0,62607/0,63190 vs prior0,65708;referência para ablações,sem AUC/submissão |
| L01 / H-44 | Professor atual versus negação antes/depois por idioma, escopo por cláusula e exclusão da indicação clínica | V01 para labels; V02 para efeito visual / baixo + treino | DIAGNÓSTICO PARCIAL — AV-017; treino299,17discordâncias/723comparáveis; não substituir professor; revisão e efeito visual pendentes |
| L02 / H-44 | Acrescentar consequências OA com atribuição ao compartimento e severidade, mantendo os demais alvos/labels | L01; referência de avaliação congelada / baixo + treino | PENDENTE |
| L03 / H-44 | BCE atual versus BCE com máscara de não mencionado e peso reduzido para incerto; distinguir gravidade de confiança | V02; labels com estados auditáveis / alto | PENDENTE |
| L04 / H-44 | Teacher atual versus consenso apenas nos casos discordantes, com abstenção se faltar evidência; regras ou LLM local já disponível | V01/V02; não repetir média irrestrita já negativa / alto | PENDENTE |
| G01 / H-45 | Triplets adjacentes nativos versus amostragem espaçada, mantendo centros, número de views, crop e encoder | V02; primeiro medir gaps reais do loader / alto | DIAGNÓSTICO CACHE V02 — AV-018; quartis com gaps2–80; ablação visual e outros loaders pendentes |
| G02 / H-45 | Cache 16 versus 32 fatias com mesma banda; medir fatias únicas/gaps e treinar cada receita compatível | G01; ramo que de fato usa cache 16 / alto | PENDENTE |
| G03 / H-45 | Banda central versus ampla mantendo densidade física semelhante; depois ablação da densidade com banda fixa | G01; contagem pode mudar para preservar densidade / alto | PENDENTE |
| G04 / H-45 | 224 versus 336; 384 só se 336 ganhar, crop físico fixo, treino e inferência compatíveis | V02; melhor amostragem congelada / alto | PENDENTE |
| G05 / H-45 | Global 140 mm versus crop anatômico menor de 100–110 mm, resolução fixa; avaliar periferia e estruturas finas | V02; definir centro reproduzível / alto | PENDENTE |
| M01 / H-46 | Média de grupos versus atenção aprendida por alvo, preservando posição física e máscara de presença | V02; pesos treinados para cada cabeça / alto | TESTADA AV-022 no V02 — mean/shared/target,2sementes; sem ganho consistente, shared mantida; coordenadas físicas novas fora desta rodada |
| M02 / H-46 | Especialista público Renta somente em Medial Meniscus, com peso fixado antes da avaliação e 11 colunas preservadas | A00; fonte/receita própria auditada / médio | PENDENTE |
| M03 / H-46 | Modelo próprio global versus global + ramo local para menisco/MCL; atenção espacial como alternativa ao crop fixo | G05/M01; escolher um mecanismo por rodada / alto | PENDENTE |
| M04 / H-46 | Adicionar posição em mm e máscara de protocolo à atenção; comparar com mesma cabeça sem posição | V02/M01 / alto | PENDENTE |
| R01 / robustez | Normalização por volume/série versus receita atual, com MONOCHROME1, rescale e paridade de intensidade verificados | V02; preservar contrato dos pesos públicos / alto | PENDENTE |
| R02 / robustez | Estresse de slot ausente/ruidoso; se houver queda, comparar treino normal com dropout de slots | V02; mistura de perdas somente depois do diagnóstico / médio + treino | PENDENTE |
| E01 / ensemble | Âncora + melhor componente elegível: peso global fixo pequeno, definido no desenvolvimento; medir correlação e delta por caso | A/V com predições comparáveis / médio | PENDENTE |
| E02 / ensemble | Probabilidade versus rank no mesmo conjunto de componentes, pesos e partição; evitar grid target-wise | E01 / baixo | PENDENTE |
| E03 / eficiência | Compartilhar decode; compartilhar prefixo congelado só se os tensores forem idênticos; distribuir braços nas duas T4 | referência estável / médio | APROVADA_LOCAL — AV-015; paridade completa/−6,99% no benchmark e smoke3/15 aprovado; confirmação oculta da nova receita pendente |
| E04 / eficiência | Destilar o ensemble aprovado em DINOv2-S ou CNN menor; comparar AUC, tempo e memória | E01 aprovado; teacher sem exposição ao fold avaliado / alto | PENDENTE |
| X01 / exploração | Head auxiliar de dependência entre alvos versus cabeça atual, com regularização; relações aprendidas apenas no treino | V02/M01; rótulos suficientes / alto | PENDENTE |
| X02 / exploração | Pré-treino auto-supervisionado nas imagens de treino de cada fold; depois fine-tuning com labels fixos | baseline próprio em plateau e orçamento disponível / alto | PENDENTE |

A geração de labels e o teste de efeito visual são etapas distintas. Marcar a hipótese testada somente depois da comparação visual.

## Ordem de execução e alternativas a bloqueios

1. Começar por A00. Se as fontes permitirem reprodução integral, executar A01;
   se uma fonte restrita/indisponível bloquear, registrar e seguir A03.
2. Concluir V01 em paralelo conceitual com a reprodução: pode usar CPU enquanto
   um kernel existente trabalha. No Mac, limitar a uma tarefa pesada local e
   memória limitada por lote, considerando o travamento relatado.
3. Produzir V02 e testar L01–L04. Os primeiros 700 estudos weak já baixados são
   um ponto de partida para inventário, não promessa de cobertura completa dos
   seis slots. Completar só as séries necessárias ao experimento.
4. Percorrer G01–G05 e M01–M04. Alternar uma família de geometria com uma de
   modelagem quando as dependências permitirem, para não concentrar toda a
   cota numa única hipótese.
5. Executar R01/R02 e E01/E02 após os primeiros candidatos aprovados.
6. Usar E03 para ganhar capacidade de executar os finalistas. E04, X01 e X02
   ficam na fila para explorar conforme custo e resultados permitirem.
7. Se duas experiências consecutivas da mesma família falharem, passar à
   próxima família elegível. Não descartar resultados negativos: eles evitam
   repetir as mesmas tentativas.
8. Se o ganho desaparecer na confirmação, manter a âncora e registrar a
   regressão. Nunca zerar o histórico ao trocar de estratégia.

## Limites de interpretação que corrigem a pesquisa anterior

- Os números OOF 0,7931/0,7942 vêm do tópico de tamanho de encoder
  (fonte 3 abaixo); o tópico Base vs Small em 224 é outra experiência,
  de uma dobra/semente. São evidências distintas.
- H-42 já organiza três cortes por slab. Banda central estreita não demonstra
  falta de adjacência; G01 mede os índices e distâncias antes de alterar código.
  G02 não se aplica automaticamente a esse loader.
- H-36/H-38 já têm 62 janelas. Aumentar resolução ou fatias de um checkpoint
  sem respeitar seu treino não equivale ao ganho relatado por outro autor.
- Associação entre labels não autoriza regras como “OA implica menisco”.
  Usar correlações apenas como hipótese e validar contra a cabeça de controle.
- Um holdout weak maior reduz parte da variância, mas não elimina viés dos
  labels, exposição dos pesos ou seleção repetida. O tamanho 250 é uma proposta,
  não garantia estatística.
- Presets públicos parent/halfway podem reduzir extremos dos pesos; ainda não
  demonstramos que generalizem melhor no privado.

## Fontes e reutilização

Reavaliar termos e disponibilidade dos artefatos usados antes da execução.
“Equity-free” e permissão comercial são questões diferentes. As licenças
declaradas CC0/Apache são um filtro inicial dos artefatos; não bastam para
liberar dados, direitos de terceiros ou uso clínico/comercial de toda a cadeia.

O stack integral contém fontes NC e outras marcadas “other”. A00 deve apurar
esses componentes. A03 oferece continuidade com artefatos de licença declarada
permissiva; sua saída deve ser identificada como variante, sem herdar o score
0,941 do stack completo.

Pesquisa-base, consultada na rodada anterior:

1. [Stack público reestruturado](https://www.kaggle.com/code/maverickss26/rsna-knee-0941-restructured) e [inventário de fontes](https://www.kaggle.com/code/starkhushi/rsna-knee-0-940-source-checklist-4-diffs).
2. [Geometria/cobertura Raptor](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737696) e [adjacência](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737597).
3. [Tamanho do encoder e normalização](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735154); [outra ablação Base/Small em 224](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/738096).
4. [Validação em 58 estudos](https://www.kaggle.com/code/starkhushi/58-study-validation-set-is-lying-to-you) e [22 submissões e resultados negativos](https://www.kaggle.com/code/yosukeinada/rsna-knee-0-937-22-submissions-6-lessons).
5. [Vocabulário OA](https://www.kaggle.com/code/busyaprime/osteoarthritis-is-almost-never-written-as-oa), [negação multilíngue](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734106).
6. [Especialista de menisco](https://www.kaggle.com/code/renta0426/rsna-knee-0-937-weak-label-dinov2-meniscus-resid), [compartilhamento de execução 2xT4](https://www.kaggle.com/code/jiweiliu/rsna-knee-fast-2xt4-inference).

## Registro de cada AVANCE

Estados válidos: PENDENTE, EM_EXECUCAO, APROVADA_PARA_CONFIRMACAO,
REPRODUZIDA, REJEITADA, INCONCLUSIVA, BLOQUEADA, NAO_APLICAVEL.
A tabela registra o estado atual; o log preserva toda tentativa anterior.

Copiar este registro para cada execução em docs/LOG_EXPERIMENTOS.md e resumir
a decisão aqui, junto do cursor. Arquivos volumosos ficam em reports/,
models/, data/ e submissions/ no HD, fora do Git.

```text
Execução: AV-001 | experiência: A00 | hipótese: H-43A
Data e estado:
Pergunta / única mudança / referência:
Commit, código, fontes, versões e hashes:
Partição, IDs/grupos, labels congelados e exposição dos pesos:
Sementes, hardware, orçamento e critério definidos antes do teste:
Comando reproduzível / checkpoint de retomada:
Métricas por alvo, macro, cobertura, delta pareado e incerteza:
Tempo / RAM / VRAM / caminhos dos artefatos:
Kaggle: kernel, versão, submission_id, status, score público:
Resultado: observado | alegado pela fonte | não medido
Decisão e razão:
Próxima experiência elegível / bloqueio:
```

### Rodada de planejamento — 14/09/2026

Fila criada com 27 alternativas, dependências e protocolo AVANCE.
Nenhum treino, inferência ou envio novo nesta rodada. Próximo: A00.

### AV-001 — 14/09/2026 — A00 + V01

- A00: fonte H-43A fixada por SHA; 12 datasets, 2 outputs e modelo oficial.
  Manifesto DINO atual tem 20 membros/5 folds; A5 lista 5 folds; Raptor são
  4 views/3 checkpoints, e CoAt residual são 3 checkpoints. Heads E13 baixados,
  63,5 MB, hash idêntico ao exigido. Versões reais dos mounts, demais hashes e
  contagem integral de runtime ainda serão conferidos em A01.
- Risco encontrado: fonte captura falha do CoAt e prossegue sem esse ramo;
  também tolera falha na promoção DINO e preenche predições ausentes com 0,5.
  Não enviar apenas porque um CSV passou no schema. A01 exige composição
  completa e parent explícito 0,60, sem escolher pesos pelo gold.
- V01: 700 estudos/2.100 caches existentes; excluído 1 por compartilhar laudo
  com gold. Congelados 299 treino/250 desenvolvimento/150 confirmação, em
  296/249/147 grupos disjuntos. Todos os alvos com positivos e negativos.
  Labels 0,5 continuam incertos; confirmação sem avaliações de modelo.
- Manifesto HD: data/processed/validation_weak_v1/manifest.json; SHA-256
  365566b0f830e398e78bd36bd9118a7365c3af340f588e774c8bcc351398b6df.
  Segunda execução idempotente; 6 testes passaram, incluindo exclusões,
  grupos, NaN, incerteza e proteção contra sobrescrita.
- Evidência detalhada no repo: docs/AV001_AUDITORIA_E_VALIDACAO.md;
  código scripts/freeze_weak_validation.py; downloads em reports/avance_av001_sources/.
- Resultado observado: inventário e partição, não ganho de AUC. Sem novo
  treino/kernel/envio. H-38 0,929 preservada; próximo A01, alternativa A03.

### AV-002 — 14/09/2026 — A01 iniciada

- Código implementado: scripts/h43_integrity.py e scripts/prepare_h43_parent.py.
  Fixa parent 0,60, exige todos os membros, rejeita ausência/NaN e falhas
  dos ramos obrigatórios. Somente o gate final publica submission.csv;
  intermediários ficam em h43_candidate.partial.csv.
- 16 testes passaram, incluindo hashes incorretos, membros duplicados/ausentes,
  IDs, NaN, proteção contra drift da fonte e falha sem publicação.
- Preflight real na CPU Kaggle COMPLETE: 56 arquivos, 4.483.039.152 bytes
  para hash, 48,39 s. Kernel rsna-knee-h43-artifact-preflight v1, ID 134328050.
- Piloto rsna-knee-h43-parent-strict-pilot v1, ID 134328295, iniciado e
  consultado RUNNING. Offline, T4 solicitada, código exige 2 GPUs T4,
  limite 1.800 s. Build local h43_parent_strict_locked_v2.ipynb em
  reports/avance_av002_build/, com lock dos 56 hashes observados.
- Artefatos/log detalhado: docs/AV002_PARENT_ESTRITO.md. Outputs do piloto
  serão recuperados em reports/avance_av002_pilot_v1/.
- Cota antes do piloto: 24,38 h GPU; refresh informado 19/09 00:00 UTC.
  Nenhuma submissão, nenhum score novo; H-38 0,929 preservada.
- Retomar o piloto existente. Se completar, verificar recibo e cobertura;
  ainda falta medir lote representativo antes da versão de submissão.
  Se inviável, A03 é a alternativa. V01 permanece congelada.

### AV-003 — 14/09/2026 (15/09 UTC) — A01 piloto aprovado

- Piloto v1 COMPLETE: 20 fingerprints/membros DINO, cinco folds A5, Rad com
  calibrador, quatro views Raptor e três CoAt. CoAt sem fallback/erros.
  CSV 3×13 validado e hash conferido:
  7d3b8bd4e76b309171e2c44a58301e71da26104d1049cc9f8c445c17eee3c770.
- Gate aos 289,22 s; log termina aos 301,06 s. São três exemplos, não
  validação de AUC nem confirmação do score público do autor.
- Iniciado benchmark v1, ID 134423935, jvlegend/rsna-knee-h43-runtime-benchmark:
  36 estudos de treino V01/205 séries, seleção por quantidade de séries,
  sem labels para seleção, sem AUC, desenvolvimento/confirmação intocados.
  Último status RUNNING, offline, duas T4 exigidas, limite 1h.
- Implementados adaptador de benchmark e análise de cenários de runtime;
  22 testes passaram. Saída de treino se chama benchmark_predictions.csv;
  nunca submeter este kernel à competição.
- Detalhes no repo: docs/AV003_PILOTO_APROVADO_BENCHMARK.md. Outputs esperados
  em reports/avance_av003_benchmark_v1/. Não duplicar a execução existente.
- Próximo: medir tempos/limites e conferir cobertura. Se não couber, E03/A03.
  Nenhuma submissão ou score novo; H-38 0,929 preservada.

### AV-004 — 15/09/2026 — A01 benchmark aprovado, confirmação preparada

- Benchmark v1 COMPLETE; 36×13, 205 séries, cinco ramos, 56 hashes e T4x2
  verificados. CoAt com três checkpoints, zero fallback/falhas. Sem AUC.
- DINO 201,31 s; A5 29,48 s; Rad 49,43 s; Raptor/CoAt/fusão 387,50 s.
  Este último subtotal representa 58% do tempo e orientará E03.
- Projeção com margem 1,25: 8,57 h para 1.322 e 12,94 h para 2.000 estudos
  hipotéticos. Há risco de timeout; não conhecemos o tamanho real oculto.
  Limite de 9h/offline/Notebook-only revalidado no overview oficial.
- Criado kernel dedicado rsna-knee-h43-parent-strict-submission v1,
  ID 134484564, COMPLETE, teto 32.400 s, mesmo código/hash do piloto aprovado.
  CSV 3×13 idêntico ao piloto, IDs oficiais conferidos e cinco ramos íntegros.
  Último evento aos 281,49 s; CoAt zero fallback/falhas.
- Submissão efetivada: ref 56253529, scriptVersionId 350055640, às 08:47
  São Paulo. PENDING, sem score; um envio hoje, quatro restantes. Não
  reenviar enquanto aguarda avaliação. H38 0,929 e seleção final preservadas.
- 22 testes passaram; V01 permanece congelada e confirmação intocada.
  Reprodução pública; OOF independente indisponível, sem liberação comercial.
- Evidência: docs/AV004_BENCHMARK_E_SUBMISSAO.md; benchmark e projeção no HD
  em reports/avance_av004_benchmark_v1/. Saída dedicada em
  reports/avance_av004_submission_v1/. Retomar a execução existente.

### AV-005 — 15/09/2026 — A01 0,939 confirmada; A02 probe22

- H43 parent ref 56253529 COMPLETE, 0,939; delta público exibido +0,010
  contra H38 0,929. Nova referência pública, sem alterar seleção final.
  Score privado e OOF independente indisponíveis; não atribuir causalidade
  isolada a um componente do stack.
- Escolhido unicamente probe22 publicado: peso externo 0,75 ACL, 0,80 Medial
  Meniscus, 1,00 Lateral Meniscus, 0,75 Lateral OA/Fracture; outros sete 0,60.
  Mistura interna 60/40 e todos os modelos/geometrias permanecem iguais.
  Sem grade própria, sem consulta à confirmação V01 ou ajuste nos 58.
- Builder ancorado no SHA do parent e teste de identidade das demais células;
  gate exige RUN inteiro fixo. Comparador verifica parent diagnóstico contra
  execução anterior, sete alvos inalterados, IDs/hashes/recibos e CoAt íntegro.
  28 testes passaram; manifesto V01 manteve o mesmo SHA.
- Kernel rsna-knee-h43-probe22-strict-submission v1, ID 134536133,
  COMPLETE; T4x2/offline/teto9h, último log 248,28 s. Build SHA
  beb65225a4ed75d1ce2d60f0afdd5ee46c39ed843c151e465ff07e6eb80d572c.
  CSV 3×13 validado; parent idêntico ao anterior, sete alvos intactos,
  CoAt zero fallback/falhas. SHA novo
  ff848c73ba6f31e487d175161304d26df307e6e777189c01dfb508a76532ddfc.
- Submissão 56263721, scriptVersionId 350168027, enviada às 19:51:17 São Paulo.
  PENDING, sem score/erro informado. Único envio nesta AVANCE; três restantes hoje.
  Retomar pelo submission_id, sem duplicar; H43 parent 0,939 preservada.
- Detalhes: docs/AV005_PARENT_0939_E_PROBE22.md; outputs esperados no HD
  em reports/avance_av005_probe22_v1/, com paired_integrity.json aprovado.

### AV-006 — 15/09/2026 à noite (16/09 UTC) — E03 iniciada

- Probe22 56263721 ainda PENDING, sem score/erro informado. Não duplicada.
  H43 parent 0,939, H38 e seleção final preservados. Sem nova submissão.
- Implementado prefetch ordenado de um estudo, produtor CPU único, fechamento
  e espera antes da troca de receita; funções de modelo/geometria preservadas.
  33 testes passaram. Não é ganho de velocidade demonstrado ainda.
- Kernel rsna-knee-e03-prefetch-abba v1, ID 134542682 RUNNING; 12 casos/
  70 séries de treino V01, quatro passagens serial-prefetch-prefetch-serial.
  Offline/T4/teto1.800s. Só Raptor, nunca submeter. Sem validação/AUC.
- Critério: hashes de volumes/máscaras, IDs, probabilidades por ramo e ranks
  exatamente iguais; speedup ≥1,05 tanto ABBA quanto contra o serial final
  aquecido permite apenas teste futuro no stack completo, não promoção automática.
- Build SHA 223ec2a6ee262aae8202aa46866ee31df916454e9fa2a90399b39b258f3eeeb2.
  Registro repo docs/AV006_E03_PREFETCH.md; outputs esperados no HD em
  reports/avance_av006_e03_v1/. Retomar ambos os IDs antes de executar de novo.

### AV-007 — 15/09/2026 à noite (16/09 UTC) — E03 isolado aprovado

- ABBA COMPLETE: serial 123,55/86,11 s; prefetch 64,08/63,37 s. Média B 63,73 s,
  speedup 1,645× no ABBA e 1,351× contra serial final aquecido (25,99% menos tempo).
  Ganho só do Raptor em 12 estudos; não ganho comprovado do stack ou do oculto.
- Auditoria local recalculou métricas e conferiu quatro NPZs: IDs, imagens/
  máscaras, probabilidades dos quatro ramos e ranks exatamente iguais; delta0.
  Outputs reais em reports/avance_av007_e03_v1/, local_assessment.json aprovado.
  O serial final também teve CUDA 8,02 GB: não atribuir redução de RAM ao prefetch.
- Iniciado teste no stack completo, mesmos 36 estudos/205 séries de treino V01:
  rsna-knee-e03-fullstack36-prefetch v1, ID 134543253 RUNNING, offline/T4x2/1h.
  Apenas duas células adaptadas. Exigir CSV igual ao benchmark anterior,
  cinco ramos íntegros e avaliar tempos antes de usar no candidato real.
- 39 testes passaram; V01 preservada. Build SHA
  7d2dc6c5f3538c451ae332c22ae92949b29370c1f02a82b3f86af6123285cb5c.
  Registro docs/AV007_E03_GANHO_E_FULLSTACK.md; próximos outputs em
  reports/avance_av007_fullstack_v1/. Benchmark nunca deve ser submetido.
- Probe22 56263721 segue PENDING, sem erro/score novo; nenhum reenvio.
  H43 parent 0,939, H38 e seleção final preservados. Nenhuma automação nova.

### AV-008 — 15/09/2026 à noite (16/09 UTC) — ganho e diagnóstico de empates

- Stack36 COMPLETE: etapas 667,7184 → 598,6004 s (−10,351%); subtotal
  Raptor/CoAt/fusão 387,4955 → 328,8618 s (−15,131%). Workers diferentes:
  não atribuir todo o ganho ao prefetch. 56 hashes/T4x2/cinco ramos íntegros.
- Gate FAILED_EXACT_CSV_PARITY: quatro Lateral OA e dois Effusion mudam.
  Raptor/CoAt, native DINO e legacy-fold idênticos; DINO público muda primeiro.
- Captura DINO36 v1 ID 134544795 COMPLETE. Replay das MESMAS previsões com
  as duas ordens dos logs reproduziu ambos os CSVs históricos sem diferenças.
  Troca dos membros 8476b29285/72081758ce muda dois valores LOA e dois Effusion.
  Causa reproduzida: soma float64 sequencial seguida de novo rank rompe empates.
- Correção candidata: somar ranks médios duplicados como inteiros, sem dividir
  antes do rank final. Pesos uniformes/cobertura completa apenas. Zero diferença
  nas duas ordens e em 20 permutações; muda 14 valores contra o legado
  (Medial Meniscus2, Lateral OA4, PF OA2, Baker's6). Isso não prova melhora de AUC.
- Implementados auditor fullstack, captura e replay; 47 testes passaram.
  V01 congelada intacta; nenhum modelo avaliado em dev/confirmation.
  Fonte: docs/AV008_FULLSTACK_E_EMPATES_DINO.md; artifacts em reports/avance_av008_*.
- Decisão: manter H43 0,939 e probe22 56263721 PENDING, sem reenviar.
  Não promover o prefetch nem alterar seleção final. Próximo: baseline DINO
  estável + comparação serial/prefetch no stack completo, com gates explícitos.

### AV-009 — 15/09/2026 à noite (16/09 UTC) — DINO estável no par completo

- Implementado h43_stable_rank.py: ranks médios duplicados somados em int64,
  apenas 20 membros públicos com pesos unitários/12 alvos/cobertura completa.
  Valida IDs, finitude, shapes, duplicatas e limite de acumulação. Não muda
  _combine ponderado, native DINO, legacy-fold, A5, Rad, CoAt ou checkpoints.
- Teste sobre captura real bate com replay independente em 20 permutações.
  Builders serial/prefetch preservam todos os demais ramos e diferem somente
  nas declarações de modo. São candidatos distintos do parent histórico.
- 59 testes passaram, incluindo auditoria negativa de hash/modo/CSV.
- Kernels privados v1: rsna-knee-stable36-serial ID 134546480 RUNNING e
  rsna-knee-stable36-prefetch ID 134546487 RUNNING. Cada um offline/T4/teto
  1.800s; mesmos 36 estudos/205 séries de treino, cinco ramos e lock56.
- Auditor assess_h43_stable_fullstack.py preparado para comparar CSVs, inputs,
  raw e composição. Paridade e ganho de tempo remotos ainda não demonstrados.
  Não submeter estes benchmarks. V01 intacta; nenhum uso de dev/confirmation.
- A API de cota oscilou: primeira leitura 6h permitidas/10,31h usadas;
  duas seguintes 30h, com 19,69h livres antes do despacho. Gate exigiu ≥1h
  para os dois tetos de 30min. Nenhum recurso pago habilitado.
- Probe22 56263721 permanece PENDING; H43 0,939, H38 e seleção final preservados.
  Nenhuma nova submissão/automação. Detalhes, hashes e comandos:
  docs/AV009_DINO_ESTAVEL_STACK_PAREADO.md. Retomar os kernels existentes.

### AV-010 — 15/09/2026 à noite (16/09 UTC) — paridade final e variação Raptor

- Par stable36 COMPLETE; CSV final SHA
  87ee1ac74c3e58d7dadfef705634eb23745f2322ad002e7772b6d03bad2ee777 igual.
  DINO público e raw alinhado idênticos; replay independente aprovado.
  Etapas serial710,0229/prefetch605,9772s (−14,654%); subtotal Raptor/CoAt/
  fusão421,6460/337,2915s (−20,006%). Workers distintos: não isola todo efeito.
- Gate completo reprovado: Raptor raw1727/1728 valores mudam, máx0,0004003644;
  2 ranks Baker's mudam. Imagens/máscaras são idênticas. CSV native DINO
  muda2 Medial OA; demais componentes exceto _raptor.csv são idênticos.
  Igualdade final nos 36 casos não permite ignorar deriva intermediária.
- Raptor usa cudnn.benchmark=True e AMP fp16. Autotuning/backend é hipótese,
  não causa comprovada. Iniciado Raptor36 ABBA determinístico, v1ID134547620,
  offline/T4/1.800s: algoritmos determinísticos, benchmark/TF32 desligados,
  CUBLAS_WORKSPACE_CONFIG antes do preflight, seed2026, versões registradas.
  Não muda pesos/decode; flags mudam receita numérica, diagnóstico separado.
- Auditoria ampliada com deltas raw e suporte explícito 12/36 mantendo default12.
  Builder de smoke protegido implementado, mas gate real recusou a geração.
  66 testes passaram; V01 intacta. Sem avaliação AUC/dev/confirmation.
- Melhor0,939; probe22 PENDING, sem envio/seleção final/automação novos.
  Artefatos e próximo passo: docs/AV010_PARIDADE_FINAL_E_VARIACAO_RAPTOR.md.

### AV-011 — 15/09/2026 à noite (16/09 UTC) — determinismo local aprovado

- Raptor36 determinístico COMPLETE. Auditoria local conferiu flags, versões,
  56 hashes/T4x2, amostra de treino, quatro NPZs e 108 pares receita-estudo
  por passagem. Inputs, probabilidades e ranks exatamente iguais; delta zero.
  Raw SHA: 1f42760aec1b82bb875e0c456f9dd5b11a653546640f1daa29a91d7977e4c1f5.
- Tempos ABBA: 287,4890 / 177,1051 / 175,4631 / 240,7954s. Prefetch médio
  176,2841s; 1,4984× no ABBA e 1,3660× vs serial aquecido (−26,79% tempo).
  Só Raptor na mesma sessão; não prova causalidade do autotuner nem AUC.
- Iniciada nova sessão prefetch v1 ID 134548619, offline/T4/teto1.800s;
  mesmo código de modelo, decode e flags do ABBA. Builder só aceita auditoria
  bem-sucedida. Auditor de repetição preparado, sem tolerância numérica.
- Auditada promoção native/public: quatro constantes native (0/0,25/0,75/1)
  geram o mesmo arquivo principal público; diagnóstico native preservado.
  Público ausente/IDs inválidos causam erro. Código extraído por SHA, inferência
  simulada, sem GPU/pesos/treino. Não é prova geral de dependências dinâmicas.
- 71 testes passaram; V01 intacta; nenhum novo uso de labels/dev/confirmation.
  Sem alteração do gate completo, serving, seleção final ou automação.
  H43 0,939, probe22 PENDING. Ver docs/AV011_RAPTOR_DETERMINISTICO_E_ROUTING.md.

### AV-012 — 16/09/2026 — probe22 0,941; integração determinística

- API confirmou probe22 ref56263721 COMPLETE **0,941**, parent56253529
  COMPLETE0,939. +0,002 absoluto; reprodução pública, OOF independente
  indisponível. Seleção final oficial inalterada, nenhuma nova submissão.
- Repetição Raptor36 ID134548619 passou: inputs/raw/ranks/ambiente/pesos
  idênticos ao ABBA. Tempo193,9379s, contra175,4631s na referência B2.
  Não confundir estabilidade com speedup entre sessões ou prova de causalidade.
- Implementado par completo com CUBLAS antes do preflight e flags Raptor
  após definição da fonte/antes do forward. Estado backend e RNG restaurados
  em finally, inclusive se falhar. Ambiente/receita registrados no worker.
  CUBLAS é ajuste de processo e herdado por filhos; não alegamos isolar
  causalmente seu efeito no restante do stack.
- Native weighted usa ordem fixa por ID em ambos os modos, mantendo pesos.
  DINO público continua soma inteira exata. Receitas candidatas diferentes do
  parent histórico; não comparar seu score como se já tivessem sido submetidas.
  Gate inclui native e demais CSVs, raw/inputs, versões/flags e âncora isolada.
- 75 testes passaram; manifesto V01 intacto; dev/confirmation não consultados.
  Cota antes do despacho18,9291h. Par T4/offline/1.800s:
  serial134588338, prefetch134588340, v1. Não é submissão, nunca enviar treino.
- Próximo: auditar par; se passar e houver ganho, adaptar smoke à receita
  realmente validada. Builder de smoke anterior não contém essas flags.
  Detalhes/hashes/comandos: docs/AV012_PROBE22_0941_E_STACK_DETERMINISTICO.md.

### AV-013 — 16/09/2026 — Raptor integrado; CoAt sob investigação

- Dois kernels completos. CSV final SHA87ee1ac7… igual; Raptor raw/ranks
  iguais aos dois modos e à âncora. DINO público e native agora exatos.
  Gate reprovado por CoAt, não promovido:71/1296 diferenças raw em2estudos,
  delta máx0,00003410876;2ranks ACL do e06 e2valores CoAt mudam.
- Tempos de etapas613,5509→602,0923s (−1,868%); subtotal Raptor/CoAt/fusão
  −8,863%. Workers distintos; não demonstra ganho robusto ou melhora de AUC.
- Corrigido auditor local: legacy_fold_diagnostics.csv é recibo de5folds×4,
  não matriz de previsões. Valida schema/conteúdo/bytes; inventários CSV devem
  coincidir. Nenhum arquivo de previsão excluído, tolerância continua zero.
- Fonte CoAt baixada no HD (49.679bytes), SHA b11e58f8… igual ao preflight.
  Código consome FIRST_COMPLETED e forma batch2 com ordem de preparo variável.
  Lotes podem alterar microblocos e padding em fp16. Hipótese, não causa
  retrospectiva comprovada; teste CPU do loop real mostra pares diferentes.
- Experimento isolado134608117 v1: completion/ordered/ordered/completion,
  mesma amostra36/205 e3checkpoints. Ordered aguarda menor índice pendente;
  mantém preparo concorrente limitado, pesos, inferência e flags originais.
  Inputs/batches/ambiente capturados, sem treino ou labels. Teto1.800s,
  cota antes do despacho18,5429h. Auditor independente preparado.
- 83 testes passaram; V01 intacta; dev/confirmation intocados. Melhor0,941,
  nenhuma nova submissão/seleção final/automação. Registro e comandos em
  docs/AV013_COAT_LOTES_E_GATE_COMPLETO.md.

### AV-014 — 16/09/2026 — CoAt ordenado aprovado no teste isolado

- ABBA COMPLETE, auditoria independente PASSED_ORDERED_REPEAT: mesmos
  inputs e ambiente nas4passagens; raw/ranks/ensemble ordered exatamente
  iguais nas2repetições, delta0. Completion varia248/1296 valores raw
  em7estudos, máx0,0001143664; nesta amostra os ranks/CSV ficam iguais.
  Não há logs dos lotes da antiga divergência ACL para provar retrospectivamente
  sua causa. Não confundir correção de estabilidade com melhora de AUC.
- Tempos116,5058/113,9788/115,8829/115,7087s. Ordered médio114,9309s.
  Não declarar speedup robusto; duas repetições na mesma sessão.
- Builder e auditor do par completo com CoAt ordered implementados.
  Só célula57 muda contra cada build determinístico anterior: carrega cópia
  derivada do runtime com SHA31c4acb5…; preserva originais montados, pesos,
  flags e inferência. DINO/Raptor/fusão não alterados. Fallback de microbatch
  aborta. Recibo de receita, código e log do filho preservados.
- Novo auditor mantém todos os gates anteriores e compara CoAt raw/ranks,
  inputs, lotes e ambiente com run1 ordered auditado. Sem exclusões/tolerância.
  87 testes passaram, inclusive parse do código filho completo; V01 intacta.
- Par serial134609177/prefetch134609181 v1 iniciado, offline/T4/1.800s.
  Cota antes18,3958h; nenhum uso de dev/confirmation, novo envio, seleção
  final, automação ou serviço pago. Melhor0,941 preservado.
  Comandos e hashes: docs/AV014_COAT_ORDENADO_E_INTEGRACAO.md.

### AV-015 — 16/09/2026 — paridade completa e smoke oficial aprovados

- Par ordered COMPLETE e auditado: todos os componentes/CSV/diagnósticos
  exatos, DINO raw/replay iguais, Raptor e CoAt iguais às âncoras isoladas.
  CoAt inclui inputs/lotes/ambiente exatos. Nenhum gate excluído ou tolerância.
- Tempos624,3039→580,6483s (−6,9927%); subtotal Raptor/CoAt/fusão−13,573%.
  Ganho observado em36estudos/205séries; não é intervalo estatístico ou AUC.
  CSV final SHA87ee1ac7… igual. Cenários7,4614h/11,2585h para1.322/2.000
  estudos hipotéticos com margem; não comprovam tamanho/runtime ocultos.
- Builder smoke deriva do exato build prefetch aprovado, SHA11ddd8e5… .
  Remove seleção de treino e restaura descoberta dos inputs oficiais.
  Mesmos modelos/flags/ranks/prefetch/CoAt ordenado; saída não submetível.
  Os nomes dos protocolos nos recibos preservam a origem validada; o recibo
  smoke registra root/IDs/contagem reais. Outerparent0,60, não probe22.
- Auditor smoke preparado:3IDs oficiais, todos5ramos/56hashes, DINO e CoAt
  com replay independente, Raptor com cobertura/raw, sem fallback ou caminhos
  de benchmark. Leitor CoAt aceita número de estudos/series explícito,
  mantendo o contrato36/205 dos testes anteriores.
- 91 testes passaram; V01 intacta. Kernel134610738 v1 COMPLETE,
  privado/offline/T4/teto1.800s; cota antes18,0020h. Nenhum uso de dev/
  confirmation, novo envio, seleção final ou automação. Melhor0,941 preservado.
- Smoke auditado PASSED_ORDERED_OFFICIAL_SMOKE:3estudos/15séries,
  cinco ramos completos,56hashes, replay DINO/CoAt aprovado, sem fallback.
  CSV SHA7d3b8bd4… igual ao piloto histórico visível; não prova paridade
  no oculto. Total das etapas117,6857s, sem extrapolar a partir de3casos.
  Auditoria: reports/avance_av015_audit/smoke_v1.json.
- Próximo: candidata estável com preset probe22 fixo, após testes de routing,
  regras vigentes, cota e duplicidade. Não submeter o smoke de30min.
  Retomada: docs/AV015_PARIDADE_COMPLETA_E_SMOKE_OFICIAL.md.

### AV-016 — 16/09/2026 — candidata estável probe22 enviada

- Builder herda o smoke validado, altera apenas células3/4/5/61, preserva
  toda inferência e fixa preset probe22 já publicado. Gate confere RUN inteiro.
  Captura os ranks dos dois lados da mistura para replay independente.
- Auditor verifica componentes/CSVs e raw DINO/Raptor/CoAt contra smoke,
  cinco ramos/56hashes/T4x2, receita, IDs, zero fallback e replay da fusão.
  97 testes passaram; V01 intacta; dev/confirmation não consultados.
- Kernel jvlegend/rsna-knee-ordered-probe22-submission v1 ID134618609,
  COMPLETE offline/T4/32.400s. Build SHAfcf4ec54… . Regras notebook/9h
  revalidadas via API; cota antes17,9238h e cinco envios disponíveis.
- PASSED_ORDERED_PROBE22_CANDIDATE:3estudos/15séries, componentes/raw iguais
  ao smoke, replay aprovado; CSV SHAff848c73… igual ao probe22 histórico
  visível. Etapas114,5048s; não inferir AUC/runtime ocultos a partir de3casos.
- Envio único ref56281610, scriptVersionId350342219,16/09 12:00:59 São Paulo.
  PENDING, sem score/erro. Cota antes do envio17,8690h; após envio1hoje/4restantes.
  Melhor0,941 histórico não é score desta receita. Seleção final preservada.
  Próximo: acompanhar sem duplicar e seguir V02/L01, sem grade de pesos.
  Detalhes: docs/AV016_CANDIDATA_ESTAVEL_PROBE22.md.

### AV-017 — 16/09/2026 — L01 diagnóstico de treino, sem promover labels

- Ref56281610 PENDING, sem score/erro nas consultas; melhor0,941 preservado,
  sem reenvio, alteração de seleção final, automação ou GPU nova.
- Implementada variante lexical conservadora por cláusula/seção, cues antes/
  depois, incerteza/conflito/ausência distintos. Vocabulário de alvos e extrator
  antigo preservados; múltiplos alvos e anatomia sem achado suportado abstêm.
- Auditoria real só no treino V01:299estudos/296grupos,3.588pares.
  169sinais direcionais alterados vs legado;17discordâncias em723comparáveis
  com professor. Não são erros comprovados.725sinais definidos,653abstenções
  sobre sinais antigos;2.210pares sem match no vocabulário. Sem AUC/treino.
- Fila cega60casos com laudo completo e chave separada, privada no HD.
  Nenhuma adjudicação realizada. Outputs reports/avance_av017_l01_v2/;
  v1preservada, mesmos estados/totais.102testes passaram, incluindo40casos
  sintéticos; hashes V01/professor intactos, dev/confirmation não analisados.
- Decisão: não substituir professor; L01 permanece parcial até revisão e
  comparação visual. Próximo V02 original/pretreino genérico, com preflight
  de pixels/cache/custo e checkpoint. Docs/AV017_AUDITORIA_L01_TREINO.md.

### AV-018 — 16/09/2026 — cache V02 aprovado e piloto genérico iniciado

- Submissão56281610 PENDING; não reenviada. Melhor0,941 e seleção final intactos.
- Cache treino auditado em leitura serial:299estudos/897séries/2.691canais,
  94.557.531bytes, formato/índices/variância/hashes válidos; nenhum array de
  pixels duplicado dentro do treino. Gaps2–80: canais espaçados, não adjacentes.
  Não atesta geometria completa ou ausência de pacientes duplicados entre splits.
- Pretreino oficial metaresearch/dinov2/PyTorch/small/1, hash esperado conferido
  contra LFS facebook/dinov2-small; worker deverá validar bin/config antes de
  carregar. Nenhum peso/cabeça ajustado à competição no novo baseline.
- Implementados builder, runtime e auditor piloto:12estudos/36séries só do
  treino, reconstrução108DICOMs vs pixels do HD, DINO congelado fp32, atenção
  compartilhada própria,16passos de engenharia e retomada de checkpoint.
  Sem AUC/dev/confirmation;105testes locais. Manifesto/professor intactos.
- Kernel134631088 v1 RUNNING, privado/offline/T4/teto1.800s, usa1das2GPUs.
  Cota antes17,8690h. Logs expiraram na consulta; não concluir falha/sucesso.
  Build SHAe310c754… em reports/avance_av018_v02/. Não duplicar.
- Próximo: auditar outputs e custo antes de treino completo V02 com duas
  sementes. Nenhuma nova submissão/automação. Docs/AV018_PREFLIGHT_V02_GENERICO.md.

### AV-019 — 16/09/2026 à noite — 0,941 confirmada e diagnóstico de pixels V02

- 56281610 COMPLETE0,941: empate com probe22 histórico, sem novo melhor,
  sem prova de igualdade das predições ocultas. Nenhum envio ou seleção alterada.
- Piloto134631088 v1 ERROR no gate exato de pixels, antes do treino.
  Rebuild local serial108DICOMs/36séries coincide com cache; controle da primeira
  série com versões NumPy2.0.2/Pillow11.3.0/pydicom3.0.2 também coincide.
  Causa por estágio ainda não demonstrada; não afrouxar o gate.
- Instrumentados hashes/arrays intermediários e ambiente; encoder carregado
  após o gate de pixels. Teste de regressão da falha/evidências incluído.
  Suíte completa134testes/44subtestes passou; gates GPU continuam pendentes.
- Mesmo kernel v2 privado/offline/T4/1.800s, RUNNING; cota antes12,1254h.
  Build SHA2b174453…; outputs reports/avance_av019_pilot_v2/.
  Próximo: comparar estágios, corrigir causa, repetir gate e só então treinar.
  Nenhum dev/confirmation lido. Docs/AV019_PARIDADE_PIXELS_V02.md.

### AV-020 — 17/09/2026 — precisão corrigida e piloto GPU aprovado

- V2 ERROR na série axial do primeiro estudo. DICOM/raw iguais; NumPy2.0.2
  calcula percentil99=297,65234375 vs297,6499938964844 no cache,125pixels
  finais diferentes. Reproduzido localmente com mesmas versões; resize exato.
- Correção isolada V02: percentis lineares float64, limites float32, hashes
  congelados preservados.36/36séries exatas tanto NumPy2.0.2 quanto2.5.3;
  código antigo32/36 no2.0.2. Teste adicional limita/poda descoberta de anexos.
- Kernel134631088 v3 COMPLETE e PASSED_V02_PILOT_AUDIT.12estudos/36séries/
  108DICOMs iguais; features válidas; treino/checkpoint/retomada exata.
  Descoberta0,001893s; processamento6,8730s sem imports; pico alocado153MiB.
  Não comparar esses6,9s diretamente aos710s de log v2 como speedup causal.
- 136testes/44subtestes passaram. Nenhuma AUC/dev/confirmation ou novo envio.
  Baseline completo ainda pendente. Próximo: protocolo/cache train299/dev250,
  sementes2026/42, orçamento1.800s por execução; confirmação150fechada.
  Evidências: docs/AV020_CORRECAO_PERCENTIS_V02.md.

### AV-021 — 17/09/2026 — primeiro baseline próprio completo no split V01

- Protocolo commit69876d6 registrado antes de avaliar;299treino/250dev,
  professor original, DINO genérico congelado, atenção compartilhada,
  duas sementes2026/42,20épocas,menor softBCE dev com primeira época em empate.
- Auditoria local549estudos/1.647arrays aprovada; mesmos4.941DICOMs/pixels
  reconstruídos no worker. Confirmação150sem pixels lidos ou avaliação;
  V01/teacher intactos (metadados dos splits usados apenas para integridade).
- Kernel134788472 v1 COMPLETE e PASSED_V02_BASELINE_AUDIT; buildSHA0049a9b0… .
  Prior0,65707840;seed2026época10=0,62607019;seed42época6=0,63189521.
  Epoch20dev0,64938283/0,66013783: não estender treino automaticamente.
  Baker's piora nas duas sementes; ganhos repetidos maiores em Effusion/OA.
- 141testes/44subtestes. Independente replay NumPy logits até1,04e−6;
  features164,57s,total175,04s fora imports,153MiB pico alocado.
- Referência aceita para ablações, não nova submissão. M01 primeiro(média
  vs atenção por alvo, mesmas features); G01 em seguida. Nenhum score clínico
  ou LB inferido de softBCE. Melhor público0,941 e seleção final intactos.
  Evidências/retomada: docs/AV021_BASELINE_PROPRIO_V02.md.

### AV-022 — 17/09/2026 — ablação M01 concluída sem promoção

- Protocolo commit538a0e9 antes dos resultados; features549x3x384/IDs/labels
  congelados,299treino/250dev,3arquiteturas×2seeds×20épocas. Sem reextrair
  DICOM, consultar confirmação ou mudar geometria/otimizador/teacher.
- Kernel134795832 v1 COMPLETE e PASSED_M01_AUDIT. Controle shared reproduz
  AV-021; replay NumPy até1,21e−6, checkpoints/seleção/BCE/IDs verificados.
- SoftBCE2026/42:shared0,62607019/0,63189521;
  mean0,62733778/0,62448577;target0,63315980/0,62473064.
  Médias entre sementes0,62898270/0,62591178/0,62894522 respectivamente.
  Mean melhora a média mas não ambas sementes; target também inconsistente.
- Regra anterior aos resultados mantém shared; mean em reserva como controle
  simples promissor, não ganho confirmado. Diferenças incluem capacidade e
  inicialização; dev seleciona épocas, não é teste final. Nenhuma AUC/LB inferida.
- 146testes+44subtestes;36,02s sem imports,76,74MiB pico alocado.
  Código GitHub, artefatos privados no HD; melhor público0,941 inalterado,
  nenhuma submissão/final selection alterada. Próximo G01, geometria isolada.
  Evidências: docs/AV022_ABLACAO_POOLING_M01.md.
