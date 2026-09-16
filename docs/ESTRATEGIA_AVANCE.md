# Estratégia de execução — comando AVANCE

#JoaoVictor #Kaggle #Tecnologia #Academia

Criado em 14/09/2026. Plano operacional aprovado pelo pedido do JV para
transformar as pesquisas em alternativas e testá-las a cada comando AVANCE.
Atualizado na AV-011: H43 segue 0,939; probe22 ref 56263721 PENDING.
Raptor36 determinístico aprovado dentro da sessão: raw/ranks/inputs idênticos,
26,79% menos tempo com prefetch contra serial aquecido. Repetição iniciada
em nova sessão; ainda sem promoção do stack ou nova submissão.


Espelho operacional da fonte de verdade no vault:
/Users/iaparamedicos/Documents/GitHub/SuperJV/01_Projects/Competicoes/RSNA_Knee_Abnormality_Detection/07_Estrategia_AVANCE.md.
Ao executar AVANCE, atualizar a nota canônica e este espelho. Evidências do
repositório: docs/PLANO_MELHORIAS_2026-09-05.md e docs/LOG_EXPERIMENTOS.md.

## Objetivo e ponto de partida

Melhorar o modelo de imagem para os 12 alvos, com experimentos comparáveis,
submissões rastreáveis e custo medido. Trabalhar em três frentes:
reprodução pública forte, treino próprio e combinação/eficiência.

Melhor referência pública: H43 parent, público 0,939, ref 56253529.
H-38 0,929 e H-36 0,928 permanecem disponíveis; seleção final não alterada.
H-42 terminou com 0,881 e não será promovida. Sua avaliação local de 0,995527
já tinha exposição dos pesos ao gold; a queda no leaderboard não identifica,
sozinha, qual componente causou a diferença.

O parent público 0,939 foi reproduzido pela H43 na AV-005, superando H38.
O preset probe22 relata 0,941 na fonte: esse ainda não é nosso resultado.
Superar 0,94 continua sendo uma meta; ganho público não garante o privado.

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

- Próxima ação: A02 — consultar a submissão probe22 ref 56263721, sem duplicar.
  Nunca submeter o piloto de 30 minutos nem o benchmark com casos de treino.
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
- Repetição em execução: jvlegend/rsna-knee-raptor36-deterministic-repeat v1
  ID 134548619 RUNNING, offline/T4/teto1.800s; uma passagem prefetch, mesmos
  36 estudos/205 séries, pesos e flags. Sessão nova; host físico não identificável.
  Recuperar em reports/avance_av011_raptor_repeat_v1/, sem duplicar.
- Próximo: assess_h43_raptor_repeat contra o ABBA aprovado; exigir ambiente,
  modelos, IDs, inputs e raw/ranks iguais. Se passar, preparar par do stack
  com a mesma receita determinística, preservando H43 histórica e sem promover.
- Native DINO classificado por teste isolado: diagnóstico preservado, enquanto
  o DINO público substitui o arquivo principal. Quatro variantes native não
  alteram a saída; público ausente/inválido falha. Não alteramos o gate completo.
  Causa cuDNN não confirmada. Retomada: docs/AV011_RAPTOR_DETERMINISTICO_E_ROUTING.md.
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
  São Paulo, PENDING sem score, reconsultado na AV-011 (16/09 UTC).
  Um envio na AV-005; nenhum novo na AV-006/007/008/009/010/011, não reenviar.
- Melhor candidato confirmado: H43 parent 0,939; H38 0,929 preservada.
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
| A02 / H-43A | Comparar parent com um único preset publicado escolhido previamente: halfway OU probe22; aproveitar previsões dos mesmos membros | A01 / baixo após inferência | EM_EXECUCAO — AV-005; probe22 íntegro, submissão 56263721 PENDING |
| A03 / H-43B | Se o stack integral bloquear, testar Native384Dense com sua receita nativa como alternativa ao MaxSpan H-36 | A00; pesos disponíveis e receita compatível / médio | PENDENTE |
| V01 / validação | Inventariar cobertura local e congelar treino/desenvolvimento/confirmacão por grupos, labels e hashes; avaliar ausência de classes | início / baixo | REPRODUZIDA — AV-001; 699 elegíveis, 692 grupos, 6 testes OK |
| V02 / validação | Treinar baseline próprio DINOv2-S 2.5D + atenção por estudo, partindo de pretreino genérico; repetir duas sementes para medir variação | V01 / alto | PENDENTE |
| L01 / H-44 | Professor atual versus negação antes/depois por idioma, escopo por cláusula e exclusão da indicação clínica | V01 para labels; V02 para efeito visual / baixo + treino | PENDENTE |
| L02 / H-44 | Acrescentar consequências OA com atribuição ao compartimento e severidade, mantendo os demais alvos/labels | L01; referência de avaliação congelada / baixo + treino | PENDENTE |
| L03 / H-44 | BCE atual versus BCE com máscara de não mencionado e peso reduzido para incerto; distinguir gravidade de confiança | V02; labels com estados auditáveis / alto | PENDENTE |
| L04 / H-44 | Teacher atual versus consenso apenas nos casos discordantes, com abstenção se faltar evidência; regras ou LLM local já disponível | V01/V02; não repetir média irrestrita já negativa / alto | PENDENTE |
| G01 / H-45 | Triplets adjacentes nativos versus amostragem espaçada, mantendo centros, número de views, crop e encoder | V02; primeiro medir gaps reais do loader / alto | PENDENTE |
| G02 / H-45 | Cache 16 versus 32 fatias com mesma banda; medir fatias únicas/gaps e treinar cada receita compatível | G01; ramo que de fato usa cache 16 / alto | PENDENTE |
| G03 / H-45 | Banda central versus ampla mantendo densidade física semelhante; depois ablação da densidade com banda fixa | G01; contagem pode mudar para preservar densidade / alto | PENDENTE |
| G04 / H-45 | 224 versus 336; 384 só se 336 ganhar, crop físico fixo, treino e inferência compatíveis | V02; melhor amostragem congelada / alto | PENDENTE |
| G05 / H-45 | Global 140 mm versus crop anatômico menor de 100–110 mm, resolução fixa; avaliar periferia e estruturas finas | V02; definir centro reproduzível / alto | PENDENTE |
| M01 / H-46 | Média de grupos versus atenção aprendida por alvo, preservando posição física e máscara de presença | V02; pesos treinados para cada cabeça / alto | PENDENTE |
| M02 / H-46 | Especialista público Renta somente em Medial Meniscus, com peso fixado antes da avaliação e 11 colunas preservadas | A00; fonte/receita própria auditada / médio | PENDENTE |
| M03 / H-46 | Modelo próprio global versus global + ramo local para menisco/MCL; atenção espacial como alternativa ao crop fixo | G05/M01; escolher um mecanismo por rodada / alto | PENDENTE |
| M04 / H-46 | Adicionar posição em mm e máscara de protocolo à atenção; comparar com mesma cabeça sem posição | V02/M01 / alto | PENDENTE |
| R01 / robustez | Normalização por volume/série versus receita atual, com MONOCHROME1, rescale e paridade de intensidade verificados | V02; preservar contrato dos pesos públicos / alto | PENDENTE |
| R02 / robustez | Estresse de slot ausente/ruidoso; se houver queda, comparar treino normal com dropout de slots | V02; mistura de perdas somente depois do diagnóstico / médio + treino | PENDENTE |
| E01 / ensemble | Âncora + melhor componente elegível: peso global fixo pequeno, definido no desenvolvimento; medir correlação e delta por caso | A/V com predições comparáveis / médio | PENDENTE |
| E02 / ensemble | Probabilidade versus rank no mesmo conjunto de componentes, pesos e partição; evitar grid target-wise | E01 / baixo | PENDENTE |
| E03 / eficiência | Compartilhar decode; compartilhar prefixo congelado só se os tensores forem idênticos; distribuir braços nas duas T4 | referência estável / médio | EM_VALIDACAO — AV-011; ABBA36 determinístico aprovado, −26,79% tempo vs serial aquecido; repetição134548619 RUNNING; sem promoção |
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
