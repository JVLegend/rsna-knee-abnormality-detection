# Estratégia de execução — comando AVANCE

#JoaoVictor #Kaggle #Tecnologia #Academia

Criado em 14/09/2026. Plano operacional aprovado pelo pedido do JV para
transformar as pesquisas em alternativas e testá-las a cada comando AVANCE.
Atualizado na AV-001: scores reconciliados por CLI em 14/09/2026; nenhuma
nova métrica de imagem. Inventário A00 e partição V01 executados abaixo.


Espelho operacional da fonte de verdade no vault:
/Users/iaparamedicos/Documents/GitHub/SuperJV/01_Projects/Competicoes/RSNA_Knee_Abnormality_Detection/07_Estrategia_AVANCE.md.
Ao executar AVANCE, atualizar a nota canônica e este espelho. Evidências do
repositório: docs/PLANO_MELHORIAS_2026-09-05.md e docs/LOG_EXPERIMENTOS.md.

## Objetivo e ponto de partida

Melhorar o modelo de imagem para os 12 alvos, com experimentos comparáveis,
submissões rastreáveis e custo medido. Trabalhar em três frentes:
reprodução pública forte, treino próprio e combinação/eficiência.

Referência protegida: H-38, público 0,929; alternativa H-36, público 0,928.
H-42 terminou com 0,881 e não será promovida. Sua avaliação local de 0,995527
já tinha exposição dos pesos ao gold; a queda no leaderboard não identifica,
sozinha, qual componente causou a diferença.

O stack público que declara 0,939–0,941 é uma referência a reproduzir.
Esse resultado não é nosso nem uma promessa de ganho. O objetivo inicial é
superar H-38 com evidência; atingir 0,94 é uma meta aspiracional.

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

- Próxima ação: A01 — preparar parent estrito, eliminar substituições silenciosas
  e conferir todos os membros/mounts/cota antes de executar em duas T4.
- Experiência em execução: nenhuma iniciada por este plano.
- Kernel/submissão em andamento: nenhum novo lançado na AV-001.
- Último resultado confirmado por CLI: H-42 COMPLETE, 0,881, ref 56217840.
- Melhor candidato confirmado: H-38, 0,929, ref 55916072.
- Alternativa se A01 bloquear: A03 — Native384Dense com receita nativa.
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
- O manifesto atual cobre apenas os 58 gold: V01 precisa criar uma partição
  nova para os estudos weak. Não reutilizar aquele manifesto como se cobrisse
  o corpus inteiro.
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
| A01 / H-43A | Reproduzir primeiro um único parent fixado, sem pesos próprios; verificar predições e execução antes da confirmação Kaggle | A00 com fontes utilizáveis / médio | PENDENTE |
| A02 / H-43A | Comparar parent com um único preset publicado escolhido previamente: halfway OU probe22; aproveitar previsões dos mesmos membros | A01 / baixo após inferência | PENDENTE |
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
| E03 / eficiência | Compartilhar decode; compartilhar prefixo congelado só se os tensores forem idênticos; distribuir braços nas duas T4 | referência estável / médio | PENDENTE |
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
