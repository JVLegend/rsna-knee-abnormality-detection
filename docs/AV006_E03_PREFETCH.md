# AV-006 — E03: preparo de estudos sobreposto à inferência

#RSNA #Kaggle #Pesquisa #Tecnologia

15/09/2026 à noite em São Paulo; consultas e execução em 16/09 UTC.
Continuação de [AV-005](AV005_PARENT_0939_E_PROBE22.md), frente E03 do
[plano AVANCE](ESTRATEGIA_AVANCE.md).

## Estado reconciliado antes de agir

- Probe22, submissão **56263721**, scriptVersionId 350168027, permanece
  **PENDING**, sem score ou erro informado na consulta de 16/09 00:37 UTC.
  Não reenviar nem concluir que falhou pelo tempo de espera.
- H43 parent, ref 56253529: **COMPLETE, 0,939**. Melhor público preservado,
  assim como H38 e a seleção final oficial.
- CLI informou 20,21 h GPU restantes, renovação em 19/09 00:00 UTC;
  cinco submissões disponíveis após virar o dia UTC. Não usamos um slot.

## Hipótese predefinida

Raptor representou aproximadamente 276 s no benchmark anterior de 36 casos;
Raptor/CoAt/fusão, 387,50 s de 667,72 s das etapas. A fonte já agrupa receitas
iguais, evitando repetir o decode para MaxSpan direto/reverso. Não reivindicar
essa otimização existente como uma mudança nova.

O loop continua preparando cada estudo antes de inferi-lo. E03 testa um único
produtor CPU para preparar o estudo seguinte enquanto a GPU processa o atual.
Não troca checkpoint, arquitetura, crop, ordem, quantidade de fatias, janelas,
normalização, precisão, peso ou função de ranking.

O preparo usa NumPy/OpenCV/pydicom, não a GPU. Só existe um produtor, mantendo
o leitor fora de concorrência consigo próprio. O consumidor fecha o gerador
e espera o preparo ativo antes de mudar os globais da próxima receita.
No máximo um volume futuro além do corrente; isso não limita os temporários
internos do decoder nem demonstra limite de memória do ensemble completo.

## Experimento isolado

- **12 estudos / 70 séries**, selecionados deterministicamente por número
  de séries entre os 36 estudos já usados no benchmark; todos pertencem aos
  299 de treino V01. Quantidades: 4,4,4,5,5,5,5,5,6,7,9,11.
- Nenhum label para seleção/AUC; desenvolvimento 250 e confirmação 150
  intocados. Este não é um teste de generalização.
- Sequência **A–B–B–A**, A=serial, B=one-ahead prefetch. Quatro passagens
  completas dos mesmos quatro ramos/ três receitas Raptor, em um worker.
- Mesmo dispositivo CUDA e mesmos defaults do Raptor original. Preflight
  exige duas T4, mas Raptor usa uma delas como na referência; não alegar
  paralelismo em duas GPUs ou benchmark do restante do ensemble.
- Todas as passagens medem tempo de ponta a ponta do `main`, incluindo cargas
  dos checkpoints. Receipts por passagem, probabilidades por ramo/ranks/IDs
  em NPZ e hashes dos volumes/máscaras de cada estudo/receita.
- O primeiro A pode ter caches mais frios. Reportar média ABBA e comparação
  do A final aquecido contra média B; ABBA não elimina todo viés de cache,
  aquecimento ou deriva. Só duas observações por modo, sem intervalo de confiança.
- RSS via `resource` é máximo cumulativo do processo Linux, não pico por modo.
  CUDA reservado registrado separadamente; somas de tempo de preparo se
  sobrepõem à GPU e não devem ser somadas ao wall time.

Critério registrado antes do resultado: identidade exata de IDs, volumes,
máscaras, probabilidades dos quatro ramos e ranks em todas as passagens.
Se qualquer paridade falhar, não promover. Somente se a identidade passar e
o speedup for ≥1,05 tanto no ABBA quanto no comparador aquecido, a variante
fica elegível para um teste posterior do stack completo. Nunca habilitar
produção ou submeter automaticamente apenas pelo microbenchmark.

## Código e verificações

- `scripts/ordered_prefetch.py`: fila ordenada com um worker/um futuro,
  propagação de exceções e shutdown antes da troca de receita.
- `scripts/h43_e03_runtime.py`: instrumentação, sequência ABBA e paridade.
- `scripts/prepare_h43_e03.py`: builder ancorado no SHA do parent 0,939;
  altera apenas loop de entrega, root de benchmark e saída diagnóstica.
- `tests/test_h43_e03.py`: ordem, limite de lookahead, encerramento, exceções,
  drift e identidade AST das funções de modelo/preparo. **33 testes passaram**
  incluindo as suítes anteriores. Parent/probe22 originais não foram alterados.

Build HD: `reports/avance_av006_build/h43_e03_prefetch_abba_v1.ipynb`, SHA
`223ec2a6ee262aae8202aa46866ee31df916454e9fa2a90399b39b258f3eeeb2`.
Fonte parent SHA
`a103e27720ba52f584c377ac6ffa470aad062c0a4e8558df22da06535a6b0306`.
Manifesto V01 fixado em
`365566b0f830e398e78bd36bd9118a7365c3af340f588e774c8bcc351398b6df`.

## Execução iniciada e retomada

[RSNA Knee E03 Prefetch ABBA](https://www.kaggle.com/code/jvlegend/rsna-knee-e03-prefetch-abba),
v1, ID **134542682**, privado/offline/T4, teto **1.800 s**. Fontes iguais às
auditadas no parent; despacho via API tipada, sem editar configuração local.
Último estado **RUNNING**. Ainda não há resultado de velocidade ou paridade
real. Não chamar esta implementação de ganho de performance demonstrado.

**Nunca submeter este notebook:** usa estudos de treino e mede só Raptor.
Não cria `submission.csv` elegível na raiz de working; `_raptor.csv` é um
intermediário diagnóstico sobrescrito entre passagens. O sample com valores
constantes vive apenas na subpasta de entrada sintética do benchmark.

No próximo AVANCE:

1. Consultar a submissão 56263721 sem duplicar; comparar eventual score com 0,939.
2. Recuperar esta execução, não lançar cópia. Outputs em
   `reports/avance_av006_e03_v1/`: `e03_comparison.json`, quatro receipts,
   quatro NPZs, seleção/preflight e log.
3. Verificar os quatro runs e paridade. Se falhar, investigar input/rank/raw
   antes de atribuir a falha ao paralelismo; repetição serial também é controle.
4. Se não ganhar pelo critério, rejeitar esta variante; seguir V02 ou outro
   gargalo medido, sem alterar H43. Se ganhar, escalar benchmark/stack completo
   antes de qualquer uso na submissão.

Nenhuma nova submissão, seleção final ou automação nesta AVANCE.
