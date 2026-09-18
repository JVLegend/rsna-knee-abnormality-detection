# AV-017 — diagnóstico de escopo/negação nos laudos de treino

#RSNA #Kaggle #Pesquisa

16/09/2026. Fonte de verdade: 07_Estrategia_AVANCE do vault,
[espelho](ESTRATEGIA_AVANCE.md). Continua [AV-016](AV016_CANDIDATA_ESTAVEL_PROBE22.md).

## Resultado e decisão

Submissão **56281610 continua PENDING**, sem score/erro nas duas consultas
desta rodada. Melhor confirmado0,941 (56263721). Nenhum reenvio, treino GPU,
serviço pago, automação ou mudança da seleção final. Cota consultada17,8690h.

Executado L01 diagnóstico em **299 estudos / 296 grupos / 3.588 pares
estudo–alvo**, exclusivamente no treino V01. A regra por cláusula alterou
169 sinais direcionais contra o extrator lexical legado. Houve17discordâncias
com o professor em723pares comparáveis. **Isso não são17erros confirmados**:
professor weak não é verdade clínica e a nova regra também pode errar.

**Decisão: não substituir o professor, não produzir labels de treino novos.**
A ferramenta fica como triagem de auditoria. V02 ainda precisa fornecer a
comparação visual antes de concluir L01 ou alegar ganho no leaderboard.

## O que foi testado

O extrator legado src/rsna_knee_baseline/lexicon.py procura cues somente nos
90caracteres anteriores. Reproduções sintéticas, não transcrições dos dados:

- `ACL normal.`: legado+1; candidato reconhece normalidade depois da menção.
- `No fracture. Joint effusion present.`: legado−1 para Effusion; candidato
  restringe a negação à primeira frase e retorna menção afirmada para Effusion.
- `Indication: ACL tear. Findings: ACL normal.`: candidato não transforma
  a hipótese da indicação clínica em achado afirmado do exame.

Novo módulo src/rsna_knee_baseline/scoped_mentions.py preserva o vocabulário
de alvos e o código legado. Separa seções explícitas e cláusulas por pontuação/
contraste; procura cues antes/depois; registra incerteza, conflito, ausência
de menção e seção excluída. Menção anatômica sem achado suportado e cláusula
com vários alvos fazem a regra abster-se. Não preenche ausência com negativo.

Regras de cues para en/pt/es/fr/de/nl, mas **sem identificação confiável do
idioma do laudo**. O vocabulário legado é limitado; cues de idiomas distintos
podem coincidir. As contagens de idiomas no JSON não são cobertura linguística.
Negações complexas, abreviações, quebras de linha e elipses ainda podem falhar.
Não é algoritmo médico validado nem substituto de anotação especializada.

## Distribuição observada

| Estado de sinal textual | Pares |
|---|---:|
| Menção afirmada | 269 |
| Menção negada/normalidade explícita | 456 |
| Escopo/achado não resolvido | 590 |
| Incerto | 21 |
| Afirmação e negação em conflito | 34 |
| Apenas em seção excluída | 8 |
| Não mencionado pelo vocabulário | 2.210 |

Só725/3.588pares (20,2%) recebem sinal afirmado/negado; dois têm professor0,5
e não entram na comparação direcional.653sinais legados passam a abstenção.
Essa perda de cobertura impede tratar a versão conservadora como professor
substituto. Ausência de match lexical não prova ausência de informação no laudo.

Discordâncias com professor: Medial Meniscus5; Baker's3; ACL/MCL/Lateral OA/
PF OA2cada; Effusion1. Os demais alvos não tiveram discordância nos pares
comparáveis, o que não significa precisão perfeita ou cobertura suficiente.
27registros de evidência foram excluídos por seção clínica; aliases sobrepostos
podem gerar mais de uma evidência para o mesmo trecho, não são27estudos.

## Evidências, revisão e proteção dos dados

Script scripts/audit_scoped_training_mentions.py exige o hash V01 e o hash
do train.csv original, verifica grupos/IDs disjuntos, hash de cada laudo de
treino, ausência de gold e cobertura completa. Só analisa textos de IDs de treino.
Os arquivos originais são lidos para filtragem/integridade, sem extrair ou
avaliar texto/labels de desenvolvimento/confirmação.

Artefatos finais, **somente no HD externo e ignorados pelo Git**:
reports/avance_av017_l01_v2/{summary,records,blind_review,review_key}.json.

- summary: agregados, hashes do código e limites de interpretação.
- records: pares de treino, estados, teacher e evidências; acesso restrito.
- blind_review:60casos de discordância/abstenção selecionados por hash,
  com laudo completo e seções, sem respostas do professor/legado/candidato.
- review_key: chave separada; não abrir antes da revisão cega.

A fila inclui6dos17desacordos com o professor; os17estão em records.
É amostra de casos problemáticos, não estimativa representativa de precisão.
Nenhum caso foi adjudicado nesta rodada. Laudos e UIDs não vão para docs/GitHub.
v1foi preservada; v2acrescenta contexto completo à revisão, com os mesmos
estados e totais. Não sobrescrever artefatos ou chave de revisão.

## Verificação e reprodução

**102 testes passaram**, incluindo40casos sintéticos de comportamento do parser,
reprodução das falhas legadas, offsets na string normalizada, ausência≠negativo,
exclusão de IDs reservados, rejeição de gold/hash alterado e grupos cruzados.
Os40casos estão dentro de um teste parametrizado; não são40laudos reais.

```sh
uv run --no-project --with pandas --with scikit-learn python -m scripts.audit_scoped_training_mentions \
  --output-dir reports/avance_av017_l01_v2
```

scikit-learn/scipy são necessários pelo import preexistente do pacote;
ambiente efêmero uv, sem alteração de configurações. Sem processo pesado local.
Manifesto V01 SHA365566b0f830e398e78bd36bd9118a7365c3af340f588e774c8bcc351398b6df
e professor SHAeeca4a86d1842dec04e2f18de43856a371f9b3098710a7de82c616e85ad42d00
permanecem intactos. Nenhuma AUC medida, rótulo alterado ou comparação visual.

## Próxima ação

Consultar56281610 sem duplicar. Avançar **V02**, baseline DINOv2-S 2.5D com
atenção por estudo e pretreino genérico, usando professor original e treino299.
Primeiro verificar cache/pixels, proveniência do pretreino e custo com checkpoint
retomável; só então iniciar treino orçado. Não usar pesos públicos ajustados
na competição como encoder de validação independente. Selecionar no dev250,
manter confirmação150fechada até finalistas. Revisão L01 e comparação visual
ficam pendentes; a fila de60casos não é bloqueio para começar o baseline V02.
