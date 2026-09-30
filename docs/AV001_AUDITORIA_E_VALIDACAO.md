# AV-001 — inventário H-43A e validação weak congelada

#RSNA #Kaggle #Tecnologia #Pesquisa

14/09/2026. Referência: [plano AVANCE](ESTRATEGIA_AVANCE.md).
Auditoria por CLI autenticada e leitura estática; nenhum código baixado ou
checkpoint de terceiros foi executado nesta rodada.

## Resultado e decisão

- A00: inventário estático concluído. **Não é reprodução do score 0,941**.
- V01: executada, 699 estudos elegíveis em 692 grupos; partição congelada.
- A01: próxima ação é preparar execução estrita do `parent`, com preflight
  de fontes/receitas, composição integral e orçamento; ainda não apta a envio.
- H-38 continua `0,929`, submission `55916072`, COMPLETE; H-42 `0,881`,
  submission `56217840`, COMPLETE. Consultados novamente via CLI nesta rodada.
- Nenhuma submissão, kernel ou treino novo. Não houve nova métrica de imagem.

## A00 — fonte, contratos e riscos encontrados

Fonte: [maverickss26/rsna-knee-0941-restructured](https://www.kaggle.com/code/maverickss26/rsna-knee-0941-restructured),
ID numérico `133723210`. O notebook declara parent `0,939` e probe22 `0,941`;
são números do autor, não resultados nossos. A cópia local foi fixada por hash,
não por um número de versão do notebook confirmado pela API.

- Notebook SHA-256: `5b133e41d951aae8dc9efa9361d1975d5259a8f98f956cc1e84d78ea1aed80e7`.
- Metadados CLI SHA-256: `d94b98914e47ee72a58f673740a86339b871e6d6890c4bc0b6cbe345f5867a1f`.
- 12 datasets, 2 outputs de kernels e modelo `metaresearch/dinov2/PyTorch/small/1`.
- IDs `datasetVersion` embutidos: `18229736, 18673450, 18716507, 18839182,
  18842180, 18879001, 18956429, 19120128, 19122845, 19134209, 19395055, 19395706`.
  Não associar por posição aos slugs: a ordenação das duas listas pode diferir.
- IDs `kernelVersion`: `342671664, 342849430`; `modelInstanceVersion`: `4533`.
- Metadados CLI pedem T4 e internet desligada; o metadata interno do ipynb
  contradiz o CLI em `isGpuEnabled`. Confirmar hardware real no preflight.
  O subprocesso CoAt exige explicitamente **duas GPUs**.

| Bloco | Inventário/receita observada | Pendência de execução |
|---|---|---|
| DINO Pilkwang | Manifesto atual lista 20 membros, 4 por fold em 5 folds; 336 px, 12 cortes/slot, grupos de 3, crop 130 mm, banda 0,20–0,80, 6 slots | Conferir manifesto da versão anexada, todos os pesos e fingerprints; contagem executada deve ser 20, não o texto legado “24-member” |
| A5 Mattia | Listagem atual contém `m_f0.pt` até `m_f4.pt`; 5 folds, fusão 0,45 | Conferir cfg de cada checkpoint e receita efetiva; não presumir contrato igual ao DINO |
| RadImageNet | Dois bundles públicos com 5 heads cada + E13 com 5 heads; E13 também aplicado no layout E11; calibrador embutido | Identidade não se deduz do nome do arquivo; conferir hashes, layouts e execução do calibrador |
| Raptor | 4 views usando 3 checkpoints distintos: MaxSpan v5, v5 reverse, Native384Dense v10, Native384 v8 | v5 e reverse compartilham peso, não previsões; não contar views como checkpoints únicos |
| CoAt residual | 3 checkpoints e4/e6/e8, peso interno 0,40 | Verificar manifesto, wheel OpenCV fixada, duas T4 e zero fallback |

Pilkwang `manifest.json` baixado, SHA-256:
`496949a3a3e789bc1f4ccff595205c911e471c5b5ef669366a2dd0a58e125844`.
O total final de modelos/passes precisa de recibo de runtime; a tabela não
inclui uma contagem especulativa dos objetos do calibrador embutido.

Raptor: pesos de views `0,55 / 0,15 / 0,10 / 0,20` respectivamente v5,
v5 reverse, v10 e v8. V10 usa cache **384**, slots `18/14/12/8/12`, banda
`0,02–0,98`, 62 janelas e crop 140 mm. V5 usa cache 336 com a mesma contagem;
V8 usa cache 384, slots `12/10/8/6/8`, banda `0,06–0,94`, 42 janelas.
H-38 é H-36 + residual DINOv3 20/80 conforme o log; não é este stack integral.

### Bloqueios para promoção automática

1. Célula 48 captura falha do CoAt, restaura o Raptor público e prossegue.
   Isso pode produzir CSV válido de um modelo diferente do pretendido.
2. Célula 35 admite falha na promoção do public-frontier DINO; o writer ainda
   preenche ausências com 0,5. A célula 31 pode descartar membro degenerado ou
   omitir voto com janelas incompletas. Schema/finitude sozinhos não bastam.
3. O `parent` usa peso externo CoAt 0,60 em todos os alvos. O default da fonte
   é `probe22`, com pesos por alvo até 1,00. Fixar explicitamente parent antes
   da primeira reprodução, sem selecionar pelo gold.
4. Ainda faltam verificação de todos os hashes nos mounts reais, auditoria do
   calibrador, cobertura, runtime representativo e cota. Não lançar inferência
   integral simplesmente porque o notebook foi baixado.

Próxima implementação A01: abortar nas substituições de ensemble acima,
exigir recibo de todos os membros/views e separar os arquivos diagnósticos do
CSV elegível para submissão. Se as dependências inviabilizarem o stack, A03
Native384Dense permanece alternativa. Não corrigir aritmética do parent no
mesmo teste: uma alteração dessas deve ser outra variante.

### Disponibilidade e licenças declaradas

Metadados atuais consultados por CLI:

- CC0-1.0: três Raptor `dreaddevelopment`, folds Mattia, CoAt residual Mattia,
  labels Pilkwang e pesos Pilkwang.
- CC-BY-NC-SA-4.0: RadImageNet Marwan e heads Antoine E11/E9.
- `other`: heads Prvsiyan; README descreve dependência de RadImageNet e
  necessidade de respeitar os termos upstream, inclusive NC/share-alike.
- `unknown`: dataset do wheel OpenCV Mattia. Não inferir a licença do wheel
  a partir desse campo; verificar distribuição/upstream antes da execução.
- Licenças dos outputs Sofia, do notebook e de toda a cadeia não estão
  liberadas comercialmente por esta auditoria. Disponível para download não
  significa autorização irrestrita. Este registro não é parecer jurídico.

Os outputs E11/E13 estavam disponíveis. A listagem retornava tamanho 885
para os heads, mas ambos os arquivos baixados têm **63.534.726 bytes**:

- E11 SHA-256 `f05d92ee59dcce270cfb7983fa219940abf9b0c2e247ff12a88a81129841ca34`.
- E13 SHA-256 `ad9f19af73bfdf4e49263c0e45060dc3cb239e1195039b26dc8c0a3a6bcd1a8a`:
  coincide com `_RAD_E13_HEADS_SHA256` da fonte. Ambos se chamam
  `v52_e11_heads.pt`; resolver por conteúdo, não pelo primeiro nome encontrado.

Os downloads são os outputs atuais; a correspondência com os kernelVersion
anexados ainda precisa de preflight, apesar do hash E13 coincidir.
Material bruto no HD em `reports/avance_av001_sources/`, ignorado pelo Git.

## V01 — partição implementada e executada

Código: [freeze_weak_validation.py](../scripts/freeze_weak_validation.py).
Testes: [test_freeze_weak_validation.py](../tests/test_freeze_weak_validation.py).

Inventário: 700 estudos/2.100 séries, com 2.100 arrays locais não vazios nos
três planos. **Um estudo compartilha laudo normalizado com o gold** e foi
excluído. Os 58 gold e seus grupos são protegidos globalmente. Mesmos laudos
não cruzam partições. Nenhum laudo completo foi copiado para o Git.

| Partição | Estudos | Grupos | Menor nº positivo entre os 12 alvos | Menor nº negativo |
|---|---:|---:|---:|---:|
| Treino | 299 | 296 | 23 | 85 |
| Desenvolvimento | 250 | 249 | 23 | 67 |
| Confirmação | 150 | 147 | 14 | 48 |

Labels suaves congelados. Nos counts/AUC binária: positivo >0,5, negativo
<0,5; exatamente 0,5 é incerto e excluído, não convertido em positivo.
Counts por alvo completos no manifesto. Há classes suficientes para calcular
AUC, mas isso não garante precisão estatística; alguns alvos continuam raros.

Manifesto: `data/processed/validation_weak_v1/manifest.json`.
SHA-256: `365566b0f830e398e78bd36bd9118a7365c3af340f588e774c8bcc351398b6df`.
Inclui IDs, grupos, labels, caminhos/tamanhos de arrays e hashes dos três
inputs. Segunda execução retornou `created=false`, mesmo hash, sem alteração.
Código recusa sobrescrever uma partição diferente no mesmo arquivo.

```sh
python3 -m unittest tests/test_freeze_weak_validation.py -v
python3 scripts/freeze_weak_validation.py --output data/processed/validation_weak_v1/manifest.json
```

**6 testes passaram**: normalização, grupos inteiros/disjuntos e determinismo,
quotas inviáveis, tratamento de incerteza, imutabilidade e inventário com
gold compartilhado/laudo vazio/plano ausente/NaN/série duplicada.

Limitações: auditada existência/tamanho do cache, não integridade de todos os
pixels nem volumes DICOM. Três planos não são seis protocolos. Os labels são
weak e a confirmação é nova **neste protocolo**, não um corpus historicamente
intocado. Nenhum modelo consultou a confirmação nesta rodada; só foram
inventariadas as classes. Usar pretreino genérico no novo treino; pesos públicos
com exposição desconhecida continuam inadequados para chamar esta avaliação
de independente. CPU leve, nenhuma GPU usada; pico de RAM/tempo global não
instrumentados, sem estimativa de aceleração ou score.
