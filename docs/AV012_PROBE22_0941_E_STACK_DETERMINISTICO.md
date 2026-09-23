# AV-012 — probe22 0,941 e stack determinístico

#RSNA #Kaggle #Pesquisa

16/09/2026. Fonte de verdade: 07_Estrategia_AVANCE do vault,
[espelhada aqui](ESTRATEGIA_AVANCE.md). Continua [AV-011](AV011_RAPTOR_DETERMINISTICO_E_ROUTING.md).

## Resultado público confirmado

API Kaggle: probe22 **56263721 COMPLETE 0,941**, sem erro;
parent **56253529 COMPLETE 0,939**. Ganho absoluto **0,002**.
Envio existente da AV-005, scriptVersion350168027, não reenvio nesta rodada.
Reprodução pública; OOF independente indisponível. Não prova ganho no privado.
Nenhuma alteração da seleção final, serviço pago ou automação.

## Repetição Raptor

Kernel134548619 COMPLETE. Auditoria independente:
`reports/avance_av012_audit/repeat_v1.json`, PASSED_CROSS_SESSION_PARITY.
Mesmos36 estudos/205 séries, 56 hashes/T4x2, ambiente, IDs, imagens/máscaras,
probabilidades e ranks exatos, delta zero. Tempo193,937866927s vs175,463111368s
na passagem B2 de referência. Não é ganho de velocidade entre sessões.
Host físico desconhecido. Só Raptor, não validação AUC nem stack completo.

## Integração implementada

- `prepare_h43_deterministic_fullstack.py` reaudita os arquivos da repetição
  e parte do benchmark fixado por SHA. Preserva DINO público inteiro exato.
- CUBLAS=:4096:8 antes do preflight. Flags determinísticas após definição
  Raptor e antes do forward, iguais à repetição. Backend/RNG restaurados em
  finally. CUBLAS é processo/filhos: não alegar isolamento causal dos demais ramos.
- `_combine` weighted native ordena membros por ID antes de somar, preservando
  pesos. Remove dependência da ordem dos workers no diagnóstico; não exclui
  esse CSV do gate. Teste com20 permutações e pesos não uniformes passou.
- Só células3/33/57 mudam contra stable36; fonte Raptor/modelo/decode intacta.
  Não promete equivalência ao parent histórico. Modos compartilham a receita.
- `assess_h43_deterministic_fullstack.py` mantém integralmente o auditor anterior,
  inclusive native, e acrescenta recibo de receita, ambiente e comparação raw
  com a repetição isolada. Sem tolerância ajustada depois do resultado.
- 75 testes passaram, inclusive restauração de estado em erro. V01 SHA intacto:
  `365566b0f830e398e78bd36bd9118a7365c3af340f588e774c8bcc351398b6df`.
  Dev/confirmation não consultados.

## Par em execução

Privados, offline, T4x2 solicitado, teto1.800s cada; API aceitou todos os anexos.
Cota antes do despacho18,9291h. Não são submissões; contêm apenas treino.

| Modo | Kernel | ID / versão |
|---|---|---|
| Serial | jvlegend/rsna-knee-deterministic36-serial | 134588338 / v1 |
| Prefetch | jvlegend/rsna-knee-deterministic36-prefetch | 134588340 / v1 |

Builds em `reports/avance_av012_build/`:

- `h43_deterministic36_serial_v1.ipynb`:
  `7e4fce5475b117e7301f2590f9edbcbd4c881fa7b0305dd2e5386e630a6e4f6d`.
- `h43_deterministic36_prefetch_v1.ipynb`:
  `a6a49f5ef3b84f7ce0118235f6dbdc57aacbbe5418bf7dfd6e05a2c3edebe760`.

## Retomada

Consultar os dois slugs, sem duplicar; após COMPLETE:

```sh
uvx --from kaggle kaggle kernels output jvlegend/rsna-knee-deterministic36-serial \
  -p reports/avance_av012_serial_v1 --file-pattern '.*csv|.*json|.*npz'
uvx --from kaggle kaggle kernels output jvlegend/rsna-knee-deterministic36-prefetch \
  -p reports/avance_av012_prefetch_v1 --file-pattern '.*csv|.*json|.*npz'
uv run --no-project --with numpy python -m scripts.assess_h43_deterministic_fullstack \
  --reference reports/avance_av012_serial_v1 --candidate reports/avance_av012_prefetch_v1 \
  --output reports/avance_av012_audit/fullstack_v1.json
```

Se passar e houver ganho de tempo, adaptar smoke para a MESMA receita e exigir
testes de proveniência. `prepare_h43_stable_smoke.py` antigo não contém estas
flags: não usá-lo como se tivesse sido validado aqui. Se falhar, diagnosticar
primeira divergência, preservar gate e não promover por CSV final coincidente.
