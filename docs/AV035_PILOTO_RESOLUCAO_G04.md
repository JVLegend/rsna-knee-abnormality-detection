# AV-035 — piloto técnico de resolução

#RSNA #Kaggle #Pesquisa

30/09/2026. A submissão H46 original ref56696639 continua PENDING durante
esta rodada. Melhor score confirmado: H43A0,941. Não há nova submissão.

## Execução

G04 retomado conforme o protocolo AV-029:20exames do treino,60séries,
resize224vs336, DINOv2-S oficial congelado, mesmos canais/decoder/geometria.
Não contém labels ou exames da confirmação. Fonte congelada66.245bytes,
SHAca3dd37aea22cdacfc7899938d2a68f6475fc68bf55b90aa645546a857dfeb21.

Os8testes de seleção/arrays/custo/fonte passaram. Launcher próprio
scripts/launch_g04_preflight.py, commit4a141ce, dry-run validado. Verificado
slug404 e somente placeholder vazio na busca antes de lançar. Ambiente
Docker igual ao V03, anexo V03scale1000, competição e DINOv2small/1.

Kernel **135300098**, versão1, slug
`jvlegend/rsna-knee-g04-resolution-preflight`; terminou ERROR em39,36s,
antes da extração: `Pilot physical sampling drift`.
T4×2/offline, somente cuda:0 usada, teto480s/guard360s. Cota normalizada
antes do envio:81.872,277241s disponíveis. Não confundir oJSON bruto do
SDK (que serializa a duração permitida incorretamente) com timedelta real.

Recibo privado: reports/avance_av035_g04/launch.json. Não relançar.
Baixar outputs para reports/avance_av035_g04/output após conclusão e usar
scripts.assess_g04_preflight. Este piloto avalia compatibilidade e custo;
não mede qualidade clínica ou score e não produz submission.csv.

## Correção do contrato de geometria

36registros V03 têm apenas índices/arquivos/posições/gaps;24registros G01
incluem também pixel_sha256 dentro do dicionário selected. physical_plan
retorna só geometria, então a igualdade bruta entre os dicionários impediria
o sucesso desses24casos mesmo com pixels iguais. O builder agora usa o mesmo
esquema geométrico para ambas fontes; a checagem separada do hash dos pixels
224 permanece exata, assim como seleção/posições e tolerâncias de features.
Isso corrige um defeito comprovado; o log v1 não identifica qual foi o
primeiro registro divergente. V2 grava actual/expected caso ainda haja drift.

9testes passaram, incluindo regressão do esquema histórico. Commit98a5499.
Build v2 SHA37b721468b9a5273403014f9569707b32b23a023c3703baaef6e70346fb0d528,
64.527bytes. Kernel135300098v2 aceito após reconciliar ERRORv1; recibo em
reports/avance_av035_g04/launch_v2.json. Não lançar outra cópia; auditar v2.

## Resultado auditado da versão2

COMPLETE e PASSED_G04_PREFLIGHT_NOT_MODEL_VALIDATION;20exames/60séries,
31,533s no piloto (log total74,21s). Hashes/pixels/séries/ordem passaram;
features224 idênticas ao cacheV03: diferença máxima0. Features336 finitas,
20×3×384 e diferentes de224. Não houve treino ou consulta ao dev/confirm.

- Decode com ambos resizes:1,352s/exame em média.
- Forward224:0,02326s/exame; forward336:0,05163s/exame; razão2,21996.
- Pico alocado224:163.883.520bytes;336:213.534.208bytes (~204MiB).
- Planejamento conservador para1300casos/ambasresoluções:5570s (~93min),
  extrapolação de20exames que não garante o custo de I/O frio ou treino futuro.

Saídas privadas: reports/avance_av035_g04/v2_output; auditoria independente
reports/avance_av035_g04/audit_v2.json, reciboSHA
2250c8ff8338410bb5fe53e7e338d7074ec32f1ef23489a4c9bddb3bc51b6306.
Auditor verificou código/artefatos/IDs/paridade/tempos/geometria.336 não foi
reproduzido com uma segunda implementação; não se mediu acurácia ou AUC.

Decisão: resolução336 tecnicamente elegível para treino pareado. Melhor
score confirmado0,941; ref56696639 continuaPENDING na consulta final.

## Próxima decisão

Se os gates passarem, fixar o protocolo224vs336 com cabeças separadas no
mesmo treino1000/dev300/seeds2026e42 antes da primeira consulta ao novo dev.
Confirmação300 continua fechada. Se falhar, registrar o motivo sem relaxar
paridade/identidade após observar o resultado.
