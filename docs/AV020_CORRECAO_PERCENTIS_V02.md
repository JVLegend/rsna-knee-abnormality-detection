# AV-020 — correção reprodutível dos percentis do V02

#RSNA #Kaggle #Pesquisa

17/09/2026. Fonte de verdade: 07_Estrategia_AVANCE do vault;
[espelho](ESTRATEGIA_AVANCE.md). Continua [AV-019](AV019_PARIDADE_PIXELS_V02.md).

## Causa demonstrada

Piloto134631088 v2 terminou ERROR. A primeira divergência ocorre na série
axial do primeiro estudo, não na primeira série sagital. Portanto o controle
anterior da primeira série não excluía diferença de versão do NumPy.

Arquivos DICOM e pixels float32 coincidem exatamente nos3canais diagnosticados.
No canal0, percentil99 remoto297,65234375 contra cache297,6499938964844.
Isso muda64.180valores normalizados (máximo7,8678e−6) e125pixels após resize
(máximo1nível uint8). Outros2canais iguais. Aplicar localmente o resize à
imagem normalizada remota reproduz exatamente a saída remota: nesse caso,
a divergência está na normalização, não no DICOM nem no resize.

Controle causal local com NumPy2.0.2/Pillow11.3.0/pydicom3.0.2 reproduz
o percentil e a normalização remotos. O código instalado do NumPy2.0.2
calcula q/100 no dtype da entrada quando q é escalar Python; em float32,
isso altera a posição fracionária na estatística de ordem. NumPy2.5.3 local
reproduz o cache. Não generalizar para toda operação NumPy ou para todo dado.

Referência primária do método: [numpy.percentile](https://numpy.org/doc/2.0/reference/generated/numpy.percentile.html).
O método linear interpola entre estatísticas de ordem; a atribuição causal
acima vem dos nossos arrays e testes pareados, não apenas da documentação.

## Mudança e verificação

`normalize_cache_compatible` no runtime V02: percentis lineares em float64,
limites convertidos explicitamente a float32, normalização float32 e restante
do pipeline preservado. Método1/99%, arquivos e canais continuam iguais.
O helper histórico compartilhado não foi modificado. Não mudamos hashes do
cache, pesos, rótulos ou partição para acomodar a divergência.

`scripts.audit_v02_pixel_fix` reconstrói108DICOMs/36séries do piloto e compara
bytes dos arrays com cache congelado:

| Ambiente local | Receita antiga exata | Receita corrigida exata |
|---|---:|---:|
| NumPy2.0.2 / Pillow11.3.0 | 32/36 | **36/36** |
| NumPy2.5.3 / Pillow12.3.0 | 36/36 | **36/36** |

Artefatos privados no HD: reports/avance_av020_v02/pixel_fix_numpy202_v1.json
e pixel_fix_numpy253_v1.json, ambos PASSED_FIXED_PIXEL_REBUILD. Nenhuma leitura
de pixels dev/confirmation. Isso ainda não verifica todas897séries do treino
nem750séries dev; igualdade com cache permanece gate por série no worker.

Também substituímos duas buscas recursivas nos anexos por descoberta que
não entra em train_series/test_series/kaggle_test_series, com limite256pastas.
A versão2 levou710,77s até a falha; não atribuímos toda demora à descoberta,
pois não havia cronômetro isolado. A versão3 mede esse tempo separadamente.

**136 testes e44subtestes passaram**. Seis testes V02 passaram também no
NumPy2.0.2, incluindo limites dos percentis e poda/limite da busca de anexos.

## Execução GPU e retomada

Mesmo kernel privado jvlegend/rsna-knee-v02-generic-dino-pilot,
ID134631088, **v3**, offline/T4x2/teto1.800s, usa cuda:0. Anexos: competição
e DINOv2 genérico oficialsmall/1. Cota disponível antes11,60728909h.
**COMPLETE e PASSED_V02_PILOT_AUDIT**. Build privado reports/avance_av020_v02/v02_fixed_v3.py,
SHA212fb597d5d9152c0c2cd061483ad939fb0d03beeb636cad89b7be9c2d0bbb18.
Não é treino completo nem submissão elegível.

Saída baixada em reports/avance_av020_pilot_v3/. Auditoria executada:

```sh
uv run --no-project --with numpy python -m scripts.assess_v02_pilot \
  --directory reports/avance_av020_pilot_v3 \
  --cache reports/avance_av018_v02/cache_audit_v1.json \
  --build reports/avance_av020_v02/v02_fixed_v3.py \
  --output reports/avance_av020_v02/pilot_audit_v3.json
```

Gates aprovados:36séries/108DICOMs iguais ao cache, features12x3x384 finitas,
encoder congelado, máscara de slots, treino da cabeça,16passos e checkpoint
retomado com perdas/parâmetros exatos na mesma sessão GPU.
Tempo: descoberta0,001893s,reconstrução3,2128s,encoder0,8088s,total6,8730s
fora imports; log até conclusão do script33,34s. Pico de memória alocada
160.398.848bytes (~153MiB), não memória total reservada do worker.

Próximo baseline299treino/250dev: projeção aproximada188,74s antes de margem,
377,48s com margem2x para20épocas/batch4. Não garante runtime de mais dados;
propor teto1.800s por execução, cache auditado integralmente e receitas/seeds
2026/42 congeladas antes de avaliar. Retomada foi testada na mesma sessão,
não entre máquinas; qualidade do modelo não medida. Features/pesos piloto
não são um baseline completo nem justificam submissão.

Confirmação150permanece fechada. Melhor público0,941, nenhuma nova submissão
ou alteração de seleção final na AV-020.
