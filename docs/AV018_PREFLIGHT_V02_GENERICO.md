# AV-018 — preflight do baseline visual próprio V02

#RSNA #Kaggle #Pesquisa

16/09/2026. Fonte de verdade: 07_Estrategia_AVANCE do vault,
[espelho](ESTRATEGIA_AVANCE.md). Continua [AV-017](AV017_AUDITORIA_L01_TREINO.md).

## Estado e decisão

56281610 ainda PENDING, sem score/erro na consulta inicial; melhor0,941
preservado. Não reenviar. Avançamos V02 com professor original e pretreino
genérico, sem cabeças ou encoders ajustados nesta competição.

Auditoria local **PASSED_TRAIN_CACHE_PIXELS**:299estudos,897séries,
2.691canais,94.557.531bytes de arrays. Formato uint8[3,224,224], CRC/leitura,
tamanho congelado, índices, variância e hashes verificados. Sem canal constante
ou hash de array de pixels duplicado dentro do treino. Leitura serial, memória
limitada; nenhuma inferência/treino pesado no Mac. O HD tem cerca354GiB livres.

Os três canais representam quantis25/50/75% da série: gaps de2a80posições,
**não cortes adjacentes**. Esta é a receita inicial preservada, não uma correção
de geometria já validada. G01 poderá comparar adjacência depois do baseline.
Não revalidamos orientação anatômica completa nem descartamos duplicatas
visuais/pacientes entre splits apenas com hashes dentro do treino.

## Proveniência do encoder

Modelo fixado: [Meta DINOv2 Small, versão1 no Kaggle](https://www.kaggle.com/models/metaresearch/dinov2/PyTorch/small/1).
API oficial confirmou versão1, arquivos de08/08/2023, pytorch_model.bin
88.297.097bytes, treinamento indicado no [repositório DINOv2](https://github.com/facebookresearch/dinov2).
Licença indicada pela API: Apache2.0; isso não libera os dados médicos ou
outros checkpoints públicos para qualquer uso.

Referência independente do arquivo: [facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small),
commit ed25f3a31f01632728cabb09d1542f84ab7b0056, metadata LFS consultada.
SHA256 esperado do bin:
1051e25b2ed69ddad24f3c41e7b6eed6e7f7d012103ea227e47eb82e87dc2050.
Config oficial SHA1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d.
O worker deve verificar ambos antes de carregar com weights_only=True e
load_state_dict(strict=True); nenhum código remoto do modelo é executado.

## Piloto executável — teste de engenharia, não baseline treinado

- scripts/preflight_v02_cache.py:audita todos os arrays do treino e escolhe
  12estudos distribuídos por custo aproximado de pixels, sem usar os labels
  na seleção. Inclui36séries/108DICOMs selecionados, hashes de pixels locais.
- scripts/prepare_v02_pilot.py:gate de partição, cobertura299/897,
  piloto pertencente exatamente ao treino, teacher e arrays fixos.
- scripts/v02_pilot_runtime.py:reconstrói os mesmos108DICOMs no Kaggle e
  exige igualdade pixel a pixel por SHA antes de extrair features.
- DINOv2-S/14 congelado, FP32, CLS384; entrada224, quartis como3canais,
  divisão por255 e normalização ImageNet explícita. Não é a transformação
  padrão completa do AutoImageProcessor nem adjacência nativa.
- Cabeça própria, inicialização aleatória: atenção compartilhada por estudo
  384→64→1 sobre os3planos e classificador384→12. Ainda não é atenção
  específica por alvo; M01 permanece ablação futura.
- BCEWithLogits com professor suave original, incluindo0,5. AdamW,
  lr1e−3, weight_decay1e−4, batch4, seed2026,16passos de engenharia.
- Salva modelo/otimizador/RNG no passo8; continua4passos, restaura e repete
  esses4. Exige perdas/parâmetros exatamente iguais; testa máscara e bag vazio.
  Retomada na mesma sessão, não promessa de determinismo entre máquinas.
- scripts/assess_v02_pilot.py:auditoria independente de fonte/pixels/features/
  recibos/checkpoint/perdas e projeção de custo. Não mede AUC nem publica CSV
  elegível; checkpoint de piloto não vira baseline/finalista automaticamente.

Build privado no HD: reports/avance_av018_v02/v02_generic_pilot_v1.py.
SHAe310c75490da6f71e68facf4e8bb14742bbd5832a109d0ceccf80561191a8aa9.
Auditoria cache: reports/avance_av018_v02/cache_audit_v1.json.
Código gerado contém IDs/labels de treino; não publicar no GitHub.

## Execução

[Kernel privado V02](https://www.kaggle.com/code/jvlegend/rsna-knee-v02-generic-dino-pilot),
v1 ID134631088, **RUNNING**, T4x2 alocada, usa apenas cuda:0; offline,
teto1.800s. Únicos anexos: competição e DINOv2 oficialsmall/1.
Cota antes17,86903667h. Busca de slug sem duplicata; despacho sem erros.
Não é submissão à competição nem treino completo nos299estudos.

**105 testes locais passaram**. Asserções CUDA/checkpoint serão confirmadas
na execução remota, não contam como testes locais já executados.
V01 SHA365566b0… e professor SHAeeca4a86… intactos. Não lemos pixels
de desenvolvimento/confirmação, não avaliamos modelos nesses splits.

## Retomada

Consultar o kernel existente, sem duplicar. Após COMPLETE, baixar recibo,
NPZ e checkpoint em reports/avance_av018_pilot_v1/ e executar:

```sh
uv run --no-project --with numpy python -m scripts.assess_v02_pilot \
  --directory reports/avance_av018_pilot_v1 \
  --cache reports/avance_av018_v02/cache_audit_v1.json \
  --build reports/avance_av018_v02/v02_generic_pilot_v1.py \
  --output reports/avance_av018_v02/pilot_audit_v1.json
```

Se falhar, corrigir a causa sem afrouxar paridade de pixels/pesos e registrar
nova versão. Se passar, fixar orçamento e treino V02 com299treino/250dev,
professor original e duas sementes(2026/42). Preparar/validar caches dev antes
de uso; confirmação150permanece fechada até seleção dos finalistas. Não
extrapolar runtime sem margem; custo ainda não medido no piloto em andamento.
