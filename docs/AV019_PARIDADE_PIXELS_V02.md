# AV-019 — diagnóstico de pixels V02 e resultado da candidata estável

#RSNA #Kaggle #Pesquisa

16/09/2026 à noite, São Paulo. Fonte de verdade: 07_Estrategia_AVANCE do
vault; [espelho operacional](ESTRATEGIA_AVANCE.md). Continua
[AV-018](AV018_PREFLIGHT_V02_GENERICO.md).

## Submissão confirmada

Consulta autenticada: **56281610 COMPLETE, público0,941**, sem erro.
Empata com56263721; não é novo melhor. Mesmo score arredondado não implica
paridade das predições ocultas nem garante score privado. Nenhum novo envio
ou mudança de seleção final nesta rodada.

## Falha identificada e teste delimitado

Piloto134631088 v1 terminou ERROR em `Rebuilt pixels differ from HD cache`.
Log preservado em reports/avance_av018_pilot_v1/. A verificação falhou antes
da extração de features/treino; não há baseline, checkpoint aprovado ou CSV
elegível nessa versão. O log não identifica a etapa que introduziu a diferença.

Reexecução local serial dos108DICOMs selecionados: **36/36séries coincidem
exatamente** com SHA de pixels do cache. Artefato
reports/avance_av019_v02/local_rebuild_v1.json. Apenas12estudos do treino;
desenvolvimento/confirmação não acessados. Isso não prova qual biblioteca ou
operação causou a divergência no worker remoto.

Controle adicional local: primeira série exata também com Python3.12,
NumPy2.0.2, Pillow11.3.0 e pydicom3.0.2 (mesmas versões de processamento
informadas pelo worker). Não basta atribuir a falha a versões diferentes;
arquitetura/decodificação/resize permanecem hipóteses até comparar os estágios.

Instrumentação adicionada em scripts/v02_pilot_runtime.py:

- salva versões de bibliotecas no início;
- registra SHA do DICOM, pixels float32, imagem normalizada, imagem final e
  percentis usados;
- na primeira divergência, salva recibo e arrays intermediários privados no
  diretório de saída e **interrompe**; não usa tolerância nem fallback;
- carrega o encoder somente após a reconstrução passar; pesos/config e
  contrato do modelo continuam fixos;
- teste sintético cobre caminho exato e persistência de evidência na falha.

## Diagnóstico remoto e retomada

Mesmo kernel privado `jvlegend/rsna-knee-v02-generic-dino-pilot`, ID134631088,
**versão2**, offline/T4x2/teto1.800s, somente competição e DINOv2small/1.
Despacho sem erro, cota disponível antes12,1254h. Não é submissão.
Build privado reports/avance_av019_v02/v02_diagnostic_v2.py,
SHA2b17445319627953df96f0d182bf53b334fdb2cf61ec719a7ac588e605ea8f27.

Se falhar novamente, comparar v02_pixel_mismatch.npz e JSON com os mesmos
arquivos locais antes de corrigir a transformação. Não trocar hashes esperados
para aprovar artificialmente o teste. Se passar, executar o auditor V02 e
verificar retomada/custo antes do treino completo.

O ambiente local foi verificado: macOS/zsh, Python3.14.6 e Node22.22.2;
sem `python`, PowerShell, GNO ou venv ativa. Execuções isoladas usam Python
gerenciado pelo uv, sem alteração de configuração do projeto.

## Verificação local

**134 testes e44subtestes passaram** em95,23s:

```sh
PYTHONPATH=src:. uv run --no-project --with pytest --with numpy --with pandas \
  --with scikit-learn --with pillow --with pydicom --with torch --with torchvision \
  python -m pytest tests -q
```

Tentativa anterior com unittest tinha seis falhas de importação por ausência
de pytest/torch no ambiente efêmero; repetida com dependências e runner completos.
Isso não equivale a aprovação do piloto CUDA ou validação do modelo.
