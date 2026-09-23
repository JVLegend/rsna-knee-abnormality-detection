# AV-023 — pesquisa Kaggle e geometria G01

#RSNA #Kaggle #Pesquisa

17/09/2026. Fonte de verdade:07_Estrategia_AVANCE do vault.
Continua [M01](AV022_ABLACAO_POOLING_M01.md). Protocolo abaixo fixado antes
de qualquer avaliação das variantes G01; não é promessa de melhora no placar.

## Novidades verificadas e implicações

A API oficial listou25notebooks ordenados por dateRun. Publicação/execução
recente não implica modelo novo nem ganho reproduzido. Fontes baixadas como
texto, nunca executadas: reports/avance_av023_sources/. Nenhum laudo enviado
a serviço externo. Datas relativas da busca indexada não provam data de publicação.

- [The bee's knees](https://www.kaggle.com/code/prvsiyan/the-bee-s-knees-final-rsna-push),
  execução17/09/2026 09:12UTC: o próprio autor identifica receita herdada e
  cinco variações que não superaram0,940. Duas alterações de intensidade
  regrediram para0,922. Não repetir essa linha esperando ganho garantido.
- [Geometry to6Slots](https://www.kaggle.com/code/xiaoleilian/rsna-2026-knee-geometry-to-6-slots):
  código de ordenação IPP·normal(IOP), grupos adjacentes, contraste/máscaras,
  crop130mm e janela por volume. É EDA, não experimento isolado provando ganho.
  Incorporaremos primeiro geometria, com código próprio e testes, sem copiar
  todos os limiares ou o fallback0,5 sugerido no texto.
- [Labeling Deathmatch](https://www.kaggle.com/code/joshuaziel/rsna-slightly-overexhaustive-labeling-deathmatch),
  execução17/09/2026 20:18UTC: novos prompts/modelos mais caros não mostraram
  vantagem clara no pequeno conjunto avaliado. Separar confiança no texto
  de concordância com leitura da imagem; não importar labels ou executar APIs
  comerciais. Os autores não fornecem prova de ganho para nosso pipeline.
- [Bottleneck: labels, pipeline or capacity](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735826):
  discussão reforça testes isolados de geometria e ruído do professor. Relatos
  de participantes, não conclusões dos organizadores ou validação clínica.
- [Encoder scaling / comentários de amostragem](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735154):
  fonte já conhecida, revisitada. A comparação3vs9mistura localização e
  quantidade; não reproduzir esse confundimento. Preservar1view/3canais.

Também consultado [Head and shoulders](https://www.kaggle.com/code/prvsiyan/head-and-shoulders-knees-and-toes).
Não inferir score pelo título de um notebook. Nenhum ranking atual nosso
foi confirmado nesta pesquisa;0,941 é o melhor score previamente confirmado.

## Experimento registrado antes dos resultados

Três braços, mesma receita AV-021:

1. `control`: features originais V02, sem modificar cache/ordem histórica.
2. `physical_quartiles`: ordenar por geometria da série; amostrar25/50/75%.
3. `physical_adjacent`: mesma ordem física e mesmo índice central do braço2;
   usar centro−1/centro/centro+1, com clip só nas bordas de séries curtas.

Braço2−1avalia troca de ordenação;3−2avalia espaçamento/cobertura, sem mudar
centro, quantidade de views ou encoder. Cobertura mais estreita é parte do
tratamento de adjacência, não se deve chamar de ganho puro de resolução.
IOP/IPP ausente, orientação inconsistente ou projeção repetida bloqueiam
a experiência; não omitir estudos nem mudar denominador silenciosamente.

3planos/1view por plano/3canais/224px, sem crop novo, slots adicionais,
janela por volume ou mudança de laterality. Normalização por fatia permanece
exatamente V02. Geometria nova fica em artefatos próprios no HD.

Encoder DINOv2-S oficial congelado; sharedattention; professor original;
299treino/250dev; seeds2026/42;20épocas,batch4,AdamWlr0,001,decay0,0001.
Escolher menor softBCE média dev, primeira época em empate. Sem AUC clínica,
sem ler pixels ou avaliar confirmação150. Reproduzir controle AV-021.

Comparação: reportar ambos deltas por semente, não só melhor execução.
Promover nova referência apenas se melhor que controle nas duas sementes
por >2e−6; se ambas elegíveis, menor média, empate favorece quartis físicos.
Replay NumPy logits atol1e−4/rtol1e−5 e métricas2e−6; pixels exigem hash exato.
Teto1.800s, T4x2 usando cuda:0, offline, quota existente. Sem CSV/submissão.

## Alternativas seguintes, ainda não testadas nesta rodada

- Janela compartilhada por volume, isolada da adjacência e do crop.
- Crop físico respeitando PixelSpacing diferente nos dois eixos.
- Slots por contraste/máscara verdadeira, sem preencher slot ausente errado.
- Mais estudos de treino e fine-tuning parcial, com validação por grupo;
  os299estudos atuais são um teste de engenharia, não todo o potencial da base.

Não combinar tudo numa única experiência nem continuar apenas ajustando
o ensemble público. A skill Obsidian mantém fontes e protocolo no vault.

## Estado

**Encerramento18/09/2026:** kernel134798342 v1 COMPLETE e PASSED_G01_AUDIT.
Adjacentes0,62426164/0,62611569 vencem controle0,62607019/0,63189521
nas duas sementes; promovidos como referência própria conforme protocolo.
Quartis físicos0,62624314/0,63235150 não melhoram o controle.
597,66s total,549,01s pré-processamento; gap mediano3,5mm vs26,4mm.
Confirmação não avaliada; nenhum score clínico/LB inferido.
Auditoria em reports/avance_av023_g01/g01_audit_v1.json.
Continuação: [V03/AV-024](AV024_ESCALA_TREINO_V03.md).
O texto abaixo preserva o histórico do lançamento em17/09.

Protocolo/código inicial commitb769ee4.160testes+44subtestes passaram.
Buildlocalv1SHA cbe7ec84…(1.107.464bytes) recusado pela API:
HTTP400, fonte precisa ter menos de1MB. Nenhum kernel/treino criado.
Correção de transporte: contagens em vetor alinhado ao manifesto, eliminando
UIDs repetidos; mesmo experimento/dados, fonte legível sem código comprimido.
Builder passa a bloquear >1.000.000bytes. Resultados ainda não observados.

Correção commit6183f46. Buildlocalv2,888.413bytes,SHA256
986a3ffbb16bc3cb489e320496cb93346703e32e616ff70e32c39883dc23fbc3.
Kaggle kernel `jvlegend/rsna-knee-g01-physical-adjacency`,ID134798342,
versão1,RUNNING. Anexos: competição, DINOv2-S oficial e outputs privados
da baselineAV-021. Internetoff,T4x2,teto1.800s; quota anterior4,667horas.
Último log consultado:100/549estudos aprovados,136,52s de log. Não estimar
tempo total só com esse trecho; nenhuma perda G01 observada ainda.
Suíte final:162testes+44subtestes passaram. Código/protocolo no GitHub.

Implementação:

- src/rsna_knee_baseline/physical_triplets.py:projeção/consistência/índices/gaps.
- scripts/prepare_g01_ablation.py:manifesto congelado e limite de fonte.
- scripts/g01_ablation_runtime.py:geometria de todas séries antes do treino,
  reprodução de hashes V02, features e três braços×duas sementes.
- scripts/assess_g01_ablation.py:replay NumPy, identidade, seleção,
  máscara de escopo, comparação e decisão pré-especificada.

## Retomada

Não lançar duplicata. Consultar status do kernel; em COMPLETE baixar versão1
para reports/avance_av023_g01_v1/. Em ERROR baixar inclusive logs e eventual
g01_failure.json; não afrouxar o gate para forçar ganho.

Auditoria, com dependências NumPy/Pandas/PyTorch e PYTHONPATH=src:.:

```sh
python -m scripts.assess_g01_ablation \
  --directory reports/avance_av023_g01_v1 \
  --build reports/avance_av023_g01/g01_v2.py \
  --baseline reports/avance_av021_baseline_v1 \
  --output reports/avance_av023_g01/g01_audit_v1.json
```

Kernel não é submissão, mesmo se terminar com sucesso. Promoção exige
auditoria dos três braços e das duas sementes; melhor confirmado0,941
continua pertencendo ao ensemble público anterior.

Após auditoria: nova prioridade V03, aumentar número de estudos de treino
sem tocar nos250dev/150confirmação nem seus grupos/gold. Primeiro inventariar
labels e dados disponíveis, congelar seleção e custo. Depois fine-tuning
parcial em experiência separada. V03 ainda não implementada nesta rodada.
Manifestos maiores devem ir em anexo privado, não inflar o código >1MB.

Hashes das fontes baixadas (SHA256; não são nossos modelos):

- bees:08ad918d5d470abe8b37fa20de738cae41aebbd7b48956b53e6f70d4bce64407
- heads:7adebc548a11281954b61700ee5106ed1750c7c7b373d962efe9a5e0b0937d5c
- geometry:296df1d993d85ae9bc49d39fe9dce4e37c668b1826aab6a9de65e69d2d528448
- labels:22f6cdd6fef1f81d2c29fae77ad5d2eb417efd9805a8ede04dac78207d15b345
