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

Implementação iniciada; resultados ainda não observados.
