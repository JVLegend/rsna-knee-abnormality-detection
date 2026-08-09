# Log de experimentos

#RSNA #Kaggle #Pesquisa #Importante

| Data | Versão | Hipótese | Validação local | Leaderboard | Decisão |
|---|---|---|---:|---:|---|
| 07/08/2026 | `setup` | Repositório, regras, protocolo e smoke test sintético | N/A | N/A | Infraestrutura pronta |
| 07/08/2026 | `v0_report_metadata` | TF-IDF de laudo + sexo + atributos de séries, um classificador por alvo | 5 folds por estudo; macro-AUC `0.556609`; contrato OK | A executar | Benchmark de referência, sem envio |
| 08/08/2026 | `v0_1_report_metadata_c32` | Mesmo modelo com regularização C=32 | Média de 2 seeds: macro-AUC `0.565655`; ganho de `0.002888` sobre C=2 | A executar | Candidata para primeiro envio, sem DICOM |
| 08/08/2026 | `weak_lexicon_audit` | Léxico multilíngue com janela simples de negação | Auditoria nos 58 estudos: score-AUC diagnóstico `0.646785`; precisão varia `0.43–0.88` | — | Não criar pseudo-rótulos; testar como feature de confiança |
| 08/08/2026 | `v0_2_report_metadata_lexicon` | Adicionar ao v0.1 uma feature {-1, 0, 1} por alvo, calculada do laudo e sem rótulos | 5 folds por estudo; seeds 42/2026: macro-AUC `0.628815`/`0.630918`; média `0.629867`; saída local válida e entrypoint Kaggle idêntico | A executar | Candidata principal para primeiro envio; sem DICOM |
| 08/08/2026 | `v0_2_lexicon_weight_grid` | Ajustar somente a escala da feature lexical | Médias em seeds 42/2026: peso `0,5`=`0.627766`, `1`=`0.629867`, `2`=`0.622799`, `4`=`0.611707` | — | Manter peso `1`; alternativas rejeitadas |
| 08/08/2026 | `dicom_series_smoke` | Confirmar aquisição incremental e decodificação antes de baixar o conjunto integral | 1 série real, 30/30 fatias, `320×320`, `uint16`, `MONOCHROME2`, ~6,19 MB; `pydicom` OK | — | Manter lote integral pendente; implementar seleção/normalização antes de ampliar |
| 08/08/2026 | `dicom_subset_manifest` | Selecionar séries fluido-sensíveis, cobrindo positivo/negativo por alvo | Manifesto local com 10 estudos únicos e 1 série por estudo; ambos os valores presentes nos 12 alvos | — | Seleção congelada para o primeiro lote visual |
| 08/08/2026 | `dicom_subset_download` | Baixar somente as séries do manifesto, sem varrer o dataset integral | 10 séries, 289/289 arquivos, `193.721.314` bytes; leitura DICOM e pixels OK; dimensões de `256×256` a `800×800` | — | Lote local pronto para o pipeline visual; aquisição integral não autorizada ainda |
| 08/08/2026 | `dicom_subset_download_v2` | Ampliar a cobertura sem repetir estudos do primeiro manifesto | 8 estudos/séries, 232/232 arquivos, `169.086.016` bytes; os dois lotes somam 18 estudos e 521 fatias; leitura DICOM e pixels OK | — | Manter aquisição incremental; iniciar pipeline visual antes de novo lote |
| 08/08/2026 | `dicom_25d_v0` | Construir representação visual compacta para validar a esteira 2.5D | 18/18 estudos processados; 3 fatias por série; arrays `(3, 224, 224) uint8`; `index.json` íntegro | — | Pipeline de leitura/normalização pronto; próximo gate é embedding e pooling |
| 08/08/2026 | `efficientnet_b0_embedding_v0` | Usar um encoder visual pré-treinado disponível localmente, sem rede, para reduzir cada estudo a um vetor | 18/18 estudos; matriz `(18, 1280)` `float32`; todos os valores finitos; 18 linhas distintas; pesos com SHA-256 registrado no índice local | — | Pré-processamento validado; repetir nos 58 estudos anotados antes da fusão |
| 08/08/2026 | `dicom_labeled_completion_audit` | Evitar rebaixar arquivos e separar séries completas das incompletas | 34/40 séries adicionais completas; 1.296/1.428 arquivos presentes; 132 arquivos faltantes em 6 séries | — | Processar os 52 estudos completos e retomar apenas o residual |
| 08/08/2026 | `dicom_25d_labeled_v0` | Ampliar a representação 2.5D para o universo visual já completo | 52/52 estudos processados; arrays `(3, 224, 224) uint8`; índice íntegro | — | Gate de pré-processamento ampliado; aguardar 6 séries restantes |
| 08/08/2026 | `efficientnet_b0_embedding_52` | Extrair embeddings EfficientNet-B0 para os 52 estudos completos | Matriz `(52, 1280)` `float32`, finita; SHA-256 dos pesos registrado | — | Dados prontos para baseline visual e futura fusão |
| 08/08/2026 | `visual_embedding_logistic_52` | Medir o sinal visual com regressão logística e split por estudo | Macro-AUC `0,593222` (seed 42) e `0,584142` (seed 2026); média `0,588682` | — | Resultado preliminar; não substituir a v0.2 textual até completar os 58 |
| 08/08/2026 | `fusion_probability_blend_52` | Misturar probabilidades textuais e visuais sem aumentar a arquitetura | Peso visual `0,25`/textual `0,75`: macro-AUC `0,643647` (seed 42) e `0,660061` (seed 2026), média `0,651854`; texto sozinho no mesmo subconjunto: `0,638499` | — | Candidato multimodal atual; repetir nos 58 antes de congelar o peso |
| 08/08/2026 | `kaggle_kernel_v0_2` | Executar a candidata v0.2 em kernel privado, sem internet e com a competição anexada | Versão 1: módulo não foi empacotado; versão 2: código autocontido, mas `train.csv` não foi montado em `/kaggle/input/rsna-knee-abnormality-detection/` | — | Corrigir aceite/vínculo da fonte; não enviar leaderboard |
| 08/08/2026 (UTC 09/08) | `kaggle_submission_55364182` | Enviar a referência textual v0.2 após corrigir a autodetecção do dataset | Kernel privado concluído; CSV íntegro e sem internet | Público `0,505` | Benchmark inicial |
| 08/08/2026 (UTC 09/08) | `kaggle_submission_55364637` | Misturar texto e EfficientNet-B0 2.5D com peso visual `0,4` | Worker concluído; 58/58 estudos rotulados e 3/3 testes válidos | Público `0,607` | Referência visual inicial |
| 08/08/2026 (UTC 09/08) | `kaggle_submission_55365036` | Calibrar peso visual por alvo | Worker concluído; CSV íntegro | Público `0,582` | Rejeitar por overfit dos pesos targetwise |
| 08/08/2026 (UTC 09/08) | `kaggle_submission_55365263` | Suavizar a calibração targetwise em direção ao blend global | Worker concluído; CSV íntegro | Público `0,605` | Rejeitar; não supera `0,607` |
| 08/08/2026 (UTC 09/08) | `kaggle_submission_55365537` | Selecionar uma série fluido-sensível por plano e agregar multi-view | Worker concluído em `97,7 s`; `183 views`; 58/58 treino e 3/3 teste válidos | Público `0,635` | Nova referência; ganho de `+0,028` sobre `55364637` |
| 08/08/2026 (UTC 09/08) | `forum_audit_2026_08_08` | Ler discussões e respostas oficiais para revisar a hipótese de modelagem | Host confirma rótulo visual autoritativo e ausência de `Report` no teste; comunidade alerta para instabilidade dos 58 estudos e shortcut de scanner | — | Usar weak supervision nos 4.407 relatórios, sem enviar texto a API externa |
| 09/08/2026 (UTC 09/08) | `v3_weak_visual_local_smoke` | Transferir sinal explícito dos laudos para o classificador visual, preservando o rótulo oficial como âncora | 4.410 registros percorridos em `130,8 s`; CSV válido; no HD somente 58 estudos têm DICOM completo | — | Hipótese aprovada para worker; ausência de menção nunca vira negativo |
| 09/08/2026 (UTC 09/08) | `kaggle_kernel_v3_weak_visual_v6` | Rodar a v3 no kernel multi-view histórico, preservando o anexo de pesos offline; limiar `0,85` e peso weak máximo `0,10` | Versão 6 ficou `RUNNING` sem logs/artefato por mais de 15 min e foi supersedida | — | Não promover; preservar como tentativa CPU sem saída |
| 09/08/2026 (UTC 09/08) | `kaggle_kernel_v3_weak_visual_v7` | Acelerar a mesma candidata com GPU genérica | Worker recebeu Tesla P100, incompatível com o PyTorch instalado (`sm_60`; suporte `sm_70+`); terminou `ERROR` antes do modelo | — | Diagnóstico confirmado; não usar P100 neste ambiente |
| 09/08/2026 (UTC 09/08) | `kaggle_kernel_v3_weak_visual_v8` | Executar a candidata com GPU explicitamente `NvidiaTeslaT4`, internet desligada e pesos anexados | Kernel histórico multi-view, versão 8, status atual `RUNNING`; aguardando primeiros logs | — | Aguardar CSV; promover somente após saída íntegra e score futuro superior a `0,635` |
| 09/08/2026 (UTC 09/08) | `kaggle_kernel_v3_weak_visual_v9` | Repetir a v8 com fallback automático de GPU antiga para CPU | Worker T4 `COMPLETE` em `74,8 s`; `183 views`, CSV íntegro, mas o pacote enviado manteve `weak_visual=False` e reproduziu a referência v2 | — | Não submeter; usar como regressão de runtime e corrigir o default do pacote |
| 09/08/2026 (UTC 09/08) | `kaggle_kernel_v3_weak_visual_v10` | Executar de fato a weak supervision visual com o fallback CUDA atualizado | `COMPLETE` em `3.727,8 s`; T4; `weak_visual=True`; 4.407/4.407 estudos e 13.230 views válidos; CSV íntegro arquivado em `submissions/submission_v3_weak_visual_v10.csv` | — | Candidata pronta para o primeiro envio após o reset UTC; as cinco submissões do dia já foram usadas |

## Detalhe da candidata v0.2

- `C=32`, TF-IDF de palavras/caracteres, sexo, metadados de séries e léxico auditável.
- O léxico é uma feature de entrada; não cria pseudo-rótulos, não transforma ausência de anotação em negativo e não consulta o teste para ajustar regras.
- A diferença entre os dois seeds foi `0.002103` de macro-AUC; a média superou a v0.1 em `0.064212`.
- A submissão foi gerada por `scripts/run_baseline.py` e reproduzida exatamente por `kaggle/rsna_knee_v0.py` no conjunto local de smoke.
- Artefatos ignorados pelo Git: `reports/v0_2_lexicon_seed42.json`, `reports/v0_2_lexicon_seed2026.json` e `submissions/submission_v0_2_report_metadata_lexicon.csv`.

## Convenção

- `v0`, `v1`, ... identificam famílias de modelagem, não submissões automáticas.
- A coluna de validação deve conter macro-AUC e AUC de cada alvo, além de seed/split.
- Score público e privado ficam separados; não ajustar pesos usando o leaderboard.
- Cada entrada deve apontar para o commit e para os artefatos locais ignorados pelo Git.
