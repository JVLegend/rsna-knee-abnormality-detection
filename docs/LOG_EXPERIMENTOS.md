# Log de experimentos

#RSNA #Kaggle #Pesquisa #Importante

| Data | Versão | Hipótese | Validação local | Leaderboard | Decisão |
|---|---|---|---:|---:|---|
| 07/08/2026 | `setup` | Repositório, regras, protocolo e smoke test sintético | N/A | N/A | Infraestrutura pronta |
| 07/08/2026 | `v0_report_metadata` | TF-IDF de laudo + sexo + atributos de séries, um classificador por alvo | 5 folds por estudo; macro-AUC `0.556609`; contrato OK | A executar | Benchmark de referência, sem envio |
| 08/08/2026 | `v0_1_report_metadata_c32` | Mesmo modelo com regularização C=32 | Média de 2 seeds: macro-AUC `0.565655`; ganho de `0.002888` sobre C=2 | A executar | Candidata para primeiro envio, sem DICOM |
| 08/08/2026 | `weak_lexicon_audit` | Léxico multilíngue com janela simples de negação | Auditoria nos 58 estudos: score-AUC diagnóstico `0.646785`; precisão varia `0.43–0.88` | — | Não criar pseudo-rótulos; testar como feature de confiança |

## Convenção

- `v0`, `v1`, ... identificam famílias de modelagem, não submissões automáticas.
- A coluna de validação deve conter macro-AUC e AUC de cada alvo, além de seed/split.
- Score público e privado ficam separados; não ajustar pesos usando o leaderboard.
- Cada entrada deve apontar para o commit e para os artefatos locais ignorados pelo Git.
