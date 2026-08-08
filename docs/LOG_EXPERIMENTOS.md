# Log de experimentos

#RSNA #Kaggle #Pesquisa #Importante

| Data | Versão | Hipótese | Validação local | Leaderboard | Decisão |
|---|---|---|---:|---:|---|
| 07/08/2026 | `setup` | Repositório, regras, protocolo e smoke test sintético | N/A | N/A | Infraestrutura pronta |
| 07/08/2026 | `v0_report_metadata` | TF-IDF de laudo + sexo + atributos de séries, um classificador por alvo | A executar após download | A executar | Candidato inicial |

## Convenção

- `v0`, `v1`, ... identificam famílias de modelagem, não submissões automáticas.
- A coluna de validação deve conter macro-AUC e AUC de cada alvo, além de seed/split.
- Score público e privado ficam separados; não ajustar pesos usando o leaderboard.
- Cada entrada deve apontar para o commit e para os artefatos locais ignorados pelo Git.
