# Pesquisa Kaggle — 23/09/2026

#JoaoVictor #Kaggle #Tecnologia

Fonte de verdade: vault SuperJV,
`01_Projects/Competicoes/RSNA_Knee_Abnormality_Detection/05_Forum_Kaggle.md`,
seção “Atualização — 23/09/2026: novos candidatos além de 0,941”.
Fila: [ESTRATEGIA_AVANCE.md](ESTRATEGIA_AVANCE.md).

## Estado e candidatos

Consulta ao [leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard):
DataRockstar 929º/4.229, público0,941,26entradas; líder0,958. Retrato datado,
não score privado. Nenhum treino, submissão ou execução de fonte de terceiros.

- P1: auditar e reproduzir receita fixa
  [Maverick V3](https://www.kaggle.com/code/maverickss26/rsna-knee-restructured-version-3),
  público0,943 V1/scriptVersionId351863321. Verificar dependências, licenças,
  inventário real, recibos, fallback e custo antes de executar.
- P2: auditar recibos de
  [Fracture OOF](https://www.kaggle.com/code/sushanthtiruvaipati/rsna-knee-0942-rad-fracture-oof),
  público0,942 V1, igual ao parent. Não é ganho público comprovado.
- P3: fine-tuning parcial e professores visuais cross-fitted, preservando
  grupos/seleção interna/holdout. O
  [ConvNeXt público](https://www.kaggle.com/code/forwardidear/rsna-knee-selftrain-convnext)
  seleciona épocas pelos58 oficiais: não copiar esse protocolo.
- P4: investigar contrato de canais RadImageNet com treino compatível,
  como hipótese, sem alterar inputs de checkpoints existentes às cegas.

V04 continua NOT_CONFIRMED; V05 não virou OOF executado. G04 continua piloto
preparado; concorrência/quota de21/09 precisam ser consultadas novamente.

## Fontes congeladas no HD

Downloads para leitura estática em `reports/research_20260923/`:

| Pasta | SHA-256 do notebook |
|---|---|
| maverick-v3 | `7dc49666e01c46e5017d4975960b06b359e869d8fd916d1be41cb90561beb522` |
| fracture-oof | `f245c4da2a6d27a89bbf5f2409e41d6c5676d009ec284a5b82d5d38610673196` |
| mattia-current | `5e03fc9f7799b2e46bcf67395d1ad5177d1e9f19b58cc93825c9fc02da6d0155` |
| convnext | `aef63e3c88275b00767d8d8ad11cdf23b043de98f9b052ee229610ed2b2fd821` |

## Discussões lidas

- [Plateau0,941](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742050)
- [Modelos individuais](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304)
- [Raptor e comentários recentes](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737696)
- [Replicação HARD/SOFT](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734105)

Interpretações, limitações e gates detalhados no vault. Não autoriza APIs
externas com laudos, gastos, publicação nem mudança da seleção final.
