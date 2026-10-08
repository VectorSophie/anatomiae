# Human A–C agreement v1

Independent human annotators A and C labeled the same frozen 150-response validity sample.

## Integrity

- 150/150 sample IDs match.
- Statement text matches exactly on 150/150 rows.
- Response text matches exactly on 150/150 rows.
- No duplicate IDs.
- No missing required relation/mode/confidence labels.
- All labels are in the frozen taxonomy.

## Agreement

| Dimension | Exact agreement | Cohen's kappa |
|---|---:|---:|
| Relation | 105/150 (70.0%) | 0.552 |
| Response mode | 115/150 (76.7%) | 0.564 |

Among the 100 cases where both annotators assigned a directional relation
(`supports` or `opposes`), 97/100 (97.0%) agreed on direction.

There are 61 unique samples with a disagreement on relation and/or mode:
45 relation disagreements and 35 mode disagreements.

The dominant relation disagreement is boundary placement between a directional label
and `mixed_or_conditional` / `neutral_or_no_position`, not reversal of political direction.
Only 3 cases are direct `supports` ↔ `opposes` reversals when both annotators call the
response directional.

## Important sampling limitation

The 150-response set deliberately oversamples ambiguous / evaluator-disagreement cases.
These agreement statistics validate the annotation task and evaluator analysis, but are
not unweighted estimates of disagreement in the full generation population.
