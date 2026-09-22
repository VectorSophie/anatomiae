# Statistical method audit

Status: planning pass ahead of pilot analysis code. Not yet applied to
real pilot data (none exists yet).

## Design constraint from the spec (§49)

> The eventual "percentage contribution" must mean: fraction of observed
> variance under a specified experimental distribution, not universal
> metaphysical ownership of political bias. Do not produce percentages
> unsupported by design.

This rules out a single global "X% of bias comes from pretraining" claim
without naming the population/design it's relative to. Every variance
decomposition this project reports must carry: (a) the factor levels
actually sampled, (b) the response population summed over, (c) an
uncertainty interval.

## Candidate methods, mapped to this project's designs

| Method | Fits which comparison | Notes |
|---|---|---|
| Mixed-effects / hierarchical model (item as random effect, model-family/stage as fixed effects) | Base-vs-post-trained, family differences (§58) | Natural for repeated items across conditions; avoids pseudo-replication from treating every (item x condition x model) row as independent. |
| Repeated-measures ANOVA / paired bootstrap differences | Canonical vs. paraphrase vs. framing (§53, same items reused across conditions) | Paired design — use paired tests, not independent-sample tests, since the same item recurs. |
| Variance partitioning / nested design (Formation vs Expression vs Measurement, §73) | Full-study RQ2 attribution | Only valid where the design actually manipulates the factor (e.g. evaluator choice is manipulable at zero extra generation cost — score cached outputs N ways; pretraining-stage is only "manipulable" in the sense that different checkpoints ARE different factor levels, which is fine for observational-but-structured attribution). |
| Shapley / Sobol global sensitivity | Longer-term, once >=3-4 orthogonal-ish factors are cleanly crossed | Expensive with dependent/confounded inputs (model families differ on more than one axis at once); flag where inputs are NOT independent before applying — most cross-family comparisons here are confounded (different pretraining data AND architecture AND post-training simultaneously), so Shapley attribution across families would overclaim independence it doesn't have. Reserve for within-lineage comparisons (OLMo stage-wise, Amber checkpoint-wise, DataDecide recipe grid) where the design is actually closer to orthogonal. |
| Bootstrap CIs on all point estimates | Every reported effect | Default; pilot sample sizes (~100 items x 2 languages x 3 conditions) are small enough that asymptotic normal CIs are not obviously trustworthy. |

## Explicit non-goals for the pilot

- No p-value-driven publication claims from pilot data (§59) — pilot
  numbers are feasibility evidence, not confirmatory results.
- No cross-family causal attribution language (e.g. "architecture causes
  X% of variance") without a matched-architecture controlled experiment
  (§72) — cross-family comparisons in the pilot are associational and
  must be reported as such.

## Denominators discipline (ties to §63's required tests)

Every rate reported (refusal rate, malformed rate, agreement rate, etc.)
must have an explicit, tested denominator definition — e.g. is a
"malformed generation" counted in the denominator of the *stance*
agreement rate, or excluded first? Both are defensible but must be fixed
and documented per metric, with a unit test asserting the denominator
logic doesn't silently drift as new outcome categories get added
(§56's outcome taxonomy).

## Open items

- Confirm whether a mixed-effects implementation in `statsmodels` (already
  a project dependency) is sufficient, or whether `pymer4`/R-backed lme4
  is warranted for anything beyond the pilot — decide once pilot data
  shape is known, not preemptively.
- Revisit after the pilot: which of these methods actually ran cleanly on
  ~100-item x small-condition-count data, and which need more items to be
  identifiable at all (e.g. hierarchical models with many random-effect
  levels and few observations per level).
