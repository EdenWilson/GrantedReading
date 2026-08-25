# Calibrating the reading level model

The grade estimate is a linear model over two features:

```
estimated_grade = GRADE_INTERCEPT
                + GRADE_COEF_WORD_FREQ        * mean_log_word_freq
                + GRADE_COEF_SENTENCE_LENGTH  * mean_sentence_length
```

All three coefficients live in `reading_level/config.py` and are currently
**uncalibrated placeholders**. This document explains how to replace them with
fitted values.

## Current status

Production coefficients are still the fixture placeholders. They have **not**
been replaced.

| Constant | Value | Basis |
|---|---|---|
| `GRADE_COEF_WORD_FREQ` | -2.60 | Eyeballed against three fixture passages |
| `GRADE_COEF_SENTENCE_LENGTH` | 0.50 | Eyeballed against three fixture passages |
| `GRADE_INTERCEPT` | 4.46 | Eyeballed against three fixture passages |

A large 3–12 ELA excerpt corpus was evaluated and removed: it confirmed
the signs (rarer words and longer sentences raise the grade) but flattened
K–2 and secondary fixtures, so those coefficients were never shipped.
The next fit needs independently labeled **worksheet-like narrative**,
balanced across K–6.

Measured against `reading_level/data/fixtures`, the current placeholders produce:

| Fixture | mean_log_word_freq | mean_sentence_length | estimated_grade |
|---|---|---|---|
| `easy_primary` | 2.22 | 5.8 | 1.60 |
| `medium_upper_elementary` | 1.90 | 13.1 | 6.08 |
| `hard_secondary` | 1.22 | 23.8 | 13.00 (clamped) |

The ordering is correct and the magnitudes are plausible. Three hand-written
passages are not a calibration set, and nothing here should be reported to a
teacher as an accurate grade level until the fit below is done.

## What data is needed

A corpus of passages with **independently assigned grade levels**, ideally:

- **300+ passages minimum**, 1000+ preferred, for a two-feature linear fit with
  usable confidence intervals.
- **Full K-12 coverage**, roughly balanced across bands. A corpus skewed toward
  grades 4-8 will fit those well and extrapolate badly at the ends, which is
  exactly where special education material lives.
- **100+ words per passage.** Anything shorter is below the reliability floor
  (`MIN_RELIABLE_WORDS`) and adds noise rather than signal.
- **Prose matching the target domain.** Coefficients fitted on encyclopedia
  text will misjudge narrative fiction, which is what this app generates.

Candidate sources: teacher- or publisher-leveled worksheet stories, basal
readers with stated grade levels, state-released narrative passages, or
graded readers with a documented leveling scheme. Excerpt corpora built for
high-school ELA will not transfer.
Whatever is used, record its provenance — a model fitted on one leveling scheme
inherits that scheme's assumptions.

## Fitting procedure

1. **Extract features.** For each passage, call
   `reading_level.scorer.score_text` and keep `features.mean_log_word_freq` and
   `features.mean_sentence_length`. Do not use `estimated_grade`; that is the
   thing being replaced.

2. **Hold out a test set.** Split 80/20, stratified by grade band, before
   looking at anything.

3. **Fit ordinary least squares** with the labeled grade as the dependent
   variable:

   ```python
   import numpy as np
   design = np.column_stack([
       np.ones(len(rows)),
       [r.mean_log_word_freq for r in rows],
       [r.mean_sentence_length for r in rows],
   ])
   coefficients, *_ = np.linalg.lstsq(design, labels, rcond=None)
   intercept, coef_freq, coef_sentence_length = coefficients
   ```

4. **Check the signs.** `coef_freq` must be negative (rarer words are harder)
   and `coef_sentence_length` positive (longer sentences are harder). A sign
   flip means the feature extraction or the labels are wrong; do not ship it.

5. **Evaluate on the held-out set.** Report mean absolute error in grade levels
   and the share of passages landing in the correct band. Band accuracy matters
   more than MAE here: the system gates on bands.

6. **Compare against the cross-checks.** `features.flesch_kincaid_grade` and
   `features.dale_chall_grade` are computed on every passage. If the fitted
   model does not beat both on held-out band accuracy, it is not earning its
   complexity and one of the published formulas should be used instead.

7. **Write the values into `config.py`** and delete the `UNCALIBRATED` comment
   above them. Record the corpus name, size, date, and held-out error in the
   comment that replaces it.

## Recalibrating the band width

`BAND_HALF_WIDTH` is 0.5, so grade N spans `[N-0.5, N+0.5]`. Once held-out
error is known, that width should be revisited: a band narrower than the
model's standard error will reject passages the model cannot actually
distinguish, and the retry ladder will burn model calls on noise.

## Other values that should be empirically set

These were chosen by reasoning about the frequency distribution, not fitted:

- `HARD_WORD_LOGFREQ_THRESHOLD` (1.20) — the point at which a content word
  becomes a substitution target. Should be set from the vocabulary actually
  known at each grade, not from a single global cut.
- `MIN_SENSE_SIMILARITY` (0.50) and `MIN_FREQ_GAIN` (0.15) — the substitution
  acceptance floors. Should be tuned against human judgements of whether a
  substitution preserved meaning. This is the highest-value calibration target
  after the grade coefficients, because a wrong substitution is a worse failure
  than an uncorrected hard word.
- `MAX_SENTENCE_LENGTH_INTERCEPT` (8.0) and `MAX_SENTENCE_LENGTH_SLOPE` (1.0) —
  the per-band sentence length ceiling. Should come from measured sentence
  lengths in grade-appropriate published material.
- `QUESTION_BAND_TOLERANCE` (1.5) — how far a question stem may drift before it
  is worth reporting.
- `FAMILIAR_WORD_PERCENTILE` (95.0) — stands in for the Dale-Chall 3000-word
  list. If a licensed copy of that list is available, replace the percentile
  cut with a real membership test.
- `OOV_LOGFREQ` (0.11) — the 5th percentile of the shipped table. Recompute
  whenever the table is rebuilt; the build script prints the percentiles.

## Logging as a calibration dataset

`ai.py` emits one JSON log record per stage (`generated`, `edit_applied`,
`corrected`, `questions_scored`, `gate_failed`). Every passage that needed
several iterations, and every passage that failed the gate, is a labeled
example of the model disagreeing with the generator. Collecting these is the
cheapest available path to a domain-matched calibration set.
