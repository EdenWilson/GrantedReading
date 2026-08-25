# reading_level

Measures how hard a passage is to read, repairs it when it falls outside a
target grade band, and refuses to ship anything it cannot bring into band.

The pipeline is `generate → score → repair → re-score → gate`. This package
owns measurement and text surgery. Generation and the gate policy live in the
caller (`ai.py`).

Layout:

- `data/frequency_table.json.gz` — word-frequency table the scorer reads
- `data/fixtures/` — short passages used by the tests
- `tests/` — pytest suite for this pipeline
- `scripts/` — offline frequency-table build

## What it measures

Two features dominate every mainstream readability metric, and both are used
here:

- **Semantic difficulty** — mean log10 word frequency across content words.
  Stopwords and proper nouns are excluded: function words are uniformly common
  and dilute the signal, and a student's own name is not a vocabulary burden.
- **Syntactic difficulty** — mean sentence length, measured with a real
  sentence segmenter (`pysbd`), not `text.split('.')`.

Flesch-Kincaid and a Dale-Chall-style score are computed alongside as a cheap
cross-check. They are stored on `TextFeatures` and never exposed through the
HTTP API.

## What it deliberately does not claim

- **These are not certified readability scores.** They are estimated grade
  bands produced by coefficients that have never been fitted against a labeled
  corpus. Trust the ordering; do not trust the absolute numbers.
- **Word sense disambiguation is best-effort.** Substitution resolves the sense
  from context where it can and otherwise falls back to WordNet's dominant
  sense, which stopped "deliberate" becoming "careful". It cannot separate two
  words WordNet itself calls synonyms, so a misleading swap inside the right
  synset is still possible. Output is meant to be reviewed by a teacher, not
  shipped unread.
- **The drift check verifies, it does not understand.** It catches a rewrite
  that wandered onto another topic, dropped a protected term, or negated the
  source. It will not catch a fluent rewrite that quietly changes a number, a
  date, or a name while keeping the topic and polarity intact.
- **The Dale-Chall cross-check is Dale-Chall-shaped, not Dale-Chall.** The
  published 3000-word familiar list is not bundled; the top ~5% of the
  frequency table stands in for it.
- **Short text cannot be measured.** Below 50 words the score is returned with
  `confidence="low"` and a `short_text` flag. Question stems always hit this,
  which is why they are logged rather than gated.
- **Deterministic repair never makes text harder.** Split and synonym swap
  only simplify. A draft below the band is reported as `below_target_band`
  and left unchanged by this package. Raising difficulty is Operator D in
  `ai.py`, which now runs on both sides of the band.

## Calibration status

**Uncalibrated.** `GRADE_COEF_WORD_FREQ`, `GRADE_COEF_SENTENCE_LENGTH`, and
`GRADE_INTERCEPT` in `config.py` were eyeballed against the three fixture
passages in `data/fixtures` so that the ordering is right and the magnitudes
are plausible. Three hand-written passages are not a calibration set. See
`docs/calibration.md` for how to fit them properly and what data is needed.

The shipped frequency table was built from `wordfreq`, not the SUBTLEX-US
reference corpus, because SUBTLEX-US is not redistributed with this project.
The build script supports both; the source is recorded in the artifact's
`_meta` block.

## Public API

```python
from reading_level import (
    score_text, correct_text, CorrectionResult, ReadingLevelError,
    build_diagnostics, PassageDiagnostics, check_semantic_drift, DriftReport,
)
```

The first four are the original surface. The rest exist to support the LLM
rewrite pass that lives in `ai_rewrite.py`: this package describes the problem
and judges the answer, but never calls a model. Nothing else is supported;
modules prefixed with an underscore (`_nlp`, `_errors`) are internal.

```python
score = score_text("The dog ran to the park.")
score.estimated_grade   # 1.8
score.confidence        # "low" -- too short to measure
score.flags             # ["short_text", "single_sentence"]

result = correct_text(passage, target_grade=4, protected_terms=["photosynthesis"])
result.in_band          # False
result.failure_reason   # "no_applicable_edit"
result.edits            # [RepairEdit(...), ...]
result.score_trajectory # [5.812, 5.44, 5.21] -- one entry per iteration

# Only worth building when the above failed.
diagnostics = build_diagnostics(result.text, 4, prior_result=result)
diagnostics.grade_gap                    # 0.71
diagnostics.hard_words[0].sense_gloss    # "bring about abruptly"
diagnostics.hard_words[0].suggested_replacements  # ["cause"]

drift = check_semantic_drift(original, rewritten, protected_terms=["photosynthesis"])
drift.passed            # False
drift.reasons           # ["protected terms dropped: photosynthesis"]
```

`correct_text` never raises for content reasons — it returns a result with
`gate_passed=False` and a `failure_reason`. `ReadingLevelError` is raised only
for setup problems (a missing model or frequency table) and by the caller's
gate.

## Repair operators

Repair is targeted surgery on failing sentences, never regeneration:
regenerating would discard the topical content the passage was built around.

- **A — sentence splitting.** Splits at coordinating conjunctions and, where
  the relationship can be restored with an explicit connective, at subordinate
  clause boundaries. Both halves must have a subject on the ROOT and a finite
  verb; a fragment is worse than a long sentence, so unsafe splits are skipped.
- **B — constrained synonym substitution.** Candidates come from WordNet, not
  vector search, because nearest neighbours in embedding space include antonyms
  and co-hyponyms. Every candidate is a lemma of a synset the original word
  belongs to, so synonymy holds by construction. Which synset is chosen matters
  as much as synonymy: the operator uses the sense NLTK's Lesk resolves from the
  surrounding sentence, and falls back to WordNet's dominant sense when Lesk
  cannot decide. It never considers the unrestricted sense list, which is what
  turned "a deliberate attempt" into "a careful attempt". A sense Lesk picks
  that drifts far from the dominant one must additionally clear a Wu-Palmer
  taxonomy-distance floor. Word vectors are not used: spaCy's pruned table
  scored unrelated words at 1.0 and real synonyms at 0.47. Each candidate must
  also clear a frequency gain floor and part-of-speech agreement verified by
  re-parsing. Irregular verbs are skipped rather than regularised into forms
  like "dealed". A word introduced by one edit is never replaced by a later one,
  so meaning cannot drift one defensible hop at a time. `protected_terms` is a
  hard constraint.

  Substitutions are applied in batches of `SUBSTITUTION_BATCH_SIZE` between
  full-passage re-scores, because a single substitution moves the estimate by
  only 0.03–0.05 grade levels: the score averages over every content word, so
  one re-score per edit paid parse costs out of proportion to the movement. Each
  substitution is still recorded as its own edit.

  After WordNet and frequency, each surviving lemma is dropped into a ±3-token
  window around the original word and compared with `all-mpnet-base-v2`. A
  candidate below `MIN_SUBSTITUTION_SENTENCE_SIMILARITY` (0.92) is discarded.
  The window is required because a full-sentence compare lets one swapped word
  hide behind the rest of a long sentence: postpone→table is 0.933 over 23
  words and 0.851 in the window.

  **Known limitation.** "ensuing weeks" → "resulting weeks" scores 0.960 in
  that same window -- the encoder treats them as near-paraphrases, so the
  filter cannot reject it. Review substitutions before they reach a student.
- **C — clause simplification.** Removes parenthetical asides and converts
  by-agent passives to active voice, but only for regular verbs where the
  conjugation can be verified. Off by default
  (`ENABLE_CLAUSE_SIMPLIFICATION`): highest meaning-distortion risk.
- **D — LLM rewrite.** Not in this package. A, B, and C plateau on large gaps
  because substitution moves the passage score by only 0.03–0.05 grade levels
  per edit and WordNet often has no easier synonym at all, so Operator D
  escalates to a model rewrite — but only when deterministic repair has already
  failed its gate, never as the default path. This package contributes the two
  deterministic halves: `diagnostics.build_diagnostics` describes exactly what
  is wrong (which words, in which sense, with which pre-vetted replacements;
  which sentences are over the ceiling and by how much), and `verify` plus the
  existing scorer decide whether what comes back is acceptable. The model call
  and the prompt live in `ai_rewrite.py`, because nothing in this package may
  touch a network.

## How much a rewrite may change

`PassageDiagnostics.passage_type` decides how much freedom the rewrite prompt
grants, and it is **caller-supplied, never inferred**. Nothing here can tell a
story from a science paragraph, and the question it answers — is this detail
load-bearing for the lesson? — belongs to whoever asked for the passage.

- **`narrative`** — theme, topic, and protected terms hold; everything else is
  negotiable. Plot, ending, order of events, and what the characters do may all
  change. The student never sees the pre-rewrite draft, so there is no original
  to be unfaithful to, and pinning the details is what forced word-level swaps
  like "deliberate" → "careful" in the first place: told to keep every detail,
  the model's only remaining lever was a synonym inside a sentence it could not
  restructure.
- **`informational`** — structure is free, content is not. Wording, sentence
  shape, and organisation may all change; every fact must survive, because the
  comprehension questions are generated from them.

Omitting the field logs a warning and selects `informational`, the stricter
tier. That direction is deliberate: granting plot freedom to an unlabelled
factual passage fails quietly, whereas the reverse merely costs a rewrite some
freedom it could have had. The type also decides whether the numbers-and-facts
gate below runs, so a mislabelled passage loses a check as well as gaining
licence — an unrecognised value raises rather than resolving to a tier.

## Verifying a rewrite

A word swap constrained to a WordNet sense cannot say something the original
did not; an LLM rewrite can. `verify.check_semantic_drift` is the check the
deterministic operators never needed, and it is three independent gates:

- **Protected terms** — hard and non-negotiable. A term present in the original
  and absent from the rewrite fails immediately, whatever the similarity says.
- **Negation delta** — hard. Embeddings are blind to polarity: a rewrite
  asserting the opposite of the source scored *higher* (0.695) than a faithful
  one (0.704) in measurement. Counting negation markers separates that case
  cleanly.

  **Known limitation.** It counts markers, so it cannot tell inversion from
  paraphrase. Negative polarity carried by a word rather than a particle is
  invisible in the original but explicit in a plain-English rewrite of it:
  "Persistently inadequate disclosure could precipitate penalties" becomes "If
  they do not share enough information, they could face big fines" — identical
  in meaning, `negation_delta` of +1, rejected. With `MAX_NEGATION_DELTA = 0`
  this was the single largest cause of rejection once the narrative tier was
  loosened, accounting for every drift failure in a 4-run measurement on the
  business fixture. The threshold trades these false positives for certainty
  about the failure it exists to catch; that trade has not been calibrated.
- **Sentence-level similarity** — graded, using `all-mpnet-base-v2`. Compared
  per sentence with best-match alignment rather than as a passage average,
  because an average lets one badly drifted sentence hide behind six good ones.

- **Numbers and named facts** — hard, and the only gate that is tier-dependent.
  Runs for `informational` passages, skipped for `narrative` ones, because a
  story dropping a detail is the freedom that tier grants and a science passage
  dropping a number is not. Numbers are compared by value, so "three" and "3"
  are one fact; entities are compared leniently, so "Maya Chen" → "Maya" passes
  while "Jupiter" → "Saturn" fails.

  This is the third failure mode the other gates cannot see. "Water boils at
  100 degrees Celsius" coming back as 50 scores 0.82 passage-level and 0.67 at
  sentence level with a negation delta of zero — every graded check waves it
  through. Measured on live rewrites it added no false positives: across six
  runs of the business and boiling-point fixtures, `numbers_missing` and
  `entities_missing` were empty on every rewrite a human would accept.

spaCy's vectors were evaluated for this first and rejected on measurement, not
reputation: averaged over a passage, `en_core_web_md` scored an unrelated
passage at 0.900 against the business fixture versus 0.934 for a faithful
rewrite of it, and `en_core_web_lg` was worse (0.865 versus 0.868). Both give
negative separation, so no threshold exists. A static-embedding alternative was
also rejected. The labeled cases are in `tests/drift_cases.py` and are the same
ones the model choice was made on.

Reading level and meaning are judged separately and by different machinery. The
deterministic scorer remains the single source of truth for whether a passage
hit its band, regardless of which operator produced it; the model is never
asked to assess its own output.

Substitution proceeds worst-offender-first and stops the moment the passage
enters band. Over-substituting produces flat, patronising prose and can push
the score below the band floor, which is its own failure.

Each iteration offers both operators, applies whichever lands closer to the
band, and records the comparison on the edit. The iteration budget scales with
the size of the gap (`BASE_ITERATIONS_PER_GRADE_GAP`, capped by
`ITERATION_HARD_CEILING`) rather than being fixed, because splitting moves the
score in large steps and substitution in small ones. `CorrectionResult.
score_trajectory` records the estimated grade after each iteration, so a run
can be read as converging, plateauing, or oscillating without instrumenting the
loop.

## Determinism

Nothing in this package calls a language model, and nothing touches the
network. The whole pipeline is unit-testable offline. The only I/O is loading
the frequency table and the spaCy model, both lazy, cached, and read-only after
load.

## Rebuilding the frequency table

```bash
# Reference corpus (not redistributed -- supply your own copy)
python -m reading_level.scripts.build_frequency_table \
    --source subtlex --corpus /path/to/SUBTLEXusfrequencyabove1.csv

# Bootstrap source, no manual download
python -m reading_level.scripts.build_frequency_table --source wordfreq
```
