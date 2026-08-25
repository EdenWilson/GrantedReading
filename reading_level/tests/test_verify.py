"""Meaning-drift detection.

The cases live in drift_cases.py because they were also the evidence used to
choose the embedding model: spaCy vectors (md and lg) and static embeddings all
scored an unrelated passage as similar to the source as a faithful rewrite, and
were rejected on these exact examples.
"""

import pytest
from drift_cases import (
    CASES,
    DRIFT_FACTS_INVERTED,
    DRIFT_TOPIC_SWAPPED,
    INFORMATIONAL_FACT_KEPT,
    INFORMATIONAL_NUMBER_ALTERED,
    INFORMATIONAL_ORIGINAL,
    LEGIT_FULL_REWORD,
    LEGIT_MILD_REWORD,
    NARRATIVE_NEW_ENDING,
    NARRATIVE_ORIGINAL,
    ORIGINAL,
    UNRELATED,
)

from reading_level import config
from reading_level.verify import check_semantic_drift


def test_identical_text_passes():
    report = check_semantic_drift(ORIGINAL, ORIGINAL)
    assert report.passed
    assert report.reasons == []
    assert report.passage_similarity == pytest.approx(1.0, abs=1e-3)


def test_a_faithful_rewrite_that_changes_almost_every_word_passes():
    """The case a weak similarity metric reports as drift.

    This is what a good grade-6 rewrite looks like: same facts, same order,
    nearly no shared vocabulary. Rejecting it would make Operator D useless.
    """
    report = check_semantic_drift(ORIGINAL, LEGIT_FULL_REWORD)

    assert report.passed, report.reasons
    assert report.min_sentence_similarity >= config.MIN_SENTENCE_SIMILARITY


def test_a_mild_rewrite_passes():
    report = check_semantic_drift(ORIGINAL, LEGIT_MILD_REWORD)
    assert report.passed, report.reasons


def test_a_rewrite_about_a_different_subject_fails():
    report = check_semantic_drift(ORIGINAL, DRIFT_TOPIC_SWAPPED)
    assert not report.passed
    assert report.worst_sentence_index is not None


def test_an_unrelated_passage_fails():
    report = check_semantic_drift(ORIGINAL, UNRELATED)
    assert not report.passed


def test_a_rewrite_asserting_the_opposite_fails():
    """Embeddings alone score this above a faithful rewrite; negation catches it."""
    report = check_semantic_drift(ORIGINAL, DRIFT_FACTS_INVERTED)

    assert not report.passed
    assert report.negation_delta > config.MAX_NEGATION_DELTA


@pytest.mark.parametrize(
    "label,rewrite,should_pass", CASES, ids=[case[0] for case in CASES]
)
def test_every_labeled_case_lands_correctly(label, rewrite, should_pass):
    report = check_semantic_drift(ORIGINAL, rewrite)
    assert report.passed is should_pass, report.reasons


# --------------------------------------------------------------------------
# Protected terms are a hard constraint, checked before anything graded.
# --------------------------------------------------------------------------


def test_a_dropped_protected_term_fails_regardless_of_similarity():
    """Identical text minus one protected word: similarity is ~1.0 anyway."""
    original = "Plants use photosynthesis to make food from sunlight every day."
    rewritten = "Plants use a special process to make food from sunlight every day."

    report = check_semantic_drift(
        original, rewritten, protected_terms=["photosynthesis"]
    )

    assert not report.passed
    assert report.protected_terms_missing == ["photosynthesis"]
    assert any("photosynthesis" in reason for reason in report.reasons)


def test_a_preserved_protected_term_does_not_fail():
    original = "Plants use photosynthesis to make food from sunlight every day."
    rewritten = "Plants use photosynthesis to make food from the sun each day."

    report = check_semantic_drift(
        original, rewritten, protected_terms=["photosynthesis"]
    )

    assert report.protected_terms_missing == []
    assert report.passed, report.reasons


def test_protected_terms_are_matched_case_insensitively():
    report = check_semantic_drift(
        "Photosynthesis feeds the plant.",
        "photosynthesis feeds the plant.",
        protected_terms=["Photosynthesis"],
    )
    assert report.protected_terms_missing == []


def test_a_term_absent_from_the_original_is_not_required():
    report = check_semantic_drift(
        "The dog ran to the park.",
        "The dog ran to the yard.",
        protected_terms=["photosynthesis"],
    )
    assert report.protected_terms_missing == []


# --------------------------------------------------------------------------
# Negation, the failure embeddings are blind to.
# --------------------------------------------------------------------------


def test_a_single_inserted_negation_fails():
    original = "The experiment worked and the plants grew tall."
    rewritten = "The experiment did not work and the plants grew tall."

    report = check_semantic_drift(original, rewritten)

    assert report.negation_delta > 0
    assert not report.passed


def test_removing_a_negation_is_also_caught_when_it_flips_meaning():
    """Delta is directional; a dropped 'not' shows up as a negative delta.

    That direction is not gated -- the similarity check has to carry it -- so
    this records the current behaviour rather than asserting a rejection.
    """
    original = "The plants did not grow in the dark room."
    rewritten = "The plants grew in the dark room."

    report = check_semantic_drift(original, rewritten)
    assert report.negation_delta < 0


def test_rephrasing_without_changing_polarity_keeps_the_delta_at_zero():
    original = "The long negotiation finally concluded after many months."
    rewritten = "The long talks finally ended after many months."

    report = check_semantic_drift(original, rewritten)
    assert report.negation_delta == 0


# --------------------------------------------------------------------------
# Numbers and named facts
# --------------------------------------------------------------------------


def test_an_altered_number_fails_an_informational_check():
    """The failure every graded check is blind to.

    The two passages differ by one digit and score 0.82 passage-level, 0.67 at
    sentence level, with a negation delta of zero. Nothing but a direct
    comparison of the values catches it.
    """
    report = check_semantic_drift(
        INFORMATIONAL_ORIGINAL,
        INFORMATIONAL_NUMBER_ALTERED,
        passage_type="informational",
    )

    assert not report.passed
    assert report.numbers_missing == ["100"]
    assert report.negation_delta == 0
    assert report.min_sentence_similarity > config.MIN_SENTENCE_SIMILARITY


def test_a_reorganised_informational_rewrite_keeping_its_number_passes():
    """Structural freedom is not what the fact check restricts."""
    report = check_semantic_drift(
        INFORMATIONAL_ORIGINAL,
        INFORMATIONAL_FACT_KEPT,
        passage_type="informational",
    )

    assert report.passed, report.reasons
    assert report.numbers_missing == []
    assert report.facts_checked


def test_a_narrative_may_drop_a_detail_the_fact_check_would_reject():
    """The tier split, stated as one pair of passages.

    The rewrite loses "misplaying three notes" entirely. Under the narrative
    tier that is the freedom being granted; under the informational tier the
    same text is a dropped fact. Only the passage type differs.
    """
    narrative = check_semantic_drift(
        NARRATIVE_ORIGINAL, NARRATIVE_NEW_ENDING,
        protected_terms=["violin"], passage_type="narrative",
    )
    informational = check_semantic_drift(
        NARRATIVE_ORIGINAL, NARRATIVE_NEW_ENDING,
        protected_terms=["violin"], passage_type="informational",
    )

    assert narrative.passed, narrative.reasons
    assert not narrative.facts_checked

    assert not informational.passed
    assert informational.numbers_missing == ["3"]


def test_an_unspecified_passage_type_checks_facts():
    """Silence takes the strict tier here too."""
    report = check_semantic_drift(
        INFORMATIONAL_ORIGINAL, INFORMATIONAL_NUMBER_ALTERED
    )

    assert report.facts_checked
    assert not report.passed


def test_spelled_out_and_numeric_forms_are_the_same_fact():
    report = check_semantic_drift(
        "The team waited three months for the report.",
        "The team waited 3 months for the report.",
        passage_type="informational",
    )

    assert report.numbers_missing == []


def test_shortening_a_name_is_not_a_dropped_fact():
    """Lenient by design: an over-strict hard gate rejects good rewrites."""
    report = check_semantic_drift(
        "Maya Chen practiced every day before the audition.",
        "Maya practiced every day before the audition.",
        passage_type="informational",
    )

    assert report.entities_missing == []


def test_a_swapped_name_is_a_dropped_fact():
    report = check_semantic_drift(
        "Jupiter is the largest planet in the solar system.",
        "Saturn is the largest planet in the solar system.",
        passage_type="informational",
    )

    assert report.entities_missing == ["Jupiter"]
    assert not report.passed


def test_vague_dates_are_not_gated_on():
    """"quarterly" -> "every three months" is a legitimate simplification.

    spaCy tags both as DATE. Gating on that label would reject the rewrite
    this whole operator exists to produce.
    """
    report = check_semantic_drift(
        "Officers must file the quarterly documentation.",
        "Officers must send in papers every three months.",
        passage_type="informational",
    )

    assert report.entities_missing == []
    assert report.numbers_missing == []


# --------------------------------------------------------------------------
# Report shape
# --------------------------------------------------------------------------


def test_the_worst_sentence_is_identified_by_index():
    original = (
        "The dog ran to the park. The cat slept on the mat. "
        "The bird sang in the tree."
    )
    rewritten = (
        "The dog ran to the park. Quantum chromodynamics governs the strong "
        "nuclear force. The bird sang in the tree."
    )

    report = check_semantic_drift(original, rewritten)

    assert report.worst_sentence_index == 1
    assert not report.passed


def test_every_failure_carries_a_reason():
    report = check_semantic_drift(ORIGINAL, UNRELATED, protected_terms=["nowhere"])
    assert not report.passed
    assert report.reasons


def test_a_pass_carries_no_reasons():
    report = check_semantic_drift(ORIGINAL, LEGIT_MILD_REWORD)
    assert report.passed
    assert report.reasons == []
