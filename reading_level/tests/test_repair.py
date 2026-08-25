import pytest

from reading_level import correct_text
from reading_level import config
from reading_level._nlp import get_nlp
from reading_level.repair import (
    OPERATOR_SPLIT,
    OPERATOR_SUBSTITUTE,
    REASON_BELOW_BAND,
    REASON_EMPTY,
    _candidate_replacements,
    _hard_words,
    _is_independent_clause,
)
from reading_level.scorer import parse_sentences, split_sentences

COMPOUND = (
    "The students planted a garden behind the school and they watered it "
    "every morning before class started. The tomatoes struggled because the "
    "ground stayed too dry throughout the entire month of July."
)


def test_splitting_never_produces_fragments(table):
    result = correct_text(COMPOUND, 3, table=table)

    split_edits = [e for e in result.edits if e.operator == OPERATOR_SPLIT]
    for edit in split_edits:
        for part in split_sentences(edit.after):
            assert _is_independent_clause(part), f"fragment produced: {part!r}"


def test_every_output_sentence_stands_alone(table):
    result = correct_text(COMPOUND, 3, table=table)
    if not any(e.operator == OPERATOR_SPLIT for e in result.edits):
        pytest.skip("no split was applied to this passage")

    for sentence in split_sentences(result.text):
        assert _is_independent_clause(sentence)


def test_substitution_never_touches_protected_terms(fixture_text, table):
    text = fixture_text("hard_secondary")
    protected = ["proliferation", "methane", "composting", "contamination"]

    result = correct_text(text, 4, protected_terms=protected, table=table)

    substitutions = [e for e in result.edits if e.operator == OPERATOR_SUBSTITUTE]
    replaced = {edit.before.lower() for edit in substitutions}
    assert replaced.isdisjoint({term.lower() for term in protected})

    lowered = result.text.lower()
    for term in protected:
        assert term.lower() in lowered


def test_protected_terms_are_reported(fixture_text, table):
    result = correct_text(
        fixture_text("hard_secondary"),
        4,
        protected_terms=["methane", "notpresentanywhere"],
        table=table,
    )
    assert "methane" in result.protected_terms_preserved
    assert "notpresentanywhere" not in result.protected_terms_preserved


def test_substitutions_use_real_word_forms(fixture_text, table):
    """Regular inflection rules must not invent forms like "dealed"."""
    result = correct_text(fixture_text("hard_secondary"), 5, table=table)
    for edit in result.edits:
        if edit.operator == OPERATOR_SUBSTITUTE:
            assert table.contains(edit.after.lower())


def test_easier_than_target_is_reported_not_complicated(fixture_text, table):
    result = correct_text(fixture_text("easy_primary"), 9, table=table)

    assert result.edits == []
    assert result.failure_reason == REASON_BELOW_BAND
    assert result.text == result.original_text


def test_in_band_text_is_left_alone(fixture_text, table):
    text = fixture_text("easy_primary")
    result = correct_text(text, 2, table=table)

    assert result.in_band
    assert result.gate_passed
    assert result.failure_reason is None
    assert result.edits == []


def test_failure_reason_is_set_whenever_the_gate_fails(fixture_text, table):
    result = correct_text(fixture_text("hard_secondary"), 2, table=table)
    assert not result.gate_passed
    assert result.failure_reason is not None


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "Dog.",
        "Tyrannosaurus Rex met Mia. Sam and Alvarez greeted Rex.",
        " ".join(["the extraordinarily complicated proliferation of apparatus"] * 70)
        + ".",
    ],
    ids=["empty", "whitespace", "single_word", "proper_nouns", "runaway_sentence"],
)
def test_loop_terminates_on_adversarial_input(text, table):
    result = correct_text(text, 3, max_iterations=5, table=table)
    assert result.iterations <= 5
    if not result.gate_passed:
        assert result.failure_reason is not None


def test_empty_text_never_passes_the_gate(table):
    result = correct_text("", 0, table=table)
    assert not result.gate_passed
    assert result.failure_reason == REASON_EMPTY


def test_max_iterations_is_respected(fixture_text, table):
    result = correct_text(fixture_text("hard_secondary"), 3, max_iterations=2, table=table)
    assert result.iterations <= 2


def test_invalid_grade_level_is_rejected(table):
    with pytest.raises(ValueError):
        correct_text("The dog ran to the park today.", "not-a-grade", table=table)


def test_original_text_is_preserved_on_the_result(fixture_text, table):
    text = fixture_text("hard_secondary")
    result = correct_text(text, 5, table=table)
    assert result.original_text == text


# --------------------------------------------------------------------------
# Regressions: substitution candidates were being vetoed by spaCy's pruned
# vector table, which scored unrelated words at 1.0 and real synonyms at 0.47.
# --------------------------------------------------------------------------


def _candidates_for(word, sentence, table):
    return {c.word for c in _replacements_for(word, sentence, table)}


def _replacements_for(word, sentence, table):
    doc = get_nlp()(sentence)
    token = next(t for t in doc if t.lemma_.lower() == word)
    return _candidate_replacements(token, table)


def test_melancholic_to_melancholy_is_a_known_false_positive(table):
    """A good morphological swap scores like a bad same-synset swap.

    'She felt melancholic' vs 'She felt melancholy' is 0.875, sitting on
    top of motive→need (0.874) and below postpone→table (0.903). The 0.92
    floor that kills those also kills this. Recorded rather than papered
    over by lowering the floor.
    """
    candidates = _candidates_for(
        "melancholic", "She felt melancholic that evening.", table
    )
    assert "melancholy" not in candidates


def test_meaning_guard_does_not_depend_on_word_vectors(table):
    """Guarding happens in WordNet, so a vectorless model must not break it."""
    candidates = _candidates_for("arduous", "The climb was arduous.", table)
    assert candidates


def test_hard_words_mostly_produce_candidates(fixture_text, table):
    """The window filter must not return us to the old 'almost no swaps' state.

    Before WordNet replaced the vector veto, 2 of 39 hard words had a
    candidate. The ±3-token filter is stricter than WordNet alone, but it
    still has to leave a usable set.
    """
    docs = [get_nlp()(d.text) for d in parse_sentences(fixture_text("astronaut_mixed"))]
    hard = _hard_words(docs, table, set())
    with_candidates = [
        token for *_, token in hard if _candidate_replacements(token, table)
    ]
    assert len(hard) >= 15
    assert len(with_candidates) >= 5


def test_both_operators_run_when_both_problems_are_present(fixture_text, table):
    """Splitting used to consume the whole iteration budget."""
    result = correct_text(fixture_text("astronaut_mixed"), 5, table=table)
    operators = {edit.operator for edit in result.edits}

    assert OPERATOR_SPLIT in operators
    assert OPERATOR_SUBSTITUTE in operators


def test_operator_choice_is_recorded_on_each_edit(fixture_text, table):
    result = correct_text(fixture_text("astronaut_mixed"), 5, table=table)
    assert result.edits
    for edit in result.edits:
        assert edit.selection
        assert edit.operator in edit.selection


def test_substitution_lowers_the_estimated_grade(fixture_text, table):
    result = correct_text(fixture_text("astronaut_mixed"), 5, table=table)
    assert result.final_score.estimated_grade < result.initial_score.estimated_grade


# --------------------------------------------------------------------------
# Regressions: substitutions were drawn from whichever WordNet sense ranked
# highest, not the sense the sentence was using, so "a deliberate attempt"
# became "a careful attempt" and "favorable terms" became "lucky terms".
# --------------------------------------------------------------------------

NEGOTIATION = (
    "The board made a deliberate attempt to negotiate more favorable terms."
)

# Judged by hand. Both belong to WordNet's "carefully thought out in advance"
# sense of "deliberate", which is the one this sentence uses. Note that the
# obvious paraphrases -- intentional, purposeful, planned -- are NOT reachable:
# WordNet gives "deliberate" only two adjective senses, and none of the three
# appears in either.
DELIBERATE_ACCEPTABLE = {"calculated", "measured"}


def test_deliberate_does_not_become_careful(table):
    candidates = _candidates_for("deliberate", NEGOTIATION, table)

    assert "careful" not in candidates
    assert candidates <= DELIBERATE_ACCEPTABLE


def test_favorable_does_not_become_lucky(table):
    assert "lucky" not in _candidates_for("favorable", NEGOTIATION, table)


def test_wrong_sense_substitutions_do_not_reach_the_passage(table):
    """End to end, not just at the candidate level."""
    result = correct_text(NEGOTIATION, 4, table=table)

    replacements = {
        edit.after.lower()
        for edit in result.edits
        if edit.operator == OPERATOR_SUBSTITUTE
    }
    assert not replacements & {"careful", "lucky"}


def test_a_resolved_sense_is_recorded_on_the_edit(table):
    """Substitutions carry the sense path so edits are reviewable after the fact."""
    replacements = _replacements_for(
        "motive",
        "Investigators could not establish the underlying motive for the decision.",
        table,
    )
    assert replacements
    assert replacements[0].disambiguated
    assert replacements[0].sense_label


def test_motive_does_not_become_need(table):
    """Same synset is not enough; the local window has to stay close.

    WordNet lists 'need' under motivation.n.01. In this sentence the ±3
    window scores 0.745, below MIN_SUBSTITUTION_SENTENCE_SIMILARITY. A
    full-sentence compare of the same pair is 0.955 and would have let it
    through -- that is why the filter uses a window, not the whole sentence.
    """
    assert "need" not in _candidates_for(
        "motive",
        "Analysts nonetheless remained skeptical about the underlying "
        "motive behind this unprecedented contractual concession.",
        table,
    )


def test_postpone_does_not_become_table(table):
    """Parliamentary 'table' shares a synset with 'postpone' and is not a swap.

    The 23-word sentence is the case the full-sentence filter missed (0.933).
    The ±3 window is 0.851 and rejects it.
    """
    candidates = _candidates_for(
        "postpone",
        "The committee's decision to postpone the merger was met with "
        "considerable skepticism, particularly among shareholders who had "
        "anticipated a swift resolution.",
        table,
    )
    assert "table" not in candidates


def test_a_sense_preserving_swap_still_survives_the_sentence_filter(table):
    candidates = _candidates_for(
        "detrimental",
        "The delay would be detrimental to an already unstable financial situation.",
        table,
    )
    assert "damaging" in candidates


def test_words_with_no_wordnet_swap_do_not_gain_one(table):
    """The encoder vetoes candidates; it does not invent them."""
    assert _candidates_for(
        "exacerbate",
        "Further delay would only exacerbate an already unstable financial situation.",
        table,
    ) == set()


def test_fallback_path_still_substitutes_and_is_flagged(table):
    """Lesk returns None on thin context; the dominant sense still applies."""
    replacements = _replacements_for(
        "detrimental",
        "The delay would be detrimental to an already unstable financial situation.",
        table,
    )

    assert replacements, "fallback produced no candidate at all"
    assert all(not c.disambiguated for c in replacements)
    assert all("not disambiguated" not in c.word for c in replacements)


def test_fallback_is_visible_on_the_resulting_edit(table):
    result = correct_text(NEGOTIATION, 4, table=table)
    substitutions = [e for e in result.edits if e.operator == OPERATOR_SUBSTITUTE]
    if not substitutions:
        pytest.skip("no substitution fired on this short passage")

    for edit in substitutions:
        assert isinstance(edit.sense_disambiguated, bool)
        expected = "sense:" if edit.sense_disambiguated else "not disambiguated"
        assert expected in edit.rationale


# --------------------------------------------------------------------------
# Regressions: one substitution moves the passage score by only 0.03-0.05
# grade levels, so a fixed five-iteration budget applying one edit per pass
# could not close a real gap.
# --------------------------------------------------------------------------


def test_iteration_budget_scales_with_the_gap(fixture_text, table):
    """A wide gap must buy more passes than the base default of five."""
    result = correct_text(fixture_text("astronaut_mixed"), 5, table=table)
    assert result.iterations > config.MAX_ITERATIONS


def test_iteration_budget_respects_the_hard_ceiling(fixture_text, table):
    result = correct_text(fixture_text("hard_secondary"), 1, table=table)
    assert result.iterations <= config.ITERATION_HARD_CEILING


def test_one_pass_applies_a_batch_of_substitutions(fixture_text, table):
    """Batching is what makes the scaled budget affordable."""
    result = correct_text(fixture_text("business_register"), 6, table=table)
    substitutions = [e for e in result.edits if e.operator == OPERATOR_SUBSTITUTE]

    assert len(substitutions) > result.iterations
    assert len(substitutions) <= result.iterations * config.SUBSTITUTION_BATCH_SIZE


def test_batched_edits_are_recorded_individually(fixture_text, table):
    result = correct_text(fixture_text("business_register"), 6, table=table)
    substitutions = [e for e in result.edits if e.operator == OPERATOR_SUBSTITUTE]

    assert len({(e.before, e.after) for e in substitutions}) == len(substitutions)
    for edit in substitutions:
        assert edit.before and edit.after
        assert edit.selection


def test_a_batch_still_competes_against_splitting(fixture_text, table):
    """Batching must not let substitution bypass per-iteration operator choice."""
    result = correct_text(fixture_text("astronaut_mixed"), 5, table=table)
    operators = {edit.operator for edit in result.edits}

    assert operators == {OPERATOR_SPLIT, OPERATOR_SUBSTITUTE}
    for edit in result.edits:
        assert edit.operator in edit.selection


def test_score_trajectory_tracks_every_iteration(fixture_text, table):
    result = correct_text(fixture_text("business_register"), 6, table=table)

    assert len(result.score_trajectory) == result.iterations
    assert result.score_trajectory[-1] == pytest.approx(
        result.final_score.estimated_grade
    )


def test_score_trajectory_is_empty_when_nothing_is_applied(fixture_text, table):
    result = correct_text(fixture_text("easy_primary"), 9, table=table)
    assert result.score_trajectory == []


def test_substitutions_never_chain(fixture_text, table):
    """A word introduced by one edit must not be replaced by a later one.

    "tenacity" -> "persistence" -> "continuity" is two defensible-looking hops
    that together assert something the passage never said.
    """
    result = correct_text(fixture_text("astronaut_mixed"), 5, table=table)
    substitutions = [e for e in result.edits if e.operator == OPERATOR_SUBSTITUTE]

    introduced = {e.after.lower() for e in substitutions}
    replaced = {e.before.lower() for e in substitutions}
    assert introduced.isdisjoint(replaced)
