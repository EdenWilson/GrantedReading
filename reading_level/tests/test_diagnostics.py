"""build_diagnostics must be pure, offline, and agree with Operator B."""

import logging

import pytest

from reading_level import bands, config, correct_text
from reading_level._nlp import get_nlp
from reading_level.diagnostics import build_diagnostics
from reading_level.repair import replacement_options, substitution_targets
from reading_level.scorer import parse_sentences


def test_diagnostics_needs_no_network(monkeypatch, fixture_text, table):
    """Nothing in the package may reach a model, including this path."""

    def explode(*args, **kwargs):
        raise AssertionError("build_diagnostics made a network call")

    monkeypatch.setattr("requests.post", explode, raising=False)
    monkeypatch.setattr("requests.get", explode, raising=False)

    diagnostics = build_diagnostics(
        fixture_text("business_register"), 6, table=table
    )
    assert diagnostics.current_grade > 0


def test_diagnostics_carry_the_two_scorer_levers(fixture_text, table):
    from reading_level.scorer import score_text

    text = fixture_text("medium_upper_elementary")
    diagnostics = build_diagnostics(text, 4, table=table)
    score = score_text(text, table=table)

    assert diagnostics.mean_sentence_length == pytest.approx(
        score.features.mean_sentence_length
    )
    assert diagnostics.mean_log_word_freq == pytest.approx(
        score.features.mean_log_word_freq
    )


def test_grade_gap_is_measured_to_the_near_band_edge(fixture_text, table):
    text = fixture_text("business_register")
    diagnostics = build_diagnostics(text, 6, table=table)
    band = bands.target_band(6)

    assert diagnostics.grade_gap == pytest.approx(
        diagnostics.current_grade - band.high
    )


def test_in_band_text_reports_no_gap(fixture_text, table):
    diagnostics = build_diagnostics(fixture_text("easy_primary"), 2, table=table)
    assert diagnostics.grade_gap == 0.0


def test_below_band_text_reports_a_negative_gap(fixture_text, table):
    diagnostics = build_diagnostics(fixture_text("easy_primary"), 9, table=table)
    band = bands.target_band(9)

    assert diagnostics.grade_gap == pytest.approx(
        diagnostics.current_grade - band.low
    )
    assert diagnostics.grade_gap < 0


@pytest.mark.parametrize("passage_type", config.PASSAGE_TYPES)
def test_a_stated_passage_type_is_carried_through(
    fixture_text, table, passage_type
):
    diagnostics = build_diagnostics(
        fixture_text("business_register"), 6, table=table,
        passage_type=passage_type,
    )

    assert diagnostics.passage_type == passage_type
    assert diagnostics.passage_type_specified is True


def test_an_omitted_passage_type_falls_back_strictly_and_says_so(
    fixture_text, table, caplog
):
    """The fallback keeps an unlabelled call safe; the warning keeps it visible.

    Nothing in this package can tell a story from a science paragraph, and
    guessing wrong toward the permissive tier would let a rewrite restructure
    away a fact. So the strict tier is assumed and the assumption is logged.
    """
    with caplog.at_level(logging.WARNING, logger="reading_level.diagnostics"):
        diagnostics = build_diagnostics(
            fixture_text("business_register"), 6, table=table
        )

    assert diagnostics.passage_type == config.FALLBACK_PASSAGE_TYPE
    assert diagnostics.passage_type == config.PASSAGE_TYPE_INFORMATIONAL
    assert diagnostics.passage_type_specified is False
    assert "no passage_type supplied" in caplog.text


def test_an_unrecognised_passage_type_raises(fixture_text, table):
    with pytest.raises(ValueError, match="passage_type"):
        build_diagnostics(
            fixture_text("business_register"), 6, table=table,
            passage_type="expository",
        )


def test_hard_words_match_what_operator_b_targets(fixture_text, table):
    """The diagnosis must describe the real operator, not a parallel copy."""
    text = fixture_text("business_register")
    diagnostics = build_diagnostics(text, 6, table=table)

    sentences = [doc.text for doc in parse_sentences(text)]
    expected = [
        token.text for _, token in substitution_targets(sentences, table, set())
    ]

    assert [word.word for word in diagnostics.hard_words] == expected[
        : config.MAX_DIAGNOSTIC_HARD_WORDS
    ]
    assert all(word.sentence for word in diagnostics.hard_words)


def test_suggestions_match_what_operator_b_would_consider(fixture_text, table):
    text = fixture_text("astronaut_mixed")
    diagnostics = build_diagnostics(text, 5, table=table)

    sentences = [doc.text for doc in parse_sentences(text)]
    tokens = {
        token.text: token
        for _, token in substitution_targets(sentences, table, set())
    }

    checked = 0
    for word in diagnostics.hard_words:
        if not word.suggested_replacements:
            continue
        expected = [
            option.word
            for option in replacement_options(tokens[word.word], table)
        ][: config.MAX_DIAGNOSTIC_SUGGESTIONS]
        assert word.suggested_replacements == expected
        checked += 1

    assert checked, "no word carried a suggestion, so nothing was verified"


def test_hard_words_are_capped_and_ordered_worst_first(fixture_text, table):
    diagnostics = build_diagnostics(
        fixture_text("hard_secondary"), 4, table=table
    )
    frequencies = [word.logfreq for word in diagnostics.hard_words]

    assert len(diagnostics.hard_words) <= config.MAX_DIAGNOSTIC_HARD_WORDS
    assert frequencies == sorted(frequencies)


def test_protected_terms_are_excluded_from_hard_words(fixture_text, table):
    text = fixture_text("hard_secondary")
    diagnostics = build_diagnostics(
        text, 4, protected_terms=["proliferation", "methane"], table=table
    )

    flagged = {word.word.lower() for word in diagnostics.hard_words}
    assert not flagged & {"proliferation", "methane"}
    assert diagnostics.protected_terms == ["proliferation", "methane"]


def test_long_sentences_are_measured_against_the_band_ceiling(
    fixture_text, table
):
    text = fixture_text("astronaut_mixed")
    diagnostics = build_diagnostics(text, 5, table=table)
    limit = bands.max_sentence_length(bands.target_band(5))
    nlp = get_nlp()

    assert diagnostics.long_sentences
    for sentence in diagnostics.long_sentences:
        actual = sum(1 for token in nlp(sentence.text) if token.is_alpha)
        assert sentence.word_count == actual
        assert sentence.exceeds_limit_by == actual - limit
        assert sentence.exceeds_limit_by > 0


def test_long_sentences_are_ordered_worst_first(fixture_text, table):
    diagnostics = build_diagnostics(
        fixture_text("hard_secondary"), 3, table=table
    )
    overage = [s.exceeds_limit_by for s in diagnostics.long_sentences]
    assert overage == sorted(overage, reverse=True)


def test_business_passage_has_no_splittable_sentence_for_grade_six(
    fixture_text, table
):
    """This fixture exists to isolate substitution; guard that property."""
    diagnostics = build_diagnostics(
        fixture_text("business_register"), 6, table=table
    )
    assert diagnostics.long_sentences == []
    assert diagnostics.hard_words


def test_prior_result_supplies_the_edit_history(fixture_text, table):
    text = fixture_text("business_register")
    result = correct_text(text, 6, table=table)

    diagnostics = build_diagnostics(
        result.text, 6, prior_result=result, table=table
    )

    assert diagnostics.deterministic_edits_applied == result.edits
    assert diagnostics.failure_reason == result.failure_reason


def test_diagnostics_without_a_prior_result_is_still_valid(fixture_text, table):
    diagnostics = build_diagnostics(
        fixture_text("business_register"), 6, table=table
    )
    assert diagnostics.deterministic_edits_applied == []
    assert diagnostics.failure_reason is None


def test_diagnostics_does_not_mutate_its_input(fixture_text, table):
    text = fixture_text("business_register")
    protected = ["counterparties"]

    build_diagnostics(text, 6, protected_terms=protected, table=table)

    assert protected == ["counterparties"]
    assert text == fixture_text("business_register")
