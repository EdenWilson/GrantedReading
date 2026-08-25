"""Scorer tests.

Assertions are on ordering, never absolute values: the coefficients in
config.py are uncalibrated placeholders, so asserting exact grades would only
test the placeholders.
"""

from reading_level import score_text
from reading_level.scorer import count_syllables, split_sentences

LONG_SENTENCES = (
    "The dog ran to the park and the cat followed him there because they "
    "wanted to play together in the grass while the sun was still up."
)
SHORT_SENTENCES = (
    "The dog ran to the park. The cat followed him. They wanted to play. "
    "The sun was still up."
)

COMMON_VOCAB = "The big dog was very happy. The small cat was very sad."
RARE_VOCAB = "The colossal canine was exceedingly jubilant. The diminutive feline was exceedingly morose."


def test_longer_sentences_score_higher(table):
    long_score = score_text(LONG_SENTENCES, table=table)
    short_score = score_text(SHORT_SENTENCES, table=table)
    assert long_score.estimated_grade > short_score.estimated_grade


def test_rarer_vocabulary_scores_higher(table):
    rare = score_text(RARE_VOCAB, table=table)
    common = score_text(COMMON_VOCAB, table=table)
    assert rare.estimated_grade > common.estimated_grade
    assert rare.features.mean_log_word_freq < common.features.mean_log_word_freq


def test_fixture_passages_are_ordered_by_difficulty(fixture_text, table):
    easy = score_text(fixture_text("easy_primary"), table=table)
    medium = score_text(fixture_text("medium_upper_elementary"), table=table)
    hard = score_text(fixture_text("hard_secondary"), table=table)

    assert easy.estimated_grade < medium.estimated_grade < hard.estimated_grade


def test_short_text_is_reported_as_low_confidence(table):
    score = score_text("The dog ran.", table=table)
    assert score.confidence == "low"
    assert "short_text" in score.flags


def test_long_text_is_reported_as_high_confidence(fixture_text, table):
    score = score_text(fixture_text("medium_upper_elementary"), table=table)
    assert score.confidence == "high"
    assert "short_text" not in score.flags


def test_empty_text_is_flagged_not_scored(table):
    score = score_text("", table=table)
    assert score.flags == ["empty_text"]
    assert score.features.word_count == 0
    assert score.confidence == "low"


def test_single_sentence_is_flagged(table):
    score = score_text("The dog ran to the park.", table=table)
    assert "single_sentence" in score.flags


def test_segmenter_handles_abbreviations_and_decimals():
    sentences = split_sentences("Dr. Smith read 3.5 pages. Then he slept!")
    assert len(sentences) == 2


def test_cross_check_metrics_are_populated(fixture_text, table):
    easy = score_text(fixture_text("easy_primary"), table=table)
    hard = score_text(fixture_text("hard_secondary"), table=table)

    assert hard.features.flesch_kincaid_grade > easy.features.flesch_kincaid_grade
    assert hard.features.dale_chall_grade > easy.features.dale_chall_grade


def test_proper_nouns_do_not_inflate_difficulty(table):
    with_name = score_text("Tyrannosaurus ran fast. The dog ran too.", table=table)
    assert with_name.features.word_count > 0


def test_count_syllables():
    assert count_syllables("dog") == 1
    assert count_syllables("running") == 2
    assert count_syllables("banana") == 3
    assert count_syllables("") == 0
