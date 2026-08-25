"""Reading difficulty measurement.

The estimate is driven by the two features that dominate every mainstream
readability metric: semantic difficulty (how common the content words are) and
syntactic difficulty (how long the sentences are).

Flesch-Kincaid and a Dale-Chall-style score are computed alongside as a cheap
cross-check and fallback. Coefficients live in config.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config
from ._nlp import get_nlp, get_segmenter
from .frequency import FrequencyTable, get_default_table

_VOWELS = frozenset("aeiouy")

# Tokens with these coarse tags never count as content words.
_NON_CONTENT_POS = frozenset({"PUNCT", "SYM", "SPACE", "NUM", "X", "PROPN"})

# Tags that mark a subordinate or relative clause.
_CLAUSE_MARKER_TAGS = frozenset({"WDT", "WP", "WP$", "WRB"})


@dataclass(frozen=True)
class TextFeatures:
    mean_log_word_freq: float
    mean_sentence_length: float
    sentence_count: int
    word_count: int
    hard_word_ratio: float
    mean_syllables_per_word: float
    max_sentence_length: int
    clause_density: float
    flesch_kincaid_grade: float
    dale_chall_grade: float


@dataclass(frozen=True)
class ReadingLevelScore:
    estimated_grade: float
    features: TextFeatures
    confidence: str
    flags: list[str]


def count_syllables(word: str) -> int:
    """Heuristic syllable count.

    Vowel-group counting with a silent-e correction. Approximate by design:
    exact syllabification needs a pronunciation dictionary, and the error is
    small relative to the uncalibrated coefficients it feeds.
    """
    word = word.lower().strip()
    if not word:
        return 0

    count = 0
    previous_was_vowel = False
    for char in word:
        is_vowel = char in _VOWELS
        if is_vowel and not previous_was_vowel:
            count += 1
        previous_was_vowel = is_vowel

    if word.endswith("e") and count > 1 and not word.endswith(("le", "ee", "ye")):
        count -= 1

    return max(1, count)


def split_sentences(text: str) -> list[str]:
    """Segment text into sentences using a real segmenter."""
    if not text or not text.strip():
        return []
    segments = get_segmenter().segment(text)
    return [segment.strip() for segment in segments if segment.strip()]


def parse_sentences(text: str):
    """Return one spaCy Doc per sentence.

    Segmentation is done by pysbd, then each sentence is parsed individually so
    that character offsets within a sentence stay local to it. Shared with the
    repair operators.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []
    return list(get_nlp().pipe(sentences))


def _is_content_word(token, table: FrequencyTable) -> bool:
    if not token.is_alpha or token.is_stop:
        return False
    if token.pos_ in _NON_CONTENT_POS:
        return False
    # Proper nouns are exempt: a name is not a vocabulary burden.
    return not table.is_proper(token.text)


def _dale_chall_grade(raw_score: float) -> float:
    for upper_bound, grade in config.DALE_CHALL_GRADE_TABLE:
        if raw_score <= upper_bound:
            return grade
    return config.DALE_CHALL_MAX_GRADE


def _empty_score() -> ReadingLevelScore:
    features = TextFeatures(
        mean_log_word_freq=0.0,
        mean_sentence_length=0.0,
        sentence_count=0,
        word_count=0,
        hard_word_ratio=0.0,
        mean_syllables_per_word=0.0,
        max_sentence_length=0,
        clause_density=0.0,
        flesch_kincaid_grade=0.0,
        dale_chall_grade=0.0,
    )
    return ReadingLevelScore(
        estimated_grade=config.GRADE_FLOOR,
        features=features,
        confidence="low",
        flags=["empty_text"],
    )


def score_text(text: str, *, table: FrequencyTable | None = None) -> ReadingLevelScore:
    """Estimate the reading difficulty of `text` as a continuous grade level.

    Pure with respect to its arguments: no network access and no mutation of
    shared state. The returned grade is an estimate produced by uncalibrated
    coefficients; treat ordering as meaningful and absolute values as not.
    """
    table = table if table is not None else get_default_table()

    docs = parse_sentences(text)
    if not docs:
        return _empty_score()

    words = []
    content_words = []
    sentence_lengths = []
    clause_markers = 0

    for doc in docs:
        doc_words = [token for token in doc if token.is_alpha]
        sentence_lengths.append(len(doc_words))
        words.extend(doc_words)
        content_words.extend(
            token for token in doc_words if _is_content_word(token, table)
        )
        clause_markers += sum(
            1
            for token in doc
            if token.pos_ == "SCONJ" or token.tag_ in _CLAUSE_MARKER_TAGS
        )

    word_count = len(words)
    sentence_count = len(docs)

    if word_count == 0:
        return _empty_score()

    flags: list[str] = []

    frequency_source = content_words or words
    if not content_words:
        flags.append("no_content_words")

    mean_log_word_freq = sum(
        table.logfreq(token.text) for token in frequency_source
    ) / len(frequency_source)

    hard_words = sum(
        1
        for token in frequency_source
        if table.logfreq(token.text) <= config.HARD_WORD_LOGFREQ_THRESHOLD
    )
    hard_word_ratio = hard_words / len(frequency_source)

    syllables = [count_syllables(token.text) for token in words]
    mean_syllables_per_word = sum(syllables) / word_count

    mean_sentence_length = word_count / sentence_count
    max_sentence_length = max(sentence_lengths)
    clause_density = clause_markers / sentence_count

    flesch_kincaid_grade = (
        config.FK_COEF_SENTENCE_LENGTH * mean_sentence_length
        + config.FK_COEF_SYLLABLES * mean_syllables_per_word
        + config.FK_INTERCEPT
    )

    unfamiliar = sum(
        1
        for token in words
        if not token.is_stop
        and not table.is_proper(token.text)
        and table.percentile(token.text) < config.FAMILIAR_WORD_PERCENTILE
    )
    percent_unfamiliar = 100.0 * unfamiliar / word_count
    dale_chall_raw = (
        config.DALE_CHALL_COEF_DIFFICULT * percent_unfamiliar
        + config.DALE_CHALL_COEF_SENTENCE_LENGTH * mean_sentence_length
    )
    if percent_unfamiliar > config.DALE_CHALL_PENALTY_THRESHOLD:
        dale_chall_raw += config.DALE_CHALL_PENALTY

    features = TextFeatures(
        mean_log_word_freq=mean_log_word_freq,
        mean_sentence_length=mean_sentence_length,
        sentence_count=sentence_count,
        word_count=word_count,
        hard_word_ratio=hard_word_ratio,
        mean_syllables_per_word=mean_syllables_per_word,
        max_sentence_length=max_sentence_length,
        clause_density=clause_density,
        flesch_kincaid_grade=flesch_kincaid_grade,
        dale_chall_grade=_dale_chall_grade(dale_chall_raw),
    )

    estimated_grade = (
        config.GRADE_INTERCEPT
        + config.GRADE_COEF_WORD_FREQ * mean_log_word_freq
        + config.GRADE_COEF_SENTENCE_LENGTH * mean_sentence_length
    )
    estimated_grade = min(
        max(estimated_grade, config.GRADE_FLOOR), config.GRADE_CEILING
    )

    if word_count < config.MIN_RELIABLE_WORDS:
        flags.append("short_text")
    if sentence_count == 1:
        flags.append("single_sentence")

    confidence = "low" if word_count < config.MIN_RELIABLE_WORDS else "high"

    return ReadingLevelScore(
        estimated_grade=estimated_grade,
        features=features,
        confidence=confidence,
        flags=flags,
    )
