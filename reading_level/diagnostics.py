"""A structured description of what is specifically wrong with a passage.

This is what gets handed to an LLM rewrite pass instead of "make this easier".
Every field is assembled from machinery the deterministic operators already
use, so the rewrite is steered by the same constraints that will judge it.

Pure and offline, like everything else in this package: no LLM, no network, no
mutation of shared state. The rewrite that consumes this lives in ai.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from . import bands, config, repair
from ._nlp import get_nlp
from .frequency import FrequencyTable, get_default_table
from .repair import CorrectionResult, RepairEdit
from .scorer import parse_sentences, score_text

logger = logging.getLogger(__name__)

PassageType = Literal["narrative", "informational"]


@dataclass(frozen=True)
class WordDiagnostic:
    word: str
    sentence_index: int
    logfreq: float
    # The sentence the word appears in, so a rewrite can replace it in place
    # rather than guessing from the isolated token.
    sentence: str = ""
    # WordNet gloss for the sense the word is used in, when Lesk resolved it.
    sense_gloss: str | None = None
    # Pre-vetted by Operator B: frequency-safe, sense-matched, correct POS.
    suggested_replacements: list[str] = field(default_factory=list)
    sense_disambiguated: bool = False


@dataclass(frozen=True)
class SentenceDiagnostic:
    index: int
    text: str
    word_count: int
    # Words over the band ceiling; 0 when the sentence is within limit.
    exceeds_limit_by: int


@dataclass(frozen=True)
class PassageDiagnostics:
    current_grade: float
    target_band: bands.Band
    grade_gap: float
    # The two numbers the scorer actually reads. A rewrite that only splits
    # "long" sentences can still miss the band if the average stays high.
    mean_sentence_length: float = 0.0
    mean_log_word_freq: float = 0.0
    hard_words: list[WordDiagnostic] = field(default_factory=list)
    long_sentences: list[SentenceDiagnostic] = field(default_factory=list)
    protected_terms: list[str] = field(default_factory=list)
    deterministic_edits_applied: list[RepairEdit] = field(default_factory=list)
    failure_reason: str | None = None
    # Caller-supplied, never inferred: how much structural freedom a rewrite of
    # this passage may take. Guessing it from the text would put the decision
    # about whether a fact is load-bearing in the hands of the component least
    # able to know.
    passage_type: PassageType = config.FALLBACK_PASSAGE_TYPE
    # False when the caller did not say, and the strict fallback was used. Kept
    # separate from `passage_type` so a rewrite prompt built on an assumption is
    # distinguishable in the logs from one built on a stated fact.
    passage_type_specified: bool = False

    @property
    def sentence_limit(self) -> int:
        return bands.max_sentence_length(self.target_band)


def _word_diagnostics(sentences, table: FrequencyTable, protected: set[str]):
    diagnostics = []
    for sentence_index, token in repair.substitution_targets(
        sentences, table, protected
    )[: config.MAX_DIAGNOSTIC_HARD_WORDS]:
        sense = repair.resolved_sense(token)
        options = repair.replacement_options(token, table)
        diagnostics.append(
            WordDiagnostic(
                word=token.text,
                sentence_index=sentence_index,
                logfreq=table.logfreq(token.text),
                sentence=sentences[sentence_index],
                sense_gloss=sense.definition() if sense is not None else None,
                suggested_replacements=[
                    option.word
                    for option in options[: config.MAX_DIAGNOSTIC_SUGGESTIONS]
                ],
                sense_disambiguated=sense is not None,
            )
        )
    return diagnostics


def _sentence_diagnostics(sentences, band: bands.Band):
    """Over-length sentences, worst first.

    Word counts match what Operator A measures, so the numbers in the prompt
    agree with the numbers the split operator acted on.
    """
    limit = bands.max_sentence_length(band)
    nlp = get_nlp()

    diagnostics = []
    for index, sentence in enumerate(sentences):
        word_count = sum(1 for token in nlp(sentence) if token.is_alpha)
        if word_count <= limit:
            continue
        diagnostics.append(
            SentenceDiagnostic(
                index=index,
                text=sentence,
                word_count=word_count,
                exceeds_limit_by=word_count - limit,
            )
        )

    diagnostics.sort(key=lambda item: item.exceeds_limit_by, reverse=True)
    return diagnostics


def _resolve_passage_type(passage_type: str | None) -> tuple[PassageType, bool]:
    """Validate a caller's passage type, or fall back to the strict tier loudly.

    An unknown value raises rather than falling back: a typo like "story" would
    otherwise silently buy the caller the strict tier while looking correct at
    the call site.
    """
    if passage_type is None:
        logger.warning(
            "no passage_type supplied; assuming %r, the stricter tier. The "
            "rewrite will be told to preserve every fact, which is wrong for a "
            "story and will cost freedom it should have had. Pass one of %s.",
            config.FALLBACK_PASSAGE_TYPE,
            list(config.PASSAGE_TYPES),
        )
        return config.FALLBACK_PASSAGE_TYPE, False

    if passage_type not in config.PASSAGE_TYPES:
        raise ValueError(
            f"passage_type must be one of {list(config.PASSAGE_TYPES)}, "
            f"got {passage_type!r}"
        )
    return passage_type, True


def build_diagnostics(
    text: str,
    target_grade: int | str,
    *,
    protected_terms: list[str] | None = None,
    prior_result: CorrectionResult | None = None,
    table: FrequencyTable | None = None,
    passage_type: str | None = None,
) -> PassageDiagnostics:
    """Describe what is wrong with `text` relative to `target_grade`.

    `text` should be the passage as it stands after deterministic repair, and
    `prior_result` the run that produced it: the edits already applied are part
    of the diagnosis, because they say which levers are spent.

    `passage_type` decides how much a rewrite may restructure. Omitting it is
    allowed but logs a warning and selects the stricter tier.
    """
    resolved_type, type_specified = _resolve_passage_type(passage_type)
    table = table if table is not None else get_default_table()
    band = bands.target_band(target_grade)
    protected = list(protected_terms or [])
    protected_lower = {term.lower() for term in protected}

    score = score_text(text, table=table)
    sentences = [doc.text for doc in parse_sentences(text)]

    # Signed distance to the near edge. Positive is too hard, negative is too
    # easy, zero is in band. The rewrite prompt needs the direction, not only
    # the overshoot.
    if score.estimated_grade > band.high:
        gap = score.estimated_grade - band.high
    elif score.estimated_grade < band.low:
        gap = score.estimated_grade - band.low
    else:
        gap = 0.0

    return PassageDiagnostics(
        current_grade=score.estimated_grade,
        target_band=band,
        grade_gap=gap,
        mean_sentence_length=score.features.mean_sentence_length,
        mean_log_word_freq=score.features.mean_log_word_freq,
        hard_words=_word_diagnostics(sentences, table, protected_lower),
        long_sentences=_sentence_diagnostics(sentences, band),
        protected_terms=protected,
        deterministic_edits_applied=(
            list(prior_result.edits) if prior_result is not None else []
        ),
        failure_reason=prior_result.failure_reason if prior_result is not None else None,
        passage_type=resolved_type,
        passage_type_specified=type_specified,
    )
