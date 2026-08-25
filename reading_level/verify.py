"""Meaning-preservation check for rewritten passages.

Operators A, B, and C are safe by construction -- a word swap constrained to a
WordNet sense, or a split verified to leave two independent clauses, cannot say
something the original did not. An LLM rewrite has no such guarantee, so it
gets a check the deterministic operators never needed.

Deterministic and offline. The embedding model runs locally; nothing here calls
an LLM, and the rewrite it judges is never allowed to judge itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import config
from ._nlp import get_nlp, get_sentence_encoder
from .scorer import split_sentences


@dataclass(frozen=True)
class DriftReport:
    passage_similarity: float
    min_sentence_similarity: float
    worst_sentence_index: int | None
    protected_terms_missing: list[str] = field(default_factory=list)
    passed: bool = False
    reasons: list[str] = field(default_factory=list)
    # Not in the original sketch, but this is a hard gate's input: burying it
    # in a prose reason string would make a rejection impossible to debug.
    negation_delta: int = 0
    # Facts present in the original and gone from the rewrite. Empty when the
    # check did not run, which `facts_checked` distinguishes from "nothing was
    # dropped" -- the two mean very different things to someone reading a
    # report and deciding whether a passage needs a human.
    numbers_missing: list[str] = field(default_factory=list)
    entities_missing: list[str] = field(default_factory=list)
    facts_checked: bool = False


def _missing_protected_terms(original: str, rewritten: str, protected: list[str]):
    """Protected terms present in the original but gone from the rewrite."""
    source, target = original.lower(), rewritten.lower()
    return [
        term
        for term in protected
        if term.lower() in source and term.lower() not in target
    ]


def _count_negations(text: str) -> int:
    """Negation markers, by dependency label and by surface form.

    The parser catches "not"/"n't" as `neg`; the word list catches the ones it
    does not, such as "rarely" and "without", which flip an assertion just as
    completely.
    """
    doc = get_nlp()(text)
    return sum(
        1
        for token in doc
        if token.dep_ == "neg" or token.lower_ in config.NEGATION_MARKERS
    )


def _normalize_number(text: str) -> str:
    """Canonical form of a numeric token, so "three" and "3" are one fact."""
    cleaned = text.lower().strip().replace(",", "").rstrip(".%")
    if cleaned in config.NUMBER_WORDS:
        return str(config.NUMBER_WORDS[cleaned])
    try:
        value = float(cleaned)
    except ValueError:
        return cleaned
    return str(int(value)) if value.is_integer() else str(value)


def _numbers(doc) -> set[str]:
    return {_normalize_number(token.text) for token in doc if token.like_num}


def _named_facts(doc):
    """Named entities worth gating on, as (surface form, proper-noun tokens)."""
    facts = []
    for entity in doc.ents:
        if entity.label_ not in config.FACT_ENTITY_LABELS:
            continue
        anchors = {
            token.text.lower()
            for token in entity
            if token.pos_ in {"PROPN", "NOUN"} or token.text[:1].isupper()
        }
        facts.append((entity.text, anchors or {entity.text.lower()}))
    return facts


def _missing_facts(original: str, rewritten: str):
    """Numbers and named entities the rewrite dropped or changed.

    Entities match leniently: any one of an entity's naming tokens surviving
    counts as preserved, so "Maya Chen" -> "Maya" passes while "Jupiter" ->
    "Saturn" does not. A stricter rule would reject legitimate shortening, and
    the negation gate has already demonstrated what an over-strict hard check
    costs. Numbers match by value and are not lenient: a changed quantity is
    the failure this whole check exists for.
    """
    nlp = get_nlp()
    source, target = nlp(original), nlp(rewritten)

    target_numbers = _numbers(target)
    numbers_missing = sorted(_numbers(source) - target_numbers)

    target_text = rewritten.lower()
    entities_missing = [
        surface
        for surface, anchors in _named_facts(source)
        if not any(anchor in target_text for anchor in anchors)
    ]

    return numbers_missing, entities_missing


def _similarities(original: str, rewritten: str):
    """Passage similarity, plus the worst-matched original sentence.

    Sentences are matched best-against-best rather than by position, because a
    rewrite is expected to split and merge them. Each original sentence is
    scored against its closest counterpart in the rewrite; the weakest of those
    is what the gate reads. An average over the passage would let one sentence
    assert something new and hide behind six that did not.
    """
    encoder = get_sentence_encoder()

    original_sentences = split_sentences(original) or [original]
    rewritten_sentences = split_sentences(rewritten) or [rewritten]

    texts = [original, rewritten] + original_sentences + rewritten_sentences
    vectors = encoder.encode(texts, normalize_embeddings=True)

    passage_similarity = float(vectors[0] @ vectors[1])

    offset = 2 + len(original_sentences)
    source_vectors = vectors[2:offset]
    target_vectors = vectors[offset:]

    worst_similarity = 1.0
    worst_index = None
    for index, vector in enumerate(source_vectors):
        best = max(float(vector @ other) for other in target_vectors)
        if best < worst_similarity:
            worst_similarity, worst_index = best, index

    return passage_similarity, worst_similarity, worst_index


def check_semantic_drift(
    original: str,
    rewritten: str,
    *,
    protected_terms: list[str] | None = None,
    passage_type: str | None = None,
) -> DriftReport:
    """Judge whether `rewritten` still says what `original` said.

    Four checks, all of which must pass. Protected terms, negation, and facts
    are hard and deterministic; similarity is the graded one. They cover
    different failures: embeddings catch a rewrite that wandered onto another
    topic but are blind to one that asserts the opposite or that quietly
    changes a number, and the deterministic checks are the reverse.

    `passage_type` gates the fact check only. A narrative is allowed to drop a
    detail, so numbers and entities are not checked for one; everything else
    applies to both. Omitting it checks facts, matching the strict fallback
    that `diagnostics.build_diagnostics` takes.
    """
    protected = list(protected_terms or [])
    reasons: list[str] = []

    missing = _missing_protected_terms(original, rewritten, protected)
    if missing:
        reasons.append(f"protected terms dropped: {', '.join(missing)}")

    check_facts = (
        passage_type or config.FALLBACK_PASSAGE_TYPE
    ) in config.FACT_CHECKED_PASSAGE_TYPES
    numbers_missing: list[str] = []
    entities_missing: list[str] = []
    if check_facts:
        numbers_missing, entities_missing = _missing_facts(original, rewritten)
        if numbers_missing:
            reasons.append(
                f"numbers changed or dropped: {', '.join(numbers_missing)}; a "
                f"comprehension question may be built on one of them"
            )
        if entities_missing:
            reasons.append(
                f"named facts dropped: {', '.join(entities_missing)}"
            )

    negation_delta = _count_negations(rewritten) - _count_negations(original)
    if negation_delta > config.MAX_NEGATION_DELTA:
        reasons.append(
            f"rewrite introduced {negation_delta} negation marker(s); the "
            f"passage may now assert the opposite of the original"
        )

    passage_similarity, sentence_similarity, worst_index = _similarities(
        original, rewritten
    )
    if passage_similarity < config.MAX_PASSAGE_DRIFT:
        reasons.append(
            f"passage similarity {passage_similarity:.2f} below "
            f"{config.MAX_PASSAGE_DRIFT:.2f}"
        )
    if sentence_similarity < config.MIN_SENTENCE_SIMILARITY:
        reasons.append(
            f"sentence {worst_index} similarity {sentence_similarity:.2f} below "
            f"{config.MIN_SENTENCE_SIMILARITY:.2f}"
        )

    return DriftReport(
        passage_similarity=passage_similarity,
        min_sentence_similarity=sentence_similarity,
        worst_sentence_index=worst_index,
        protected_terms_missing=missing,
        passed=not reasons,
        reasons=reasons,
        negation_delta=negation_delta,
        numbers_missing=numbers_missing,
        entities_missing=entities_missing,
        facts_checked=check_facts,
    )
