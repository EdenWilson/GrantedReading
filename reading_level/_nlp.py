"""Lazily loaded NLP resources shared by the scorer and the repair operators.

Private to the package. Loading is deferred until first use so that importing
`reading_level` stays cheap: Flask startup and pytest collection must not pay
for a spaCy model load.
"""

from __future__ import annotations

import threading

from . import config
from ._errors import ReadingLevelError

_nlp = None
_nlp_lock = threading.Lock()

_wordnet = None
_wordnet_lock = threading.Lock()
_wordnet_failed = False

_lesk = None
_lesk_failed = False

_encoder = None
_encoder_lock = threading.Lock()


def get_nlp():
    """Return the shared spaCy pipeline, loading it on first use."""
    global _nlp
    if _nlp is not None:
        return _nlp

    with _nlp_lock:
        if _nlp is None:
            try:
                import spacy
            except ImportError as exc:
                raise ReadingLevelError(
                    "spaCy is required for reading level analysis. "
                    "Install it with: pip install -r requirements.txt"
                ) from exc

            try:
                _nlp = spacy.load(config.SPACY_MODEL)
            except OSError as exc:
                raise ReadingLevelError(
                    f"spaCy model '{config.SPACY_MODEL}' is not installed. "
                    f"Install it with: python -m spacy download {config.SPACY_MODEL}"
                ) from exc
    return _nlp


def get_segmenter():
    """Return a sentence segmenter.

    A real segmenter is required: splitting on '.' mangles abbreviations,
    decimals, and ellipses, and sentence length is half of the score.
    """
    import pysbd

    return pysbd.Segmenter(language="en", clean=False)


def get_wordnet():
    """Return the WordNet corpus reader, or None if its data is unavailable.

    Returns None rather than raising so that substitution can be skipped and
    reported instead of taking down the whole correction run.
    """
    global _wordnet, _wordnet_failed
    if _wordnet is not None or _wordnet_failed:
        return _wordnet

    with _wordnet_lock:
        if _wordnet is None and not _wordnet_failed:
            try:
                from nltk.corpus import wordnet

                wordnet.synsets("test")
                _wordnet = wordnet
            except Exception:
                _wordnet_failed = True
    return _wordnet


def get_lesk():
    """Return NLTK's Lesk word sense disambiguator, or None if unavailable.

    Ships with NLTK, so no new dependency. Returns None rather than raising for
    the same reason as get_wordnet: substitution falls back to the dominant
    sense instead of taking down the correction run.
    """
    global _lesk, _lesk_failed
    if _lesk is not None or _lesk_failed:
        return _lesk

    with _wordnet_lock:
        if _lesk is None and not _lesk_failed:
            try:
                from nltk.wsd import lesk

                _lesk = lesk
            except Exception:
                _lesk_failed = True
    return _lesk


def get_sentence_encoder():
    """Return the sentence embedding model used for substitution filtering
    and meaning-drift detection.

    Raises rather than returning None, unlike get_wordnet and get_lesk. Those
    guard an optional improvement, so degrading quietly is right. This one
    backs a safety gate: if it cannot load, the correct outcome is a visible
    failure, not a swap or a drift check that silently passes everything.

    Loaded lazily -- it pulls in torch, which is far too slow to import at
    Flask startup. First use on a process is the slow one; after that the
    model stays in memory.
    """
    global _encoder
    if _encoder is not None:
        return _encoder

    with _encoder_lock:
        if _encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ReadingLevelError(
                    "sentence-transformers is required to verify rewritten "
                    "passages. Install it with: pip install -r requirements.txt"
                ) from exc

            try:
                _encoder = SentenceTransformer(config.DRIFT_EMBEDDING_MODEL)
            except Exception as exc:
                raise ReadingLevelError(
                    f"Could not load the embedding model "
                    f"'{config.DRIFT_EMBEDDING_MODEL}'. It downloads on first "
                    f"use and is cached afterwards, so this usually means no "
                    f"network on a cold cache."
                ) from exc
    return _encoder
