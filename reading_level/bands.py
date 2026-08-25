"""Grade band definitions and conversions.

Bands are ranges, not points. Nothing in this package targets an exact grade
number; a passage is either inside a band or outside it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import config


@dataclass(frozen=True)
class Band:
    label: str
    display: str
    low: float
    high: float

    @property
    def center(self) -> float:
        return (self.low + self.high) / 2.0


def _build_bands() -> tuple[Band, ...]:
    return tuple(
        Band(
            label=label,
            display=display,
            low=grade - config.BAND_HALF_WIDTH,
            high=grade + config.BAND_HALF_WIDTH,
        )
        for label, display, grade in config.BAND_DEFINITIONS
    )


BANDS: tuple[Band, ...] = _build_bands()

_BY_LABEL = {band.label: band for band in BANDS}

_WORD_ALIASES = {
    "k": "grade_k",
    "kindergarten": "grade_k",
    "kg": "grade_k",
}


def band_for_grade(grade: float) -> Band:
    """Return the band containing a continuous grade estimate.

    Estimates outside the defined range clamp to the first or last band.
    """
    if grade <= BANDS[0].high:
        return BANDS[0]
    for band in BANDS:
        if band.low <= grade <= band.high:
            return band
    return BANDS[-1]


def target_band(grade_level: int | str) -> Band:
    """Resolve a caller-supplied grade level to a band.

    Accepts integers, numeric strings, ordinals such as "3rd", and the
    kindergarten aliases "K"/"kindergarten".
    """
    if isinstance(grade_level, bool):
        raise ValueError(f"Invalid grade level: {grade_level!r}")

    if isinstance(grade_level, (int, float)):
        numeric = float(grade_level)
    else:
        raw = str(grade_level).strip().lower()
        if not raw:
            raise ValueError("Grade level is required")

        alias = _WORD_ALIASES.get(raw)
        if alias:
            return _BY_LABEL[alias]

        if raw in _BY_LABEL:
            return _BY_LABEL[raw]

        match = re.match(r"^(\d+)", raw)
        if not match:
            raise ValueError(f"Invalid grade level: {grade_level!r}")
        numeric = float(match.group(1))

    for band in BANDS:
        if abs(band.center - numeric) < 1e-9:
            return band

    raise ValueError(f"Grade level out of supported range: {grade_level!r}")


def in_band(score, band: Band) -> bool:
    """True when a score's estimated grade falls inside the band."""
    return band.low <= score.estimated_grade <= band.high


def max_sentence_length(band: Band) -> int:
    """Sentence-length ceiling appropriate for a band."""
    limit = (
        config.MAX_SENTENCE_LENGTH_INTERCEPT
        + config.MAX_SENTENCE_LENGTH_SLOPE * band.center
    )
    return int(round(limit))


def widen(band: Band, tolerance: float) -> Band:
    """Return a copy of the band widened by `tolerance` grade levels each way.

    Used for question stems, which are too short to measure reliably.
    """
    return Band(
        label=band.label,
        display=band.display,
        low=band.low - tolerance,
        high=band.high + tolerance,
    )
