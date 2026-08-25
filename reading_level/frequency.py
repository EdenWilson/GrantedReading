"""Word frequency lookup.

The table is built offline by scripts/build_frequency_table.py and shipped as
a gzipped JSON artifact. At runtime it is loaded once, lazily, and treated as
read-only.
"""

from __future__ import annotations

import bisect
import gzip
import json
import threading
from pathlib import Path

from . import config
from ._errors import ReadingLevelError


class FrequencyTable:
    """O(1) lookup of how common a word is.

    Loading is deferred to first use; the instance is safe to share across
    threads because the data is read-only once loaded.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = Path(path) if path is not None else config.FREQUENCY_TABLE_PATH
        self._entries: dict[str, dict] | None = None
        self._meta: dict = {}
        self._sorted_logfreqs: list[float] = []
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def meta(self) -> dict:
        self._ensure_loaded()
        return dict(self._meta)

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._entries)

    def _ensure_loaded(self) -> None:
        if self._entries is not None:
            return

        with self._lock:
            if self._entries is not None:
                return

            if not self._path.exists():
                raise ReadingLevelError(
                    f"Frequency table not found at {self._path}. "
                    "Build it with: python -m reading_level.scripts.build_frequency_table "
                    "--source wordfreq"
                )

            with gzip.open(self._path, "rt", encoding="utf-8") as handle:
                raw = json.load(handle)

            self._meta = raw.pop("_meta", {})
            self._entries = raw
            self._sorted_logfreqs = sorted(
                entry["logfreq"] for entry in self._entries.values()
            )

    def logfreq(self, word: str) -> float:
        """Log10 frequency per million. Returns OOV_LOGFREQ for unknown words."""
        self._ensure_loaded()
        entry = self._entries.get(word.lower())
        if entry is None:
            return config.OOV_LOGFREQ
        return entry["logfreq"]

    def is_proper(self, word: str) -> bool:
        """True when the word appears predominantly capitalised in the corpus.

        Proper nouns are exempt from difficulty penalties: a student's own name
        or "Tyrannosaurus" is not a vocabulary burden in the same way an
        unfamiliar common word is.
        """
        self._ensure_loaded()
        entry = self._entries.get(word.lower())
        if entry is None:
            return False
        return bool(entry.get("proper", False))

    def contains(self, word: str) -> bool:
        self._ensure_loaded()
        return word.lower() in self._entries

    def percentile(self, word: str) -> float:
        """0-100, where 0 is rarest. Used to rank repair candidates."""
        self._ensure_loaded()
        if not self._sorted_logfreqs:
            return 0.0
        value = self.logfreq(word)
        rank = bisect.bisect_left(self._sorted_logfreqs, value)
        return 100.0 * rank / len(self._sorted_logfreqs)


_default_table: FrequencyTable | None = None
_default_lock = threading.Lock()


def get_default_table() -> FrequencyTable:
    """Return the process-wide shared table."""
    global _default_table
    if _default_table is not None:
        return _default_table

    with _default_lock:
        if _default_table is None:
            _default_table = FrequencyTable()
    return _default_table


def reset_default_table() -> None:
    """Drop the cached table. Intended for tests."""
    global _default_table
    with _default_lock:
        _default_table = None
