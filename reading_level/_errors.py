"""Exception types for the reading_level package."""


class ReadingLevelError(Exception):
    """Raised when text cannot be brought into its target grade band.

    This is a content-quality failure, not a server fault. Callers should
    surface it to the user rather than treating it as an internal error.
    """

    def __init__(self, message, *, reason=None, score=None, band=None):
        super().__init__(message)
        self.reason = reason
        self.score = score
        self.band = band
