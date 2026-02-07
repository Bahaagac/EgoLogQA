"""Backward-compatible reader import path."""

from egologqa.io.reader import InMemoryMessageSource, MCapMessageSource, MessageSource

__all__ = ["MessageSource", "InMemoryMessageSource", "MCapMessageSource"]
