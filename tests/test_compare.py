"""Tests for src/compare.py"""

import pytest

from src.compare import (
    ComparisonResult,
    TranscriptionComparer,
)
from src.transcribe import TranscriptionSegment


class TestCalculateSimilarity:
    def test_identical_texts(self):
        comparer = TranscriptionComparer()
        assert comparer.calculate_similarity("hello world", "hello world") == 1.0

    def test_completely_different(self):
        comparer = TranscriptionComparer()
        score = comparer.calculate_similarity("abc", "xyz")
        assert score < 0.5

    def test_empty_strings(self):
        comparer = TranscriptionComparer()
        assert comparer.calculate_similarity("", "") == 1.0
        assert comparer.calculate_similarity("abc", "") == 0.0


class TestTranscriptionComparer:
    def test_align_segments_basic(self):
        primary = [
            TranscriptionSegment(id=0, start=0.0, end=2.0, text="hello"),
            TranscriptionSegment(id=1, start=3.0, end=5.0, text="world"),
        ]
        secondary = [
            TranscriptionSegment(id=0, start=0.5, end=2.5, text="hello"),
            TranscriptionSegment(id=1, start=6.0, end=8.0, text="world"),
        ]
        comparer = TranscriptionComparer()
        aligned = comparer.align_segments(primary, secondary)
        assert len(aligned) == 2
        assert aligned[0][1] is not None  # first segment should match
        assert aligned[1][1] is None  # second segment has low overlap

    def test_compare_segments_identical(self):
        seg = TranscriptionSegment(id=0, start=0.0, end=2.0, text="hello", confidence=0.95)
        comparer = TranscriptionComparer()
        result = comparer.compare_segments(seg, seg)
        assert result.has_disagreement is False
        assert result.priority == "low"

    def test_compare_segments_missing_secondary(self):
        seg = TranscriptionSegment(id=0, start=0.0, end=2.0, text="hello", confidence=0.95)
        comparer = TranscriptionComparer()
        result = comparer.compare_segments(seg, None)
        assert result.has_disagreement is True
        assert result.priority == "high"
        assert "missing_secondary" in result.flags

    def test_compare_segments_low_confidence(self):
        seg = TranscriptionSegment(id=0, start=0.0, end=2.0, text="hello", confidence=0.5)
        comparer = TranscriptionComparer()
        result = comparer.compare_segments(seg, seg)
        assert result.has_low_confidence is True
        assert "low_confidence" in result.flags


class TestComparisonResult:
    def test_dataclass_creation(self):
        result = ComparisonResult(
            segment_id=1,
            start=0.0,
            end=2.0,
            text_primary="hello",
            text_secondary="hallo",
            primary_confidence=0.9,
            secondary_confidence=0.8,
            similarity_score=0.8,
            has_disagreement=True,
            has_low_confidence=False,
            priority="high",
            flags=["disagreement"],
        )
        assert result.segment_id == 1
        assert result.priority == "high"
