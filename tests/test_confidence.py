"""Tests for src/confidence.py"""

import pytest

from src.confidence import (
    ConfidenceExtractor,
    SegmentConfidence,
    extract_confidence_signals,
)


class TestSegmentConfidence:
    def test_dataclass_creation(self):
        sc = SegmentConfidence(
            segment_id=0,
            start=0.0,
            end=2.0,
            text="hello world",
            avg_logprob=-0.3,
            alignment_score=0.8,
        )
        assert sc.segment_id == 0
        assert sc.priority_score == 0.0  # default
        assert sc.flags == []  # default


class TestConfidenceExtractor:
    def test_extract_from_whisperx_basic(self):
        segments = [
            {"start": 0.0, "end": 2.0, "text": "hello", "avg_logprob": -0.3},
            {"start": 2.0, "end": 4.0, "text": "world", "avg_logprob": -0.8},
        ]
        extractor = ConfidenceExtractor()
        results = extractor.extract_from_whisperx(segments)
        assert len(results) == 2
        assert results[0].avg_logprob == -0.3
        assert results[1].avg_logprob == -0.8

    def test_compute_priority_ranks_segments(self):
        segments = [
            SegmentConfidence(
                segment_id=0, start=0.0, end=2.0, text="hello",
                avg_logprob=-0.3, alignment_score=0.9, no_speech_prob=0.1
            ),
            SegmentConfidence(
                segment_id=1, start=2.0, end=4.0, text="world",
                avg_logprob=-0.9, alignment_score=0.4, no_speech_prob=0.6
            ),
        ]
        extractor = ConfidenceExtractor()
        results = extractor.compute_priority(segments)
        # The second segment should have higher priority (worse scores)
        assert results[0].segment_id == 1  # highest priority first
        assert results[0].priority_score > results[1].priority_score
        assert results[0].priority_rank == 1
        assert results[1].priority_rank == 2

    def test_compute_priority_flags(self):
        segments = [
            SegmentConfidence(
                segment_id=0, start=0.0, end=2.0, text="hello",
                avg_logprob=-0.9, alignment_score=0.4, no_speech_prob=0.8,
                compression_ratio=3.0, temperature=0.5
            ),
        ]
        extractor = ConfidenceExtractor()
        results = extractor.compute_priority(segments)
        flags = results[0].flags
        assert "low_logprob" in flags
        assert "low_alignment" in flags
        assert "high_no_speech_prob" in flags
        assert "high_compression" in flags
        assert "temperature_fallback" in flags

    def test_export_review_list(self, temp_dir):
        segments = [
            SegmentConfidence(
                segment_id=1, start=2.0, end=4.0, text="world",
                priority_score=0.8, priority_rank=1, flags=["low_alignment"]
            ),
            SegmentConfidence(
                segment_id=0, start=0.0, end=2.0, text="hello",
                priority_score=0.3, priority_rank=2, flags=[]
            ),
        ]
        extractor = ConfidenceExtractor()
        output = temp_dir / "review.txt"
        extractor.export_review_list(segments, output)
        assert output.exists()
        content = output.read_text()
        assert "PRIORITIZED REVIEW LIST" in content
        assert "Rank 1" in content
        assert "low_alignment" in content


class TestExtractConfidenceSignals:
    def test_empty_segments(self):
        results = extract_confidence_signals([])
        assert results == []

    def test_with_comparison_results(self):
        segments = [
            {"start": 0.0, "end": 2.0, "text": "hello", "avg_logprob": -0.3},
        ]
        comparison = [
            {"segment_id": 0, "similarity_score": 0.7},
        ]
        results = extract_confidence_signals(segments, comparison_results=comparison)
        assert len(results) == 1
        assert round(results[0].model_disagreement, 10) == 0.3  # 1 - 0.7


class TestHardRules:
    """Tests for hard-rules that flag high-risk content regardless of decoder signals."""

    def test_contains_numbers_flagged(self):
        """Segments with digits should always be flagged."""
        segments = [
            SegmentConfidence(
                segment_id=0, start=0.0, end=2.0, text="pris er 1500 kroner",
            ),
        ]
        extractor = ConfidenceExtractor()
        results = extractor.compute_priority(segments)
        flags = results[0].flags
        assert any(f.startswith("contains_numbers") for f in flags)
        assert results[0].priority_score > 0.3  # boosted by digit rule

    def test_multiple_numbers_higher_priority(self):
        """More digits = higher priority boost."""
        seg1 = SegmentConfidence(segment_id=0, start=0.0, end=2.0, text="tallet 5")
        seg2 = SegmentConfidence(segment_id=1, start=2.0, end=4.0, text="tall 15 20 100")
        extractor = ConfidenceExtractor()
        results = extractor.compute_priority([seg1, seg2])
        # seg2 has more digits, should rank higher
        assert results[0].segment_id == 1

    def test_capitalized_oov_flagged(self):
        """Capitalized words not at sentence start should be flagged as proper nouns."""
        segments = [
            SegmentConfidence(
                segment_id=0, start=0.0, end=2.0, text="jeg snakket med Ola i går",
            ),
        ]
        extractor = ConfidenceExtractor()
        results = extractor.compute_priority(segments)
        flags = results[0].flags
        assert any(f.startswith("possible_proper_noun") for f in flags)

    def test_sentence_start_capitalized_not_flagged(self):
        """Capitalized word at sentence start is not a proper noun flag."""
        segments = [
            SegmentConfidence(
                segment_id=0, start=0.0, end=2.0, text="Ola kom i går",
            ),
        ]
        extractor = ConfidenceExtractor()
        results = extractor.compute_priority(segments)
        flags = results[0].flags
        assert not any(f.startswith("possible_proper_noun") for f in flags)

    def test_repeated_words_flagged(self):
        """3+ identical consecutive words = hallucination flag."""
        segments = [
            SegmentConfidence(
                segment_id=0, start=0.0, end=2.0, text="dette er er er veldig",
            ),
        ]
        extractor = ConfidenceExtractor()
        results = extractor.compute_priority(segments)
        assert "repeated_words" in results[0].flags

    def test_english_words_flagged(self):
        """Common English words in Norwegian text = language confusion."""
        segments = [
            SegmentConfidence(
                segment_id=0, start=0.0, end=2.0, text="jeg the and you vet",
            ),
        ]
        extractor = ConfidenceExtractor()
        results = extractor.compute_priority(segments)
        assert any(f.startswith("english_words") for f in results[0].flags)

    def test_very_short_segment_flagged(self):
        """Segments under 2 seconds should be flagged."""
        segments = [
            SegmentConfidence(
                segment_id=0, start=0.0, end=1.5, text="hei",
            ),
        ]
        extractor = ConfidenceExtractor()
        results = extractor.compute_priority(segments)
        assert "very_short_segment" in results[0].flags

    def test_all_caps_flagged(self):
        """All-caps tokens should be flagged as acronyms/abbreviations."""
        segments = [
            SegmentConfidence(
                segment_id=0, start=0.0, end=2.0, text="møtet er i NAV og KS",
            ),
        ]
        extractor = ConfidenceExtractor()
        results = extractor.compute_priority(segments)
        assert any(f.startswith("all_caps") for f in results[0].flags)

    def test_incomplete_ending_flagged(self):
        """Segment ending with hyphen or ellipsis = likely truncation."""
        segments = [
            SegmentConfidence(
                segment_id=0, start=0.0, end=2.0, text="han sa at han kom-",
            ),
        ]
        extractor = ConfidenceExtractor()
        results = extractor.compute_priority(segments)
        assert "incomplete_ending" in results[0].flags

    def test_lowercase_start_flagged(self):
        """All-lowercase segment start (not continuation) should be flagged."""
        segments = [
            SegmentConfidence(
                segment_id=0, start=0.0, end=2.0, text="dette er en setning",
            ),
        ]
        extractor = ConfidenceExtractor()
        results = extractor.compute_priority(segments)
        assert "lowercase_start" in results[0].flags

    def test_continuation_not_flagged(self):
        """Continuation words (og, men, så, ja, nei) should not trigger lowercase flag."""
        segments = [
            SegmentConfidence(
                segment_id=0, start=0.0, end=2.0, text="og så gikk han",
            ),
        ]
        extractor = ConfidenceExtractor()
        results = extractor.compute_priority(segments)
        assert "lowercase_start" not in results[0].flags
