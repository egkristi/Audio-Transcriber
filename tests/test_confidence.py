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
