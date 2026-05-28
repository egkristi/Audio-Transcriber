"""Tests for src/diarize.py"""

import pytest

from src.diarize import (
    DiarizationSegment,
    Diarizer,
    check_hf_auth,
)


class TestCheckHfAuth:
    def test_returns_bool(self, monkeypatch):
        # Ensure no token is present so we get a predictable result
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
        result = check_hf_auth()
        assert isinstance(result, bool)


class TestDiarizationSegment:
    def test_dataclass_creation(self):
        seg = DiarizationSegment(start=0.0, end=2.0, speaker="SPEAKER_00", confidence=1.0)
        assert seg.speaker == "SPEAKER_00"
        assert seg.confidence == 1.0


class TestDiarizer:
    def test_init_defaults(self):
        d = Diarizer()
        assert d.model is None
        assert d.device == "cpu"

    def test_init_with_config(self):
        d = Diarizer({"model": "custom-model", "threshold": 0.7})
        assert d.config["model"] == "custom-model"

    def test_get_speaker_timeline(self):
        d = Diarizer()
        segments = [
            DiarizationSegment(start=0.0, end=2.0, speaker="SPEAKER_00", confidence=1.0),
            DiarizationSegment(start=2.0, end=4.0, speaker="SPEAKER_01", confidence=1.0),
            DiarizationSegment(start=4.0, end=6.0, speaker="SPEAKER_00", confidence=1.0),
        ]
        timeline = d.get_speaker_timeline(segments, audio_duration=6.0)
        assert "SPEAKER_00" in timeline
        assert "SPEAKER_01" in timeline
        assert len(timeline["SPEAKER_00"]) == 2
        assert len(timeline["SPEAKER_01"]) == 1
