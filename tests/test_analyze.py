"""Tests for src/analyze.py"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.analyze import (
    AudioMetadata,
    detect_bandwidth,
    detect_stereo_separation,
    get_ffprobe_info,
)


class TestDetectBandwidth:
    def test_narrowband(self):
        audio = np.zeros(8000)
        assert detect_bandwidth(8000, audio) == "narrowband"

    def test_wideband(self):
        audio = np.zeros(16000)
        assert detect_bandwidth(16000, audio) == "wideband"

    def test_fullband(self):
        audio = np.zeros(48000)
        assert detect_bandwidth(48000, audio) == "fullband"


class TestDetectStereoSeparation:
    def test_mono_returns_false(self):
        audio = np.zeros(16000)
        assert detect_stereo_separation(audio, 16000) is False

    def test_identical_channels_returns_false(self):
        channel = np.ones(16000)
        stereo = np.stack([channel, channel])
        assert detect_stereo_separation(stereo, 16000) == False

    def test_different_channels_returns_true(self):
        t = np.linspace(0, 1, 16000)
        left = np.sin(2 * np.pi * 440 * t)
        right = np.sin(2 * np.pi * 440 * t + np.pi)  # phase-inverted
        stereo = np.stack([left, right])
        assert detect_stereo_separation(stereo, 16000) == True


class TestAudioMetadata:
    def test_dataclass_creation(self):
        metadata = AudioMetadata(
            file_path="/tmp/test.wav",
            file_name="test.wav",
            file_size_mb=1.5,
            duration_seconds=10.0,
            sample_rate=16000,
            channels=1,
            bit_rate_kbps=128,
            codec="pcm_s16le",
            bandwidth_type="wideband",
            has_stereo_separation=False,
            language="no",
            has_speech=True,
            loudness_lufs=-16.0,
            peak_db=-1.0,
            dynamic_range_db=15.0,
        )
        assert metadata.file_name == "test.wav"
        assert metadata.language == "no"
        assert metadata.has_speech is True


class TestGetFfprobeInfo:
    def test_missing_file_raises(self, temp_dir):
        missing = temp_dir / "nonexistent.wav"
        with pytest.raises(Exception):
            get_ffprobe_info(missing)
