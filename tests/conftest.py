"""Pytest configuration and shared fixtures."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config(temp_dir):
    """Create a sample config.yaml for testing."""
    config_data = {
        "analysis": {"vad_model": "silero", "language_detection": True},
        "preprocessing": {
            "target_sample_rate": 16000,
            "target_channels": 1,
            "high_pass_filter_hz": 80,
            "low_pass_filter_hz": 8000,
            "loudness_target_lufs": -16,
            "denoising": False,
        },
        "diarization": {
            "model": "pyannote/speaker-diarization-3.1",
            "min_speakers": None,
            "max_speakers": None,
        },
        "transcription": {
            "primary_model": "NbAiLab/nb-whisper-large-verbatim",
            "beam_size": 5,
            "word_timestamps": True,
            "vad_filter": True,
            "language": "no",
            "low_confidence_threshold": 0.85,
        },
        "comparison": {
            "alignment_method": "wer",
            "min_agreement_score": 0.95,
        },
        "batch": {"max_workers": 4, "chunk_size": 10, "timeout_per_file": 3600},
        "output": {"format": "srt", "include_confidence": True},
        "logging": {"level": "INFO", "format": "text", "save_to_file": False},
        "performance": {"device": "cpu", "compute_type": "int8"},
    }
    config_path = temp_dir / "config.yaml"
    import yaml

    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return config_path


@pytest.fixture
def sample_audio_data():
    """Generate a short synthetic mono audio signal for testing."""
    sr = 16000
    duration = 2.0  # seconds
    t = np.linspace(0, duration, int(sr * duration))
    # Simple sine wave at 440 Hz
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sr


@pytest.fixture
def sample_stereo_audio_data():
    """Generate a short synthetic stereo audio signal for testing."""
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    # Two different sine waves for each channel
    left = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    right = np.sin(2 * np.pi * 880 * t).astype(np.float32)
    stereo = np.stack([left, right])
    return stereo, sr
