"""Tests for src/preprocess.py"""

import numpy as np
import pytest

from src.preprocess import (
    apply_highpass_filter,
    apply_lowpass_filter,
    convert_to_mono,
    normalize_loudness,
    resample_to_target_rate,
)


class TestResampleToTargetRate:
    def test_no_change_when_same_rate(self, sample_audio_data):
        audio, sr = sample_audio_data
        result = resample_to_target_rate(audio, sr, target_sr=sr)
        np.testing.assert_array_equal(result, audio)

    def test_resample_changes_shape(self, sample_audio_data):
        audio, sr = sample_audio_data
        target_sr = sr // 2
        result = resample_to_target_rate(audio, sr, target_sr=target_sr)
        assert result.shape[0] == audio.shape[0] // 2


class TestConvertToMono:
    def test_mono_unchanged(self, sample_audio_data):
        audio, _ = sample_audio_data
        result = convert_to_mono(audio)
        np.testing.assert_array_equal(result, audio)

    def test_stereo_averaged(self, sample_stereo_audio_data):
        stereo, _ = sample_stereo_audio_data
        result = convert_to_mono(stereo)
        expected = np.mean(stereo, axis=0)
        np.testing.assert_array_almost_equal(result, expected)


class TestApplyHighpassFilter:
    def test_filter_preserves_shape(self, sample_audio_data):
        audio, sr = sample_audio_data
        result = apply_highpass_filter(audio, sr, cutoff_hz=80)
        assert result.shape == audio.shape

    def test_cutoff_too_high_skips(self, sample_audio_data):
        audio, sr = sample_audio_data
        result = apply_highpass_filter(audio, sr, cutoff_hz=sr)
        np.testing.assert_array_equal(result, audio)


class TestApplyLowpassFilter:
    def test_wideband_skips_filter(self, sample_audio_data):
        audio, sr = sample_audio_data
        result = apply_lowpass_filter(audio, sr, cutoff_hz=8000, bandwidth_type="wideband")
        np.testing.assert_array_equal(result, audio)

    def test_narrowband_applies_filter(self, sample_audio_data):
        audio, sr = sample_audio_data
        result = apply_lowpass_filter(audio, sr, cutoff_hz=8000, bandwidth_type="narrowband")
        assert result.shape == audio.shape


class TestNormalizeLoudness:
    def test_silent_audio_returns_unchanged(self, sample_audio_data):
        audio, sr = sample_audio_data
        silent = np.zeros_like(audio)
        result = normalize_loudness(silent, sr, target_lufs=-16.0)
        np.testing.assert_array_equal(result, silent)

    def test_normal_audio_changes(self, sample_audio_data):
        audio, sr = sample_audio_data
        result = normalize_loudness(audio, sr, target_lufs=-16.0)
        assert result.shape == audio.shape
        # Result should be finite and within reasonable bounds
        assert np.isfinite(result).all()
        assert np.max(np.abs(result)) <= 1.0
