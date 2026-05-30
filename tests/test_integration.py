"""Integration tests for the pipeline orchestrator (process_single_file).

Tests the glue between pipeline steps: config wiring, path building,
step sequencing, and result aggregation. Heavy dependencies (WhisperX,
pyannote) are mocked to keep tests fast and deterministic.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY

import numpy as np
import pytest
import soundfile as sf

from scripts.run_pipeline import AudioTranscriberPipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_wav(temp_dir):
    """Create a tiny synthetic WAV file for pipeline testing."""
    sr = 16000
    duration = 0.5  # 500 ms
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    path = temp_dir / "test_input.wav"
    sf.write(str(path), audio, sr)
    return path


@pytest.fixture
def pipeline(temp_dir):
    """Create a pipeline instance with a temp config."""
    return AudioTranscriberPipeline()


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_mock_metadata(file_path, duration=0.5):
    """Create a realistic AudioMetadata instance."""
    from src.analyze import AudioMetadata
    return AudioMetadata(
        file_path=str(file_path),
        file_name=Path(file_path).name,
        file_size_mb=0.001,
        duration_seconds=duration,
        sample_rate=16000,
        channels=1,
        bit_rate_kbps=256,
        codec="pcm_f32le",
        bandwidth_type="narrowband",
        has_stereo_separation=False,
        language="no",
        has_speech=True,
        loudness_lufs=-20.0,
        peak_db=-3.0,
        dynamic_range_db=17.0,
        total_confidence=None,
        segments_count=None,
        flagged_segments_count=None,
    )


def _make_mock_segments():
    """Create mock TranscriptionSegments for pipeline testing."""
    from src.transcribe import TranscriptionSegment
    return [
        TranscriptionSegment(
            id=0,
            start=0.0,
            end=0.3,
            text="Hei dette er en test",
            speaker=None,
            words=[
                {"word": "Hei", "start": 0.0, "end": 0.1, "score": 0.95},
                {"word": "dette", "start": 0.1, "end": 0.18, "score": 0.90},
                {"word": "er", "start": 0.18, "end": 0.22, "score": 0.92},
                {"word": "en", "start": 0.22, "end": 0.26, "score": 0.88},
                {"word": "test", "start": 0.26, "end": 0.3, "score": 0.85},
            ],
            confidence=0.90,
            avg_logprob=-0.15,
            no_speech_prob=0.02,
            compression_ratio=1.2,
            temperature=0.0,
        ),
        TranscriptionSegment(
            id=1,
            start=0.3,
            end=0.5,
            text="Kor e du hen",
            speaker=None,
            words=[
                {"word": "Kor", "start": 0.3, "end": 0.36, "score": 0.80},
                {"word": "e", "start": 0.36, "end": 0.40, "score": 0.85},
                {"word": "du", "start": 0.40, "end": 0.45, "score": 0.90},
                {"word": "hen", "start": 0.45, "end": 0.5, "score": 0.75},
            ],
            confidence=0.82,
            avg_logprob=-0.30,
            no_speech_prob=0.05,
            compression_ratio=1.1,
            temperature=0.0,
        ),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineOrchestrator:
    """Test the pipeline orchestrator glue."""

    @patch("scripts.run_pipeline.analyze_audio")
    @patch("scripts.run_pipeline.preprocess_audio")
    @patch("scripts.run_pipeline.transcribe_audio")
    @patch("scripts.run_pipeline.extract_confidence_signals")
    def test_full_pipeline_runs_all_steps(
        self,
        mock_confidence,
        mock_transcribe,
        mock_preprocess,
        mock_analyze,
        pipeline,
        synthetic_wav,
        temp_dir,
    ):
        """Verify all pipeline steps are called in sequence."""
        mock_analyze.return_value = _make_mock_metadata(synthetic_wav)
        mock_preprocess.return_value = (np.zeros((16000,), dtype=np.float32), 16000)
        mock_transcribe.return_value = (_make_mock_segments(), str(temp_dir / "test.srt"))
        mock_confidence.return_value = []

        results = pipeline.process_single_file(
            synthetic_wav,
            temp_dir,
            diarize=False,
            compare_models=False,
        )

        assert results["status"] == "complete"
        mock_analyze.assert_called_once_with(synthetic_wav)
        mock_preprocess.assert_called_once()
        mock_transcribe.assert_called_once()
        mock_confidence.assert_called_once()

    @patch("scripts.run_pipeline.analyze_audio")
    @patch("scripts.run_pipeline.preprocess_audio")
    @patch("scripts.run_pipeline.transcribe_audio")
    @patch("scripts.run_pipeline.extract_confidence_signals")
    def test_pipeline_with_diarization(
        self,
        mock_confidence,
        mock_transcribe,
        mock_preprocess,
        mock_analyze,
        pipeline,
        synthetic_wav,
        temp_dir,
    ):
        """Verify diarization step is called when --diarize is set."""
        mock_analyze.return_value = _make_mock_metadata(synthetic_wav)
        mock_preprocess.return_value = (np.zeros((16000,), dtype=np.float32), 16000)
        mock_transcribe.return_value = (_make_mock_segments(), str(temp_dir / "test.srt"))
        mock_confidence.return_value = []

        with patch("scripts.run_pipeline.diarize_audio") as mock_diarize:
            mock_diarize.return_value = ([], None)
            results = pipeline.process_single_file(
                synthetic_wav,
                temp_dir,
                diarize=True,
            )

        assert results["status"] == "complete"
        mock_diarize.assert_called_once()
        assert "diarize" in results["steps"]

    @patch("scripts.run_pipeline.analyze_audio")
    @patch("scripts.run_pipeline.preprocess_audio")
    @patch("scripts.run_pipeline.transcribe_audio")
    @patch("scripts.run_pipeline.extract_confidence_signals")
    def test_pipeline_without_diarization(
        self,
        mock_confidence,
        mock_transcribe,
        mock_preprocess,
        mock_analyze,
        pipeline,
        synthetic_wav,
        temp_dir,
    ):
        """Verify diarization is NOT called when --diarize is not set."""
        mock_analyze.return_value = _make_mock_metadata(synthetic_wav)
        mock_preprocess.return_value = (np.zeros((16000,), dtype=np.float32), 16000)
        mock_transcribe.return_value = (_make_mock_segments(), str(temp_dir / "test.srt"))
        mock_confidence.return_value = []

        with patch("scripts.run_pipeline.diarize_audio") as mock_diarize:
            results = pipeline.process_single_file(
                synthetic_wav,
                temp_dir,
                diarize=False,
            )

        assert results["status"] == "complete"
        mock_diarize.assert_not_called()
        assert "diarize" not in results["steps"]

    @patch("scripts.run_pipeline.analyze_audio")
    @patch("scripts.run_pipeline.preprocess_audio")
    @patch("scripts.run_pipeline.transcribe_audio")
    @patch("scripts.run_pipeline.extract_confidence_signals")
    def test_pipeline_with_spell_check(
        self,
        mock_confidence,
        mock_transcribe,
        mock_preprocess,
        mock_analyze,
        pipeline,
        synthetic_wav,
        temp_dir,
    ):
        """Verify spell-check step is called when --spell-check is set."""
        mock_analyze.return_value = _make_mock_metadata(synthetic_wav)
        mock_preprocess.return_value = (np.zeros((16000,), dtype=np.float32), 16000)
        mock_transcribe.return_value = (_make_mock_segments(), str(temp_dir / "test.srt"))
        mock_confidence.return_value = []

        with patch("scripts.run_pipeline.check_transcription") as mock_spell:
            mock_spell.return_value = {"error_count": 0, "enabled": True}
            results = pipeline.process_single_file(
                synthetic_wav,
                temp_dir,
                spell_check=True,
            )

        assert results["status"] == "complete"
        mock_spell.assert_called_once()
        assert results["steps"]["spell_check"]["enabled"] is True

    @patch("scripts.run_pipeline.analyze_audio")
    @patch("scripts.run_pipeline.preprocess_audio")
    @patch("scripts.run_pipeline.transcribe_audio")
    @patch("scripts.run_pipeline.extract_confidence_signals")
    def test_pipeline_with_normalization(
        self,
        mock_confidence,
        mock_transcribe,
        mock_preprocess,
        mock_analyze,
        pipeline,
        synthetic_wav,
        temp_dir,
    ):
        """Verify normalization step is called when --normalize is set."""
        mock_analyze.return_value = _make_mock_metadata(synthetic_wav)
        mock_preprocess.return_value = (np.zeros((16000,), dtype=np.float32), 16000)
        mock_transcribe.return_value = (_make_mock_segments(), str(temp_dir / "test.srt"))
        mock_confidence.return_value = []

        with patch("scripts.run_pipeline.normalize_transcription_segments") as mock_norm:
            mock_norm.return_value = (
                [
                    {"id": 0, "text": "Hei dette er en test"},
                    {"id": 1, "text": "Kor e du hen"},
                ],
                [],
            )
            results = pipeline.process_single_file(
                synthetic_wav,
                temp_dir,
                normalize=True,
            )

        assert results["status"] == "complete"
        mock_norm.assert_called_once()
        assert results["steps"]["normalization"]["status"] == "complete"

    @patch("scripts.run_pipeline.analyze_audio")
    @patch("scripts.run_pipeline.preprocess_audio")
    @patch("scripts.run_pipeline.transcribe_audio")
    @patch("scripts.run_pipeline.extract_confidence_signals")
    def test_pipeline_without_normalization(
        self,
        mock_confidence,
        mock_transcribe,
        mock_preprocess,
        mock_analyze,
        pipeline,
        synthetic_wav,
        temp_dir,
    ):
        """Verify normalization is skipped when --normalize is not set."""
        mock_analyze.return_value = _make_mock_metadata(synthetic_wav)
        mock_preprocess.return_value = (np.zeros((16000,), dtype=np.float32), 16000)
        mock_transcribe.return_value = (_make_mock_segments(), str(temp_dir / "test.srt"))
        mock_confidence.return_value = []

        with patch("scripts.run_pipeline.normalize_transcription_segments") as mock_norm:
            results = pipeline.process_single_file(
                synthetic_wav,
                temp_dir,
                normalize=False,
            )

        assert results["status"] == "complete"
        mock_norm.assert_not_called()
        assert results["steps"]["normalization"]["status"] == "skipped"

    @patch("scripts.run_pipeline.analyze_audio")
    @patch("scripts.run_pipeline.preprocess_audio")
    @patch("scripts.run_pipeline.transcribe_audio")
    @patch("scripts.run_pipeline.extract_confidence_signals")
    def test_pipeline_with_vocabulary(
        self,
        mock_confidence,
        mock_transcribe,
        mock_preprocess,
        mock_analyze,
        pipeline,
        synthetic_wav,
        temp_dir,
    ):
        """Verify vocabulary is loaded when dialect is specified."""
        mock_analyze.return_value = _make_mock_metadata(synthetic_wav)
        mock_preprocess.return_value = (np.zeros((16000,), dtype=np.float32), 16000)
        mock_transcribe.return_value = (_make_mock_segments(), str(temp_dir / "test.srt"))
        mock_confidence.return_value = []

        with patch("scripts.run_pipeline.load_vocabulary") as mock_vocab:
            mock_vocab_instance = MagicMock()
            mock_vocab_instance.generate_initial_prompt.return_value = "Hei kor e du"
            mock_vocab.return_value = mock_vocab_instance

            results = pipeline.process_single_file(
                synthetic_wav,
                temp_dir,
                dialect="northern_norwegian",
            )

        assert results["status"] == "complete"
        mock_vocab.assert_called_once_with(
            use_default_norwegian=True,
            dialect="northern_norwegian",
        )

    @patch("scripts.run_pipeline.analyze_audio")
    @patch("scripts.run_pipeline.preprocess_audio")
    @patch("scripts.run_pipeline.transcribe_audio")
    @patch("scripts.run_pipeline.extract_confidence_signals")
    def test_pipeline_with_compare_models(
        self,
        mock_confidence,
        mock_transcribe,
        mock_preprocess,
        mock_analyze,
        pipeline,
        synthetic_wav,
        temp_dir,
    ):
        """Verify secondary model comparison is called when --compare-models is set."""
        mock_analyze.return_value = _make_mock_metadata(synthetic_wav)
        mock_preprocess.return_value = (np.zeros((16000,), dtype=np.float32), 16000)
        mock_transcribe.return_value = (_make_mock_segments(), str(temp_dir / "test.srt"))
        mock_confidence.return_value = []

        with (
            patch("scripts.run_pipeline.compare_transcriptions") as mock_compare,
            patch("scripts.run_pipeline.transcribe_audio") as mock_transcribe2,
        ):
            # First call = primary, second call = secondary
            mock_transcribe2.side_effect = [
                (_make_mock_segments(), str(temp_dir / "test_primary.srt")),
                (_make_mock_segments(), str(temp_dir / "test_secondary.srt")),
            ]
            mock_compare.return_value = ([], str(temp_dir / "comparison.txt"))

            results = pipeline.process_single_file(
                synthetic_wav,
                temp_dir,
                compare_models=True,
            )

        assert results["status"] == "complete"
        assert mock_transcribe2.call_count == 2  # primary + secondary
        mock_compare.assert_called_once()

    @patch("scripts.run_pipeline.analyze_audio")
    @patch("scripts.run_pipeline.preprocess_audio")
    @patch("scripts.run_pipeline.transcribe_audio")
    @patch("scripts.run_pipeline.extract_confidence_signals")
    def test_pipeline_result_structure(
        self,
        mock_confidence,
        mock_transcribe,
        mock_preprocess,
        mock_analyze,
        pipeline,
        synthetic_wav,
        temp_dir,
    ):
        """Verify the result dict has the expected structure."""
        mock_analyze.return_value = _make_mock_metadata(synthetic_wav)
        mock_preprocess.return_value = (np.zeros((16000,), dtype=np.float32), 16000)
        mock_transcribe.return_value = (_make_mock_segments(), str(temp_dir / "test.srt"))
        mock_confidence.return_value = []

        results = pipeline.process_single_file(
            synthetic_wav,
            temp_dir,
            diarize=False,
        )

        assert "file" in results
        assert "status" in results
        assert "steps" in results
        assert results["status"] == "complete"
        assert "analyze" in results["steps"]
        assert "preprocess" in results["steps"]
        assert "transcribe" in results["steps"]
        assert "confidence" in results["steps"]

    @patch("scripts.run_pipeline.analyze_audio")
    @patch("scripts.run_pipeline.preprocess_audio")
    @patch("scripts.run_pipeline.transcribe_audio")
    @patch("scripts.run_pipeline.extract_confidence_signals")
    def test_pipeline_error_handling(
        self,
        mock_confidence,
        mock_transcribe,
        mock_preprocess,
        mock_analyze,
        pipeline,
        synthetic_wav,
        temp_dir,
    ):
        """Verify pipeline handles errors gracefully."""
        mock_analyze.side_effect = RuntimeError("Analysis failed")

        results = pipeline.process_single_file(
            synthetic_wav,
            temp_dir,
        )

        assert results["status"] == "failed"
        assert "error" in results
        assert "Analysis failed" in results["error"]

    @patch("scripts.run_pipeline.analyze_audio")
    @patch("scripts.run_pipeline.preprocess_audio")
    @patch("scripts.run_pipeline.transcribe_audio")
    @patch("scripts.run_pipeline.extract_confidence_signals")
    def test_pipeline_confidence_aggregation(
        self,
        mock_confidence,
        mock_transcribe,
        mock_preprocess,
        mock_analyze,
        pipeline,
        synthetic_wav,
        temp_dir,
    ):
        """Verify confidence aggregation is computed from segments."""
        mock_analyze.return_value = _make_mock_metadata(synthetic_wav)
        mock_preprocess.return_value = (np.zeros((16000,), dtype=np.float32), 16000)
        segments = _make_mock_segments()
        mock_transcribe.return_value = (segments, str(temp_dir / "test.srt"))
        mock_confidence.return_value = []

        results = pipeline.process_single_file(
            synthetic_wav,
            temp_dir,
        )

        assert results["status"] == "complete"
        # Verify metadata was updated with confidence stats
        assert mock_analyze.return_value.total_confidence is not None
        assert mock_analyze.return_value.segments_count == 2
        # Both segments have confidence > 0.7, so 0 flagged
        assert mock_analyze.return_value.flagged_segments_count == 0

    @patch("scripts.run_pipeline.analyze_audio")
    @patch("scripts.run_pipeline.preprocess_audio")
    @patch("scripts.run_pipeline.transcribe_audio")
    @patch("scripts.run_pipeline.extract_confidence_signals")
    def test_pipeline_step_filtering(
        self,
        mock_confidence,
        mock_transcribe,
        mock_preprocess,
        mock_analyze,
        pipeline,
        synthetic_wav,
        temp_dir,
    ):
        """Verify --step filtering runs only the requested step.

        Note: When steps=["analyze"], the pipeline skips preprocess/transcribe
        but still tries to load existing preprocessed audio. If none exists,
        it falls through to running preprocess. This test verifies that
        analyze is called and the pipeline completes.
        """
        mock_analyze.return_value = _make_mock_metadata(synthetic_wav)
        mock_preprocess.return_value = (np.zeros((16000,), dtype=np.float32), 16000)

        results = pipeline.process_single_file(
            synthetic_wav,
            temp_dir,
            steps=["analyze"],
        )

        assert results["status"] == "complete"
        mock_analyze.assert_called_once()
