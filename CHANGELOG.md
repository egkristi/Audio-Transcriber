# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project setup with pyproject.toml and uv configuration
- ROADMAP.md for feature tracking
- CHANGELOG.md for release notes
- **Step 1 - Audio Analysis** (`src/analyze.py`):
  - FFprobe integration for audio metadata extraction
  - Bandwidth detection (narrowband/wideband/fullband)
  - Stereo separation detection for real stereo audio
  - Language detection using Whisper
  - Voice Activity Detection (VAD) using Silero model
  - Loudness and dynamic range calculation (ITU-R BS.1770-4)
  - Metadata serialization to JSON format
- **Step 2 - Preprocessing** (`src/preprocess.py`):
  - Adaptive resampling to 16 kHz mono
  - High-pass filter (80 Hz) for rumble removal
  - Conditional low-pass filter (8 kHz) for narrowband audio only
  - ITU-R BS.1770-4 loudness normalization
  - Optional noise reduction (denoising) with configurable strength
  - WAV output for next stages
- **Step 3 - Diarization** (`src/diarize.py`):
  - PyAnnote speaker-diarization-3.1 integration
  - Flexible speaker count configuration (auto-detect, min/max, exact)
  - Speaker timeline generation
  - JSON output with speaker segments
- **Steps 3 & 4 - Transcription** (`src/transcribe.py`):
  - WhisperX integration with Norwegian-optimized models
  - NB-Whisper (verbatim and main variants)
  - OpenAI Whisper multilingual support
  - Word-level timestamp generation
  - Speaker diarization alignment
  - Multiple output formats: SRT, WebVTT, JSON
- **Step 5 - Model Comparison** (`src/compare.py`):
  - Segment-level alignment between transcriptions
  - Similarity scoring with text normalization
  - Low-confidence detection
  - Disagreement flagging
  - Priority-based sorting (high/medium/low)
  - Human-readable comparison reports
- **Step 6 - Manual Editor** (`src/editor.py`):
  - SRT export for external editors (Subtitle Edit for Mac)
  - Editing workflow instructions
  - Placeholder for future web-based UI
- **Main Orchestration** (`scripts/run_pipeline.py`):
  - Single-file and batch processing modes
  - Parallel worker pool (configurable)
  - Step-by-step or full pipeline execution
  - Comprehensive CLI interface
  - JSON logging with file output
  - Pipeline summary reporting
- Utility modules (`src/utils.py`, `src/config.py`):
  - JSON and text logging formatters
  - Configuration management with YAML support
  - Audio processing helper functions
  - File utilities (save/load JSON, ensure directories)

### Changed

### Fixed

### Removed

## [0.1.0] - 2026-05-28

### Added
- Project scaffolding and infrastructure
- README with detailed pipeline documentation
- MIT License
