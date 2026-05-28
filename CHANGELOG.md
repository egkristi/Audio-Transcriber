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
- Utility modules (`src/utils.py`, `src/config.py`):
  - JSON and text logging formatters
  - Configuration management with YAML support
  - Audio processing helper functions

### Changed

### Fixed

### Removed

## [0.1.0] - 2026-05-28

### Added
- Project scaffolding and infrastructure
- README with detailed pipeline documentation
- MIT License
