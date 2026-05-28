# Audio-Transcriber Roadmap

## Core Pipeline Implementation

### Phase 1: Project Infrastructure
- [x] Create pyproject.toml with uv configuration
- [x] Create config.yaml with pipeline settings
- [x] Create directory structure (src/, scripts/)
- [ ] Setup Hugging Face authentication mechanism

### Phase 2: Core Modules
- [x] Implement `analyze.py` (Step 1: Audio analysis and metadata)
- [x] Implement `preprocess.py` (Step 2: Adaptive audio preprocessing)
- [x] Implement `diarize.py` (Step 3: Speaker diarization)
- [x] Implement `transcribe.py` (Steps 3 & 4: Whisper transcription)
- [x] Implement `compare.py` (Step 5: Model comparison and deviation marking)
- [x] Implement `editor.py` (Step 6: Web-based audio editor)

### Phase 3: Orchestration & CLI
- [x] Implement `run_pipeline.py` (Main orchestration script)
- [x] Add CLI argument parsing (--input, --output-dir, --step, --workers, etc.)
- [x] Add batch processing support
- [x] Add single-file processing support
- [x] Add worker pool for parallel processing

### Phase 4: Configuration & Utilities
- [x] Create `utils.py` (Helper functions, logging, error handling)
- [x] Create `config.py` (Configuration loading and validation)
- [x] Setup logging system
- [x] Add database module (`database.py`) for tracking and logging
- [x] Add spell-checking module (`spell_check.py`) for Norwegian text
- [x] Add vocabulary module (`vocabulary.py`) for custom words and initial prompts
- [ ] Add VAD (Voice Activity Detection) optimization

### Phase 5: Features & Enhancements
- [ ] Word-level confidence filtering
- [ ] Norwegian spell-checking integration
- [ ] Custom word list support (proper nouns, technical terms)
- [ ] Automatic initial prompt generation for Whisper
- [ ] SQLite database for logging and performance tracking
- [ ] Web UI for subtitle editing with waveform visualization

### Phase 6: Performance Optimization
- [ ] Apple Silicon optimization (CoreML/MLX support)
- [ ] GPU acceleration support (CUDA for Linux/Windows)
- [ ] Model caching optimization
- [ ] Memory usage optimization for batch processing

### Phase 7: Testing & Documentation
- [ ] Add unit tests
- [ ] Add integration tests
- [ ] Add example workflows to README
- [ ] Add troubleshooting guide
- [ ] Add API documentation

### Known Issues & Blockers
- [ ] Faster-whisper currently CPU-only on Mac M1/M2
- [ ] Need to evaluate CoreML conversion overhead
- [ ] PyAnnote privacy concerns for speaker diarization
- [ ] Whisper confidence score calibration issues

### Future Considerations
- Language model fine-tuning on domain-specific vocabulary
- Real-time transcription support
- REST API endpoint for pipeline
- Docker containerization
- Support for other Nordic languages (Swedish, Danish, Finnish)
