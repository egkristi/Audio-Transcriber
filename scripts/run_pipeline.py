#!/usr/bin/env python3
"""
Main orchestration script for Audio-Transcriber pipeline.

Coordinates all 6 pipeline stages:
1. Analyze - Extract metadata
2. Preprocess - Adaptive audio conditioning
3. Diarize - Speaker separation
4. Transcribe - Primary model transcription
5. Compare - Multi-model comparison (optional)
6. Editor - Manual review interface

Usage:
  uv run python scripts/run_pipeline.py --input audio.m4a --output-dir ./output
  uv run python scripts/run_pipeline.py --input folder/ --workers 4 --diarize --compare-models
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, get_logger, ensure_dir, format_duration, _NumpyEncoder
from src.config import load_config
from src.analyze import analyze_audio, save_metadata, AudioMetadata
from src.preprocess import preprocess_audio
from src.diarize import diarize_audio
from src.transcribe import transcribe_audio, TranscriptionSegment
from src.compare import compare_transcriptions
from src.editor import export_for_manual_editing
from src.database import TranscriptionDatabase
from src.vocabulary import load_vocabulary
from src.spell_check import check_transcription
from src.confidence import extract_confidence_signals

logger = get_logger("pipeline")


class AudioTranscriberPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize pipeline with configuration."""
        self.config = load_config(config_path)
        
    def _find_audio_files(self, input_path: Path) -> List[Path]:
        """Find all audio files in path, filtering out likely corrupted files."""
        AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".wma"}
        MIN_FILE_SIZE_BYTES = 1024  # Filter out empty/corrupted files
        
        if input_path.is_file():
            if input_path.suffix.lower() in AUDIO_EXTENSIONS:
                if input_path.stat().st_size < MIN_FILE_SIZE_BYTES:
                    logger.warning(f"Skipping likely corrupted file (<1KB): {input_path.name}")
                    return []
                return [input_path]
            else:
                logger.warning(f"Not an audio file: {input_path}")
                return []
        
        audio_files = []
        skipped = 0
        for ext in AUDIO_EXTENSIONS:
            for f in input_path.glob(f"**/*{ext}"):
                if f.stat().st_size >= MIN_FILE_SIZE_BYTES:
                    audio_files.append(f)
                else:
                    skipped += 1
        
        if skipped:
            logger.warning(f"Skipped {skipped} file(s) smaller than 1KB (likely corrupted)")
        
        logger.info(f"Found {len(audio_files)} audio files")
        return sorted(audio_files)
    
    def process_single_file(
        self,
        file_path: Path,
        output_dir: Path,
        steps: Optional[List[str]] = None,
        diarize: bool = True,
        compare_models: bool = False,
        primary_model: str = "NbAiLab/nb-whisper-large-verbatim",
        secondary_model: str = "openai/whisper-large-v3",
        db: Optional[TranscriptionDatabase] = None,
        vocab_file: Optional[Path] = None,
        spell_check: bool = False
    ) -> Dict:
        """
        Process a single audio file through the pipeline.
        
        Args:
            file_path: Path to audio file
            output_dir: Output directory
            steps: List of steps to run (None = all)
            diarize: Run diarization
            compare_models: Run secondary model comparison
            primary_model: Primary transcription model
            secondary_model: Secondary transcription model
            db: Optional TranscriptionDatabase for job tracking
            vocab_file: Optional custom vocabulary JSON file
            spell_check: Enable Norwegian spell-checking on output
            
        Returns:
            Dict with pipeline results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {file_path.name}")
        logger.info(f"{'='*60}\n")
        
        file_output_dir = ensure_dir(output_dir / file_path.stem)
        results = {"file": file_path.name, "status": "pending", "steps": {}}
        
        job_id = None
        if db:
            job_id = db.create_job(file_path)
        
        try:
            # Step 1: Analyze
            if not steps or "analyze" in steps:
                logger.info("STEP 1: Audio Analysis")
                metadata = analyze_audio(file_path)
                save_metadata(metadata, file_output_dir)
                results["steps"]["analyze"] = {
                    "status": "complete",
                    "metadata": {
                        "duration": metadata.duration_seconds,
                        "sample_rate": metadata.sample_rate,
                        "bandwidth": metadata.bandwidth_type,
                        "language": metadata.language,
                        "has_speech": metadata.has_speech
                    }
                }
            else:
                # Try to load existing metadata
                metadata_file = file_output_dir / f"{file_path.stem}_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata_data = json.load(f)
                        # Reconstruct minimal metadata
                        from dataclasses import make_dataclass
                        metadata = AudioMetadata(**metadata_data)
                else:
                    logger.error("Metadata not found, running analyze step")
                    metadata = analyze_audio(file_path)
                    save_metadata(metadata, file_output_dir)
            
            # Step 2: Preprocess
            if not steps or "preprocess" in steps:
                logger.info("\nSTEP 2: Audio Preprocessing")
                preprocess_config = self.config.get("preprocessing", {})
                audio_data, sr = preprocess_audio(
                    file_path,
                    metadata,
                    file_output_dir,
                    preprocess_config
                )
                results["steps"]["preprocess"] = {
                    "status": "complete",
                    "sample_rate": sr,
                    "shape": audio_data.shape
                }
                preprocessed_path = file_output_dir / f"{file_path.stem}_preprocessed.wav"
            else:
                preprocessed_path = file_output_dir / f"{file_path.stem}_preprocessed.wav"
                if not preprocessed_path.exists():
                    logger.error("Preprocessed audio not found, running preprocess step")
                    preprocess_config = self.config.get("preprocessing", {})
                    audio_data, sr = preprocess_audio(file_path, metadata, file_output_dir, preprocess_config)
            
            # Step 3: Diarization
            diarization_timeline = None
            if diarize and (not steps or "diarize" in steps):
                logger.info("\nSTEP 3: Speaker Diarization")
                diarization_config = self.config.get("diarization", {})
                segments, diarization_timeline = diarize_audio(
                    file_path,
                    metadata.duration_seconds,
                    {"diarization": diarization_config},
                    output_dir=file_output_dir
                )
                results["steps"]["diarize"] = {
                    "status": "complete",
                    "segments_count": len(segments),
                    "speakers": list(set(s.speaker for s in segments))
                }
            
            # Step 3/4: Transcription (Primary Model)
            if not steps or "transcribe" in steps:
                logger.info("\nSTEP 3/4: Transcription (Primary Model)")
                transcription_config = self.config.data
                
                # Build initial prompt from vocabulary if provided
                initial_prompt = None
                if vocab_file and vocab_file.exists():
                    vocab = load_vocabulary(vocab_file=vocab_file)
                    initial_prompt = vocab.generate_initial_prompt()
                    transcription_config = dict(transcription_config)
                    transcription_config["initial_prompt"] = initial_prompt
                
                primary_segments, primary_output = transcribe_audio(
                    preprocessed_path,
                    primary_model,
                    diarization_timeline,
                    metadata.duration_seconds,
                    transcription_config,
                    file_output_dir,
                    "srt"
                )
                results["steps"]["transcribe"] = {
                    "status": "complete",
                    "model": primary_model,
                    "segments_count": len(primary_segments),
                    "output_file": primary_output
                }
                
                # Step: Confidence-flagging for review prioritization
                logger.info("\nSTEP: Confidence Flagging")
                try:
                    # Convert segments to dicts for confidence extractor
                    seg_dicts = [
                        {
                            "start": s.start,
                            "end": s.end,
                            "text": s.text,
                            "speaker": s.speaker,
                            "words": s.words,
                            "confidence": s.confidence,
                            "avg_logprob": s.avg_logprob,
                            "no_speech_prob": s.no_speech_prob,
                            "compression_ratio": s.compression_ratio,
                            "temperature": s.temperature,
                        }
                        for s in primary_segments
                    ]
                    confidence_config = self.config.get("transcription", {})
                    confidence_segments = extract_confidence_signals(
                        segments=seg_dicts,
                        aligned_word_segments=None,
                        comparison_results=None,
                        metadata=asdict(metadata),
                        config=confidence_config,
                    )
                    # Export review list
                    review_path = file_output_dir / f"{file_path.stem}_review_list.txt"
                    from src.confidence import ConfidenceExtractor
                    extractor = ConfidenceExtractor(confidence_config)
                    extractor.export_review_list(confidence_segments, review_path, top_n=20)
                    results["steps"]["confidence"] = {
                        "status": "complete",
                        "segments_flagged": len([s for s in confidence_segments if s.flags]),
                        "review_list": str(review_path),
                    }
                    logger.info(f"Confidence review list exported: {review_path}")
                except Exception as conf_err:
                    logger.warning(f"Confidence extraction failed: {conf_err}")
                    results["steps"]["confidence"] = {"status": "failed", "error": str(conf_err)}
                
                # Log transcription segments to database
                if db and job_id:
                    for seg in primary_segments:
                        db.log_transcription(
                            job_id=job_id,
                            model_name=primary_model,
                            segment_id=seg.id,
                            start_time=seg.start,
                            end_time=seg.end,
                            text=seg.text,
                            confidence=seg.confidence,
                            speaker=seg.speaker
                        )
                
                # Optional spell-checking
                if spell_check:
                    logger.info("Running spell-check on primary transcription")
                    spell_config = self.config.get("spell_check", {})
                    full_text = " ".join(s.text for s in primary_segments)
                    spell_results = check_transcription(full_text, spell_config)
                    results["steps"]["spell_check"] = {
                        "status": "complete",
                        "error_count": spell_results.get("error_count", 0),
                        "enabled": spell_results.get("enabled", False)
                    }
            else:
                primary_segments = []
            
            # Step 4: Secondary Model (Optional)
            if compare_models and (not steps or "compare" in steps):
                logger.info("\nSTEP 4: Secondary Model Transcription")
                secondary_segments, secondary_output = transcribe_audio(
                    preprocessed_path,
                    secondary_model,
                    diarization_timeline,
                    metadata.duration_seconds,
                    self.config.data,
                    file_output_dir,
                    "srt"
                )
                results["steps"]["secondary_transcribe"] = {
                    "status": "complete",
                    "model": secondary_model,
                    "segments_count": len(secondary_segments),
                    "output_file": secondary_output
                }
                
                # Step 5: Compare Models
                logger.info("\nSTEP 5: Model Comparison")
                comparison_config = self.config.data
                comparison_results, comparison_output = compare_transcriptions(
                    primary_segments,
                    secondary_segments,
                    comparison_config,
                    file_output_dir
                )
                results["steps"]["compare"] = {
                    "status": "complete",
                    "high_priority": sum(1 for r in comparison_results if r.priority == "high"),
                    "output_file": comparison_output
                }
            
            # Step 6: Editor
            if primary_segments and (not steps or "editor" in steps):
                logger.info("\nSTEP 6: Manual Editing")
                primary_srt = file_output_dir / f"{file_path.stem}_{primary_model.split('/')[-1]}.srt"
                if primary_srt.exists():
                    instructions = export_for_manual_editing(primary_srt, file_path)
                    results["steps"]["editor"] = {
                        "status": "ready_for_review",
                        "srt_file": str(primary_srt),
                        "instructions": instructions[:200]  # Truncate for summary
                    }
            
            results["status"] = "complete"
            logger.info("\n✓ Pipeline completed successfully")
            
            if db and job_id:
                db.update_job_status(job_id, "complete")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            results["status"] = "failed"
            results["error"] = str(e)
            if db and job_id:
                db.update_job_status(job_id, "failed", error_message=str(e))
        
        return results
    
    def process_batch(
        self,
        input_path: Path,
        output_dir: Path,
        workers: int = 4,
        **kwargs
    ) -> List[Dict]:
        """
        Process multiple audio files in parallel.
        
        Args:
            input_path: Input directory or file
            output_dir: Output directory
            workers: Number of parallel workers
            **kwargs: Additional args to pass to process_single_file
            
        Returns:
            List of results for each file
        """
        files = self._find_audio_files(input_path)
        
        if not files:
            logger.warning("No audio files found")
            return []
        
        results = []
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    self.process_single_file,
                    file_path,
                    output_dir,
                    **kwargs
                ): file_path
                for file_path in files
            }
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Task failed: {e}")
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Audio-Transcriber Pipeline - Norwegian audio transcription system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file with all steps
  uv run python scripts/run_pipeline.py \\
    --input audio.m4a --output-dir ./output \\
    --diarize --compare-models

  # Batch processing
  uv run python scripts/run_pipeline.py \\
    --input recordings/ --output-dir ./output \\
    --workers 4 --diarize

  # Single step
  uv run python scripts/run_pipeline.py \\
    --input audio.m4a --step analyze
        """
    )
    
    # Input/output
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input audio file or directory"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default="./output",
        help="Output directory (default: ./output)"
    )
    
    # Pipeline configuration
    parser.add_argument(
        "--step",
        choices=["analyze", "preprocess", "diarize", "transcribe", "compare", "editor"],
        help="Run specific pipeline step only"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config.yaml (default: project root)"
    )
    
    # Processing options
    parser.add_argument(
        "--diarize",
        action="store_true",
        default=True,
        help="Run speaker diarization (default: True)"
    )
    parser.add_argument(
        "--no-diarize",
        dest="diarize",
        action="store_false",
        help="Skip speaker diarization"
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        default=False,
        help="Compare primary and secondary models"
    )
    parser.add_argument(
        "--primary-model",
        default="NbAiLab/nb-whisper-large-verbatim",
        help="Primary transcription model"
    )
    parser.add_argument(
        "--secondary-model",
        default="openai/whisper-large-v3",
        help="Secondary transcription model"
    )
    
    # Batch processing
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for batch processing (default: 1, recommended for CPU-only)"
    )
    
    # Optional integrations
    parser.add_argument(
        "--use-database",
        action="store_true",
        default=False,
        help="Enable SQLite job tracking and transcription logging"
    )
    parser.add_argument(
        "--vocabulary-file",
        type=Path,
        default=None,
        help="Path to JSON vocabulary file for initial_prompt injection"
    )
    parser.add_argument(
        "--spell-check",
        action="store_true",
        default=False,
        help="Enable Norwegian spell-checking on transcription output"
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = args.output_dir / "logs" / "pipeline.log"
    setup_logging(
        level=args.log_level,
        format_type="json",
        log_file=log_file
    )
    
    logger.info("="*60)
    logger.info("Audio-Transcriber Pipeline Started")
    logger.info("="*60)
    
    # Validate input
    if not args.input.exists():
        logger.error(f"Input path not found: {args.input}")
        sys.exit(1)
    
    try:
        # Initialize pipeline
        pipeline = AudioTranscriberPipeline(args.config)
        
        # Initialize optional database
        db = None
        if args.use_database:
            db_path = args.output_dir / "transcriptions.db"
            db = TranscriptionDatabase(db_path)
            logger.info(f"Database enabled: {db_path}")
        
        # Process files
        steps = [args.step] if args.step else None
        
        pipeline_kwargs = {
            "steps": steps,
            "diarize": args.diarize,
            "compare_models": args.compare_models,
            "primary_model": args.primary_model,
            "secondary_model": args.secondary_model,
            "db": db,
            "vocab_file": args.vocabulary_file,
            "spell_check": args.spell_check,
        }
        
        if args.input.is_file():
            results = [pipeline.process_single_file(
                args.input,
                args.output_dir,
                **pipeline_kwargs
            )]
        else:
            results = pipeline.process_batch(
                args.input,
                args.output_dir,
                workers=args.workers,
                **pipeline_kwargs
            )
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("PIPELINE SUMMARY")
        logger.info("="*60)
        
        successful = sum(1 for r in results if r["status"] == "complete")
        failed = sum(1 for r in results if r["status"] == "failed")
        
        logger.info(f"Total files: {len(results)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        
        # Save summary
        summary_file = args.output_dir / "pipeline_summary.json"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_file, "w") as f:
            json.dump(results, f, indent=2, default=str, cls=_NumpyEncoder)
        logger.info(f"Summary saved to: {summary_file}")
        
        sys.exit(0 if failed == 0 else 1)
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
