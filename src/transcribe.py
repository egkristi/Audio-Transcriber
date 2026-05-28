"""
Steps 3 & 4: Whisper Transcription

Transcribes audio using Norwegian-optimized Whisper models or OpenAI's multilingual model.
Integrates with speaker diarization from Step 3.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np

from .utils import get_logger, save_json

logger = get_logger("transcribe")


@dataclass
class WordTimestamp:
    """Word with timestamp and confidence."""
    
    word: str
    start: float
    end: float
    confidence: float


@dataclass
class TranscriptionSegment:
    """Transcribed segment with timing and metadata."""
    
    id: int
    start: float
    end: float
    text: str
    speaker: Optional[str] = None  # From diarization
    words: Optional[List[Dict]] = None  # Word-level timestamps
    confidence: float = 1.0


class Transcriber:
    """Wrapper around WhisperX for optimized transcription."""
    
    def __init__(self, model_name: str = "NbAiLab/nb-whisper-large-verbatim", config: Optional[dict] = None):
        """
        Initialize transcriber.
        
        Args:
            model_name: Model identifier (HuggingFace or local)
            config: Configuration dict
        """
        self.model_name = model_name
        self.config = config or {}
        self.model = None
        self.processor = None
    
    def _load_model(self):
        """Lazily load the transcription model."""
        if self.model is not None:
            return
        
        try:
            import whisperx
            
            logger.info(f"Loading transcription model: {self.model_name}")
            
            device = "cpu"  # CPU for Mac compatibility
            compute_type = self.config.get("compute_type", "int8")
            
            self.model = whisperx.load_model(
                self.model_name,
                device=device,
                compute_type=compute_type,
                language="no"  # Norwegian
            )
            
            logger.info(f"Model loaded on device: {device}")
            
        except Exception as e:
            logger.error(f"Failed to load transcription model: {e}")
            raise
    
    def transcribe(
        self,
        audio_path: Path,
        language: str = "no",
        beam_size: int = 5,
        word_timestamps: bool = True,
        vad_filter: bool = True,
        condition_on_previous_text: bool = True,
        initial_prompt: Optional[str] = None
    ) -> List[TranscriptionSegment]:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to audio file (should be preprocessed)
            language: Language code
            beam_size: Beam search width
            word_timestamps: Include word-level timestamps
            vad_filter: Filter silence before transcription
            condition_on_previous_text: Use context from previous segments
            initial_prompt: Optional prompt for vocabulary injection
            
        Returns:
            List of TranscriptionSegment objects
        """
        self._load_model()
        
        logger.info(f"Transcribing: {audio_path.name} with {self.model_name}")
        
        try:
            import whisperx
            
            # Load audio
            audio = whisperx.load_audio(str(audio_path))
            
            # Build transcription kwargs from config parameters
            transcribe_kwargs = {
                "language": language,
                "batch_size": 16,
                "verbose": False,
                "beam_size": beam_size,
                "condition_on_previous_text": condition_on_previous_text,
            }
            
            if initial_prompt:
                transcribe_kwargs["initial_prompt"] = initial_prompt
                logger.debug(f"Using initial_prompt ({len(initial_prompt)} chars)")
            
            if vad_filter:
                transcribe_kwargs["vad_filter"] = True
                logger.debug("VAD filtering enabled")
            
            # Transcribe
            result = self.model.transcribe(audio, **transcribe_kwargs)
            
            # Align for word-level timestamps if requested
            if word_timestamps:
                try:
                    logger.debug("Running alignment for word-level timestamps")
                    result = self.model.align(audio, result["segments"], language)
                except Exception as align_err:
                    logger.warning(f"Word-level alignment failed: {align_err}")
            
            # Convert to our format
            segments = []
            for i, seg in enumerate(result["segments"]):
                segment = TranscriptionSegment(
                    id=i,
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"].strip(),
                    words=seg.get("words"),
                    confidence=seg.get("confidence", 1.0)
                )
                segments.append(segment)
            
            logger.info(f"Transcription complete: {len(segments)} segments")
            
            return segments
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def align_with_diarization(
        self,
        segments: List[TranscriptionSegment],
        diarization_timeline: Dict[str, List[Tuple[float, float]]]
    ) -> List[TranscriptionSegment]:
        """
        Align transcription segments with speaker diarization.
        
        Args:
            segments: Transcribed segments
            diarization_timeline: Speaker timeline from diarize.py
            
        Returns:
            Updated segments with speaker labels
        """
        logger.info("Aligning transcription with diarization")
        
        for segment in segments:
            # Find which speaker(s) are active during this segment
            segment_start = segment.start
            segment_end = segment.end
            segment_mid = (segment_start + segment_end) / 2
            
            # Assign to speaker(s) overlapping with segment midpoint
            for speaker, time_ranges in diarization_timeline.items():
                for start, end in time_ranges:
                    if start <= segment_mid <= end:
                        segment.speaker = speaker
                        break
                if segment.speaker:
                    break
        
        # Log speaker distribution
        speaker_counts = {}
        for seg in segments:
            speaker = seg.speaker or "UNKNOWN"
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        
        logger.debug(f"Speaker distribution: {speaker_counts}")
        
        return segments


def transcribe_audio(
    file_path: Path,
    model_name: str = "NbAiLab/nb-whisper-large-verbatim",
    diarization_timeline: Optional[Dict[str, List[Tuple[float, float]]]] = None,
    audio_duration: Optional[float] = None,
    config: Optional[dict] = None,
    output_dir: Optional[Path] = None,
    output_format: str = "srt"
) -> Tuple[List[TranscriptionSegment], str]:
    """
    Transcribe audio file with optional diarization alignment.
    
    Args:
        file_path: Path to preprocessed audio file
        model_name: Transcription model identifier
        diarization_timeline: Speaker timeline from Step 3
        audio_duration: Duration for validation
        config: Configuration dict
        output_dir: Optional directory to save results
        output_format: Output format ("srt", "json", "vtt")
        
    Returns:
        Tuple of (segments, output_path_or_string)
    """
    if config is None:
        config = {}
    
    transcriber = Transcriber(model_name, config)
    
    # Get transcription config
    beam_size = config.get("beam_size", 5)
    word_timestamps = config.get("word_timestamps", True)
    vad_filter = config.get("vad_filter", True)
    language = config.get("language", "no")
    
    logger.info(f"Starting transcription: {file_path.name}")
    
    # Transcribe
    segments = transcriber.transcribe(
        file_path,
        language=language,
        beam_size=beam_size,
        word_timestamps=word_timestamps,
        vad_filter=vad_filter
    )
    
    # Align with diarization if available
    if diarization_timeline:
        segments = transcriber.align_with_diarization(segments, diarization_timeline)
    
    # Save results if output dir specified
    output_text = ""
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if output_format == "srt":
            output_text = _segments_to_srt(segments)
            output_path = output_dir / f"{file_path.stem}_{model_name.split('/')[-1]}.srt"
        elif output_format == "json":
            segments_json = [asdict(s) for s in segments]
            output_text = json.dumps(segments_json, ensure_ascii=False, indent=2)
            output_path = output_dir / f"{file_path.stem}_{model_name.split('/')[-1]}.json"
        elif output_format == "vtt":
            output_text = _segments_to_vtt(segments)
            output_path = output_dir / f"{file_path.stem}_{model_name.split('/')[-1]}.vtt"
        else:
            raise ValueError(f"Unknown output format: {output_format}")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_text)
        
        logger.info(f"Transcription saved to {output_path}")
        return segments, str(output_path)
    
    return segments, output_text


def _segments_to_srt(segments: List[TranscriptionSegment]) -> str:
    """Convert segments to SRT format."""
    lines = []
    
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_format_timestamp_srt(seg.start)} --> {_format_timestamp_srt(seg.end)}")
        
        # Include speaker label if available
        if seg.speaker:
            lines.append(f"{seg.speaker}")
        
        lines.append(seg.text)
        lines.append("")
    
    return "\n".join(lines)


def _segments_to_vtt(segments: List[TranscriptionSegment]) -> str:
    """Convert segments to WebVTT format."""
    lines = ["WEBVTT", ""]
    
    for seg in segments:
        lines.append(f"{_format_timestamp_vtt(seg.start)} --> {_format_timestamp_vtt(seg.end)}")
        
        if seg.speaker:
            lines.append(f"<v {seg.speaker}> {seg.text}")
        else:
            lines.append(seg.text)
        
        lines.append("")
    
    return "\n".join(lines)


def _format_timestamp_srt(seconds: float) -> str:
    """Format timestamp for SRT (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")


def _format_timestamp_vtt(seconds: float) -> str:
    """Format timestamp for WebVTT (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
