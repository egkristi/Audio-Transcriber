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
    avg_logprob: Optional[float] = None  # Decoder confidence from faster-whisper
    no_speech_prob: Optional[float] = None
    compression_ratio: Optional[float] = None
    temperature: Optional[float] = None
    confidence_level: float = 1.0  # Computed 0-1 aggregate confidence score
    
    def compute_confidence_level(self) -> float:
        """
        Compute an aggregate confidence level (0-1) from all available signals.
        
        Combines decoder signals into a single interpretable score:
        - avg_logprob: primary signal (typical range [-1, 0])
        - no_speech_prob: hallucination risk (higher = worse)
        - compression_ratio: repetition/looping (higher = worse)
        - temperature: fallback difficulty (higher = worse)
        - confidence: WhisperX-provided score if available
        
        Returns:
            float between 0.0 (very uncertain) and 1.0 (highly confident)
        """
        scores = []
        
        # 1. Base confidence from avg_logprob (primary signal)
        if self.avg_logprob is not None:
            # Map [-1.0, 0.0] to [0.0, 1.0]
            base = max(0.0, min(1.0, 1.0 + self.avg_logprob))
            scores.append(base)
        
        # 2. WhisperX confidence if available
        if self.confidence is not None and self.confidence < 1.0:
            scores.append(self.confidence)
        
        # 3. Penalize high no_speech_prob (hallucination risk)
        if self.no_speech_prob is not None:
            penalty = max(0.0, 1.0 - self.no_speech_prob)
            scores.append(penalty)
        
        # 4. Penalize high compression_ratio (repetition)
        if self.compression_ratio is not None:
            if self.compression_ratio > 3.0:
                scores.append(0.3)
            elif self.compression_ratio > 2.0:
                scores.append(0.6)
            else:
                scores.append(1.0)
        
        # 5. Penalize temperature fallback
        if self.temperature is not None and self.temperature > 0:
            scores.append(max(0.0, 1.0 - self.temperature * 0.3))
        
        if not scores:
            return 1.0
        
        # Use geometric mean for conservative scoring
        import math
        product = 1.0
        for s in scores:
            product *= s
        return product ** (1.0 / len(scores))


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
            import torch
            
            logger.info(f"Loading transcription model: {self.model_name}")
            
            # CTranslate2 (under faster-whisper/WhisperX) does NOT support MPS.
            # Only CUDA is supported for GPU; fallback to CPU.
            if torch.cuda.is_available():
                device = "cuda"
                compute_type = self.config.get("compute_type", "float16")
            else:
                device = "cpu"
                compute_type = self.config.get("compute_type", "int8")
            
            logger.info(f"Using device: {device} (compute_type={compute_type})")
            
            # Build asr_options from config for faster-whisper decoding parameters
            asr_options = {}
            if "beam_size" in self.config:
                asr_options["beam_size"] = self.config["beam_size"]
            if "condition_on_previous_text" in self.config:
                asr_options["condition_on_previous_text"] = self.config["condition_on_previous_text"]
            if "initial_prompt" in self.config:
                asr_options["initial_prompt"] = self.config["initial_prompt"]
            if "best_of" in self.config:
                asr_options["best_of"] = self.config["best_of"]
            if "patience" in self.config:
                asr_options["patience"] = self.config["patience"]
            if "length_penalty" in self.config:
                asr_options["length_penalty"] = self.config["length_penalty"]
            
            load_kwargs = {
                "device": device,
                "compute_type": compute_type,
                "language": "no",  # Norwegian
            }
            if asr_options:
                load_kwargs["asr_options"] = asr_options
                logger.debug(f"Using asr_options: {list(asr_options.keys())}")
            
            self.model = whisperx.load_model(self.model_name, **load_kwargs)
            
            logger.info(f"Model loaded on device: {device}")
            
        except Exception as e:
            logger.error(f"Failed to load transcription model: {e}")
            raise
    
    def _align_with_whisperx(
        self,
        audio: np.ndarray,
        segments: List[Dict],
        language: str = "no",
    ) -> Dict:
        """
        Fallback alignment using whisperx's standalone wav2vec2 alignment.
        
        FasterWhisperPipeline does not have an align() method, but whisperx
        provides load_align_model() + align() that works with any transcription
        output. This gives us word-level alignment scores for confidence extraction.
        
        Args:
            audio: Audio array from whisperx.load_audio()
            segments: List of segment dicts from transcription
            language: Language code (e.g., "no" for Norwegian Bokmål)
            
        Returns:
            Updated result dict with aligned word segments containing scores.
        """
        import whisperx
        import torch
        
        # Determine device
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        
        # Map language codes for alignment model lookup
        # "no" (Norwegian Bokmål) and "nn" (Nynorsk) both have dedicated models
        align_language = language
        if language == "no":
            align_language = "no"  # NbAiLab/nb-wav2vec2-1b-bokmaal-v2
        elif language == "nn":
            align_language = "nn"  # NbAiLab/nb-wav2vec2-1b-nynorsk
        
        logger.info(f"Loading alignment model for language: {align_language}")
        align_model, align_metadata = whisperx.load_align_model(
            language_code=align_language,
            device=device,
        )
        
        logger.debug("Running standalone alignment")
        aligned_result = whisperx.align(
            segments,
            align_model,
            align_metadata,
            audio,
            device,
            return_char_alignments=False,
        )
        
        # Merge alignment data back into the original segments.
        # The aligned segments have word-level "score" fields (acoustic confidence),
        # but may lack decoder signals (avg_logprob, no_speech_prob, etc.).
        # We merge the alignment words into the original segments to preserve
        # both decoder signals and alignment scores.
        aligned_segments = aligned_result["segments"]
        
        # Build a lookup by start time for alignment segments
        aligned_by_start = {}
        for aligned_seg in aligned_segments:
            key = round(aligned_seg.get("start", 0), 2)
            aligned_by_start[key] = aligned_seg
        
        merged_segments = []
        for orig_seg in segments:
            merged = dict(orig_seg)
            key = round(orig_seg.get("start", 0), 2)
            aligned_seg = aligned_by_start.get(key)
            if aligned_seg and "words" in aligned_seg:
                # Add aligned words with scores to the original segment
                merged["words"] = aligned_seg["words"]
            merged_segments.append(merged)
        
        logger.info(f"Alignment complete: {sum(1 for s in merged_segments if s.get('words'))} segments have word-level scores")
        return {"segments": merged_segments}
    
    def transcribe(
        self,
        audio_path: Path,
        language: str = "no",
        word_timestamps: bool = True,
    ) -> List[TranscriptionSegment]:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to audio file (should be preprocessed)
            language: Language code
            word_timestamps: Include word-level timestamps
            
        Returns:
            List of TranscriptionSegment objects
        """
        self._load_model()
        
        logger.info(f"Transcribing: {audio_path.name} with {self.model_name}")
        
        try:
            import whisperx
            
            # Load audio
            audio = whisperx.load_audio(str(audio_path))
            
            # Build transcription kwargs — only pass args accepted by FasterWhisperPipeline.transcribe()
            transcribe_kwargs = {
                "language": language,
                "batch_size": 16,
                "verbose": False,
            }
            
            # Transcribe
            result = self.model.transcribe(audio, **transcribe_kwargs)
            
            # Align for word-level timestamps if requested
            if word_timestamps:
                try:
                    logger.debug("Running alignment for word-level timestamps")
                    result = self.model.align(audio, result["segments"], language)
                except AttributeError:
                    # FasterWhisperPipeline does not have an align() method.
                    # Fall back to whisperx's standalone alignment with a wav2vec2 model.
                    logger.info("Model has no align() method — using whisperx standalone alignment")
                    try:
                        result = self._align_with_whisperx(audio, result["segments"], language)
                    except Exception as wx_align_err:
                        logger.warning(f"WhisperX standalone alignment also failed: {wx_align_err}")
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
                    confidence=seg.get("confidence", 1.0),
                    avg_logprob=seg.get("avg_logprob"),
                    no_speech_prob=seg.get("no_speech_prob"),
                    compression_ratio=seg.get("compression_ratio"),
                    temperature=seg.get("temperature"),
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
    word_timestamps = config.get("word_timestamps", True)
    language = config.get("language", "no")
    
    logger.info(f"Starting transcription: {file_path.name}")
    
    # Transcribe
    segments = transcriber.transcribe(
        file_path,
        language=language,
        word_timestamps=word_timestamps,
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
    """Convert segments to SRT format with confidence levels."""
    lines = []
    
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_format_timestamp_srt(seg.start)} --> {_format_timestamp_srt(seg.end)}")
        
        # Build subtitle text with speaker and confidence
        confidence_pct = int(seg.confidence_level * 100)
        confidence_label = ""
        if confidence_pct < 50:
            confidence_label = " [LOW CONFIDENCE]"
        elif confidence_pct < 70:
            confidence_label = " [MEDIUM CONFIDENCE]"
        
        # Include speaker label inline with text (valid SRT)
        if seg.speaker:
            lines.append(f"{seg.speaker}: {seg.text}{confidence_label}")
        else:
            lines.append(f"{seg.text}{confidence_label}")
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
