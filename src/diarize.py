"""
Step 3: Speaker Diarization

Identifies and separates different speakers in audio using pyannote speaker-diarization.
Results are reused across all transcription models (Step 3 & 4).
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .utils import get_logger, save_json

logger = get_logger("diarize")


@dataclass
class DiarizationSegment:
    """A speaker segment."""
    
    start: float  # Start time in seconds
    end: float  # End time in seconds
    speaker: str  # Speaker label (e.g., "SPEAKER_00", "SPEAKER_01")
    confidence: float  # Confidence score 0-1


class Diarizer:
    """Wrapper around pyannote speaker diarization."""
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize diarizer with config."""
        self.config = config or {}
        self.model = None
        self.device = "cpu"  # Default to CPU for Mac compatibility
        
    def _load_model(self):
        """Lazily load the diarization model."""
        if self.model is not None:
            return
        
        try:
            from pyannote.audio import Pipeline
            
            model_name = self.config.get(
                "model", "pyannote/speaker-diarization-3.1"
            )
            logger.info(f"Loading diarization model: {model_name}")
            
            self.model = Pipeline.from_pretrained(
                model_name,
                use_auth_token=True  # Requires Hugging Face login
            )
            
            # Move to device (CPU for Mac)
            self.model.to(self.device)
            logger.info(f"Model loaded on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load diarization model: {e}")
            raise
    
    def diarize(
        self,
        audio_path: Path,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        num_speakers: Optional[int] = None
    ) -> List[DiarizationSegment]:
        """
        Run speaker diarization on audio file.
        
        Args:
            audio_path: Path to audio file
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            num_speakers: Exact number of speakers (overrides min/max)
            
        Returns:
            List of DiarizationSegment objects
        """
        self._load_model()
        
        logger.info(f"Running diarization on: {audio_path.name}")
        
        # Get config values
        min_speakers = min_speakers or self.config.get("min_speakers")
        max_speakers = max_speakers or self.config.get("max_speakers")
        num_speakers = num_speakers or self.config.get("num_speakers_override")
        threshold = self.config.get("threshold", 0.5)
        
        try:
            # Prepare diarization kwargs
            diarization_kwargs = {}
            
            if num_speakers is not None:
                diarization_kwargs["num_speakers"] = num_speakers
            else:
                if min_speakers is not None:
                    diarization_kwargs["min_speakers"] = min_speakers
                if max_speakers is not None:
                    diarization_kwargs["max_speakers"] = max_speakers
            
            # Run diarization
            diarization = self.model(str(audio_path), **diarization_kwargs)
            
            # Convert to our format
            segments = []
            for segment, track, speaker in diarization.itertracks(yield_label=True):
                seg = DiarizationSegment(
                    start=segment.start,
                    end=segment.end,
                    speaker=speaker,
                    confidence=1.0  # pyannote doesn't provide per-segment confidence
                )
                segments.append(seg)
            
            logger.info(
                f"Diarization complete: {len(segments)} segments, "
                f"{len(set(s.speaker for s in segments))} speakers"
            )
            
            return segments
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            raise
    
    def get_speaker_timeline(
        self, 
        segments: List[DiarizationSegment],
        audio_duration: float
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Create speaker timeline: speaker -> list of (start, end) tuples.
        
        Args:
            segments: List of DiarizationSegment objects
            audio_duration: Total audio duration for validation
            
        Returns:
            Dict mapping speaker label to list of time ranges
        """
        timeline = {}
        
        for segment in segments:
            if segment.speaker not in timeline:
                timeline[segment.speaker] = []
            timeline[segment.speaker].append((segment.start, segment.end))
        
        logger.debug(f"Speaker timeline: {list(timeline.keys())}")
        return timeline


def diarize_audio(
    file_path: Path,
    audio_duration: float,
    config: Optional[dict] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    num_speakers: Optional[int] = None,
    output_dir: Optional[Path] = None
) -> Tuple[List[DiarizationSegment], Dict]:
    """
    Perform speaker diarization on audio file.
    
    Args:
        file_path: Path to audio file
        audio_duration: Duration of audio in seconds
        config: Configuration dict
        min_speakers: Minimum speakers
        max_speakers: Maximum speakers
        num_speakers: Exact number of speakers
        output_dir: Optional directory to save results
        
    Returns:
        Tuple of (segments, timeline)
    """
    logger.info(f"Starting diarization: {file_path.name}")
    
    if config is None:
        config = {}
    
    diarizer = Diarizer(config.get("diarization", {}))
    
    segments = diarizer.diarize(
        file_path,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        num_speakers=num_speakers
    )
    
    timeline = diarizer.get_speaker_timeline(segments, audio_duration)
    
    # Save results if output dir specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save segments as JSON
        segments_json = [asdict(s) for s in segments]
        segments_path = output_dir / f"{file_path.stem}_diarization.json"
        save_json({"segments": segments_json, "timeline": timeline}, segments_path)
        logger.info(f"Diarization results saved to {segments_path}")
    
    return segments, timeline
