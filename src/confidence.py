"""
Confidence Flagging Module

Extracts multiple deterministic signals from transcription output to prioritize
segments for manual review. Signals are ranked by usefulness:

1. WhisperX alignment score (acoustic "text vs audio" confidence)
2. faster-whisper decoder signals (avg_logprob, word.probability, no_speech_prob,
   compression_ratio, temperature)
3. Cross-model disagreement (from compare.py)
4. Acoustic features from analyze.py (SNR, VAD overlap)

Phase A (current): Unweighted normalized priority score for ranking.
Phase B (future): Calibrate against ground-truth using logistic regression.

Honest limitation: Confidence-flagging catches "model knew it was uncertain"
errors but misses "confidently wrong" errors — especially plausible substitutions
of names and numbers. These get high decoder confidence because they are
linguistically plausible. Therefore: confidence is a supplement, not a
replacement. Proper nouns and numbers should be reviewed regardless of score.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .utils import get_logger, save_json

logger = get_logger("confidence")


@dataclass
class SegmentConfidence:
    """Confidence signals for a single transcription segment."""
    
    segment_id: int
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    
    # WhisperX alignment signals (acoustic confidence)
    alignment_score: Optional[float] = None  # wav2vec2 word alignment score
    min_word_alignment_score: Optional[float] = None  # worst word in segment
    
    # faster-whisper decoder signals
    avg_logprob: Optional[float] = None        # segment-level decoder confidence
    no_speech_prob: Optional[float] = None     # high = likely hallucination
    compression_ratio: Optional[float] = None  # high = repetition/looping
    temperature: Optional[float] = None        # fallback = difficult segment
    min_word_probability: Optional[float] = None  # worst word probability in segment
    
    # Cross-model disagreement (from compare.py)
    model_disagreement: Optional[float] = None  # 0-1, how much models differ
    
    # Acoustic features (from analyze.py)
    snr_db: Optional[float] = None             # signal-to-noise ratio
    vad_overlap: Optional[float] = None        # simultaneous speech overlap
    
    # Computed priority
    priority_score: float = 0.0                # higher = review first
    priority_rank: int = 0
    flags: List[str] = field(default_factory=list)


class ConfidenceExtractor:
    """Extract confidence signals from transcription output."""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.alignment_threshold = self.config.get("alignment_threshold", 0.6)
        self.logprob_threshold = self.config.get("logprob_threshold", -0.5)
        self.no_speech_threshold = self.config.get("no_speech_threshold", 0.5)
        self.compression_threshold = self.config.get("compression_threshold", 2.4)
    
    def extract_from_whisperx(
        self,
        segments: List[Dict],
        aligned_word_segments: Optional[List[Dict]] = None
    ) -> List[SegmentConfidence]:
        """
        Extract confidence signals from WhisperX transcription output.
        
        Args:
            segments: List of segment dicts from whisperx transcribe()
            aligned_word_segments: Optional word-level alignment from whisperx align()
        
        Returns:
            List of SegmentConfidence objects with signals populated.
        """
        results = []
        
        for i, seg in enumerate(segments):
            sc = SegmentConfidence(
                segment_id=i,
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                text=seg.get("text", "").strip(),
                speaker=seg.get("speaker"),
            )
            
            # Decoder signals from faster-whisper/whisperx
            sc.avg_logprob = seg.get("avg_logprob")
            sc.no_speech_prob = seg.get("no_speech_prob")
            sc.compression_ratio = seg.get("compression_ratio")
            sc.temperature = seg.get("temperature")
            
            # Word-level probabilities
            words = seg.get("words", [])
            if words:
                probs = [w.get("probability", 1.0) for w in words if "probability" in w]
                if probs:
                    sc.min_word_probability = min(probs)
            
            # Alignment scores from whisperx align()
            if aligned_word_segments:
                segment_words = [
                    w for w in aligned_word_segments
                    if sc.start <= w.get("start", 0) <= sc.end
                ]
                if segment_words:
                    scores = [w.get("score", 1.0) for w in segment_words]
                    sc.alignment_score = np.mean(scores)
                    sc.min_word_alignment_score = min(scores)
            
            results.append(sc)
        
        return results
    
    def add_model_disagreement(
        self,
        confidence_segments: List[SegmentConfidence],
        comparison_results: List[Dict]
    ) -> List[SegmentConfidence]:
        """
        Add cross-model disagreement signal from compare.py output.
        
        Args:
            confidence_segments: List of SegmentConfidence objects
            comparison_results: List of comparison result dicts from compare.py
        
        Returns:
            Updated SegmentConfidence objects with disagreement signal.
        """
        for cs in confidence_segments:
            for cr in comparison_results:
                if cr.get("segment_id") == cs.segment_id:
                    cs.model_disagreement = 1.0 - cr.get("similarity_score", 1.0)
                    break
        return confidence_segments
    
    def add_acoustic_features(
        self,
        confidence_segments: List[SegmentConfidence],
        metadata: Optional[Dict] = None
    ) -> List[SegmentConfidence]:
        """
        Add acoustic features from analyze.py metadata.
        
        Args:
            confidence_segments: List of SegmentConfidence objects
            metadata: AudioMetadata dict with SNR, VAD info, etc.
        
        Returns:
            Updated SegmentConfidence objects with acoustic signals.
        """
        if not metadata:
            return confidence_segments
        
        # Global SNR (if available in metadata)
        snr = metadata.get("snr_db")
        
        for cs in confidence_segments:
            cs.snr_db = snr
            # VAD overlap would require per-segment VAD analysis
            # For now, placeholder
            cs.vad_overlap = None
        
        return confidence_segments
    
    def compute_priority(
        self,
        segments: List[SegmentConfidence]
    ) -> List[SegmentConfidence]:
        """
        Compute priority scores for all segments.
        
        Phase A: Unweighted normalized sum of inverted signals.
        Higher score = higher priority for review.
        
        Args:
            segments: List of SegmentConfidence objects with signals populated.
        
        Returns:
            Same segments with priority_score and priority_rank set.
        """
        for seg in segments:
            flags = []
            scores = []
            
            # 1. Alignment score (acoustic confidence) — lower is worse
            if seg.alignment_score is not None:
                scores.append(1.0 - seg.alignment_score)
                if seg.alignment_score < self.alignment_threshold:
                    flags.append("low_alignment")
            
            if seg.min_word_alignment_score is not None:
                scores.append(1.0 - seg.min_word_alignment_score)
                if seg.min_word_alignment_score < self.alignment_threshold:
                    flags.append("low_word_alignment")
            
            # 2. Decoder confidence — lower avg_logprob is worse
            if seg.avg_logprob is not None:
                # Normalize: typical range [-1, 0], map to [0, 1]
                normalized = max(0.0, min(1.0, -seg.avg_logprob))
                scores.append(normalized)
                if seg.avg_logprob < self.logprob_threshold:
                    flags.append("low_logprob")
            
            # 3. no_speech_prob — high = hallucination risk
            if seg.no_speech_prob is not None:
                scores.append(seg.no_speech_prob)
                if seg.no_speech_prob > self.no_speech_threshold:
                    flags.append("high_no_speech_prob")
            
            # 4. compression_ratio — high = repetition/looping
            if seg.compression_ratio is not None:
                cr_score = max(0.0, seg.compression_ratio - self.compression_threshold)
                scores.append(cr_score)
                if seg.compression_ratio > self.compression_threshold:
                    flags.append("high_compression")
            
            # 5. Temperature — higher = more difficult
            if seg.temperature is not None and seg.temperature > 0:
                scores.append(seg.temperature / 1.0)  # normalize
                if seg.temperature > 0:
                    flags.append("temperature_fallback")
            
            # 6. Word probability — lower = worse
            if seg.min_word_probability is not None:
                scores.append(1.0 - seg.min_word_probability)
                if seg.min_word_probability < 0.5:
                    flags.append("low_word_prob")
            
            # 7. Model disagreement — higher = worse
            if seg.model_disagreement is not None:
                scores.append(seg.model_disagreement)
                if seg.model_disagreement > 0.1:
                    flags.append("model_disagreement")
            
            # 8. SNR — lower = worse (noisy audio)
            if seg.snr_db is not None:
                # Map SNR: < 10 dB = bad, > 30 dB = good
                snr_score = max(0.0, (30.0 - seg.snr_db) / 20.0)
                scores.append(snr_score)
                if seg.snr_db < 10.0:
                    flags.append("low_snr")
            
            # Compute unweighted average priority
            if scores:
                seg.priority_score = float(np.mean(scores))
            else:
                seg.priority_score = 0.0
            
            seg.flags = flags
        
        # Sort by priority descending
        segments.sort(key=lambda s: s.priority_score, reverse=True)
        for rank, seg in enumerate(segments, 1):
            seg.priority_rank = rank
        
        return segments
    
    def export_review_list(
        self,
        segments: List[SegmentConfidence],
        output_path: Path,
        top_n: Optional[int] = None
    ) -> Path:
        """
        Export a prioritized review list for manual correction.
        
        Args:
            segments: List of SegmentConfidence with priority scores
            output_path: Where to save the review list
            top_n: Only export top N segments (None = all)
        
        Returns:
            Path to exported review list
        """
        if top_n:
            segments = segments[:top_n]
        
        lines = [
            "=== PRIORITIZED REVIEW LIST ===",
            f"Total segments: {len(segments)}",
            "",
            "Review order: highest priority first",
            "",
        ]
        
        for seg in segments:
            lines.append(f"Rank {seg.priority_rank} | Priority: {seg.priority_score:.3f}")
            lines.append(f"Time: {seg.start:.1f}s - {seg.end:.1f}s")
            if seg.speaker:
                lines.append(f"Speaker: {seg.speaker}")
            lines.append(f"Text: {seg.text}")
            if seg.flags:
                lines.append(f"Flags: {', '.join(seg.flags)}")
            lines.append("")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        logger.info(f"Review list exported to {output_path}")
        return output_path


def extract_confidence_signals(
    segments: List[Dict],
    aligned_word_segments: Optional[List[Dict]] = None,
    comparison_results: Optional[List[Dict]] = None,
    metadata: Optional[Dict] = None,
    config: Optional[dict] = None,
) -> List[SegmentConfidence]:
    """
    Convenience function: extract all confidence signals and compute priorities.
    
    Args:
        segments: WhisperX transcription segments
        aligned_word_segments: Optional word-level alignment from whisperx align()
        comparison_results: Optional comparison results from compare.py
        metadata: Optional audio metadata from analyze.py
        config: Optional configuration dict
    
    Returns:
        List of SegmentConfidence with priority scores computed.
    """
    extractor = ConfidenceExtractor(config)
    
    # Step 1: Extract from transcription
    confidence_segments = extractor.extract_from_whisperx(segments, aligned_word_segments)
    
    # Step 2: Add cross-model disagreement
    if comparison_results:
        confidence_segments = extractor.add_model_disagreement(
            confidence_segments, comparison_results
        )
    
    # Step 3: Add acoustic features
    if metadata:
        confidence_segments = extractor.add_acoustic_features(
            confidence_segments, metadata
        )
    
    # Step 4: Compute priorities
    confidence_segments = extractor.compute_priority(confidence_segments)
    
    return confidence_segments
