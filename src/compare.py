"""
Step 5: Model Comparison and Deviation Marking

Compares transcriptions from multiple models and flags low-confidence segments
and disagreements for manual review.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
from difflib import SequenceMatcher

from .utils import get_logger, save_json
from .transcribe import TranscriptionSegment

logger = get_logger("compare")


@dataclass
class ComparisonResult:
    """Result of comparing two transcription versions."""
    
    segment_id: int
    start: float
    end: float
    text_primary: str
    text_secondary: str
    primary_confidence: float
    secondary_confidence: float
    
    # Comparison metrics
    similarity_score: float  # 0-1, how similar the texts are
    has_disagreement: bool
    has_low_confidence: bool  # Either model has confidence < threshold
    priority: str  # "high", "medium", "low"
    flags: List[str]  # List of issues found


class TranscriptionComparer:
    """Compare transcriptions from different models."""
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize comparer with config."""
        self.config = config or {}
        self.low_confidence_threshold = self.config.get("low_confidence_threshold", 0.85)
        self.agreement_threshold = self.config.get("min_agreement_threshold", 0.95)
    
    def align_segments(
        self,
        primary: List[TranscriptionSegment],
        secondary: List[TranscriptionSegment]
    ) -> List[Tuple[Optional[TranscriptionSegment], Optional[TranscriptionSegment]]]:
        """
        Align segments from two transcriptions based on time overlap.
        
        Returns list of (primary_segment, secondary_segment) tuples.
        """
        aligned = []
        
        for p_seg in primary:
            # Find best matching secondary segment based on time overlap
            best_overlap = 0
            best_s_seg = None
            
            for s_seg in secondary:
                # Calculate time overlap
                overlap_start = max(p_seg.start, s_seg.start)
                overlap_end = min(p_seg.end, s_seg.end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_s_seg = s_seg
            
            # Only consider it a match if overlap > 50%
            p_duration = p_seg.end - p_seg.start
            if best_overlap > p_duration * 0.5:
                aligned.append((p_seg, best_s_seg))
            else:
                aligned.append((p_seg, None))
        
        return aligned
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using SequenceMatcher.
        
        Returns score between 0 and 1.
        """
        if not text1 or not text2:
            return 1.0 if text1 == text2 else 0.0
        
        # Normalize: lowercase and remove punctuation
        text1_norm = text1.lower().replace(".", "").replace(",", "")
        text2_norm = text2.lower().replace(".", "").replace(",", "")
        
        matcher = SequenceMatcher(None, text1_norm, text2_norm)
        return matcher.ratio()
    
    def compare_segments(
        self,
        primary: TranscriptionSegment,
        secondary: Optional[TranscriptionSegment]
    ) -> ComparisonResult:
        """
        Compare two transcription segments and produce comparison result.
        """
        flags = []
        priority = "low"
        has_disagreement = False
        has_low_conf = False
        similarity = 1.0
        
        # Check for missing secondary segment
        if secondary is None:
            flags.append("missing_secondary")
            priority = "high"
            has_disagreement = True
            similarity = 0.0
        else:
            # Compare text
            similarity = self.calculate_similarity(primary.text, secondary.text)
            
            if similarity < self.agreement_threshold:
                flags.append("disagreement")
                has_disagreement = True
                priority = "high"
            
            # Check confidences
            if (primary.confidence < self.low_confidence_threshold or
                secondary.confidence < self.low_confidence_threshold):
                flags.append("low_confidence")
                has_low_conf = True
                if priority == "low":
                    priority = "medium"
        
        # Check primary model confidence
        if primary.confidence < self.low_confidence_threshold:
            flags.append("primary_low_confidence")
            has_low_conf = True
            if priority == "low":
                priority = "medium"
        
        result = ComparisonResult(
            segment_id=primary.id,
            start=primary.start,
            end=primary.end,
            text_primary=primary.text,
            text_secondary=secondary.text if secondary else "",
            primary_confidence=primary.confidence,
            secondary_confidence=secondary.confidence if secondary else 0.0,
            similarity_score=similarity,
            has_disagreement=has_disagreement,
            has_low_confidence=has_low_conf,
            priority=priority,
            flags=flags
        )
        
        return result
    
    def compare_transcriptions(
        self,
        primary: List[TranscriptionSegment],
        secondary: List[TranscriptionSegment]
    ) -> List[ComparisonResult]:
        """
        Compare two full transcriptions.
        
        Returns list of ComparisonResult objects, with high-priority items first.
        """
        logger.info(f"Comparing transcriptions: {len(primary)} vs {len(secondary)} segments")
        
        aligned = self.align_segments(primary, secondary)
        results = []
        
        for p_seg, s_seg in aligned:
            if p_seg is not None:
                result = self.compare_segments(p_seg, s_seg)
                results.append(result)
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        results.sort(key=lambda r: priority_order.get(r.priority, 3))
        
        # Log summary
        high_priority = sum(1 for r in results if r.priority == "high")
        medium_priority = sum(1 for r in results if r.priority == "medium")
        low_priority = sum(1 for r in results if r.priority == "low")
        
        logger.info(
            f"Comparison complete: "
            f"{high_priority} high, {medium_priority} medium, {low_priority} low priority"
        )
        
        return results


def compare_transcriptions(
    primary_segments: List[TranscriptionSegment],
    secondary_segments: List[TranscriptionSegment],
    config: Optional[dict] = None,
    output_dir: Optional[Path] = None
) -> Tuple[List[ComparisonResult], str]:
    """
    Compare two transcriptions and mark deviations.
    
    Args:
        primary_segments: Transcription from primary model
        secondary_segments: Transcription from secondary model
        config: Configuration dict
        output_dir: Optional directory to save comparison results
        
    Returns:
        Tuple of (comparison_results, output_path_or_string)
    """
    if config is None:
        config = {}
    
    logger.info("Starting transcription comparison")
    
    comparer = TranscriptionComparer(config.get("comparison", {}))
    results = comparer.compare_transcriptions(primary_segments, secondary_segments)
    
    output_text = ""
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed JSON
        results_json = [asdict(r) for r in results]
        
        comparison_data = {
            "total_segments": len(results),
            "high_priority_count": sum(1 for r in results if r.priority == "high"),
            "medium_priority_count": sum(1 for r in results if r.priority == "medium"),
            "low_priority_count": sum(1 for r in results if r.priority == "low"),
            "results": results_json
        }
        
        output_path = output_dir / "comparison_results.json"
        save_json(comparison_data, output_path)
        logger.info(f"Comparison results saved to {output_path}")
        
        # Also create a human-readable report
        report_path = output_dir / "comparison_report.txt"
        report_text = _create_comparison_report(results)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        logger.info(f"Comparison report saved to {report_path}")
        
        return results, str(output_path)
    
    return results, output_text


def _create_comparison_report(results: List[ComparisonResult]) -> str:
    """Create a human-readable comparison report."""
    lines = ["=== TRANSCRIPTION COMPARISON REPORT ===", ""]
    
    # Summary
    total = len(results)
    high = sum(1 for r in results if r.priority == "high")
    medium = sum(1 for r in results if r.priority == "medium")
    low = sum(1 for r in results if r.priority == "low")
    
    lines.append(f"Total segments: {total}")
    lines.append(f"  - High priority (needs review): {high}")
    lines.append(f"  - Medium priority: {medium}")
    lines.append(f"  - Low priority: {low}")
    lines.append("")
    
    # High priority items
    lines.append("=== HIGH PRIORITY ITEMS ===")
    for r in results:
        if r.priority == "high":
            lines.append(f"\nSegment {r.segment_id} ({r.start:.1f}s - {r.end:.1f}s)")
            lines.append(f"  Flags: {', '.join(r.flags)}")
            lines.append(f"  Primary:   {r.text_primary}")
            if r.text_secondary:
                lines.append(f"  Secondary: {r.text_secondary}")
            lines.append(f"  Similarity: {r.similarity_score:.0%}")
            lines.append(f"  Confidence: primary={r.primary_confidence:.2f}, secondary={r.secondary_confidence:.2f}")
    
    lines.append("")
    lines.append("=== LEGEND ===")
    lines.append("- disagreement: Models produced different text")
    lines.append("- low_confidence: Model confidence below threshold")
    lines.append("- missing_secondary: Secondary model missing this segment")
    
    return "\n".join(lines)
