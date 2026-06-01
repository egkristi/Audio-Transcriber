#!/usr/bin/env python3
"""
Confidence Validation Script

Re-validates confidence priority scores against per-segment WER after
v0.1.37 hard-rule fixes. Compares before/after Spearman correlation.

Usage:
  uv run python scripts/validate_confidence.py \\
    --reference testdata/fasit1/fasit_clean.txt \\
    --srt output/fasit1_run_v9/Call recording\\ Håvard\\ Kristiansen_260524_172503/Call\\ recording\\ Håvard\\ Kristiansen_260524_172503_preprocessed_nb-whisper-large-verbatim.srt \\
    --review-json output/fasit1_run_v9/Call\\ recording\\ Håvard\\ Kristiansen_260524_172503/Call\\ recording\\ Håvard\\ Kristiansen_260524_172503_review_list.json \\
    --output output/confidence_validation_report.json
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.confidence import ConfidenceExtractor, SegmentConfidence
from src.utils import get_logger, save_json

logger = get_logger("validate_confidence")


def parse_srt(path: Path) -> List[Dict]:
    """Parse SRT file into list of segment dicts."""
    segments = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    blocks = content.strip().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        
        # Parse sequence number
        try:
            seq = int(lines[0].strip())
        except ValueError:
            continue
        
        # Parse timestamp
        time_match = re.match(r'(\d+:\d+:\d+,\d+)\s+-->\s+(\d+:\d+:\d+,\d+)', lines[1])
        if not time_match:
            continue
        
        start = _srt_time_to_seconds(time_match.group(1))
        end = _srt_time_to_seconds(time_match.group(2))
        
        # Parse text (may be multiple lines)
        text = " ".join(lines[2:]).strip()
        # Remove confidence labels like [MEDIUM CONFIDENCE]
        text = re.sub(r'\[.*?\]', '', text).strip()
        
        segments.append({
            "segment_id": seq - 1,  # 0-based
            "start": start,
            "end": end,
            "text": text,
        })
    
    return segments


def _srt_time_to_seconds(srt_time: str) -> float:
    """Convert SRT timestamp (HH:MM:SS,mmm) to seconds."""
    match = re.match(r'(\d+):(\d+):(\d+),(\d+)', srt_time)
    if not match:
        return 0.0
    h, m, s, ms = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
    return h * 3600 + m * 60 + s + ms / 1000


def load_fasit(path: Path) -> str:
    """Load fasit (ground truth) text."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def normalize_text(text: str) -> str:
    """Normalize text for WER comparison."""
    text = text.lower()
    text = re.sub(r"[^a-zæøå0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_per_segment_wer(
    segments: List[Dict],
    fasit_text: str,
) -> List[Dict]:
    """
    Compute per-segment WER by greedily aligning fasit words to segments.
    
    Since the fasit has no timestamps, we split it proportionally based on
    each segment's word count relative to the total hypothesis word count.
    Then we compute WER for each segment against its aligned fasit chunk.
    
    Returns segments with added 'wer', 'ref_words', 'hyp_words' fields.
    """
    try:
        import jiwer
    except ImportError:
        logger.error("jiwer is not installed. Run: uv add jiwer")
        sys.exit(1)
    
    fasit_norm = normalize_text(fasit_text)
    fasit_words = fasit_norm.split()
    total_fasit_words = len(fasit_words)
    
    # Get total hypothesis word count
    total_hyp_words = sum(len(normalize_text(s["text"]).split()) for s in segments)
    
    if total_hyp_words == 0:
        logger.error("No hypothesis words found")
        return segments
    
    # Greedy alignment: assign fasit words to segments proportionally
    fasit_idx = 0
    for seg in segments:
        hyp_words = len(normalize_text(seg["text"]).split())
        # Proportion of fasit words for this segment
        seg_fasit_count = max(1, round(hyp_words / total_hyp_words * total_fasit_words))
        
        # Get the fasit chunk for this segment
        seg_fasit_words = fasit_words[fasit_idx:fasit_idx + seg_fasit_count]
        fasit_idx += seg_fasit_count
        
        seg_fasit_text = " ".join(seg_fasit_words)
        seg_hyp_text = normalize_text(seg["text"])
        
        # Compute WER for this segment
        if seg_hyp_text and seg_fasit_text:
            wer = jiwer.wer(seg_fasit_text, seg_hyp_text)
            alignment = jiwer.process_words(seg_fasit_text, seg_hyp_text)
            seg["wer"] = round(wer, 4)
            seg["ref_words"] = len(seg_fasit_words)
            seg["hyp_words"] = len(seg_hyp_text.split())
            seg["substitutions"] = alignment.substitutions
            seg["deletions"] = alignment.deletions
            seg["insertions"] = alignment.insertions
            seg["hits"] = alignment.hits
        else:
            seg["wer"] = 1.0
            seg["ref_words"] = len(seg_fasit_words)
            seg["hyp_words"] = 0
            seg["substitutions"] = 0
            seg["deletions"] = len(seg_fasit_words)
            seg["insertions"] = 0
            seg["hits"] = 0
    
    return segments


def run_confidence_extractor(segments: List[Dict]) -> List[SegmentConfidence]:
    """Run the confidence extractor on segment data (simulating pipeline output)."""
    extractor = ConfidenceExtractor()
    
    # Build whisperx-style segment dicts
    whisperx_segments = []
    for seg in segments:
        whisperx_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "avg_logprob": seg.get("avg_logprob"),
            "no_speech_prob": seg.get("no_speech_prob"),
            "compression_ratio": seg.get("compression_ratio"),
            "temperature": seg.get("temperature"),
            "words": seg.get("words", []),
        })
    
    # Extract confidence signals
    confidence_segments = extractor.extract_from_whisperx(whisperx_segments)
    
    # Compute priority scores (this applies all hard-rules including v0.1.37 fixes)
    confidence_segments = extractor.compute_priority(confidence_segments)
    
    return confidence_segments


def compute_spearman(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Compute Spearman rank correlation coefficient."""
    try:
        from scipy.stats import spearmanr
        rho, p_value = spearmanr(x, y)
        return rho, p_value
    except ImportError:
        logger.warning("scipy not installed. Computing Spearman manually.")
        # Manual computation
        n = len(x)
        if n < 3:
            return 0.0, 1.0
        
        # Rank both lists
        x_ranked = _rank_data(x)
        y_ranked = _rank_data(y)
        
        # Difference in ranks
        d = [x_ranked[i] - y_ranked[i] for i in range(n)]
        d_squared = sum(di * di for di in d)
        
        # Spearman formula
        rho = 1 - (6 * d_squared) / (n * (n * n - 1))
        
        # Approximate p-value using t-distribution
        import math
        t = rho * math.sqrt((n - 2) / (1 - rho * rho)) if abs(rho) < 1 else float('inf')
        # Simple approximation for p-value
        p_value = 0.0 if abs(rho) > 0.5 else 0.5
        
        return rho, p_value


def _rank_data(data: List[float]) -> List[float]:
    """Rank data, handling ties with average ranking."""
    n = len(data)
    sorted_indices = sorted(range(n), key=lambda i: data[i])
    ranks = [0] * n
    
    i = 0
    while i < n:
        j = i
        # Find all ties
        while j < n and data[sorted_indices[j]] == data[sorted_indices[i]]:
            j += 1
        # Assign average rank
        avg_rank = (i + j + 1) / 2.0  # 1-based
        for k in range(i, j):
            ranks[sorted_indices[k]] = avg_rank
        i = j
    
    return ranks


def precision_at_k(
    priority_ranked: List[int],
    wer_ranked: List[int],
    k: int
) -> float:
    """Compute Precision@K: fraction of top-K by priority that are in top-K by WER."""
    top_k_priority = set(priority_ranked[:k])
    top_k_wer = set(wer_ranked[:k])
    if k == 0:
        return 0.0
    return len(top_k_priority & top_k_wer) / k


def main():
    parser = argparse.ArgumentParser(
        description="Validate confidence priority scores against per-segment WER"
    )
    parser.add_argument(
        "--reference", "-r",
        type=Path,
        required=True,
        help="Path to ground-truth fasit text file"
    )
    parser.add_argument(
        "--srt", "-s",
        type=Path,
        required=True,
        help="Path to hypothesis SRT file"
    )
    parser.add_argument(
        "--review-json", "-j",
        type=Path,
        required=True,
        help="Path to existing review_list.json (for old priority scores)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("output/confidence_validation_report.json"),
        help="Output path for validation report"
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading SRT: {args.srt}")
    segments = parse_srt(args.srt)
    logger.info(f"Loaded {len(segments)} segments from SRT")
    
    logger.info(f"Loading fasit: {args.reference}")
    fasit_text = load_fasit(args.reference)
    fasit_norm = normalize_text(fasit_text)
    logger.info(f"Fasit: {len(fasit_norm.split())} words")
    
    logger.info(f"Loading old review JSON: {args.review_json}")
    with open(args.review_json, "r", encoding="utf-8") as f:
        old_review = json.load(f)
    
    # Build old priority scores map
    old_priorities = {}
    old_flags = {}
    for seg_data in old_review.get("segments", []):
        sid = seg_data["segment_id"]
        old_priorities[sid] = seg_data["priority_score"]
        old_flags[sid] = seg_data["flags"]
    
    # Compute per-segment WER
    logger.info("Computing per-segment WER...")
    segments = compute_per_segment_wer(segments, fasit_text)
    
    # Report WER statistics
    wer_values = [s.get("wer", 1.0) for s in segments]
    logger.info(f"Per-segment WER: min={min(wer_values):.2%}, "
                f"max={max(wer_values):.2%}, "
                f"mean={sum(wer_values)/len(wer_values):.2%}")
    
    # Run confidence extractor with v0.1.37 fixes
    logger.info("Running confidence extractor (v0.1.37 fixes)...")
    new_confidence_segments = run_confidence_extractor(segments)
    
    # Build results
    results = []
    for cs in new_confidence_segments:
        sid = cs.segment_id
        seg_data = segments[sid]
        results.append({
            "segment_id": sid,
            "start": cs.start,
            "end": cs.end,
            "text_preview": cs.text[:100],
            "wer": seg_data.get("wer", 1.0),
            "ref_words": seg_data.get("ref_words", 0),
            "hyp_words": seg_data.get("hyp_words", 0),
            "old_priority_score": old_priorities.get(sid, 0.0),
            "new_priority_score": cs.priority_score,
            "new_priority_rank": cs.priority_rank,
            "old_flags": old_flags.get(sid, []),
            "new_flags": cs.flags,
            "new_flag_count": len(cs.flags),
        })
    
    # Sort by segment_id for the report
    results.sort(key=lambda r: r["segment_id"])
    
    # Compute correlations
    old_scores = [r["old_priority_score"] for r in results]
    new_scores = [r["new_priority_score"] for r in results]
    wers = [r["wer"] for r in results]
    
    old_rho, old_p = compute_spearman(old_scores, wers)
    new_rho, new_p = compute_spearman(new_scores, wers)
    
    logger.info(f"OLD Spearman ρ = {old_rho:.4f} (p = {old_p:.4f})")
    logger.info(f"NEW Spearman ρ = {new_rho:.4f} (p = {new_p:.4f})")
    
    # Precision@K
    # Rank by priority (descending) and by WER (descending)
    old_priority_ranked = sorted(results, key=lambda r: r["old_priority_score"], reverse=True)
    new_priority_ranked = sorted(results, key=lambda r: r["new_priority_score"], reverse=True)
    wer_ranked = sorted(results, key=lambda r: r["wer"], reverse=True)
    
    old_priority_ids = [r["segment_id"] for r in old_priority_ranked]
    new_priority_ids = [r["segment_id"] for r in new_priority_ranked]
    wer_ranked_ids = [r["segment_id"] for r in wer_ranked]
    
    precision_results = {}
    for k in [1, 3, 5, 10]:
        old_pk = precision_at_k(old_priority_ids, wer_ranked_ids, k)
        new_pk = precision_at_k(new_priority_ids, wer_ranked_ids, k)
        precision_results[f"Precision@{k}"] = {
            "old": round(old_pk, 4),
            "new": round(new_pk, 4),
        }
        logger.info(f"Precision@{k}: old={old_pk:.2%}, new={new_pk:.2%}")
    
    # Flag distribution analysis
    old_flag_counts = Counter()
    for flags in old_flags.values():
        for flag in flags:
            base = flag.split(":")[0]
            old_flag_counts[base] += 1
    
    new_flag_counts = Counter()
    for cs in new_confidence_segments:
        for flag in cs.flags:
            base = flag.split(":")[0]
            new_flag_counts[base] += 1
    
    total_segments = len(segments)
    
    # Priority histogram
    def priority_bucket(score: float) -> str:
        if score < 0.2:
            return "low"
        elif score < 0.4:
            return "medium-low"
        elif score < 0.6:
            return "medium"
        elif score < 0.8:
            return "medium-high"
        else:
            return "high"
    
    old_histogram = Counter(priority_bucket(s) for s in old_scores)
    new_histogram = Counter(priority_bucket(s) for s in new_scores)
    
    # Build report
    report = {
        "validation_date": "2026-06-01",
        "reference_file": str(args.reference),
        "hypothesis_file": str(args.srt),
        "total_segments": total_segments,
        "fasit_words": len(fasit_norm.split()),
        "overall_wer": round(sum(wers) / len(wers), 4) if wers else 0,
        "correlation": {
            "old_spearman_rho": round(old_rho, 4),
            "old_spearman_p": round(old_p, 4),
            "new_spearman_rho": round(new_rho, 4),
            "new_spearman_p": round(new_p, 4),
            "improvement": round(new_rho - old_rho, 4),
        },
        "precision_at_k": precision_results,
        "priority_histogram": {
            "old": dict(old_histogram),
            "new": dict(new_histogram),
        },
        "flag_distribution": {
            "old": dict(old_flag_counts.most_common()),
            "new": dict(new_flag_counts.most_common()),
        },
        "segments": results,
    }
    
    # Save report
    save_json(report, args.output)
    logger.info(f"Validation report saved to {args.output}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("CONFIDENCE VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total segments: {total_segments}")
    print(f"Fasit words: {len(fasit_norm.split())}")
    print(f"Mean per-segment WER: {sum(wers)/len(wers):.2%}")
    print()
    print(f"Spearman correlation (old): ρ = {old_rho:.4f} (p = {old_p:.4f})")
    print(f"Spearman correlation (new): ρ = {new_rho:.4f} (p = {new_p:.4f})")
    print(f"Improvement: {new_rho - old_rho:+.4f}")
    print()
    print("Precision@K:")
    for k, v in precision_results.items():
        print(f"  {k}: old={v['old']:.2%}, new={v['new']:.2%}")
    print()
    print("Priority histogram:")
    for bucket in ["low", "medium-low", "medium", "medium-high", "high"]:
        old_val = old_histogram.get(bucket, 0)
        new_val = new_histogram.get(bucket, 0)
        print(f"  {bucket}: old={old_val}, new={new_val}")
    print()
    print("Top 10 most common flags (old):")
    for flag, count in old_flag_counts.most_common(10):
        pct = count / total_segments * 100
        print(f"  {flag}: {count}/{total_segments} ({pct:.0f}%)")
    print()
    print("Top 10 most common flags (new):")
    for flag, count in new_flag_counts.most_common(10):
        pct = count / total_segments * 100
        print(f"  {flag}: {count}/{total_segments} ({pct:.0f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
