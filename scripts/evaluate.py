#!/usr/bin/env python3
"""
WER Evaluation Harness for Audio-Transcriber

Compares pipeline-generated transcripts against a ground-truth (manual) transcript
and computes Word Error Rate (WER), Character Error Rate (CER), and other metrics.

Usage:
  uv run python scripts/evaluate.py \
    --reference ground_truth.txt \
    --hypothesis output/recording/nb-whisper-large-verbatim.srt \
    --output report.json

The reference should be a plain text file with the perfect manual transcript.
The hypothesis can be an SRT, JSON, or plain text file.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

from src.utils import get_logger, save_json

logger = get_logger("evaluate")


def load_text(path: Path) -> str:
    """Load plain text from file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_srt(path: Path) -> str:
    """Extract text content from SRT file."""
    text_lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines, sequence numbers, and timestamp lines
            if not line:
                continue
            if line.isdigit():
                continue
            if " --> " in line:
                continue
            # Skip speaker labels like SPEAKER_00
            if line.startswith("SPEAKER_"):
                continue
            text_lines.append(line)
    return " ".join(text_lines)


def load_json_transcript(path: Path) -> str:
    """Extract text from JSON transcript (list of segments)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        segments = data
    elif isinstance(data, dict) and "segments" in data:
        segments = data["segments"]
    else:
        raise ValueError(f"Unexpected JSON structure in {path}")
    texts = [seg.get("text", "").strip() for seg in segments]
    return " ".join(texts)


def normalize_text(text: str) -> str:
    """
    Normalize text for fair comparison.
    
    - Lowercase
    - Remove punctuation except apostrophes (Norwegian contractions)
    - Collapse multiple spaces
    """
    text = text.lower()
    # Keep letters, numbers, spaces, and apostrophes
    text = re.sub(r"[^a-zæøå0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def evaluate(
    reference_path: Path,
    hypothesis_path: Path,
    output_path: Optional[Path] = None,
) -> dict:
    """
    Compute WER, CER, and other metrics between reference and hypothesis.
    
    Returns a dict with all metrics and optionally saves to JSON.
    """
    try:
        import jiwer
    except ImportError:
        logger.error("jiwer is not installed. Run: uv add jiwer")
        sys.exit(1)

    # Load reference
    ref_text = load_text(reference_path)
    ref_norm = normalize_text(ref_text)

    # Load hypothesis based on file extension
    suffix = hypothesis_path.suffix.lower()
    if suffix == ".srt":
        hyp_text = load_srt(hypothesis_path)
    elif suffix == ".json":
        hyp_text = load_json_transcript(hypothesis_path)
    elif suffix in (".txt", ".md"):
        hyp_text = load_text(hypothesis_path)
    else:
        logger.warning(f"Unknown extension {suffix}, treating as plain text")
        hyp_text = load_text(hypothesis_path)

    hyp_norm = normalize_text(hyp_text)

    # Compute metrics
    wer = jiwer.wer(ref_norm, hyp_norm)
    cer = jiwer.cer(ref_norm, hyp_norm)
    mer = jiwer.mer(ref_norm, hyp_norm)
    wil = jiwer.wil(ref_norm, hyp_norm)
    wip = jiwer.wip(ref_norm, hyp_norm)

    # Word counts
    ref_words = len(ref_norm.split())
    hyp_words = len(hyp_norm.split())

    # Detailed alignment
    alignment = jiwer.process_words(ref_norm, hyp_norm)
    substitutions = len(alignment.substitutions)
    deletions = len(alignment.deletions)
    insertions = len(alignment.insertions)
    hits = len(alignment.hits)

    results = {
        "reference_file": str(reference_path),
        "hypothesis_file": str(hypothesis_path),
        "reference_words": ref_words,
        "hypothesis_words": hyp_words,
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "mer": round(mer, 4),
        "wil": round(wil, 4),
        "wip": round(wip, 4),
        "errors": {
            "substitutions": substitutions,
            "deletions": deletions,
            "insertions": insertions,
            "hits": hits,
        },
        "reference_text_preview": ref_norm[:200],
        "hypothesis_text_preview": hyp_norm[:200],
    }

    logger.info(f"WER: {wer:.2%} | CER: {cer:.2%} | Words: {ref_words}")
    logger.info(f"Errors: {substitutions} sub, {deletions} del, {insertions} ins, {hits} hit")

    if output_path:
        save_json(results, output_path)
        logger.info(f"Evaluation report saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate transcription quality against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate SRT against ground truth
  uv run python scripts/evaluate.py \\
    --reference ground_truth.txt \\
    --hypothesis output/recording.srt

  # Save report to JSON
  uv run python scripts/evaluate.py \\
    --reference ground_truth.txt \\
    --hypothesis output/recording.srt \\
    --output report.json
        """
    )
    parser.add_argument(
        "--reference", "-r",
        type=Path,
        required=True,
        help="Path to ground-truth reference text file"
    )
    parser.add_argument(
        "--hypothesis", "-hyp",
        type=Path,
        required=True,
        help="Path to hypothesis transcript (SRT, JSON, or text)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Optional JSON output path for detailed report"
    )

    args = parser.parse_args()

    if not args.reference.exists():
        logger.error(f"Reference file not found: {args.reference}")
        sys.exit(1)
    if not args.hypothesis.exists():
        logger.error(f"Hypothesis file not found: {args.hypothesis}")
        sys.exit(1)

    evaluate(args.reference, args.hypothesis, args.output)


if __name__ == "__main__":
    main()
