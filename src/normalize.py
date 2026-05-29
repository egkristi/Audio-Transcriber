"""
Norwegian Text Normalization Module

Post-processes WhisperX transcription output to fix common errors
specific to Norwegian language transcription.

Common Whisper errors on Norwegian:
1. Character substitution: "aa" → "å", "ae" → "æ", "oe" → "ø"
2. Missing spaces after punctuation
3. Excessive repetition (stuttering)
4. English word substitution
5. Case issues (all lowercase segments)
6. Trailing/leading whitespace

This module is conservative: it flags issues for review rather than
auto-correcting, to avoid introducing new errors.
"""

import re
from typing import Dict, List, Optional, Tuple

from .utils import get_logger

logger = get_logger("normalize")


# Norwegian character mappings: common Whisper substitutions
NORWEGIAN_CHAR_MAP = {
    # Word-level substitutions (whole words)
    r'\baa\b': 'å',
    r'\bae\b': 'æ',
    r'\boe\b': 'ø',
    r'\bAa\b': 'Å',
    r'\bAe\b': 'Æ',
    r'\bOe\b': 'Ø',
}

# Common English words that Whisper substitutes for Norwegian
ENGLISH_TO_NORWEGIAN = {
    'the': 'den/det/de',
    'and': 'og',
    'you': 'du/dere',
    'that': 'at/som',
    'have': 'har',
    'for': 'for',
    'not': 'ikke',
    'with': 'med',
    'his': 'hans',
    'they': 'de/dem',
    'say': 'si/sier',
    'her': 'henne/hennes',
    'she': 'hun',
    'will': 'vil',
    'one': 'en/ett',
    'all': 'alle',
    'would': 'ville',
    'there': 'der',
    'their': 'deres',
    'what': 'hva',
    'about': 'om',
    'which': 'som/hvilken',
    'when': 'når',
    'make': 'lage',
    'like': 'liksom/som',
    'time': 'tid',
    'just': 'akkurat/bare',
    'know': 'vet/vite',
    'take': 'ta',
    'people': 'folk/mennesker',
    'year': 'år',
    'good': 'god/bra',
    'some': 'noen',
    'come': 'komme',
    'could': 'kunne',
    'state': 'tilstand/stat',
    'only': 'bare/kun',
    'other': 'andre',
    'new': 'ny',
    'may': 'kan/må',
    'way': 'vei/måte',
    'use': 'bruke',
    'than': 'enn',
    'first': 'først',
    'water': 'vann',
    'been': 'vært',
    'call': 'ringe/kalle',
    'who': 'hvem',
    'oil': 'olje',
    'its': 'dens/dets',
    'now': 'nå',
    'find': 'finne',
    'long': 'lang',
    'down': 'ned/nedover',
    'day': 'dag',
    'did': 'gjorde',
    'get': 'få',
    'has': 'har',
    'him': 'ham',
    'how': 'hvordan',
    'man': 'mann',
    'more': 'mer',
    'much': 'mye',
    'way': 'vei',
    'too': 'også/for',
    'very': 'veldig',
    'yes': 'ja',
    'ok': 'ok',
    'okay': 'ok',
    'hello': 'hei/hallo',
    'hi': 'hei',
    'bye': 'ha det',
    'thanks': 'takk',
    'please': 'vær så snill',
    'sorry': 'beklager',
}


def normalize_norwegian_text(text: str) -> Tuple[str, List[Dict]]:
    """
    Normalize Norwegian text and return corrections with explanations.
    
    Args:
        text: Raw transcription text
        
    Returns:
        Tuple of (normalized_text, list_of_corrections)
        Each correction dict has: original, corrected, position, type, explanation
    """
    corrections = []
    normalized = text
    offset = 0
    
    # 1. Fix missing spaces after punctuation
    # Pattern: punctuation followed immediately by letter
    punct_pattern = re.compile(r'([.,;:!?])([a-zA-ZæøåÆØÅ])')
    for match in punct_pattern.finditer(text):
        pos = match.start() + offset
        # Insert space
        normalized = normalized[:pos+1] + ' ' + normalized[pos+1:]
        offset += 1
        corrections.append({
            "original": match.group(0),
            "corrected": match.group(1) + ' ' + match.group(2),
            "position": match.start(),
            "type": "missing_space",
            "explanation": "Manglende mellomrom etter tegnsetting"
        })
    
    # 2. Fix multiple spaces
    normalized = re.sub(r'  +', ' ', normalized)
    
    # 3. Fix leading/trailing whitespace
    normalized = normalized.strip()
    
    # 4. Flag character substitutions (aa→å, ae→æ, oe→ø)
    # These are conservative: we flag them but don't auto-replace
    # because context matters (e.g., "Aage" is a name, not "Åge")
    for pattern, replacement in NORWEGIAN_CHAR_MAP.items():
        for match in re.finditer(pattern, normalized, re.IGNORECASE):
            corrections.append({
                "original": match.group(0),
                "corrected": replacement,
                "position": match.start(),
                "type": "char_substitution",
                "explanation": f"Mulig '{match.group(0)}' skal være '{replacement}'"
            })
    
    # 5. Flag English words
    words = normalized.lower().split()
    for i, word in enumerate(words):
        if word in ENGLISH_TO_NORWEGIAN:
            corrections.append({
                "original": word,
                "corrected": ENGLISH_TO_NORWEGIAN[word],
                "position": i,
                "type": "english_word",
                "explanation": f"Engelsk ord '{word}' — kanskje ment '{ENGLISH_TO_NORWEGIAN[word]}'?"
            })
    
    # 6. Flag excessive repetition
    word_list = normalized.lower().split()
    for word in set(word_list):
        count = word_list.count(word)
        if count >= 3:
            corrections.append({
                "original": word,
                "corrected": word,
                "position": -1,
                "type": "repetition",
                "explanation": f"Ordet '{word}' gjentas {count} ganger — mulig hallusinasjon"
            })
    
    # 7. Flag very short segments
    if len(words) < 3:
        corrections.append({
            "original": normalized,
            "corrected": normalized,
            "position": 0,
            "type": "short_segment",
            "explanation": f"Kun {len(words)} ord — sjekk at segmentet er komplett"
        })
    
    return normalized, corrections


def normalize_transcription_segments(segments: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Normalize all segments in a transcription and collect all corrections.
    
    Args:
        segments: List of segment dicts with 'text' field
        
    Returns:
        Tuple of (normalized_segments, all_corrections)
    """
    normalized_segments = []
    all_corrections = []
    
    for seg in segments:
        text = seg.get("text", "")
        normalized_text, corrections = normalize_norwegian_text(text)
        
        # Create normalized segment
        normalized_seg = dict(seg)
        normalized_seg["text"] = normalized_text
        normalized_seg["normalization_corrections"] = corrections
        normalized_seg["has_normalization_issues"] = len(corrections) > 0
        
        normalized_segments.append(normalized_seg)
        
        # Add segment info to corrections
        for corr in corrections:
            corr["segment_id"] = seg.get("id", -1)
            corr["segment_start"] = seg.get("start", 0)
            corr["segment_end"] = seg.get("end", 0)
            all_corrections.append(corr)
    
    if all_corrections:
        logger.info(f"Text normalization: {len(all_corrections)} issues flagged across {len(segments)} segments")
    
    return normalized_segments, all_corrections


def export_normalization_report(
    corrections: List[Dict],
    output_path: Path,
    segments: Optional[List[Dict]] = None
) -> Path:
    """
    Export a human-readable normalization report.
    
    Args:
        corrections: List of correction dicts
        output_path: Where to save the report
        segments: Optional original segments for context
        
    Returns:
        Path to exported report
    """
    lines = [
        "=== NORWEGIAN TEXT NORMALIZATION REPORT ===",
        f"Total issues flagged: {len(corrections)}",
        "",
    ]
    
    # Group by type
    by_type = {}
    for corr in corrections:
        t = corr["type"]
        by_type[t] = by_type.get(t, 0) + 1
    
    lines.append("=== ISSUE BREAKDOWN ===")
    for t, count in sorted(by_type.items(), key=lambda x: -x[1]):
        lines.append(f"  {t:20s}: {count}")
    lines.append("")
    
    # List all issues
    if corrections:
        lines.append("=== DETAILED ISSUES ===")
        for corr in corrections:
            lines.append(f"\nSegment {corr.get('segment_id', '?')} ({corr.get('segment_start', 0):.1f}s - {corr.get('segment_end', 0):.1f}s)")
            lines.append(f"  Type: {corr['type']}")
            lines.append(f"  Original:   {corr['original']}")
            lines.append(f"  Suggested:  {corr['corrected']}")
            lines.append(f"  Note: {corr['explanation']}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    logger.info(f"Normalization report exported to {output_path}")
    return output_path
