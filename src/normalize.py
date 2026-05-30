"""
Norwegian Text Normalization Module

Post-processes WhisperX transcription output to fix common errors
specific to Norwegian language transcription, with awareness of
Northern Norwegian dialects (Nordland, Troms, Finnmark).

Common Whisper errors on Norwegian:
1. No punctuation — verbatim model outputs stream of lowercase words
2. Stuttering — repeated words ("jeg jeg", "kan kan")
3. Character substitution: "aa" → "å", "ae" → "æ", "oe" → "ø"
4. Missing spaces after punctuation
5. English word substitution
6. Case issues (all lowercase segments)
7. Trailing/leading whitespace
8. Dialect confusion — Whisper may normalize Northern Norwegian
   dialect words to standard Eastern Norwegian (e.g., "æ" → "jeg",
   "ikkje" → "ikke", "ka" → "hva")

This module has two modes:
- Conservative (default): flags issues for review, auto-fixes only
  punctuation, capitalization, and stuttering
- Aggressive (auto_correct=True): also applies character substitutions,
  English word replacements, and dialect normalization

Dialect notes:
- The target audio is Northern Norwegian (Nordland, Troms, Finnmark)
- Common dialect features: "æ" (jeg), "ikkje" (ikke), "ka" (hva),
  "kor" (hvor), "mæ" (meg), "dæ" (deg), "sæ" (seg), "dokker" (dere),
  "no" (noe), "bærre" (bare), "itte" (ikke, some areas)
- Whisper's nb-whisper-large-verbatim model may output either dialect
  or standard forms depending on training data
- The module flags dialect-standard mismatches but does NOT auto-correct
  them — dialect is valid Norwegian and should be preserved
"""

import re
from pathlib import Path
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

# Norwegian filler words that typically end a sentence or clause
# These are used for punctuation restoration
SENTENCE_END_FILLERS = {'ja', 'nei', 'da', 'hæ'}
CLAUSE_BREAK_WORDS = {'så', 'men', 'for', 'og', 'at'}

# Northern Norwegian dialect word mappings.
# Whisper may output either dialect or standard forms. These mappings
# help flag when the model mixes forms inconsistently.
# Format: dialect_word -> standard_equivalent (for flagging only)
# NOTE: Dialect is valid Norwegian. These are NOT auto-corrected.
NORWEGIAN_DIALECT_MAP = {
    # First person pronoun
    'æ': 'jeg',
    'eg': 'jeg',
    'e': 'jeg',
    'je': 'jeg',
    # Negation
    'ikkje': 'ikke',
    'itte': 'ikke',
    'ikke': 'ikke',  # standard, included for completeness
    # Question words
    'ka': 'hva',
    'kæ': 'hva',
    'kor': 'hvor',
    'korsn': 'hvordan',
    'kordan': 'hvordan',
    'koffer': 'hvorfor',
    'koffor': 'hvorfor',
    # Pronouns
    'mæ': 'meg',
    'dæ': 'deg',
    'sæ': 'seg',
    'dokker': 'dere',
    'dåkker': 'dere',
    'dokkeres': 'deres',
    'dåkkeres': 'deres',
    'han': 'ham',  # "han" used as object in dialect
    'ho': 'hun',
    'hu': 'hun',
    # Adverbs and other
    'no': 'noe',
    'nåkkå': 'noe',
    'nokka': 'noe',
    'bærre': 'bare',
    'berre': 'bare',
    'sæ': 'seg',
    'nån': 'noen',
    'någen': 'noen',
    'mykje': 'mye',
    'mye': 'mye',  # standard
    'lite': 'lite',  # standard
    'lita': 'liten',
    'ille': 'ikke',  # some areas
}

# Norwegian words that should always be capitalized (places, public entities)
# NOTE: Real personal names have been removed from committed source (#36).
# Load personal names from a local gitignored data file (data/proper_nouns.json)
# using load_proper_nouns() if needed.
NORWEGIAN_PROPER_NOUNS = {
    # Northern Norwegian place names (public information)
    'nordland', 'troms', 'finnmark', 'tromsø', 'bodø', 'narvik', 'harstad',
    'hammerfest', 'alta', 'vadsø', 'kirkenes', 'mo i rana', 'mosjøen',
    'fauske', 'sortland', 'stokmarknes', 'leknes', 'svolvær', 'andøya',
    'senja', 'kvaløya', 'ringvassøya', 'lyngen', 'skjervøy', 'storslett',
    'karasjok', 'kautokeino', 'lakselv', 'honningsvåg', 'mehamn',
    'berlevåg', 'båtsfjord', 'vardø', 'vadsø', 'nesna', 'hemnes',
    'rana', 'beiarn', 'saltdal', 'steigen', 'hamarøy', 'tysfjord',
    'lødingen', 'evenes', 'skånland', 'bjarkøy', 'kvæfjord', 'dyrøy',
    'sørreisa', 'målselv', 'bardu', 'salangen', 'lavangen', 'gratangen',
    # Public entities / common terms
    'sandnessjøen', 'bremdeberg', 'bergem', 'poldkaia', 'røde kors',
}


def load_proper_nouns(data_file: Optional[Path] = None) -> set:
    """
    Load proper nouns from a local data file, merged with the built-in set.
    
    The data file should be a JSON object with a "proper_nouns" key containing
    a list of strings. This allows loading personal names from a gitignored
    local file without committing them to the repository.
    
    Args:
        data_file: Path to JSON file with proper nouns. If None, returns built-in set.
        
    Returns:
        Set of proper noun strings (lowercase)
    """
    nouns = set(NORWEGIAN_PROPER_NOUNS)
    if data_file and data_file.exists():
        try:
            import json
            with open(data_file, encoding="utf-8") as f:
                data = json.load(f)
            extra = data.get("proper_nouns", [])
            nouns.update(w.lower() for w in extra)
            logger.info(f"Loaded {len(extra)} additional proper nouns from {data_file}")
        except Exception as e:
            logger.warning(f"Failed to load proper nouns from {data_file}: {e}")
    return nouns


def _fix_stuttering(words: List[str]) -> Tuple[List[str], List[Dict]]:
    """
    Remove consecutive duplicate words (stuttering artifacts).
    
    Whisper often repeats words when uncertain: "jeg jeg vil" → "jeg vil"
    
    Args:
        words: List of words from the segment
        
    Returns:
        Tuple of (cleaned_words, corrections)
    """
    corrections = []
    if not words:
        return words, corrections
    
    cleaned = [words[0]]
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            corrections.append({
                "original": f"{words[i]} {words[i]}",
                "corrected": words[i],
                "position": i,
                "type": "stuttering",
                "explanation": f"Fjernet gjentakelse: '{words[i]}'"
            })
        else:
            cleaned.append(words[i])
    
    return cleaned, corrections


def _restore_punctuation(words: List[str]) -> Tuple[List[str], List[Dict]]:
    """
    Restore basic punctuation to Norwegian conversational speech.
    
    Uses filler words and clause markers to insert periods and commas:
    - "ja", "nei", "da" at end of clause → period after
    - "så", "men", "for" at start of clause → comma before
    - "hæ" → question mark
    
    Args:
        words: List of words (post-stuttering-fix)
        
    Returns:
        Tuple of (words_with_punctuation, corrections)
    """
    corrections = []
    if not words:
        return words, corrections
    
    result = []
    for i, word in enumerate(words):
        word_lower = word.lower()
        
        # Check if this word is a sentence-ending filler
        if word_lower in SENTENCE_END_FILLERS and i < len(words) - 1:
            # Only add period if next word starts a new clause
            next_word = words[i + 1].lower()
            if next_word in CLAUSE_BREAK_WORDS or next_word in SENTENCE_END_FILLERS:
                result.append(word + '.')
                corrections.append({
                    "original": word,
                    "corrected": word + '.',
                    "position": i,
                    "type": "punctuation_period",
                    "explanation": f"La til punktum etter '{word}'"
                })
            else:
                result.append(word)
        # Check if this word is a clause break word (add comma before)
        elif word_lower in CLAUSE_BREAK_WORDS and i > 0:
            prev_word = result[-1] if result else ''
            if not prev_word.endswith(('.', '!', '?')):
                result.append(',')
                corrections.append({
                    "original": f"{prev_word} {word}",
                    "corrected": f"{prev_word}, {word}",
                    "position": i,
                    "type": "punctuation_comma",
                    "explanation": f"La til komma før '{word}'"
                })
            result.append(word)
        # Check for question: "hæ" anywhere in segment
        elif word_lower == 'hæ':
            result.append(word + '?')
            corrections.append({
                "original": word,
                "corrected": word + '?',
                "position": i,
                "type": "punctuation_question",
                "explanation": f"La til spørsmålstegn etter '{word}'"
            })
        else:
            result.append(word)
    
    # Add period or question mark at end if last word doesn't end with punctuation
    if result and not result[-1].endswith(('.', '!', '?')):
        # Check if any word in the segment is a question word
        has_question = any(w.lower().rstrip('?') in {'hæ', 'hva', 'hvem', 'hvor', 'hvordan', 'hvorfor', 'når', 'ka', 'kæ', 'kor', 'korsn', 'kordan', 'koffer', 'koffor'} for w in result)
        if has_question:
            result[-1] = result[-1] + '?'
            corrections.append({
                "original": result[-1][:-1],
                "corrected": result[-1],
                "position": len(result) - 1,
                "type": "punctuation_question_end",
                "explanation": "La til spørsmålstegn på slutten av setningen"
            })
        else:
            result[-1] = result[-1] + '.'
            corrections.append({
                "original": result[-1][:-1],
                "corrected": result[-1],
                "position": len(result) - 1,
                "type": "punctuation_period_end",
                "explanation": "La til punktum på slutten av setningen"
            })
    
    return result, corrections


def _capitalize_sentence(words: List[str]) -> List[str]:
    """
    Capitalize the first word of each sentence.
    
    After punctuation restoration, words following '.', '!', or '?'
    should be capitalized.
    """
    if not words:
        return words
    
    result = list(words)
    
    # Capitalize first word
    if result[0]:
        result[0] = result[0][0].upper() + result[0][1:]
    
    # Capitalize after sentence-ending punctuation
    for i in range(1, len(result)):
        prev = result[i - 1]
        if prev.endswith(('.', '!', '?')) and result[i]:
            result[i] = result[i][0].upper() + result[i][1:]
    
    return result


def normalize_norwegian_text(
    text: str,
    auto_correct: bool = False,
) -> Tuple[str, List[Dict]]:
    """
    Normalize Norwegian text and return corrections with explanations.
    
    Applies in order:
    1. Fix stuttering (consecutive duplicate words)
    2. Restore punctuation (periods, commas, question marks)
    3. Capitalize sentences
    4. Fix missing spaces after punctuation
    5. Flag character substitutions, English words, repetition
    
    Args:
        text: Raw transcription text
        auto_correct: If True, also apply character substitutions and
                      English word replacements (aggressive mode)
        
    Returns:
        Tuple of (normalized_text, list_of_corrections)
        Each correction dict has: original, corrected, position, type, explanation
    """
    corrections = []
    normalized = text.strip()
    
    # 0. Pre-clean: normalize whitespace
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # 1. Fix stuttering (consecutive duplicate words)
    words = normalized.split()
    words, stutter_corrections = _fix_stuttering(words)
    corrections.extend(stutter_corrections)
    
    # 2. Restore punctuation
    words, punct_corrections = _restore_punctuation(words)
    corrections.extend(punct_corrections)
    
    # 3. Capitalize sentences
    words = _capitalize_sentence(words)
    
    # Rejoin into text
    normalized = ' '.join(words)
    
    # 4. Fix missing spaces after punctuation
    # Pattern: punctuation followed immediately by letter (no space)
    punct_pattern = re.compile(r'([.,;:!?])([a-zA-ZæøåÆØÅ])')
    offset = 0
    for match in punct_pattern.finditer(normalized):
        pos = match.start() + offset
        normalized = normalized[:pos+1] + ' ' + normalized[pos+1:]
        offset += 1
        corrections.append({
            "original": match.group(0),
            "corrected": match.group(1) + ' ' + match.group(2),
            "position": match.start(),
            "type": "missing_space",
            "explanation": "Manglende mellomrom etter tegnsetting"
        })
    
    # 5. Fix multiple spaces (again, in case step 4 introduced any)
    normalized = re.sub(r'  +', ' ', normalized)
    
    # 6. Flag character substitutions (aa→å, ae→æ, oe→ø)
    for pattern, replacement in NORWEGIAN_CHAR_MAP.items():
        for match in re.finditer(pattern, normalized, re.IGNORECASE):
            corrections.append({
                "original": match.group(0),
                "corrected": replacement,
                "position": match.start(),
                "type": "char_substitution",
                "explanation": f"Mulig '{match.group(0)}' skal være '{replacement}'"
            })
    
    # 7. Flag English words
    word_list = normalized.lower().split()
    for i, word in enumerate(word_list):
        if word in ENGLISH_TO_NORWEGIAN:
            corrections.append({
                "original": word,
                "corrected": ENGLISH_TO_NORWEGIAN[word],
                "position": i,
                "type": "english_word",
                "explanation": f"Engelsk ord '{word}' — kanskje ment '{ENGLISH_TO_NORWEGIAN[word]}'?"
            })
    
    # 7b. Flag dialect-standard mismatches (informational only)
    # Northern Norwegian dialect is valid — we flag but don't correct
    for i, word in enumerate(word_list):
        if word in NORWEGIAN_DIALECT_MAP:
            standard = NORWEGIAN_DIALECT_MAP[word]
            if word != standard:
                corrections.append({
                    "original": word,
                    "corrected": standard,
                    "position": i,
                    "type": "dialect_word",
                    "explanation": f"Dialektord '{word}' — standard '{standard}' (dialekt er OK, flagges kun for informasjon)"
                })
    
    # 8. Flag excessive repetition (across whole segment, not just consecutive)
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
    
    # 9. Flag very short segments
    if len(word_list) < 3:
        corrections.append({
            "original": normalized,
            "corrected": normalized,
            "position": 0,
            "type": "short_segment",
            "explanation": f"Kun {len(word_list)} ord — sjekk at segmentet er komplett"
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
