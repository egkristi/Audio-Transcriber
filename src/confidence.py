"""
Confidence Flagging Module

Extracts multiple deterministic signals from transcription output to prioritize
segments for manual review. Signals are ranked by usefulness:

1. WhisperX alignment score (acoustic "text vs audio" confidence)
2. faster-whisper decoder signals (avg_logprob, word.probability, no_speech_prob,
   compression_ratio, temperature)
3. Cross-model disagreement (from compare.py)
4. Acoustic features from analyze.py (SNR, VAD overlap)
5. Norwegian-specific hard-rules (repetition, English words, duration, etc.)

Phase A (current): Unweighted normalized priority score for ranking.
Phase B (future): Calibrate against ground-truth using logistic regression.

Honest limitation: Confidence-flagging catches "model knew it was uncertain"
errors but misses "confidently wrong" errors — especially plausible substitutions
of names and numbers. These get high decoder confidence because they are
linguistically plausible. Therefore: confidence is a supplement, not a
replacement. Proper nouns and numbers should be reviewed regardless of score.

Norwegian-specific hard-rules (v0.1.5+):
- Repetition: 3+ repeated words (excluding filler words like "ja", "nei", "da")
  or 2+ repeated phrases = likely hallucination
- Cross-segment repetition: same word appearing 5+ times across 3+ adjacent
  segments = hallucination pattern (e.g., "svært" ×30 in v15)
- English words: common English words in Norwegian text = language confusion
- Duration: segments <2s or >60s = likely segmentation error
- Character patterns: "aa" not "å", "ae" not "æ", "oe" not "ø" = normalization issue
- Punctuation: missing spaces after punctuation = formatting error
- Unusual characters: symbols, emojis, mixed scripts = corruption
- Dialect-standard mismatch: standard forms where dialect expected = silent normalization
  (excludes common function words like "jeg", "ikke", "hva" to avoid diluting signal)
- Mixed dialect register: both dialect and standard forms in same segment = confusion
- Dialect presence: dialect words detected, flagged for awareness
- Novelty bonus: segments with rare flags get boosted priority (prevents common
  flags like repeated_words from dominating the score)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np

from .utils import get_logger, save_json, _NumpyEncoder

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
            # Check BOTH sources: (1) aligned_word_segments parameter (legacy),
            # and (2) "score" fields in words attached to the segment dict.
            # The latter is populated by _align_with_whisperx() in transcribe.py.
            # (ISSUES.md #42)
            alignment_scores = []
            
            # Source 1: aligned_word_segments parameter
            if aligned_word_segments:
                segment_words = [
                    w for w in aligned_word_segments
                    if sc.start <= w.get("start", 0) <= sc.end
                ]
                if segment_words:
                    scores = [w.get("score", 1.0) for w in segment_words]
                    alignment_scores.extend(scores)
            
            # Source 2: "score" fields in words attached to the segment
            if words:
                word_scores = [w.get("score") for w in words if "score" in w]
                if word_scores:
                    alignment_scores.extend(word_scores)
            
            if alignment_scores:
                sc.alignment_score = np.mean(alignment_scores)
                sc.min_word_alignment_score = min(alignment_scores)
            
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
        # Common Norwegian filler/function words to exclude from repetition checks.
        # These appear frequently in conversational speech and would otherwise
        # trigger false positives in both per-segment and cross-segment detection.
        filler_words = {"ja", "nei", "da", "jo", "vel", "jo", "nå", "så",
                        "å", "og", "men", "for", "at", "til", "av", "med",
                        "på", "i", "er", "det", "den", "de", "du", "han",
                        "hun", "vi", "dere", "man", "sin", "seg", "kan",
                        "skal", "vil", "må", "har", "hadde", "ble", "blir",
                        "være", "vært", "bli", "kommer", "går", "sier",
                        "tror", "vet", "ser", "litt", "bare", "også",
                        "ikke", "hva", "hvor", "hvordan", "hvorfor",
                        "når", "hvilken", "hvem", "hvis", "fordi"}
        # Extended set for cross-segment repetition detection — includes common
        # pronouns, auxiliary verbs, and high-frequency Norwegian words that
        # appear naturally across many segments in conversational speech.
        cross_segment_filler = filler_words | {
            "jeg", "meg", "deg", "seg", "være", "vært", "blitt", "fått", "hatt",
            "sagt", "gjort", "tatt", "sett", "hørt", "visst", "kunnet", "måttet",
            "skulle", "ville", "kunne", "burde", "måtte", "få", "får", "fikk",
            "gi", "gir", "ga", "gitt", "si", "sier", "sa", "sagt",
            "gå", "går", "gikk", "gått", "komme", "kommer", "kom", "kommet",
            "ta", "tar", "tok", "tatt", "se", "ser", "så", "sett",
            "ha", "har", "hadde", "hatt", "vite", "vet", "visste", "visst",
            "tro", "tror", "trodd", "tenke", "tenker", "tenkte",
            "noe", "noen", "noe", "noen", "litt", "mye", "masse",
            "sånn", "slik", "slike", "samme", "annen", "annet", "andre",
            "hver", "alle", "begge", "ingen", "ingenting",
            "opp", "ned", "ut", "inn", "bort", "fram", "tilbake",
            "her", "der", "hit", "dit", "hjem", "ute", "inne",
            "alltid", "aldri", "ofte", "sjelden", "nettopp", "akkurat",
            "kanskje", "sikkert", "egentlig", "faktisk", "nemlig", "riktig",
        }
        import re as _re
        
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
            
            # 9. HARD RULE: Numbers — always flag regardless of score
            # WhisperX alignment is weak for numeric tokens; these are high-risk.
            # "Confidently wrong" errors on numbers are invisible to acoustic signals,
            # so this rule must dominate the priority score.
            import re
            digit_count = len(re.findall(r'\d', seg.text))
            if digit_count > 0:
                # Strong boost proportional to digit count — ensures these segments
                # always appear at the top of the review list
                scores.append(min(0.9, 0.3 + 0.1 * digit_count))
                flags.append(f"contains_numbers:{digit_count}")
            
            # 10. HARD RULE: Capitalized OOV tokens (proper nouns not at sentence start)
            # These are often "confidently wrong" — high decoder score but wrong name.
            # Whisper may substitute a plausible-sounding name that is completely wrong.
            words = seg.text.split()
            proper_noun_count = 0
            for i, word in enumerate(words):
                if len(word) > 1 and word[0].isupper() and i > 0:
                    proper_noun_count += 1
            if proper_noun_count > 0:
                # Strong boost — proper noun substitutions are invisible to confidence
                scores.append(min(0.85, 0.3 + 0.15 * proper_noun_count))
                flags.append(f"possible_proper_noun:{proper_noun_count}")
            
            # 11. HARD RULE: Repetition — 3+ identical consecutive words = hallucination
            # Whisper often gets stuck repeating words when uncertain.
            # Exclude common Norwegian filler/function words that naturally repeat
            # in conversational speech (e.g., "ja", "nei", "da", "jo", "vel").
            text_lower = seg.text.lower()
            word_list = text_lower.split()
            # Strip punctuation from words for cleaner matching
            clean_words = [_re.sub(r'[^a-zæøå0-9]', '', w) for w in word_list]
            clean_words = [w for w in clean_words if w]  # Remove empty strings
            significant_repeats = [w for w in set(clean_words)
                                   if w not in filler_words and clean_words.count(w) >= 3]
            if significant_repeats:
                scores.append(0.6)
                flags.append(f"repeated_words:{','.join(significant_repeats[:3])}")
            # Also check for repeated 2-word phrases (excluding filler-only phrases)
            if len(word_list) >= 4:
                bigrams = [f"{word_list[i]} {word_list[i+1]}" for i in range(len(word_list)-1)]
                for bg in set(bigrams):
                    if bigrams.count(bg) >= 2:
                        # Check if the bigram is just filler words
                        bg_words = bg.split()
                        if not all(w in filler_words for w in bg_words):
                            scores.append(0.5)
                            flags.append(f"repeated_phrases:{bg}")
                            break
            
            # 12. HARD RULE: English words in Norwegian text = language confusion
            # Common English words that Whisper might insert
            common_english = {
                'the', 'and', 'you', 'that', 'have', 'for', 'not', 'with',
                'his', 'they', 'say', 'her', 'she', 'will', 'one', 'all',
                'would', 'there', 'their', 'what', 'about', 'which', 'when',
                'make', 'like', 'time', 'just', 'know', 'take', 'people',
                'year', 'good', 'some', 'come', 'could', 'state', 'only',
                'other', 'new', 'may', 'way', 'use', 'her', 'than',
                'first', 'water', 'been', 'call', 'who', 'oil', 'its',
                'now', 'find', 'long', 'down', 'day', 'did', 'get', 'has',
                'him', 'how', 'man', 'more', 'much', 'no', 'way', 'too',
                'very', 'what', 'who', 'why', 'yes', 'ok', 'okay', 'hello',
                'hi', 'bye', 'thanks', 'please', 'sorry', 'yes', 'no'
            }
            english_words_found = [w for w in word_list if w in common_english]
            if english_words_found:
                scores.append(min(0.5, 0.15 * len(english_words_found)))
                flags.append(f"english_words:{','.join(english_words_found[:3])}")
            
            # 13. HARD RULE: Duration heuristics
            segment_duration = seg.end - seg.start
            if segment_duration < 2.0:
                scores.append(0.3)
                flags.append("very_short_segment")
            if segment_duration > 60.0:
                scores.append(0.4)
                flags.append("very_long_segment")
            
            # 14. HARD RULE: Word count heuristics
            # Thresholds tuned for 30-second segments (typical VAD chunk).
            # 80+ words in 30s ≈ 160+ wpm — unusually dense speech.
            # 3- words in 30s ≈ near silence — likely segmentation error.
            word_count = len(words)
            if word_count < 3:
                scores.append(0.2)
                flags.append("very_few_words")
            if word_count > 80:
                scores.append(0.3)
                flags.append("very_many_words")
            
            # 15. HARD RULE: Norwegian normalization issues
            # "aa" should be "å", "ae" should be "æ", "oe" should be "ø"
            # These are common Whisper errors on Norwegian
            if re.search(r'\baa\b', seg.text.lower()):
                scores.append(0.25)
                flags.append("possible_aa_not_aa")
            if re.search(r'\bae\b', seg.text.lower()):
                scores.append(0.25)
                flags.append("possible_ae_not_ae")
            if re.search(r'\boe\b', seg.text.lower()):
                scores.append(0.25)
                flags.append("possible_oe_not_oe")
            
            # 16. HARD RULE: Formatting issues
            # Missing space after punctuation (e.g., "ja,men")
            if re.search(r'[.,;:!?][a-zA-ZæøåÆØÅ]', seg.text):
                scores.append(0.2)
                flags.append("missing_space_after_punct")
            
            # 17. HARD RULE: Unusual characters
            # Symbols, emojis, mixed scripts = corruption
            if re.search(r'[^\w\sæøåÆØÅ.,;:!?\-\'"()]', seg.text):
                scores.append(0.4)
                flags.append("unusual_characters")
            
            # 18. HARD RULE: Suspicious patterns
            # "hæ" used as filler (common in Norwegian but often hallucinated)
            hae_count = text_lower.count('hæ')
            if hae_count >= 3:
                scores.append(0.2)
                flags.append("excessive_filler_hae")
            
            # "ja" repeated excessively
            ja_count = text_lower.count(' ja ')
            if ja_count >= 4:
                scores.append(0.2)
                flags.append("excessive_filler_ja")
            
            # 19. HARD RULE: Incomplete sentence ending
            # Segment ending mid-word or with hyphen = likely truncation
            if seg.text.endswith('-') or seg.text.endswith('…') or seg.text.endswith('...'):
                scores.append(0.3)
                flags.append("incomplete_ending")
            
            # 20. HARD RULE: All-caps tokens (acronyms/abbreviations)
            # Whisper often hallucinates or misrecognizes acronyms
            all_caps = [w for w in words if len(w) >= 2 and w.isupper() and w.isalpha()]
            if all_caps:
                scores.append(min(0.7, 0.3 + 0.1 * len(all_caps)))
                flags.append(f"all_caps:{','.join(all_caps[:3])}")
            
            # 21. HARD RULE: Single-letter words (often hallucinated)
            # Whisper sometimes outputs single letters like "a", "i", "o" as artifacts
            single_letters = [w for w in word_list if len(w) == 1 and w.isalpha()]
            if len(single_letters) >= 2:
                scores.append(0.4)
                flags.append(f"single_letter_words:{','.join(single_letters[:3])}")
            
            # 22. HARD RULE: All-lowercase segment (Norwegian uses sentence case)
            # All lowercase might indicate Whisper uncertainty
            if seg.text and seg.text[0].islower() and word_count > 2:
                # Check if it's not a continuation
                if not seg.text.startswith(('og ', 'men ', 'så ', 'ja ', 'nei ')):
                    scores.append(0.15)
                    flags.append("lowercase_start")
            
            # 23. HARD RULE: Dialect-standard mismatch detection
            # Whisper may silently normalize Northern Norwegian dialect to standard
            # Eastern Norwegian. These "confidently wrong" substitutions get high
            # decoder confidence but are incorrect for the target dialect.
            # Flag segments where standard forms appear but dialect expected.
            # Dialect-standard pairs: (standard, dialect)
            # NOTE: "jeg", "ikke", "hva", "hvor", "hvordan", "hvorfor" are excluded
            # from the count because they appear in almost every conversational
            # segment and would otherwise fire on 98% of segments (diluting signal).
            dialect_pairs = [
                ("jeg", "æ"), ("meg", "mæ"), ("deg", "dæ"), ("seg", "sæ"),
                ("dere", "dokker"), ("ikke", "ikkje"), ("hva", "ka"),
                ("hvor", "kor"), ("hvordan", "korsn"), ("hvorfor", "koffer"),
                ("bare", "bærre"), ("noe", "no"), ("noen", "nån"),
                ("mye", "mykje"), ("hun", "ho"), ("være", "vær"),
                ("gjør", "gjere"), ("kommer", "kjem"), ("går", "gjære"),
                ("sier", "sei"), ("tror", "trur"), ("vet", "veit"),
                ("får", "fær"), ("blir", "bli"), ("har", "harr"),
                ("skal", "ska"), ("kan", "kann"), ("må", "må"),
                ("vil", "vill"), ("nå", "no"), ("da", "då"),
                ("opp", "oppi"), ("ned", "nedi"), ("inn", "inni"),
                ("bort", "borti"), ("fram", "frami"),
            ]
            # Common function words that appear in almost every segment — exclude
            # from dialect counting to avoid diluting the signal.
            common_function_words = {"jeg", "ikke", "hva", "hvor", "hvordan",
                                     "hvorfor", "det", "den", "de", "er", "på",
                                     "til", "av", "med", "for", "at", "og",
                                     "men", "så", "da", "nå", "jo", "vel"}
            dialect_standard_count = 0
            dialect_expected_forms = []
            for standard, dialect in dialect_pairs:
                if standard in common_function_words:
                    continue  # Skip common function words that appear everywhere
                if re.search(r'\b' + re.escape(standard) + r'\b', text_lower):
                    dialect_standard_count += 1
                    dialect_expected_forms.append(f"{standard}→{dialect}")
            if dialect_standard_count >= 2:
                # Multiple standard forms where dialect expected — likely normalization
                boost = min(0.6, 0.15 * dialect_standard_count)
                scores.append(boost)
                flags.append(f"dialect_normalized:{','.join(dialect_expected_forms[:5])}")
            elif dialect_standard_count == 1:
                # Single instance — milder flag
                scores.append(0.15)
                flags.append(f"dialect_normalized:{','.join(dialect_expected_forms[:3])}")
            
            # 24. HARD RULE: Dialect form detection (mixed dialect-standard)
            # If a segment contains BOTH dialect and standard forms of the same word,
            # it indicates Whisper is confused about the dialect register.
            dialect_words_set = {
                "æ", "mæ", "dæ", "sæ", "dokker", "dåkker", "ikkje", "itte",
                "ka", "kæ", "kor", "korsn", "kordan", "koffer", "koffor",
                "bærre", "berre", "nån", "nåkkå", "nokka", "mykje",
                "ho", "hu", "kje", "ska", "veit", "trur", "sei",
                "no", "ille", "lita", "lite",
            }
            dialect_words_found = [w for w in word_list if w in dialect_words_set]
            if dialect_words_found:
                # Check for mixed register: dialect + standard of same concept
                has_jeg = "jeg" in word_list
                has_ikke = "ikke" in word_list
                has_hva = "hva" in word_list
                has_standard_pronoun = has_jeg or "meg" in word_list or "deg" in word_list
                has_dialect_pronoun = "æ" in word_list or "mæ" in word_list or "dæ" in word_list
                if has_standard_pronoun and has_dialect_pronoun:
                    scores.append(0.35)
                    flags.append("mixed_dialect_register")
                if has_ikke and ("ikkje" in word_list or "itte" in word_list):
                    scores.append(0.3)
                    flags.append("mixed_negation")
                # General dialect presence — mild flag for awareness
                if len(dialect_words_found) >= 2:
                    scores.append(0.1)
                    flags.append(f"dialect_present:{','.join(dialect_words_found[:5])}")
            
            # Compute unweighted average priority
            if scores:
                seg.priority_score = float(np.mean(scores))
            else:
                seg.priority_score = 0.0
            
            # Novelty bonus: boost segments with rare flags (flags that appear
            # in few other segments). This prevents common flags (repeated_words,
            # dialect_normalized, low_logprob) from dominating the score and
            # ensures segments with unique issues get higher priority.
            # The bonus is proportional to how rare the flag is.
            # This is computed in a second pass (see compute_priority).
            
            seg.flags = flags
        
        # Second pass: compute flag rarity and apply novelty bonus
        # Count flag occurrences across all segments
        flag_segment_counts = {}
        for seg in segments:
            seen_bases = set()
            for flag in seg.flags:
                base = flag.split(":")[0]
                if base not in seen_bases:
                    seen_bases.add(base)
                    flag_segment_counts[base] = flag_segment_counts.get(base, 0) + 1
        
        # Apply novelty bonus: +0.1 for each rare flag (<20% of segments)
        total_segments = len(segments) if segments else 1
        for seg in segments:
            seen_bases = set()
            novelty_bonus = 0.0
            for flag in seg.flags:
                base = flag.split(":")[0]
                if base not in seen_bases:
                    seen_bases.add(base)
                    freq = flag_segment_counts.get(base, 0) / total_segments
                    if freq < 0.2:  # Rare flag: appears in <20% of segments
                        novelty_bonus += 0.1
                    elif freq < 0.4:  # Uncommon flag: appears in <40% of segments
                        novelty_bonus += 0.05
            seg.priority_score += novelty_bonus
        
        # Third pass: cross-segment repetition detection
        # Whisper sometimes hallucinates by repeating the same word across multiple
        # adjacent segments (e.g., "svært" ×30 across segment 5 in v15).
        # Per-segment repetition check misses this because each segment individually
        # may only have 1-2 occurrences.
        # 
        # Strategy: count word frequency across ALL segments. If a word appears
        # unusually often (top 1% of word frequencies), flag the segments where
        # it appears most densely.
        if len(segments) >= 3:
            from collections import Counter
            # Count word occurrences across all segments (excluding filler/function words)
            all_words = []
            for seg in segments:
                words = seg.text.lower().split()
                # Strip punctuation and exclude filler/function words
                clean = [_re.sub(r'[^a-zæøå0-9]', '', w) for w in words]
                clean = [w for w in clean if w and len(w) > 2 and w not in cross_segment_filler]
                all_words.extend(clean)
            
            word_freq = Counter(all_words)
            if word_freq:
                # Find the most repeated word(s) — top 5% of frequency distribution
                max_freq = max(word_freq.values())
                threshold = max(3, int(max_freq * 0.3))  # At least 3, or 30% of max
                
                for word, freq in word_freq.most_common(3):  # Top 3 most repeated
                    if freq >= threshold and freq >= 5:  # At least 5 occurrences
                        # Find which segments contain this word
                        affected_segments = []
                        for seg in segments:
                            seg_words = seg.text.lower().split()
                            count = seg_words.count(word)
                            if count >= 1:
                                affected_segments.append((seg.segment_id, count))
                        
                        if len(affected_segments) >= 3:  # Spread across 3+ segments
                            # Boost priority for segments with highest density
                            max_density = max(c for _, c in affected_segments)
                            for seg in segments:
                                seg_words = seg.text.lower().split()
                                count = seg_words.count(word)
                                if count >= 1:
                                    # Stronger boost for segments with more repetitions
                                    density_ratio = count / max_density if max_density > 0 else 0
                                    boost = 0.15 + 0.15 * density_ratio
                                    seg.priority_score += boost
                                    seg.flags.append(f"cross_segment_repeat:{word}×{freq}")
        
        # Sort by priority descending
        segments.sort(key=lambda s: s.priority_score, reverse=True)
        for rank, seg in enumerate(segments, 1):
            seg.priority_rank = rank
        
        return segments
    
    def export_review_list(
        self,
        segments: List[SegmentConfidence],
        output_path: Path,
        top_n: Optional[int] = None,
        export_all: bool = True,
    ) -> Path:
        """
        Export a prioritized review list for manual correction.
        
        Args:
            segments: List of SegmentConfidence with priority scores
            output_path: Where to save the review list
            top_n: Only export top N segments in the human-readable list (None = all)
            export_all: Also export a JSON file with ALL segments and full signal data
        
        Returns:
            Path to exported review list
        """
        # Human-readable review list (top N or all)
        display_segments = segments[:top_n] if top_n else segments
        
        lines = [
            "=== PRIORITIZED REVIEW LIST ===",
            f"Total segments: {len(segments)}",
            f"Displayed: {len(display_segments)}",
            "",
            "Review order: highest priority first",
            "",
        ]
        
        # Add histogram
        lines.append("=== PRIORITY HISTOGRAM ===")
        bins = [(0.0, 0.2, "low"), (0.2, 0.4, "medium-low"), (0.4, 0.6, "medium"), 
                (0.6, 0.8, "medium-high"), (0.8, 1.0, "high")]
        for low, high, label in bins:
            count = sum(1 for s in segments if low <= s.priority_score < high)
            lines.append(f"  {label:12s} ({low:.1f}-{high:.1f}): {count} segments")
        lines.append("")
        
        # Add flag distribution
        lines.append("=== FLAG DISTRIBUTION ===")
        all_flags = {}
        for seg in segments:
            for flag in seg.flags:
                all_flags[flag] = all_flags.get(flag, 0) + 1
        for flag, count in sorted(all_flags.items(), key=lambda x: -x[1]):
            lines.append(f"  {flag:30s}: {count}")
        lines.append("")
        
        for seg in display_segments:
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
        
        # Export ALL segments as JSON with full signal data
        if export_all:
            json_path = output_path.with_suffix(".json")
            all_data = []
            for seg in segments:
                all_data.append({
                    "segment_id": seg.segment_id,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "speaker": seg.speaker,
                    "priority_score": seg.priority_score,
                    "priority_rank": seg.priority_rank,
                    "flags": seg.flags,
                    "signals": {
                        "alignment_score": seg.alignment_score,
                        "min_word_alignment_score": seg.min_word_alignment_score,
                        "avg_logprob": seg.avg_logprob,
                        "no_speech_prob": seg.no_speech_prob,
                        "compression_ratio": seg.compression_ratio,
                        "temperature": seg.temperature,
                        "min_word_probability": seg.min_word_probability,
                        "model_disagreement": seg.model_disagreement,
                        "snr_db": seg.snr_db,
                        "vad_overlap": seg.vad_overlap,
                    }
                })
            
            summary = {
                "total_segments": len(segments),
                "flagged_segments": len([s for s in segments if s.flags]),
                "flag_distribution": all_flags,
                "priority_histogram": {
                    label: sum(1 for s in segments if low <= s.priority_score < high)
                    for low, high, label in bins
                },
                "segments": all_data,
            }
            
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)
            logger.info(f"Full confidence data exported to {json_path}")
        
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
