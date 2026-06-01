"""
Spell-Checking Module

Norwegian spell-checking using multiple strategies:
- symspellpy for fast detection/correction
- Transformers-based model for more accurate corrections (optional)

Dictionary source: LibreOffice Norwegian Bokmål dictionary (nb_NO.dic)
from https://github.com/LibreOffice/dictionaries. Downloaded on first use
to a local cache directory. The dictionary is GPL v2 licensed and is NOT
bundled with this package — it is fetched at runtime.
"""

from typing import List, Tuple, Optional, Dict
from pathlib import Path
import re
import urllib.request
import os

from .utils import get_logger

logger = get_logger("spell_check")


class NorwegianSpellChecker:
    """Norwegian text spell-checker."""
    
    def __init__(self, enable_transformers: bool = False, max_edits: int = 2):
        """
        Initialize spell checker.
        
        Args:
            enable_transformers: Use transformers for more accurate corrections
            max_edits: Maximum edit distance for suggestions
        """
        self.enable_transformers = enable_transformers
        self.max_edits = max_edits
        self.symspell = None
        self.transformer_model = None
        self._init_symspell()
        
        if enable_transformers:
            self._init_transformer()
    
    def _init_symspell(self):
        """Initialize SymSpell dictionary.
        
        Downloads the LibreOffice Norwegian Bokmål dictionary on first use
        to a local cache directory (~/.cache/audio-transcriber/).
        The dictionary is GPL v2 licensed and is NOT bundled — it is fetched
        at runtime from https://github.com/LibreOffice/dictionaries.
        """
        try:
            from symspellpy import SymSpell, Verbosity
            
            logger.info("Initializing SymSpell dictionary for Norwegian")
            
            self.symspell = SymSpell(max_dictionary_edit_distance=self.max_edits)
            
            # Try to load Norwegian dictionary from cache or download it
            dictionary_path = self._get_or_download_dictionary()
            
            if dictionary_path and dictionary_path.exists():
                # SymSpell expects term_index=0, count_index=1
                # The .dic format has term on each line, no count column,
                # so we use count_index=1 with a dummy count
                self.symspell.load_dictionary(
                    str(dictionary_path),
                    term_index=0,
                    count_index=1,
                )
                self.symspell_available = True
                logger.info(f"SymSpell initialized with {dictionary_path.name}")
            else:
                logger.warning(
                    "No Norwegian dictionary loaded. Spell-checking is DISABLED. "
                    "Run with --download-dictionary to fetch the Norwegian word list, "
                    "or manually place a dictionary file."
                )
                self.symspell_available = False
                self.symspell = None
            
        except ImportError:
            logger.warning("symspellpy not installed, spell-checking disabled")
            self.symspell_available = False
    
    def _get_or_download_dictionary(self) -> Optional[Path]:
        """Get or download the Norwegian dictionary.
        
        Returns:
            Path to the dictionary file, or None if unavailable.
        """
        cache_dir = Path.home() / ".cache" / "audio-transcriber"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        dic_path = cache_dir / "nb_NO.dic"
        
        # Return cached dictionary if it exists
        if dic_path.exists():
            return dic_path
        
        # Try to download from LibreOffice dictionaries repository
        url = "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/no/nb_NO.dic"
        logger.info(f"Downloading Norwegian dictionary from {url}")
        
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                # The .dic file uses ISO-8859-1 (latin-1) encoding
                text = raw.decode("latin-1")
                
                # First line is the word count, rest are words
                lines = text.splitlines()
                if len(lines) > 1:
                    # Strip affix annotations (e.g., "A-aksje/EG" -> "A-aksje")
                    # and keep only the base word form
                    words = []
                    for line in lines[1:]:  # Skip count header
                        word = line.strip()
                        if word:
                            # Remove affix annotations after '/'
                            if "/" in word:
                                word = word.split("/")[0]
                            words.append(word)
                    
                    # Write in SymSpell format: word on each line
                    with open(dic_path, "w", encoding="utf-8") as f:
                        for w in words:
                            f.write(f"{w} 1\n")  # count=1 for all words
                    
                    logger.info(f"Downloaded {len(words)} words to {dic_path}")
                    return dic_path
                else:
                    logger.error("Downloaded dictionary file is empty or invalid")
                    return None
                    
        except Exception as e:
            logger.warning(f"Failed to download Norwegian dictionary: {e}")
            return None
    
    def _init_transformer(self):
        """Initialize transformer-based spell checker."""
        try:
            from transformers import pipeline
            
            logger.info("Initializing transformer spell-checker")
            # This would use a Norwegian-specific model
            # For now, placeholder for future implementation
            logger.info("Transformer spell-checker ready (placeholder)")
            
        except ImportError:
            logger.warning("transformers not installed")
    
    def check_word(self, word: str) -> Tuple[bool, Optional[str]]:
        """
        Check if word is correctly spelled.
        
        Args:
            word: Word to check
            
        Returns:
            Tuple of (is_correct, suggestion)
        """
        word = word.strip().lower()
        
        if not word or len(word) < 2:
            return True, None
        
        # Skip if word is all uppercase (likely acronym)
        if word == word.upper() and len(word) > 1:
            return True, None
        
        # Check with SymSpell if available
        if self.symspell_available and self.symspell:
            try:
                # Use include_unknown=False so unknown words return empty list
                # rather than the word itself with distance > max_edits.
                suggestions = self.symspell.lookup(
                    word,
                    verbosity=1,  # Top suggestion only
                    max_edit_distance=self.max_edits,
                    include_unknown=False
                )
                
                if suggestions:
                    top = suggestions[0]
                    if top.distance == 0 and top.term == word:
                        # Exact match in dictionary — word is correct
                        return True, None
                    else:
                        # Suggestion with distance > 0 — word is misspelled
                        return False, top.term
                else:
                    # No suggestions at all — word is unknown (not in dictionary
                    # and no close match within edit distance)
                    return False, None
                
            except Exception as e:
                logger.debug(f"SymSpell lookup failed for '{word}': {e}")
        
        return True, None
    
    def check_text(self, text: str) -> List[Dict]:
        """
        Check entire text for spelling errors.
        
        Args:
            text: Text to check
            
        Returns:
            List of dicts with error positions and suggestions
        """
        errors = []
        
        # Split into words, preserving positions
        words = re.findall(r"\b\w+\b", text.lower())
        
        for word in words:
            is_correct, suggestion = self.check_word(word)
            
            if not is_correct:
                # Find position in original text
                pos = text.lower().find(word)
                if pos >= 0:
                    errors.append({
                        "word": word,
                        "position": pos,
                        "suggestion": suggestion,
                        "confidence": 0.8 if suggestion else 0.0  # Lower confidence for unknown words
                    })
        
        return errors
    
    def correct_text(self, text: str, auto_fix: bool = False) -> Tuple[str, List[Dict]]:
        """
        Correct text spelling errors.
        
        Args:
            text: Text to correct
            auto_fix: Automatically apply corrections
            
        Returns:
            Tuple of (corrected_text, list_of_corrections)
        """
        corrections = []
        corrected = text
        offset = 0
        
        errors = self.check_text(text)
        
        for error in errors:
            if auto_fix and error["suggestion"]:
                # Replace in corrected text
                old_word = error["word"]
                new_word = error["suggestion"]
                
                # Account for offset from previous replacements
                start = error["position"] + offset
                end = start + len(old_word)
                
                corrected = corrected[:start] + new_word + corrected[end:]
                offset += len(new_word) - len(old_word)
                
                corrections.append({
                    "original": old_word,
                    "corrected": new_word,
                    "position": error["position"]
                })
        
        return corrected, corrections
    
    def check_numbers(self, text: str) -> List[Dict]:
        """
        Check for common number transcription errors.
        
        Norwegian-specific patterns.
        """
        errors = []
        
        # Pattern: "en" might be transcribed as number "1"
        # Pattern: "to" might be transcribed as number "2"
        number_patterns = {
            r'\bone\b': '1',
            r'\btwo\b': '2',
            r'\btre\b': '3',
            r'\bfire\b': '4',
            r'\bfem\b': '5',
        }
        
        for pattern, num in number_patterns.items():
            for match in re.finditer(pattern, text.lower()):
                errors.append({
                    "type": "number",
                    "word": match.group(),
                    "position": match.start(),
                    "suggestion": num
                })
        
        return errors
    
    def check_proper_nouns(
        self,
        text: str,
        known_nouns: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Check for common proper noun errors.
        
        Args:
            text: Text to check
            known_nouns: List of known proper nouns to validate
            
        Returns:
            List of potential proper noun errors
        """
        errors = []
        
        if known_nouns is None:
            known_nouns = []
        
        # Find capitalized words that aren't in known nouns
        capitalized = re.findall(r'\b[A-Z]\w+\b', text)
        
        for word in capitalized:
            if word not in known_nouns and len(word) > 2:
                # Could be a proper noun - flag for review
                errors.append({
                    "type": "proper_noun",
                    "word": word,
                    "known": False
                })
        
        return errors


def check_transcription(
    text: str,
    config: Optional[dict] = None
) -> Dict:
    """
    Check transcription for errors.
    
    Args:
        text: Transcribed text
        config: Configuration dict
        
    Returns:
        Dict with spell-check results
    """
    if config is None:
        config = {}
    
    enabled = config.get("enabled", False)
    
    if not enabled:
        logger.debug("Spell-checking disabled")
        return {"enabled": False, "errors": []}
    
    logger.info("Running spell-check on transcription")
    
    checker = NorwegianSpellChecker(
        enable_transformers=config.get("model") == "transformers"
    )
    
    spelling_errors = checker.check_text(text)
    number_errors = checker.check_numbers(text)
    
    # Combine results
    all_errors = spelling_errors + number_errors
    
    return {
        "enabled": True,
        "text": text,
        "errors": all_errors,
        "error_count": len(all_errors),
        "suggestion_count": sum(1 for e in all_errors if "suggestion" in e)
    }
