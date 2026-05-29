"""
Spell-Checking Module

Norwegian spell-checking using multiple strategies:
- symspellpy for fast detection/correction
- Transformers-based model for more accurate corrections (optional)
"""

from typing import List, Tuple, Optional, Dict
import re

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
        """Initialize SymSpell dictionary."""
        try:
            from symspellpy import SymSpell, Verbosity
            
            logger.info("Initializing SymSpell dictionary for Norwegian")
            
            # Download or use bundled dictionary
            self.symspell = SymSpell(max_dictionary_edit_distance=self.max_edits)
            
            # CRITICAL: SymSpell requires a loaded dictionary to function.
            # We do NOT bundle a Norwegian dictionary because:
            # 1. Norwegian dictionaries (NST/UiB) have licensing restrictions
            # 2. symspellpy's bundled dictionaries are English-only
            # 3. A custom dictionary would need manual curation
            #
            # Without a dictionary, check_word() will treat ALL words as unknown
            # and return false positives. Therefore: spell-checking is effectively
            # DISABLED until a dictionary is provided.
            #
            # To enable: download a Norwegian word list and call:
            #   self.symspell.load_dictionary("no_wordlist.txt", term_index=0, count_index=1)
            #
            # See ISSUES.md for details.
            
            dictionary_loaded = False  # No dictionary bundled
            
            if not dictionary_loaded:
                logger.warning(
                    "No Norwegian dictionary loaded. Spell-checking is DISABLED. "
                    "All words will be treated as unknown without a dictionary. "
                    "Provide a word list or disable --spell-check."
                )
                self.symspell_available = False
                self.symspell = None
            else:
                self.symspell_available = True
                logger.info("SymSpell initialized with dictionary")
            
        except ImportError:
            logger.warning("symspellpy not installed, spell-checking disabled")
            self.symspell_available = False
    
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
                suggestions = self.symspell.lookup(
                    word,
                    verbosity=1,  # Top suggestion only
                    max_edit_distance=self.max_edits,
                    include_unknown=True
                )
                
                if suggestions:
                    # If first suggestion is the word itself, it's correct
                    if suggestions[0].term == word:
                        return True, None
                    else:
                        return False, suggestions[0].term
                
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
            
            if not is_correct and suggestion:
                # Find position in original text
                pos = text.lower().find(word)
                if pos >= 0:
                    errors.append({
                        "word": word,
                        "position": pos,
                        "suggestion": suggestion,
                        "confidence": 0.8  # Placeholder
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
