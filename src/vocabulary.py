"""
Custom Vocabulary Module

Manages custom word lists and generates initial prompts for Whisper
to improve transcription accuracy on domain-specific vocabulary.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set

from .utils import get_logger, load_json, save_json

logger = get_logger("vocabulary")

# Whisper's initial_prompt hard limit (tokens)
_WHISPER_PROMPT_TOKEN_LIMIT = 224
# Conservative default to stay well under the limit
_DEFAULT_MAX_TOKENS = 150

# Module-level cache for the Whisper tokenizer
_tokenizer = None


def _get_tokenizer():
    """Lazy-load and cache the Whisper tokenizer for accurate token counting."""
    global _tokenizer
    if _tokenizer is None:
        try:
            from transformers import AutoTokenizer
            _tokenizer = AutoTokenizer.from_pretrained("openai/whisper-tiny")
            logger.debug("Loaded Whisper tokenizer for vocabulary token counting")
        except Exception as e:
            logger.warning(f"Could not load Whisper tokenizer: {e}. Falling back to conservative word-count estimate.")
            _tokenizer = False  # sentinel: tried and failed
    return _tokenizer if _tokenizer is not False else None


def count_tokens(text: str) -> int:
    """Count tokens in text using the Whisper tokenizer if available."""
    tokenizer = _get_tokenizer()
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    # Conservative fallback: ~1.5 tokens per word (better than naive 2)
    return int(len(text.split()) * 1.5)


class VocabularyManager:
    """Manages custom vocabulary for transcription."""
    
    def __init__(self, vocab_file: Optional[Path] = None):
        """
        Initialize vocabulary manager.
        
        Args:
            vocab_file: Path to JSON file with custom vocabulary
        """
        self.vocab_file = vocab_file
        self.vocabulary = {}
        self.contexts = {}
        
        if vocab_file and vocab_file.exists():
            self._load_vocabulary()
    
    def _load_vocabulary(self):
        """Load vocabulary from JSON file."""
        try:
            data = load_json(self.vocab_file)
            
            # Expected format: {"vocabulary": [...], "contexts": {...}}
            if isinstance(data, list):
                # Simple list format
                self.vocabulary = {word: "" for word in data}
            elif isinstance(data, dict) and "vocabulary" in data:
                # Structured format with contexts
                self.vocabulary = {item: "" for item in data["vocabulary"]}
                self.contexts = data.get("contexts", {})
            else:
                # Assume dict format: word -> context
                self.vocabulary = data
            
            logger.info(f"Loaded {len(self.vocabulary)} vocabulary items")
            
        except Exception as e:
            logger.warning(f"Failed to load vocabulary: {e}")
    
    def add_word(self, word: str, context: Optional[str] = None):
        """Add word to vocabulary."""
        self.vocabulary[word] = context or ""
        if context:
            self.contexts[word] = context
        logger.debug(f"Added vocabulary item: {word}")
    
    def add_words(self, words: List[str], context: Optional[str] = None):
        """Add multiple words."""
        for word in words:
            self.add_word(word, context)
    
    def add_from_dict(self, word_dict: Dict[str, str]):
        """Add words from dictionary."""
        for word, context in word_dict.items():
            self.add_word(word, context)
    
    def save_vocabulary(self, output_path: Path):
        """Save vocabulary to JSON file."""
        data = {
            "vocabulary": list(self.vocabulary.keys()),
            "contexts": self.contexts
        }
        save_json(data, output_path)
        logger.info(f"Vocabulary saved to {output_path}")
    
    def generate_initial_prompt(
        self,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        include_contexts: bool = True
    ) -> str:
        """
        Generate initial prompt for Whisper from vocabulary.

        Whisper's initial_prompt helps improve recognition of specific words.
        The 224-token hard limit is enforced; if the tokenizer is unavailable,
        a conservative fallback estimate is used.

        Args:
            max_tokens: Maximum tokens in prompt (default 150, well under 224)
            include_contexts: Include context in prompt

        Returns:
            Initial prompt string
        """
        prompt_parts = []
        token_count = 0

        # Sort by context availability (with context first)
        sorted_words = sorted(
            self.vocabulary.items(),
            key=lambda x: len(x[1]) if x[1] else 0,
            reverse=True
        )

        for word, context in sorted_words:
            if token_count >= max_tokens:
                break

            if include_contexts and context and context.strip():
                # Format: "word (context)"
                item = f"{word} ({context})"
            else:
                item = word

            item_tokens = count_tokens(item)
            # +1 for the comma separator that will be added
            projected = token_count + item_tokens + (1 if prompt_parts else 0)
            if projected > max_tokens:
                break

            prompt_parts.append(item)
            token_count = projected

        if not prompt_parts:
            return ""

        # Create prompt string
        prompt = "Vocabulary: " + ", ".join(prompt_parts)

        # Final accurate count
        final_tokens = count_tokens(prompt)
        logger.info(
            f"Generated initial prompt ({final_tokens} tokens, {len(prompt_parts)} items). "
            f"Limit: {max_tokens} / hard cap {_WHISPER_PROMPT_TOKEN_LIMIT}"
        )

        if final_tokens > _WHISPER_PROMPT_TOKEN_LIMIT:
            logger.warning(
                f"Prompt exceeds Whisper 224-token hard limit ({final_tokens} tokens). "
                f"Reduce vocabulary or set max_tokens lower."
            )

        return prompt
    
    def get_vocabulary_for_transcription(
        self,
        domain: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Get vocabulary filtered by domain.
        
        Args:
            domain: Optional domain filter
            
        Returns:
            Filtered vocabulary dict
        """
        if not domain:
            return self.vocabulary
        
        # Filter by domain in context
        filtered = {
            word: context
            for word, context in self.vocabulary.items()
            if domain.lower() in context.lower()
        }
        
        logger.debug(f"Filtered vocabulary for domain '{domain}': {len(filtered)} items")
        
        return filtered
    
    def suggest_corrections(
        self,
        text: str,
        similarity_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Suggest corrections based on vocabulary.
        
        Args:
            text: Transcribed text
            similarity_threshold: Minimum similarity for suggestion
            
        Returns:
            List of suggested corrections
        """
        suggestions = []
        
        # Simple approach: look for known words that are close to vocabulary
        from difflib import SequenceMatcher
        
        text_lower = text.lower()
        words = text_lower.split()
        
        for word in words:
            if len(word) < 3:
                continue
            
            for vocab_word in self.vocabulary.keys():
                similarity = SequenceMatcher(None, word, vocab_word).ratio()
                
                if similarity > similarity_threshold and similarity < 1.0:
                    suggestions.append({
                        "original": word,
                        "suggestion": vocab_word,
                        "similarity": similarity,
                        "context": self.vocabulary.get(vocab_word, "")
                    })
        
        return suggestions


class CommonNorwegianVocabulary:
    """Common Norwegian-specific vocabulary and terminology."""
    
    COMMON_DOMAINS = {
        "medical": [
            "pasient", "diagn ose", "behandling", "medikament",
            "sykepleier", "lege", "sykehus", "infeksjon"
        ],
        "legal": [
            "dommer", "advokat", "domstol", "paragraf", "lov",
            "klager", "ankende", "dom"
        ],
        "technical": [
            "server", "database", "algoritme", "nettleser",
            "operativsystem", "programvare", "maskinvare"
        ],
        "finance": [
            "aksje", "investering", "rente", "obligasjon",
            "børs", "dividende", "tap", "gevinst"
        ]
    }
    
    COMMON_PROPER_NOUNS = [
        "Oslo", "Bergen", "Stavanger", "Tromsø", "Trondheim",
        "Norge", "Sverige", "Danmark", "Finland",
        "Stortinget", "Regjeringen",
        "NRK", "TV2", "Aftenposten", "VG"
    ]
    
    # Northern Norwegian dialect words for vocabulary injection.
    # These help Whisper recognize dialect forms instead of normalizing
    # them to standard Eastern Norwegian.
    # Grouped by category for organized prompt generation.
    DIALECT_VOCABULARY = {
        "northern_norwegian": {
            # First person pronouns (8 words)
            "pronouns": ["æ", "mæ", "dæ", "sæ", "dokker", "dåkker", "ho", "hu"],
            # Negation (3 words)
            "negation": ["ikkje", "itte", "ikke"],
            # Question words (7 words)
            "questions": ["ka", "kæ", "kor", "korsn", "kordan", "koffer", "koffor"],
            # Adverbs and particles (12 words)
            "adverbs": [
                "bærre", "berre", "nån", "nåkkå", "nokka", "mykje",
                "sånn", "slik", "nærmest", "akkurat", "kanskje", "kanke",
            ],
            # Common verbs in dialect form (14 words)
            "verbs": [
                "e", "je", "ha", "kje", "ska", "være", "gjøre",
                "komme", "gå", "sei", "trur", "veit", "får", "bli",
            ],
            # Common nouns / expressions (8 words)
            "expressions": ["no", "ille", "lita", "lite", "stor", "små", "godt", "mye"],
            # Time and quantity (12 words)
            "time_quantity": [
                "nå", "da", "sida", "sist", "førr", "etter", "oppi",
                "inni", "borti", "fram", "tilbake", "nedi",
            ],
            # Prepositions and conjunctions (10 words)
            "prepositions": [
                "oppå", "nedpå", "innpå", "bortpå", "frampå",
                "oppi", "inni", "borti", "atti", "forran",
            ],
            # Adjectives and descriptors (12 words)
            "adjectives": [
                "fin", "stygg", "bra", "dårlig", "vanskelig", "lett",
                "lang", "kort", "stor", "liten", "gammal", "ny",
            ],
            # Telephony / call vocabulary (12 words)
            "telephony": [
                "telefon", "mobil", "ring", "samtale", "beskjed",
                "melding", "nummer", "svar", "ringte", "ringer",
                "oppringt", "anrop",
            ],
            # Family and people (10 words)
            "people": [
                "mamma", "pappa", "bror", "søster", "bestefar",
                "bestemor", "tante", "onkel", "venn", "nabo",
            ],
            # Places and locations (10 words)
            "places": [
                "hjemme", "borte", "skolen", "jobben", "butikken",
                "sentrum", "byen", "landet", "sjukehus", "legevakt",
            ],
        }
    }
    
    # Dialect region detection markers.
    # Each dialect has distinctive words that identify it.
    # Format: dialect_region -> {set of distinctive words, weight}
    DIALECT_MARKERS = {
        "northern_norwegian": {
            "words": {"æ", "mæ", "dæ", "sæ", "dokker", "dåkker", "ikkje",
                      "itte", "ka", "kæ", "kor", "korsn", "kordan", "koffer",
                      "koffor", "bærre", "nåkkå", "nokka", "mykje", "ho",
                      "hu", "kje", "no", "ille"},
            "weight": 2.0,  # Strong signal
        },
        "trondersk": {
            "words": {"æ", "dæm", "hainn", "kæm", "sånn", "int", "itt",
                      "kass", "korsn", "bærre", "nå", "dæ", "mæ"},
            "weight": 2.0,
        },
        "vestlandsk": {
            "words": {"eg", "ikkje", "kva", "korleis", "kvi", "deg",
                      "meg", "seg", "no", "ikkje"},
            "weight": 2.0,
        },
        "sorlandsk": {
            "words": {"æ", "dæ", "kæm", "kordan", "itte", "kva",
                      "mæ", "sæ", "no"},
            "weight": 2.0,
        },
        "ostlandsk": {
            "words": {"jæ", "dæ", "sæ", "kæ", "sånn", "kanke",
                      "mæ", "dere", "ikke"},
            "weight": 1.5,  # Weaker signal (closer to standard)
        },
    }
    
    @staticmethod
    def detect_dialect(text: str) -> Optional[str]:
        """Auto-detect dialect region from transcribed text.
        
        Analyzes the text for distinctive dialect markers and returns
        the most likely dialect region, or None if no clear match.
        
        Args:
            text: Transcribed text to analyze
            
        Returns:
            Dialect region key (e.g. "northern_norwegian") or None
        """
        if not text or not text.strip():
            return None
        
        text_lower = text.lower()
        words = set(text_lower.split())
        
        scores = {}
        for region, markers in CommonNorwegianVocabulary.DIALECT_MARKERS.items():
            matches = words & markers["words"]
            if matches:
                score = len(matches) * markers["weight"]
                scores[region] = score
        
        if not scores:
            return None
        
        # Return region with highest score
        best = max(scores, key=scores.get)
        logger.info(
            f"Dialect detection: {best} "
            f"(scores: {dict(sorted(scores.items(), key=lambda x: -x[1]))})"
        )
        return best
    
    @staticmethod
    def get_domain_vocabulary(domain: str) -> List[str]:
        """Get vocabulary for specific domain."""
        return CommonNorwegianVocabulary.COMMON_DOMAINS.get(domain, [])
    
    @staticmethod
    def get_dialect_vocabulary(dialect: str = "northern_norwegian") -> List[str]:
        """Get dialect-specific vocabulary for Whisper prompt injection.
        
        Flattens the categorized dialect words into a single list.
        These words are injected into Whisper's initial_prompt so the
        model is more likely to transcribe dialect forms correctly
        instead of normalizing to standard Eastern Norwegian.
        
        Args:
            dialect: Dialect region key. Currently only
                     "northern_norwegian" is supported.
        
        Returns:
            List of dialect words for the requested region.
        """
        categories = CommonNorwegianVocabulary.DIALECT_VOCABULARY.get(dialect, {})
        words: List[str] = []
        for category_words in categories.values():
            words.extend(category_words)
        return words
    
    @staticmethod
    def create_manager(
        domain: Optional[str] = None,
        dialect: Optional[str] = None,
    ) -> VocabularyManager:
        """Create vocabulary manager with domain and/or dialect vocabulary.
        
        Args:
            domain: Optional domain for domain-specific vocabulary.
            dialect: Optional dialect region (e.g. "northern_norwegian")
                     to inject dialect words into Whisper's prompt.
        
        Returns:
            Initialized VocabularyManager.
        """
        manager = VocabularyManager()
        
        # Add proper nouns
        manager.add_words(
            CommonNorwegianVocabulary.COMMON_PROPER_NOUNS,
            context="proper noun"
        )
        
        # Add domain vocabulary if specified
        if domain:
            vocab = CommonNorwegianVocabulary.get_domain_vocabulary(domain)
            for word in vocab:
                manager.add_word(word, context=domain)
        
        # Add dialect vocabulary if specified
        if dialect:
            dialect_words = CommonNorwegianVocabulary.get_dialect_vocabulary(dialect)
            for word in dialect_words:
                manager.add_word(word, context=f"dialect:{dialect}")
            logger.info(
                f"Added {len(dialect_words)} dialect words for '{dialect}' "
                f"to vocabulary"
            )
        
        return manager


def load_vocabulary(
    vocab_file: Optional[Path] = None,
    domain: Optional[str] = None,
    dialect: Optional[str] = None,
    use_default_norwegian: bool = True
) -> VocabularyManager:
    """
    Load or create vocabulary manager.
    
    Args:
        vocab_file: Path to custom vocabulary file
        domain: Domain for predefined vocabulary
        dialect: Dialect region for dialect-specific vocabulary injection
                 (e.g. "northern_norwegian"). Injects dialect words into
                 Whisper's initial_prompt to improve recognition.
        use_default_norwegian: Load default Norwegian vocabulary (places, names,
            institutions) when no custom file is provided. Default True.
        
    Returns:
        Initialized VocabularyManager
    """
    if vocab_file and vocab_file.exists():
        logger.info(f"Loading custom vocabulary from {vocab_file}")
        manager = VocabularyManager(vocab_file)
        
        # Add dialect vocabulary on top of custom file if specified
        if dialect:
            dialect_words = CommonNorwegianVocabulary.get_dialect_vocabulary(dialect)
            for word in dialect_words:
                manager.add_word(word, context=f"dialect:{dialect}")
            logger.info(
                f"Added {len(dialect_words)} dialect words for '{dialect}' "
                f"on top of custom vocabulary"
            )
        
        return manager
    
    # Load default Norwegian vocabulary
    if use_default_norwegian:
        default_vocab = Path(__file__).parent.parent / "data" / "norwegian_vocabulary.json"
        if default_vocab.exists():
            logger.info(f"Loading default Norwegian vocabulary ({default_vocab})")
            manager = VocabularyManager(default_vocab)
            
            # Also add domain vocabulary if specified
            if domain:
                domain_vocab = CommonNorwegianVocabulary.get_domain_vocabulary(domain)
                for word in domain_vocab:
                    manager.add_word(word, context=domain)
                logger.info(f"Added {len(domain_vocab)} domain words for '{domain}'")
            
            # Add dialect vocabulary if specified
            if dialect:
                dialect_words = CommonNorwegianVocabulary.get_dialect_vocabulary(dialect)
                for word in dialect_words:
                    manager.add_word(word, context=f"dialect:{dialect}")
                logger.info(
                    f"Added {len(dialect_words)} dialect words for '{dialect}'"
                )
            
            return manager
        else:
            logger.warning(f"Default Norwegian vocabulary not found at {default_vocab}")
    
    # Fallback to empty or domain-only
    if domain or dialect:
        logger.info(f"Loading predefined vocabulary (domain={domain}, dialect={dialect})")
        return CommonNorwegianVocabulary.create_manager(domain=domain, dialect=dialect)
    else:
        logger.debug("Using empty vocabulary manager")
        return VocabularyManager()
