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
        max_tokens: int = 100,
        include_contexts: bool = True
    ) -> str:
        """
        Generate initial prompt for Whisper from vocabulary.
        
        Whisper's initial_prompt helps improve recognition of specific words.
        
        Args:
            max_tokens: Maximum tokens in prompt
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
            
            prompt_parts.append(item)
            # Rough estimate: 2 tokens per word
            token_count += len(item.split()) * 2
        
        if not prompt_parts:
            return ""
        
        # Create prompt string
        prompt = "Vocabulary: " + ", ".join(prompt_parts)
        
        logger.info(f"Generated initial prompt ({token_count} tokens, {len(prompt_parts)} items)")
        
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
    
    @staticmethod
    def get_domain_vocabulary(domain: str) -> List[str]:
        """Get vocabulary for specific domain."""
        return CommonNorwegianVocabulary.COMMON_DOMAINS.get(domain, [])
    
    @staticmethod
    def create_manager(domain: Optional[str] = None) -> VocabularyManager:
        """Create vocabulary manager with domain vocabulary."""
        manager = VocabularyManager()
        
        # Add proper nouns
        manager.add_words(
            CommonNorwegianVocabulary.COMMON_PROPER_NOUNS,
            context="proper noun"
        )
        
        # Add domain vocabulary if specified
        if domain:
            vocab = CommonNorwegianVocabulary.get_domain_vocabulary(domain)
            manager.add_words(vocab, context=domain)
        
        return manager


def load_vocabulary(
    vocab_file: Optional[Path] = None,
    domain: Optional[str] = None
) -> VocabularyManager:
    """
    Load or create vocabulary manager.
    
    Args:
        vocab_file: Path to custom vocabulary file
        domain: Domain for predefined vocabulary
        
    Returns:
        Initialized VocabularyManager
    """
    if vocab_file and vocab_file.exists():
        logger.info(f"Loading vocabulary from {vocab_file}")
        return VocabularyManager(vocab_file)
    elif domain:
        logger.info(f"Loading predefined vocabulary for domain: {domain}")
        return CommonNorwegianVocabulary.create_manager(domain)
    else:
        logger.debug("Using empty vocabulary manager")
        return VocabularyManager()
