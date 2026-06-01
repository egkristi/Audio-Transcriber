"""
Language Pack Module — language-agnostic abstraction for the pipeline.

A LanguagePack encapsulates everything needed to transcribe a specific language:
- Transcription model(s) and alignment model
- Language code and name
- Vocabulary (domain-specific words, proper nouns)
- Normalization rules
- Dialect packs (for languages with dialect variation)
- Confidence pairs (for dialect-standard mismatch detection)

This is the single source of truth for language-specific configuration.
All pipeline modules should load from here rather than hardcoding "no".

Usage:
    from .language_pack import LanguagePack, get_language_pack

    # Get language pack for Norwegian Bokmål
    pack = get_language_pack("no")
    model = pack.transcription_model  # "NbAiLab/nb-whisper-large-verbatim"
    dialects = pack.get_available_dialects()  # ["northern_norwegian", ...]

    # Get language pack for Swedish
    sv_pack = get_language_pack("sv")
    sv_model = sv_pack.transcription_model  # "KBLab/kb-whisper-large"
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .utils import get_logger
from .model_registry import get_model_config, get_alignment_model, get_available_languages
from .dialect_pack import DialectPack, get_available_dialects

logger = get_logger("language_pack")

# Module-level cache for loaded language packs
_language_pack_cache: Dict[str, "LanguagePack"] = {}


class LanguagePack:
    """
    Encapsulates all language-specific configuration for the pipeline.

    Each instance provides:
    - Model configuration (transcription, alignment, fallback)
    - Language metadata (code, name, script direction)
    - Dialect packs (for languages with dialect variation)
    - Normalization hints (whether to use Norwegian-style normalization)
    - Vocabulary loading helpers
    """

    def __init__(self, language_code: str):
        """
        Initialize from model registry configuration.

        Args:
            language_code: ISO 639-1 language code (e.g. "no", "sv", "da", "en")
        """
        self.language_code = language_code
        self._config = get_model_config(language_code)

        # Model configuration
        self.transcription_model: str = self._config.get("transcription", "")
        self.alignment_model: Optional[str] = self._config.get("alignment")
        self.fallback_model: str = self._config.get("fallback", "")
        self.multilingual_fallback: str = self._config.get(
            "multilingual_fallback", "openai/whisper-large-v3"
        )

        # Language metadata
        self.language_name: str = self._config.get("language_name", "Unknown")
        self.script_direction: str = self._config.get("script_direction", "ltr")
        self.written_standards: List[str] = self._config.get("written_standards", [])

        # Dialect support
        self._dialect_region_keys: List[str] = self._config.get("dialects", [])

        # Normalization hints
        self.use_norwegian_normalization: bool = self._config.get(
            "use_norwegian_normalization", False
        )
        self.use_dialect_confidence_pairs: bool = self._config.get(
            "use_dialect_confidence_pairs", False
        )

        # Loaded dialect packs (lazy)
        self._dialect_packs: Dict[str, DialectPack] = {}

        # Default language for fallback when detection is uncertain
        self.default_fallback: bool = self._config.get("default_fallback", False)

    @property
    def has_dialects(self) -> bool:
        """Whether this language has dialect packs available."""
        return len(self._dialect_region_keys) > 0

    @property
    def is_norwegian(self) -> bool:
        """Whether this is a Norwegian language variant (Bokmål or Nynorsk)."""
        return self.language_code in ("no", "nn")

    def get_available_dialects(self) -> List[str]:
        """Get list of dialect region keys for this language."""
        return list(self._dialect_region_keys)

    def load_dialect_pack(self, region: str) -> Optional[DialectPack]:
        """
        Load a specific dialect pack for this language.

        Args:
            region: Dialect region key (e.g. "northern_norwegian")

        Returns:
            DialectPack instance or None if not found
        """
        if region in self._dialect_packs:
            return self._dialect_packs[region]

        if region not in self._dialect_region_keys:
            logger.warning(
                f"Dialect '{region}' is not registered for language "
                f"'{self.language_code}'. Available: {self._dialect_region_keys}"
            )
            return None

        try:
            pack = DialectPack.load(region)
            self._dialect_packs[region] = pack
            return pack
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            logger.warning(f"Failed to load dialect pack '{region}': {e}")
            return None

    def detect_dialect(self, text: str) -> Optional[str]:
        """
        Auto-detect dialect region from transcribed text.

        Only works for languages with dialect packs (currently Norwegian).

        Args:
            text: Transcribed text to analyze

        Returns:
            Dialect region key or None
        """
        if not self.has_dialects or not text:
            return None
        return DialectPack.detect_dialect(text)

    def get_alignment_model(self, written_standard: Optional[str] = None) -> Optional[str]:
        """
        Get the alignment model for this language.

        Args:
            written_standard: Optional written standard override
                             (e.g. "nn" for Nynorsk when language is "no")

        Returns:
            Alignment model name or None
        """
        return get_alignment_model(self.language_code, written_standard)

    def to_dict(self) -> dict:
        """Serialize to dictionary for detection reports."""
        return {
            "language_code": self.language_code,
            "language_name": self.language_name,
            "transcription_model": self.transcription_model,
            "alignment_model": self.alignment_model,
            "fallback_model": self.fallback_model,
            "has_dialects": self.has_dialects,
            "dialect_regions": self._dialect_region_keys,
            "script_direction": self.script_direction,
            "written_standards": self.written_standards,
            "default_fallback": self.default_fallback,
        }


def get_language_pack(language_code: str) -> LanguagePack:
    """
    Get (or create) a LanguagePack for the given language code.

    Results are cached so the same pack is reused across the pipeline.

    Args:
        language_code: ISO 639-1 language code

    Returns:
        LanguagePack instance
    """
    if language_code not in _language_pack_cache:
        _language_pack_cache[language_code] = LanguagePack(language_code)
    return _language_pack_cache[language_code]


def get_default_language_pack() -> LanguagePack:
    """
    Get the default language pack (Norwegian Bokmål).

    Used as fallback when language detection is uncertain.
    """
    return get_language_pack("no")


def resolve_language_pack(
    detected_language: str,
    confidence: float,
    force_language: Optional[str] = None,
) -> LanguagePack:
    """
    Resolve the effective language pack to use for transcription.

    Applies confidence-based routing:
    - If force_language is set: use that language pack (CLI override)
    - If confidence >= 0.8: use detected language pack
    - If confidence >= 0.5: use detected language pack, flag for review
    - If confidence < 0.5: fall back to Norwegian Bokmål

    Args:
        detected_language: Language code from language detection
        confidence: Detection confidence (0.0-1.0)
        force_language: Optional override language code

    Returns:
        LanguagePack instance for transcription
    """
    from .model_registry import resolve_language

    effective_code = resolve_language(detected_language, confidence, force_language)
    return get_language_pack(effective_code)


def get_available_language_packs() -> List[str]:
    """Get list of language codes with registered language packs."""
    return get_available_languages()
