"""
Model Registry — maps language codes to optimal transcription and alignment models.

Provides a single source of truth for model selection across the pipeline.
Each language entry specifies:
- Transcription model (WhisperX-compatible)
- Alignment model (wav2vec2 or compatible)
- Fallback model for low-resource languages
- Supported dialects (references to dialect packs)

Usage:
    from .model_registry import get_model_config, get_available_languages

    config = get_model_config("no")
    # Returns: {
    #     "transcription": "NbAiLab/nb-whisper-large-verbatim",
    #     "alignment": "NbAiLab/nb-wav2vec2-1b-bokmaal-v2",
    #     "fallback": "NbAiLab/nb-whisper-large",
    #     "dialects": ["northern_norwegian", "trondersk", ...],
    #     "language_name": "Norwegian",
    # }
"""

from typing import Dict, List, Optional

from .utils import get_logger

logger = get_logger("model_registry")

# Language code → model configuration
# Each entry defines the optimal models for that language.
# The 'dialects' list references dialect pack region keys from data/dialects/.
_MODEL_REGISTRY: Dict[str, Dict] = {
    "no": {
        "transcription": "NbAiLab/nb-whisper-large-verbatim",
        "alignment": "NbAiLab/nb-wav2vec2-1b-bokmaal-v2",
        "alignment_nn": "NbAiLab/nb-wav2vec2-1b-nynorsk",
        "fallback": "NbAiLab/nb-whisper-large",
        "dialects": [
            "northern_norwegian",
            "trondersk",
            "vestlandsk",
            "sorlandsk",
            "ostlandsk",
        ],
        "language_name": "Norwegian",
        "multilingual_fallback": "openai/whisper-large-v3",
        "script_direction": "ltr",
        "written_standards": ["no", "nn"],
        "use_norwegian_normalization": True,
        "use_dialect_confidence_pairs": True,
        "default_fallback": True,
    },
    "nn": {
        "transcription": "NbAiLab/nb-whisper-large-verbatim",
        "alignment": "NbAiLab/nb-wav2vec2-1b-nynorsk",
        "fallback": "NbAiLab/nb-whisper-large",
        "dialects": [
            "northern_norwegian",
            "trondersk",
            "vestlandsk",
            "sorlandsk",
            "ostlandsk",
        ],
        "language_name": "Norwegian Nynorsk",
        "multilingual_fallback": "openai/whisper-large-v3",
        "script_direction": "ltr",
        "written_standards": ["nn", "no"],
        "use_norwegian_normalization": True,
        "use_dialect_confidence_pairs": True,
        "default_fallback": False,
    },
    "sv": {
        "transcription": "KBLab/kb-whisper-large",
        "alignment": "KBLab/wav2vec2-large-voxrex-swedish",
        "fallback": "openai/whisper-large-v3",
        "dialects": [],
        "language_name": "Swedish",
        "multilingual_fallback": "openai/whisper-large-v3",
        "script_direction": "ltr",
        "written_standards": ["sv"],
        "use_norwegian_normalization": False,
        "use_dialect_confidence_pairs": False,
        "default_fallback": False,
    },
    "da": {
        "transcription": "openai/whisper-large-v3",
        "alignment": "vesteinn/wav2vec2-large-xlsr-53-danish",
        "fallback": "openai/whisper-large-v3",
        "dialects": [],
        "language_name": "Danish",
        "multilingual_fallback": "openai/whisper-large-v3",
        "script_direction": "ltr",
        "written_standards": ["da"],
        "use_norwegian_normalization": False,
        "use_dialect_confidence_pairs": False,
        "default_fallback": False,
    },
    "en": {
        "transcription": "openai/whisper-large-v3",
        "alignment": "openai/whisper-large-v3",
        "fallback": "openai/whisper-large-v3",
        "dialects": [],
        "language_name": "English",
        "multilingual_fallback": "openai/whisper-large-v3",
        "script_direction": "ltr",
        "written_standards": ["en"],
        "use_norwegian_normalization": False,
        "use_dialect_confidence_pairs": False,
        "default_fallback": False,
    },
}

# Default fallback for unsupported languages
_DEFAULT_FALLBACK = {
    "transcription": "openai/whisper-large-v3",
    "alignment": None,
    "fallback": "openai/whisper-large-v3",
    "dialects": [],
    "language_name": "Unknown",
    "multilingual_fallback": "openai/whisper-large-v3",
    "script_direction": "ltr",
    "written_standards": [],
    "use_norwegian_normalization": False,
    "use_dialect_confidence_pairs": False,
    "default_fallback": False,
}

# Language detection confidence thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.8
MEDIUM_CONFIDENCE_THRESHOLD = 0.5


def get_model_config(language_code: str) -> Dict:
    """
    Get the optimal model configuration for a language code.

    Args:
        language_code: ISO 639-1 language code (e.g. "no", "sv", "da", "en")

    Returns:
        Dict with keys: transcription, alignment, fallback, dialects,
        language_name, multilingual_fallback
    """
    config = _MODEL_REGISTRY.get(language_code)
    if config is None:
        logger.warning(
            f"No model registry entry for language '{language_code}'. "
            f"Falling back to multilingual model. "
            f"Supported: {', '.join(get_available_languages())}"
        )
        return dict(_DEFAULT_FALLBACK)
    return dict(config)


def get_available_languages() -> List[str]:
    """Get list of language codes with registered model configurations."""
    return sorted(_MODEL_REGISTRY.keys())


def get_default_language() -> str:
    """
    Get the default fallback language code.

    Returns the language marked with default_fallback=True in the registry.
    Currently Norwegian Bokmål ("no").
    """
    for code, config in _MODEL_REGISTRY.items():
        if config.get("default_fallback", False):
            return code
    return "no"  # Hardcoded last resort if registry is empty


def get_alignment_model(language_code: str, written_standard: Optional[str] = None) -> Optional[str]:
    """
    Get the alignment model for a language, with optional written standard override.

    For Norwegian Bokmål ("no"), returns the Bokmål wav2vec2 model.
    For Norwegian Nynorsk ("nn"), returns the Nynorsk wav2vec2 model.
    For other languages, returns the default alignment model from the registry.

    Args:
        language_code: ISO 639-1 language code
        written_standard: Optional written standard override ("no" for Bokmål, "nn" for Nynorsk)

    Returns:
        Alignment model name or None if not available
    """
    config = get_model_config(language_code)

    # Handle Norwegian written standard routing
    if language_code == "no" and written_standard == "nn":
        return config.get("alignment_nn", config.get("alignment"))
    if language_code == "nn" and written_standard == "no":
        return config.get("alignment")  # Bokmål model is the default

    return config.get("alignment")


def resolve_language(
    detected_language: str,
    confidence: float,
    force_language: Optional[str] = None,
) -> str:
    """
    Resolve the effective language to use for transcription.

    Applies confidence-based routing:
    - If force_language is set: use it (CLI override)
    - If confidence >= HIGH_CONFIDENCE_THRESHOLD (0.8): use detected language
    - If confidence >= MEDIUM_CONFIDENCE_THRESHOLD (0.5): use detected language, flag for review
    - If confidence < MEDIUM_CONFIDENCE_THRESHOLD (0.5): fall back to Norwegian ("no")

    Args:
        detected_language: Language code from language detection
        confidence: Detection confidence (0.0-1.0)
        force_language: Optional override language code

    Returns:
        Resolved language code for transcription
    """
    if force_language is not None:
        logger.info(f"Language forced to '{force_language}' (CLI override)")
        return force_language

    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        logger.info(
            f"Language detected with high confidence: {detected_language} "
            f"({confidence:.2f})"
        )
        return detected_language

    if confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
        logger.info(
            f"Language detected with medium confidence: {detected_language} "
            f"({confidence:.2f}) — will flag for review"
        )
        return detected_language

    default_lang = get_default_language()
    logger.warning(
        f"Language detection confidence too low ({confidence:.2f}) for "
        f"'{detected_language}', falling back to '{default_lang}'"
    )
    return default_lang
