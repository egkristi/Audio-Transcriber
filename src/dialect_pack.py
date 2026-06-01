"""
Dialect Pack Module

Loads dialect-specific data from JSON files in data/dialects/.
Provides a unified interface for all dialect-related data:
- Dialect-to-standard word mappings (for flagging, not auto-correction)
- Dialect region detection markers
- Vocabulary for Whisper initial_prompt injection
- Confidence pairs for dialect-standard mismatch detection
- Common function word exclusion sets

This is the single source of truth for dialect data. All other modules
(normalize.py, vocabulary.py, confidence.py) should load from here
rather than maintaining their own hardcoded maps.

Usage:
    from .dialect_pack import DialectPack, get_available_dialects

    # Load a specific dialect pack
    pack = DialectPack.load("northern_norwegian")

    # Get dialect-to-standard map
    dialect_map = pack.dialect_map  # {"æ": "jeg", "ikkje": "ikke", ...}

    # Get dialect words set (all dialect forms)
    dialect_words = pack.dialect_words  # {"æ", "mæ", "dæ", ...}

    # Get confidence pairs for mismatch detection
    pairs = pack.confidence_pairs  # [["jeg", "æ"], ["meg", "mæ"], ...]

    # Get vocabulary for Whisper prompt
    vocab = pack.get_flat_vocabulary()  # ["æ", "mæ", "dæ", ...]

    # Detect dialect from text
    detected = DialectPack.detect_dialect("æ e ikkje sikker")
    # Returns "northern_norwegian"
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .utils import get_logger

logger = get_logger("dialect_pack")

# Default path for dialect data files
_DIALECTS_DIR = Path(__file__).resolve().parent.parent / "data" / "dialects"

# Module-level cache for loaded dialect packs
_dialect_cache: Dict[str, "DialectPack"] = {}


class DialectPack:
    """A single dialect pack loaded from a JSON data file."""

    def __init__(self, data: dict):
        """
        Initialize from a loaded JSON dictionary.

        Args:
            data: Parsed JSON content from a dialect pack file.
                  Expected keys: region, label, description, weight,
                  dialect_map, markers, vocabulary, confidence_pairs,
                  common_function_words
        """
        self.region: str = data.get("region", "unknown")
        self.label: str = data.get("label", self.region)
        self.description: str = data.get("description", "")
        self.weight: float = data.get("weight", 1.0)

        # Dialect-to-standard mapping (for flagging, not auto-correction)
        self.dialect_map: Dict[str, str] = data.get("dialect_map", {})

        # Dialect region detection markers
        markers_data = data.get("markers", {})
        self.marker_words: Set[str] = set(markers_data.get("words", []))
        self.marker_weight: float = markers_data.get("weight", self.weight)

        # Categorized vocabulary for Whisper prompt injection
        self.vocabulary: Dict[str, List[str]] = data.get("vocabulary", {})

        # Confidence pairs: [[standard, dialect], ...] for mismatch detection
        self.confidence_pairs: List[List[str]] = data.get("confidence_pairs", [])

        # Common function words to exclude from dialect counting
        self.common_function_words: Set[str] = set(
            data.get("common_function_words", [])
        )

    @property
    def dialect_words(self) -> Set[str]:
        """All dialect word forms from the dialect_map."""
        return set(self.dialect_map.keys())

    @property
    def standard_words(self) -> Set[str]:
        """All standard equivalents from the dialect_map."""
        return set(self.dialect_map.values())

    def get_flat_vocabulary(self) -> List[str]:
        """Flatten categorized vocabulary into a single list."""
        words: List[str] = []
        for category_words in self.vocabulary.values():
            words.extend(category_words)
        return words

    def get_dialect_map_forward(self) -> Dict[str, str]:
        """Get dialect → standard mapping (same as dialect_map)."""
        return dict(self.dialect_map)

    def get_dialect_map_reverse(self) -> Dict[str, List[str]]:
        """Get standard → [dialect forms] mapping."""
        reverse: Dict[str, List[str]] = {}
        for dialect, standard in self.dialect_map.items():
            if dialect == standard:
                continue
            if standard not in reverse:
                reverse[standard] = []
            reverse[standard].append(dialect)
        return reverse

    def to_dict(self) -> dict:
        """Serialize back to a dictionary."""
        return {
            "region": self.region,
            "label": self.label,
            "description": self.description,
            "weight": self.weight,
            "dialect_map": self.dialect_map,
            "markers": {
                "words": sorted(self.marker_words),
                "weight": self.marker_weight,
            },
            "vocabulary": self.vocabulary,
            "confidence_pairs": self.confidence_pairs,
            "common_function_words": sorted(self.common_function_words),
        }

    @staticmethod
    def load(region: str, dialects_dir: Optional[Path] = None) -> "DialectPack":
        """
        Load a dialect pack by region name.

        Args:
            region: Dialect region key (e.g. "northern_norwegian", "trondersk")
            dialects_dir: Path to dialects data directory. Defaults to
                          data/dialects/ relative to this file.

        Returns:
            Loaded DialectPack instance.

        Raises:
            FileNotFoundError: If no dialect pack file exists for the region.
        """
        # Check cache first
        if region in _dialect_cache:
            return _dialect_cache[region]

        dir_path = dialects_dir or _DIALECTS_DIR
        file_path = dir_path / f"{region}.json"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Dialect pack not found: {file_path}. "
                f"Available dialects: {', '.join(get_available_dialects(dir_path))}"
            )

        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
            pack = DialectPack(data)
            _dialect_cache[region] = pack
            logger.debug(f"Loaded dialect pack: {region} ({pack.label})")
            return pack
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in dialect pack {file_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load dialect pack {file_path}: {e}")

    @staticmethod
    def detect_dialect(
        text: str,
        dialects_dir: Optional[Path] = None,
    ) -> Optional[str]:
        """
        Auto-detect dialect region from transcribed text.

        Analyzes the text for distinctive dialect markers and returns
        the most likely dialect region, or None if no clear match.

        Args:
            text: Transcribed text to analyze
            dialects_dir: Path to dialects data directory

        Returns:
            Dialect region key (e.g. "northern_norwegian") or None
        """
        if not text or not text.strip():
            return None

        text_lower = text.lower()
        words = set(text_lower.split())

        dir_path = dialects_dir or _DIALECTS_DIR
        scores: Dict[str, float] = {}

        for region in get_available_dialects(dir_path):
            try:
                pack = DialectPack.load(region, dir_path)
                matches = words & pack.marker_words
                if matches:
                    score = len(matches) * pack.marker_weight
                    scores[region] = score
            except (FileNotFoundError, ValueError, RuntimeError):
                continue

        if not scores:
            return None

        best = max(scores, key=scores.get)
        logger.info(
            f"Dialect detection: {best} "
            f"(scores: {dict(sorted(scores.items(), key=lambda x: -x[1]))})"
        )
        return best

    @staticmethod
    def detect_dialect_from_segments(
        segments: List[Dict],
        dialects_dir: Optional[Path] = None,
    ) -> Optional[str]:
        """
        Auto-detect dialect from a list of transcription segments.

        Concatenates all segment text and runs dialect detection.

        Args:
            segments: List of segment dicts with 'text' field
            dialects_dir: Path to dialects data directory

        Returns:
            Dialect region key or None
        """
        full_text = " ".join(
            seg.get("text", "") for seg in segments if seg.get("text")
        )
        return DialectPack.detect_dialect(full_text, dialects_dir)

    @staticmethod
    def clear_cache():
        """Clear the module-level dialect pack cache."""
        _dialect_cache.clear()


def get_available_dialects(dialects_dir: Optional[Path] = None) -> List[str]:
    """
    Get list of available dialect region keys.

    Scans the dialects data directory for .json files.

    Args:
        dialects_dir: Path to dialects data directory

    Returns:
        Sorted list of dialect region keys (filenames without .json)
    """
    dir_path = dialects_dir or _DIALECTS_DIR
    if not dir_path.exists():
        logger.warning(f"Dialects directory not found: {dir_path}")
        return []

    regions = sorted(
        f.stem for f in dir_path.iterdir()
        if f.suffix == ".json" and not f.stem.startswith("_")
    )
    return regions


def load_dialect_map(
    region: str = "northern_norwegian",
    dialects_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """Convenience function: load dialect-to-standard map for a region."""
    pack = DialectPack.load(region, dialects_dir)
    return pack.dialect_map


def load_dialect_vocabulary(
    region: str = "northern_norwegian",
    dialects_dir: Optional[Path] = None,
) -> List[str]:
    """Convenience function: load flat dialect vocabulary for a region."""
    pack = DialectPack.load(region, dialects_dir)
    return pack.get_flat_vocabulary()


def load_confidence_pairs(
    region: str = "northern_norwegian",
    dialects_dir: Optional[Path] = None,
) -> List[List[str]]:
    """Convenience function: load confidence pairs for a region."""
    pack = DialectPack.load(region, dialects_dir)
    return pack.confidence_pairs


def load_dialect_words_set(
    region: str = "northern_norwegian",
    dialects_dir: Optional[Path] = None,
) -> Set[str]:
    """Convenience function: load dialect words set for a region."""
    pack = DialectPack.load(region, dialects_dir)
    return pack.dialect_words


def load_common_function_words(
    region: str = "northern_norwegian",
    dialects_dir: Optional[Path] = None,
) -> Set[str]:
    """Convenience function: load common function words for a region."""
    pack = DialectPack.load(region, dialects_dir)
    return pack.common_function_words
