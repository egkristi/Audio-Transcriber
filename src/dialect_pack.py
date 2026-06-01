"""
Dialect Pack Module

Loads dialect-specific data from JSON files in src/dialect_data/ (tracked)
or data/dialects/ (gitignored, for local overrides).
Provides a unified interface for all dialect-related data:
- Dialect-to-standard word mappings (for flagging, not auto-correction)
- Dialect region detection markers
- Vocabulary for Whisper initial_prompt injection
- Confidence pairs for dialect-standard mismatch detection
- Common function word exclusion sets
- VAD presets (per-dialect VAD parameters)
- Decoding presets (per-dialect Whisper decoding parameters)

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

    # Get VAD presets for this dialect
    vad = pack.get_vad_presets()  # {"vad_onset": 0.300, "vad_offset": 0.400, ...}

    # Get decoding presets for this dialect
    decoding = pack.get_decoding_presets()  # {"beam_size": 5, "temperatures": [...], ...}
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .utils import get_logger

logger = get_logger("dialect_pack")

# Default paths for dialect data files — checked in order:
# 1. src/dialect_data/ (tracked in git, canonical location)
# 2. data/dialects/ (gitignored, for local overrides)
_DIALECTS_DIRS = [
    Path(__file__).resolve().parent / "dialect_data",
    Path(__file__).resolve().parent.parent / "data" / "dialects",
]

# Module-level cache for loaded dialect packs
_dialect_cache: Dict[str, "DialectPack"] = {}

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
                  common_function_words, vad_presets, decoding_presets
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

        # VAD presets: per-dialect VAD parameters (vad_onset, vad_offset, chunk_size)
        self.vad_presets: Dict = data.get("vad_presets", {})

        # Decoding presets: per-dialect Whisper decoding parameters
        self.decoding_presets: Dict = data.get("decoding_presets", {})

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

    def get_vad_presets(self) -> Dict:
        """
        Get VAD parameter presets for this dialect.

        Returns a dict with keys: vad_onset, vad_offset, chunk_size, description.
        Returns empty dict if no VAD presets are defined.
        """
        return dict(self.vad_presets)

    def get_decoding_presets(self) -> Dict:
        """
        Get decoding parameter presets for this dialect.

        Returns a dict with keys: beam_size, temperatures, repetition_penalty,
        no_repeat_ngram_size, condition_on_previous_text, description.
        Returns empty dict if no decoding presets are defined.
        """
        return dict(self.decoding_presets)

    def apply_vad_presets(self, config: Dict) -> Dict:
        """
        Apply VAD presets to a transcription config dict.

        Merges dialect VAD presets into the config's vad_options section.
        Only overrides keys that are present in the presets.
        Returns the updated config dict (modified in place and returned).

        Args:
            config: Transcription config dict (e.g. from config.yaml transcription section)

        Returns:
            Updated config dict with VAD presets applied
        """
        if not self.vad_presets:
            return config

        vad_options = dict(config.get("vad_options", {}))
        for key in ("vad_onset", "vad_offset", "chunk_size"):
            if key in self.vad_presets:
                vad_options[key] = self.vad_presets[key]
        config["vad_options"] = vad_options
        return config

    def apply_decoding_presets(self, config: Dict) -> Dict:
        """
        Apply decoding presets to a transcription config dict.

        Merges dialect decoding presets into the config.
        Only overrides keys that are present in the presets.
        Returns the updated config dict (modified in place and returned).

        Args:
            config: Transcription config dict

        Returns:
            Updated config dict with decoding presets applied
        """
        if not self.decoding_presets:
            return config

        for key in ("beam_size", "temperatures", "repetition_penalty",
                     "no_repeat_ngram_size", "condition_on_previous_text"):
            if key in self.decoding_presets:
                config[key] = self.decoding_presets[key]
        return config

    def to_dict(self) -> dict:
        """Serialize back to a dictionary."""
        result = {
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
        if self.vad_presets:
            result["vad_presets"] = self.vad_presets
        if self.decoding_presets:
            result["decoding_presets"] = self.decoding_presets
        return result

    @staticmethod
    def _resolve_dialects_dir(dialects_dir: Optional[Path] = None) -> Path:
        """
        Resolve the dialects directory to use.

        If dialects_dir is explicitly provided, use it.
        Otherwise, search _DIALECTS_DIRS in order and return the first one
        that exists and contains dialect files.

        Args:
            dialects_dir: Explicit path override

        Returns:
            Path to the dialects directory to use
        """
        if dialects_dir is not None:
            return dialects_dir

        for d in _DIALECTS_DIRS:
            if d.exists() and any(f.suffix == ".json" for f in d.iterdir()):
                return d

        # Fall back to first dir even if it doesn't exist (will raise proper error)
        return _DIALECTS_DIRS[0]

    @staticmethod
    def load(region: str, dialects_dir: Optional[Path] = None) -> "DialectPack":
        """
        Load a dialect pack by region name.

        Searches src/dialect_data/ first (tracked in git), then
        data/dialects/ (gitignored, for local overrides).

        Args:
            region: Dialect region key (e.g. "northern_norwegian", "trondersk")
            dialects_dir: Path to dialects data directory. If None, searches
                          default directories in order.

        Returns:
            Loaded DialectPack instance.

        Raises:
            FileNotFoundError: If no dialect pack file exists for the region.
        """
        # Check cache first
        if region in _dialect_cache:
            return _dialect_cache[region]

        if dialects_dir is not None:
            dirs_to_search = [dialects_dir]
        else:
            dirs_to_search = _DIALECTS_DIRS

        last_error = None
        for dir_path in dirs_to_search:
            file_path = dir_path / f"{region}.json"
            if not file_path.exists():
                last_error = FileNotFoundError(
                    f"Dialect pack not found: {file_path}"
                )
                continue
            try:
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)
                pack = DialectPack(data)
                _dialect_cache[region] = pack
                logger.debug(f"Loaded dialect pack: {region} ({pack.label})")
                return pack
            except json.JSONDecodeError as e:
                last_error = ValueError(f"Invalid JSON in dialect pack {file_path}: {e}")
                continue
            except Exception as e:
                last_error = RuntimeError(f"Failed to load dialect pack {file_path}: {e}")
                continue

        # If we get here, all directories failed
        available = get_available_dialects(dialects_dir)
        raise FileNotFoundError(
            f"Dialect pack '{region}' not found in any search directory. "
            f"Available dialects: {', '.join(available) if available else 'none'}"
        ) from last_error

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

        dir_path = DialectPack._resolve_dialects_dir(dialects_dir)
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

    Scans all dialect data directories for .json files and returns
    the union of all found regions.

    Args:
        dialects_dir: Path to dialects data directory. If None, searches
                      all default directories.

    Returns:
        Sorted list of dialect region keys (filenames without .json)
    """
    if dialects_dir is not None:
        dirs_to_scan = [dialects_dir]
    else:
        dirs_to_scan = _DIALECTS_DIRS

    all_regions: Set[str] = set()
    for dir_path in dirs_to_scan:
        if not dir_path.exists():
            continue
        regions = {
            f.stem for f in dir_path.iterdir()
            if f.suffix == ".json" and not f.stem.startswith("_")
        }
        all_regions.update(regions)

    return sorted(all_regions)


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
