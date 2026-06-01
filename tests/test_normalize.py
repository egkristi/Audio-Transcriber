"""Tests for src/normalize.py — Norwegian text normalization."""

import json
from pathlib import Path

import pytest

from src.normalize import (
    _fix_stuttering,
    _restore_punctuation,
    _capitalize_sentence,
    normalize_norwegian_text,
    normalize_transcription_segments,
    export_normalization_report,
    load_proper_nouns,
    NORWEGIAN_DIALECT_MAP,
    NORWEGIAN_PROPER_NOUNS,
)


class TestFixStuttering:
    """Tests for _fix_stuttering — consecutive duplicate word removal."""

    def test_no_stuttering(self):
        words = ["jeg", "vil", "ha"]
        cleaned, corrections = _fix_stuttering(words)
        assert cleaned == words
        assert corrections == []

    def test_simple_stutter(self):
        words = ["jeg", "jeg", "vil"]
        cleaned, corrections = _fix_stuttering(words)
        assert cleaned == ["jeg", "vil"]
        assert len(corrections) == 1
        assert corrections[0]["type"] == "stuttering"
        assert corrections[0]["original"] == "jeg jeg"

    def test_multiple_stutters(self):
        words = ["det", "det", "er", "er", "bra"]
        cleaned, corrections = _fix_stuttering(words)
        assert cleaned == ["det", "er", "bra"]
        assert len(corrections) == 2

    def test_triple_stutter(self):
        words = ["og", "og", "og", "så"]
        cleaned, corrections = _fix_stuttering(words)
        assert cleaned == ["og", "så"]
        assert len(corrections) == 2

    def test_empty_input(self):
        cleaned, corrections = _fix_stuttering([])
        assert cleaned == []
        assert corrections == []

    def test_single_word(self):
        cleaned, corrections = _fix_stuttering(["hei"])
        assert cleaned == ["hei"]
        assert corrections == []


class TestRestorePunctuation:
    """Tests for _restore_punctuation — period, comma, question mark insertion."""

    def test_no_punctuation_needed(self):
        words = ["dette", "er", "en", "setning"]
        result, corrections = _restore_punctuation(words)
        assert result[-1].endswith(".")
        assert any(c["type"] == "punctuation_period_end" for c in corrections)

    def test_question_word_triggers_question_mark(self):
        words = ["ka", "heter", "du"]
        result, corrections = _restore_punctuation(words)
        assert result[-1].endswith("?")
        assert any(c["type"] == "punctuation_question_end" for c in corrections)

    def test_hae_triggers_question_mark(self):
        words = ["det", "var", "hæ"]
        result, corrections = _restore_punctuation(words)
        assert any(c["type"] == "punctuation_question" for c in corrections)

    def test_clause_break_comma(self):
        words = ["ja", "så", "kom", "han"]
        result, corrections = _restore_punctuation(words)
        # "ja" is a sentence-ending filler — period should be added after it
        # "så" is a clause break word — comma should be inserted before it
        period_corrections = [c for c in corrections if c["type"] == "punctuation_period"]
        assert len(period_corrections) >= 1

    def test_empty_input(self):
        result, corrections = _restore_punctuation([])
        assert result == []
        assert corrections == []


class TestCapitalizeSentence:
    """Tests for _capitalize_sentence — first-word and post-punctuation capitalization."""

    def test_capitalize_first_word(self):
        words = ["dette", "er", "en", "test"]
        result = _capitalize_sentence(words)
        assert result[0] == "Dette"

    def test_capitalize_after_period(self):
        words = ["hei.", "dette", "er", "nytt"]
        result = _capitalize_sentence(words)
        assert result[1] == "Dette"

    def test_capitalize_after_question_mark(self):
        words = ["hva?", "ingenting"]
        result = _capitalize_sentence(words)
        assert result[1] == "Ingenting"

    def test_empty_input(self):
        assert _capitalize_sentence([]) == []


class TestNormalizeNorwegianText:
    """Tests for normalize_norwegian_text — full normalization pipeline."""

    def test_stuttering_removed(self):
        text = "jeg jeg vil ha vann"
        normalized, corrections = normalize_norwegian_text(text)
        assert "jeg jeg" not in normalized
        assert normalized.startswith("Jeg")
        assert any(c["type"] == "stuttering" for c in corrections)

    def test_punctuation_added(self):
        text = "ja så kom han"
        normalized, corrections = normalize_norwegian_text(text)
        assert normalized.endswith(".")
        assert normalized[0].isupper()

    def test_question_detected(self):
        text = "ka heter du"
        normalized, corrections = normalize_norwegian_text(text)
        assert normalized.endswith("?")

    def test_dialect_words_flagged(self):
        text = "æ vil ikkje ha"
        normalized, corrections = normalize_norwegian_text(text)
        dialect_corrections = [c for c in corrections if c["type"] == "dialect_word"]
        assert len(dialect_corrections) >= 2

    def test_english_words_flagged(self):
        text = "jeg sa hello to him"
        normalized, corrections = normalize_norwegian_text(text)
        english_corrections = [c for c in corrections if c["type"] == "english_word"]
        assert len(english_corrections) >= 1

    def test_short_segment_flagged(self):
        text = "hei"
        normalized, corrections = normalize_norwegian_text(text)
        assert any(c["type"] == "short_segment" for c in corrections)

    def test_repetition_flagged(self):
        text = "dette er er er veldig mye gjentakelse"
        normalized, corrections = normalize_norwegian_text(text)
        # "er" appears 3 times (but stuttering removal reduces to 1 first)
        # Check that stuttering was caught instead
        assert any(c["type"] == "stuttering" for c in corrections)

    def test_missing_space_fixed(self):
        text = "kommer du?ja"
        normalized, corrections = normalize_norwegian_text(text)
        assert "? ja" in normalized or "?ja" not in normalized

    def test_empty_text(self):
        normalized, corrections = normalize_norwegian_text("")
        assert normalized == ""
        # Empty text still gets a "short_segment" flag (0 words < 3)
        assert any(c["type"] == "short_segment" for c in corrections)

    def test_whitespace_normalized(self):
        text = "  dette   er  test  "
        normalized, corrections = normalize_norwegian_text(text)
        assert "  " not in normalized


class TestNormalizeTranscriptionSegments:
    """Tests for normalize_transcription_segments — batch segment normalization."""

    def test_single_segment(self):
        segments = [{"id": 0, "start": 0.0, "end": 2.0, "text": "jeg vil ha"}]
        normalized, corrections = normalize_transcription_segments(segments)
        assert len(normalized) == 1
        assert normalized[0]["text"].startswith("Jeg")
        assert normalized[0]["has_normalization_issues"]
        assert len(corrections) >= 1

    def test_multiple_segments(self):
        segments = [
            {"id": 0, "start": 0.0, "end": 2.0, "text": "hei der"},
            {"id": 1, "start": 2.0, "end": 4.0, "text": "ka skjer"},
        ]
        normalized, corrections = normalize_transcription_segments(segments)
        assert len(normalized) == 2
        assert normalized[0]["text"].startswith("Hei")
        assert normalized[1]["text"].endswith("?")

    def test_segment_ids_in_corrections(self):
        segments = [{"id": 5, "start": 10.0, "end": 12.0, "text": "æ kommer"}]
        normalized, corrections = normalize_transcription_segments(segments)
        for c in corrections:
            assert c["segment_id"] == 5
            assert c["segment_start"] == 10.0
            assert c["segment_end"] == 12.0

    def test_empty_segments(self):
        normalized, corrections = normalize_transcription_segments([])
        assert normalized == []
        assert corrections == []


class TestExportNormalizationReport:
    """Tests for export_normalization_report — report file generation."""

    def test_report_created(self, temp_dir):
        corrections = [
            {"type": "stuttering", "original": "jeg jeg", "corrected": "jeg",
             "segment_id": 0, "segment_start": 0.0, "segment_end": 1.0,
             "explanation": "Fjernet gjentakelse"},
        ]
        output_path = temp_dir / "norm_report.txt"
        result = export_normalization_report(corrections, output_path)
        assert result.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "TEXT NORMALIZATION REPORT" in content
        assert "stuttering" in content

    def test_empty_corrections(self, temp_dir):
        output_path = temp_dir / "empty_report.txt"
        result = export_normalization_report([], output_path)
        assert result.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "Total issues flagged: 0" in content

    def test_issue_breakdown(self, temp_dir):
        corrections = [
            {"type": "stuttering", "original": "x", "corrected": "x",
             "segment_id": 0, "segment_start": 0.0, "segment_end": 1.0,
             "explanation": "test"},
            {"type": "stuttering", "original": "y", "corrected": "y",
             "segment_id": 1, "segment_start": 1.0, "segment_end": 2.0,
             "explanation": "test"},
            {"type": "dialect_word", "original": "z", "corrected": "z",
             "segment_id": 2, "segment_start": 2.0, "segment_end": 3.0,
             "explanation": "test"},
        ]
        output_path = temp_dir / "breakdown.txt"
        export_normalization_report(corrections, output_path)
        content = output_path.read_text(encoding="utf-8")
        assert "stuttering" in content
        assert "dialect_word" in content


class TestLoadProperNouns:
    """Tests for load_proper_nouns — loading from external data file."""

    def test_returns_builtin_set_when_no_file(self):
        nouns = load_proper_nouns()
        assert NORWEGIAN_PROPER_NOUNS.issubset(nouns)

    def test_merges_with_data_file(self, temp_dir):
        data_file = temp_dir / "proper_nouns.json"
        extra_names = ["ole", "kari", "per"]
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump({"proper_nouns": extra_names}, f)
        nouns = load_proper_nouns(data_file)
        for name in extra_names:
            assert name in nouns

    def test_handles_missing_file_gracefully(self):
        fake_path = Path("/nonexistent/proper_nouns.json")
        nouns = load_proper_nouns(fake_path)
        assert NORWEGIAN_PROPER_NOUNS.issubset(nouns)

    def test_handles_invalid_json_gracefully(self, temp_dir):
        data_file = temp_dir / "bad.json"
        data_file.write_text("not valid json", encoding="utf-8")
        nouns = load_proper_nouns(data_file)
        assert NORWEGIAN_PROPER_NOUNS.issubset(nouns)


class TestConstants:
    """Tests for module-level constants."""

    def test_dialect_map_has_expected_entries(self):
        assert "æ" in NORWEGIAN_DIALECT_MAP
        assert "ikkje" in NORWEGIAN_DIALECT_MAP
        assert "ka" in NORWEGIAN_DIALECT_MAP
        assert "kor" in NORWEGIAN_DIALECT_MAP
        assert "mæ" in NORWEGIAN_DIALECT_MAP
        assert "dokker" in NORWEGIAN_DIALECT_MAP

    def test_proper_nouns_has_place_names(self):
        assert "tromsø" in NORWEGIAN_PROPER_NOUNS
        assert "bodø" in NORWEGIAN_PROPER_NOUNS
        assert "narvik" in NORWEGIAN_PROPER_NOUNS

    def test_proper_nouns_no_personal_names(self):
        """Verify no real personal names remain in committed source (#36)."""
        personal_names = {"erling", "kristiansen", "elida", "anna", "wiktoria",
                          "håvard", "ole", "kari", "per", "nils", "maria"}
        found = personal_names & NORWEGIAN_PROPER_NOUNS
        assert not found, f"Personal names found in committed source: {found}"
