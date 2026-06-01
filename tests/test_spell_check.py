"""Tests for src/spell_check.py — Norwegian spell-checking module."""

import pytest

from src.spell_check import (
    NorwegianSpellChecker,
    check_transcription,
)


class TestNorwegianSpellChecker:
    """Tests for NorwegianSpellChecker — core spell-check operations."""

    def test_init_without_dictionary(self):
        """Checker initializes even without a dictionary (graceful fallback)."""
        checker = NorwegianSpellChecker()
        # Should not crash — symspell_available may be False if no dict
        assert checker is not None

    def test_check_word_empty(self):
        checker = NorwegianSpellChecker()
        is_correct, suggestion = checker.check_word("")
        assert is_correct is True
        assert suggestion is None

    def test_check_word_short(self):
        checker = NorwegianSpellChecker()
        is_correct, suggestion = checker.check_word("a")
        assert is_correct is True
        assert suggestion is None

    def test_check_word_acronym(self):
        """All-uppercase words (acronyms) should be skipped."""
        checker = NorwegianSpellChecker()
        is_correct, suggestion = checker.check_word("NAV")
        assert is_correct is True

    def test_check_text_empty(self):
        checker = NorwegianSpellChecker()
        errors = checker.check_text("")
        assert errors == []

    def test_check_text_no_errors(self):
        checker = NorwegianSpellChecker()
        # All words are in the Norwegian dictionary
        errors = checker.check_text("dette er en test")
        assert errors == []

    def test_check_text_with_errors(self):
        """Misspelled words should be detected."""
        checker = NorwegianSpellChecker()
        errors = checker.check_text("dette er en hestn")
        assert len(errors) >= 1
        # "hestn" should be flagged as misspelled
        hestn_errors = [e for e in errors if e["word"] == "hestn"]
        assert len(hestn_errors) >= 1
        assert hestn_errors[0]["suggestion"] == "festn"

    def test_check_text_unknown_words(self):
        """Truly unknown words (not in dict, no close match) should be flagged."""
        checker = NorwegianSpellChecker()
        errors = checker.check_text("dette er qwerty")
        assert len(errors) >= 1
        qwerty_errors = [e for e in errors if e["word"] == "qwerty"]
        assert len(qwerty_errors) >= 1
        # Unknown words have no suggestion
        assert qwerty_errors[0]["suggestion"] is None

    def test_correct_text_no_auto_fix(self):
        checker = NorwegianSpellChecker()
        corrected, corrections = checker.correct_text("dette er en test")
        assert corrected == "dette er en test"
        assert corrections == []

    def test_correct_text_with_auto_fix(self):
        checker = NorwegianSpellChecker()
        corrected, corrections = checker.correct_text("dette er en test", auto_fix=True)
        # All words are correct, no corrections
        assert corrections == []

    def test_correct_text_with_auto_fix_misspelled(self):
        """Auto-fix should correct misspelled words."""
        checker = NorwegianSpellChecker()
        corrected, corrections = checker.correct_text("dette er en hestn", auto_fix=True)
        assert len(corrections) >= 1
        hestn_corrections = [c for c in corrections if c["original"] == "hestn"]
        assert len(hestn_corrections) >= 1
        assert hestn_corrections[0]["corrected"] == "festn"

    def test_check_numbers(self):
        checker = NorwegianSpellChecker()
        errors = checker.check_numbers("en to tre fire fem")
        assert len(errors) >= 1  # At least some number patterns match

    def test_check_proper_nouns_empty_known(self):
        checker = NorwegianSpellChecker()
        errors = checker.check_proper_nouns("Ola og Kari", known_nouns=[])
        assert len(errors) >= 1  # "Ola" and "Kari" are capitalized

    def test_check_proper_nouns_with_known(self):
        checker = NorwegianSpellChecker()
        errors = checker.check_proper_nouns(
            "Ola og Kari", known_nouns=["Ola", "Kari"]
        )
        # Both are known, so no errors
        assert errors == []


class TestCheckTranscription:
    """Tests for check_transcription — top-level convenience function."""

    def test_disabled_by_default(self):
        result = check_transcription("dette er en test")
        assert result["enabled"] is False
        assert result["errors"] == []

    def test_enabled_via_config(self):
        result = check_transcription(
            "dette er en test",
            config={"enabled": True}
        )
        # Should run without crashing even without dictionary
        assert result["enabled"] is True
        assert "error_count" in result

    def test_empty_text(self):
        result = check_transcription("", config={"enabled": True})
        assert result["enabled"] is True
        assert result["errors"] == []
