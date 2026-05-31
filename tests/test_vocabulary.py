"""Tests for src/vocabulary.py — vocabulary management and prompt generation."""

import json
from pathlib import Path

import pytest

from src.vocabulary import (
    VocabularyManager,
    CommonNorwegianVocabulary,
    load_vocabulary,
    count_tokens,
)


class TestVocabularyManager:
    """Tests for VocabularyManager — core vocabulary operations."""

    def test_empty_manager(self):
        manager = VocabularyManager()
        assert manager.vocabulary == {}
        assert manager.contexts == {}

    def test_add_word(self):
        manager = VocabularyManager()
        manager.add_word("testord")
        assert "testord" in manager.vocabulary

    def test_add_word_with_context(self):
        manager = VocabularyManager()
        manager.add_word("pasient", context="medical")
        assert manager.vocabulary["pasient"] == "medical"
        assert manager.contexts["pasient"] == "medical"

    def test_add_words(self):
        manager = VocabularyManager()
        manager.add_words(["ord1", "ord2", "ord3"])
        assert len(manager.vocabulary) == 3

    def test_add_from_dict(self):
        manager = VocabularyManager()
        manager.add_from_dict({"ord1": "context1", "ord2": "context2"})
        assert manager.vocabulary["ord1"] == "context1"
        assert manager.vocabulary["ord2"] == "context2"

    def test_load_from_json_file(self, temp_dir):
        vocab_data = {"vocabulary": ["ord1", "ord2"], "contexts": {"ord1": "test"}}
        vocab_file = temp_dir / "vocab.json"
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f)
        manager = VocabularyManager(vocab_file)
        assert "ord1" in manager.vocabulary
        assert "ord2" in manager.vocabulary

    def test_load_from_simple_list(self, temp_dir):
        vocab_data = ["ord1", "ord2", "ord3"]
        vocab_file = temp_dir / "simple.json"
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f)
        manager = VocabularyManager(vocab_file)
        assert len(manager.vocabulary) == 3

    def test_save_vocabulary(self, temp_dir):
        manager = VocabularyManager()
        manager.add_word("testord", context="test")
        output_path = temp_dir / "output.json"
        manager.save_vocabulary(output_path)
        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
        assert "testord" in data["vocabulary"]

    def test_generate_initial_prompt_empty(self):
        manager = VocabularyManager()
        prompt = manager.generate_initial_prompt()
        assert prompt == ""

    def test_generate_initial_prompt_with_words(self):
        manager = VocabularyManager()
        manager.add_words(["hei", "på", "deg"])
        prompt = manager.generate_initial_prompt()
        assert prompt.startswith("The following words are important:")
        assert "hei" in prompt

    def test_generate_initial_prompt_respects_max_tokens(self):
        manager = VocabularyManager()
        # Add many words to test token limit
        for i in range(100):
            manager.add_word(f"ord_{i}")
        prompt = manager.generate_initial_prompt(max_tokens=50)
        # Should produce a short prompt within the limit
        assert len(prompt) > 0
        assert count_tokens(prompt) <= 60  # Allow small margin

    def test_get_vocabulary_for_transcription_no_domain(self):
        manager = VocabularyManager()
        manager.add_word("test")
        vocab = manager.get_vocabulary_for_transcription()
        assert "test" in vocab

    def test_get_vocabulary_for_transcription_with_domain(self):
        manager = VocabularyManager()
        manager.add_word("pasient", context="medical")
        manager.add_word("server", context="technical")
        medical = manager.get_vocabulary_for_transcription(domain="medical")
        assert "pasient" in medical
        assert "server" not in medical

    def test_suggest_corrections(self):
        manager = VocabularyManager()
        manager.add_word("pasient")
        suggestions = manager.suggest_corrections("pasienten")
        assert len(suggestions) >= 1
        assert suggestions[0]["suggestion"] == "pasient"


class TestCommonNorwegianVocabulary:
    """Tests for CommonNorwegianVocabulary — predefined vocabulary sets."""

    def test_get_domain_vocabulary_medical(self):
        vocab = CommonNorwegianVocabulary.get_domain_vocabulary("medical")
        assert "pasient" in vocab
        assert "lege" in vocab

    def test_get_domain_vocabulary_unknown(self):
        vocab = CommonNorwegianVocabulary.get_domain_vocabulary("nonexistent")
        assert vocab == []

    def test_get_dialect_vocabulary_northern_norwegian(self):
        words = CommonNorwegianVocabulary.get_dialect_vocabulary("northern_norwegian")
        assert "æ" in words
        assert "ikkje" in words
        assert "ka" in words
        assert "dokker" in words

    def test_get_dialect_vocabulary_unknown(self):
        words = CommonNorwegianVocabulary.get_dialect_vocabulary("unknown_dialect")
        assert words == []

    def test_create_manager_default(self):
        manager = CommonNorwegianVocabulary.create_manager()
        assert "Oslo" in manager.vocabulary
        assert "Norge" in manager.vocabulary

    def test_create_manager_with_domain(self):
        manager = CommonNorwegianVocabulary.create_manager(domain="medical")
        assert "pasient" in manager.vocabulary

    def test_create_manager_with_dialect(self):
        manager = CommonNorwegianVocabulary.create_manager(dialect="northern_norwegian")
        assert "æ" in manager.vocabulary
        assert manager.vocabulary["æ"] == "dialect:northern_norwegian"

    def test_create_manager_with_both(self):
        manager = CommonNorwegianVocabulary.create_manager(
            domain="medical", dialect="northern_norwegian"
        )
        assert "pasient" in manager.vocabulary
        assert "æ" in manager.vocabulary


class TestLoadVocabulary:
    """Tests for load_vocabulary — top-level factory function."""

    def test_load_default_norwegian(self):
        """Test loading default Norwegian vocabulary from data file."""
        manager = load_vocabulary(use_default_norwegian=True)
        # Should have at least some vocabulary items
        assert len(manager.vocabulary) > 0

    def test_load_with_dialect(self):
        manager = load_vocabulary(use_default_norwegian=True, dialect="northern_norwegian")
        assert "æ" in manager.vocabulary

    def test_load_with_domain(self):
        manager = load_vocabulary(use_default_norwegian=True, domain="medical")
        assert "pasient" in manager.vocabulary

    def test_load_custom_file(self, temp_dir):
        vocab_data = {"vocabulary": ["custom1", "custom2"], "contexts": {}}
        vocab_file = temp_dir / "custom.json"
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f)
        manager = load_vocabulary(vocab_file=vocab_file, use_default_norwegian=False)
        assert "custom1" in manager.vocabulary
        assert "custom2" in manager.vocabulary

    def test_load_custom_file_with_dialect(self, temp_dir):
        vocab_data = {"vocabulary": ["custom1"], "contexts": {}}
        vocab_file = temp_dir / "custom.json"
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f)
        manager = load_vocabulary(
            vocab_file=vocab_file, dialect="northern_norwegian",
            use_default_norwegian=False
        )
        assert "custom1" in manager.vocabulary
        assert "æ" in manager.vocabulary

    def test_load_empty_fallback(self):
        manager = load_vocabulary(use_default_norwegian=False)
        assert manager.vocabulary == {}


class TestCountTokens:
    """Tests for count_tokens — Whisper token counting."""

    def test_simple_word(self):
        count = count_tokens("hei")
        assert count >= 1

    def test_sentence(self):
        count = count_tokens("dette er en testsetning")
        assert count >= 5

    def test_empty_string(self):
        count = count_tokens("")
        assert count == 0

    def test_norwegian_text(self):
        count = count_tokens("æ vil ikkje ha kaffe")
        assert count >= 5
