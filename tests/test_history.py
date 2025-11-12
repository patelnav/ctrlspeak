#!/usr/bin/env python3
"""
Test suite for history functionality.
"""

import pytest
import time
from pathlib import Path
from utils.history import HistoryManager, HistoryEntry


@pytest.fixture
def temp_db(tmp_path):
    """Fixture providing temporary database path."""
    return tmp_path / "test_history.db"


@pytest.fixture
def history(temp_db):
    """Fixture providing clean history manager instance."""
    return HistoryManager(db_path=temp_db)


def test_add_entry(history):
    """Test adding a valid entry."""
    entry_id = history.add_entry(
        text="This is a test transcription",
        model="parakeet",
        duration_seconds=5.2,
        language="en"
    )
    assert entry_id is not None
    assert entry_id > 0


def test_add_multiple_entries(history):
    """Test adding multiple entries."""
    id1 = history.add_entry("First", "parakeet", 5.2, "en")
    id2 = history.add_entry("Second", "whisper", 8.7, "en")

    assert id2 > id1
    entries = history.get_recent(limit=10)
    assert len(entries) == 2


def test_get_recent_order(history):
    """Test that get_recent returns most recent first."""
    id1 = history.add_entry("First", "parakeet", 1.0)
    time.sleep(0.01)  # Ensure different timestamps
    id2 = history.add_entry("Second", "parakeet", 1.0)

    entries = history.get_recent(limit=10)

    assert entries[0].id == id2  # Most recent first
    assert entries[1].id == id1


def test_get_by_id(history):
    """Test retrieving entry by ID."""
    entry_id = history.add_entry("Test text", "parakeet", 5.2, "en")
    entry = history.get_by_id(entry_id)

    assert entry is not None
    assert entry.id == entry_id
    assert entry.text == "Test text"
    assert entry.model == "parakeet"
    assert entry.duration_seconds == 5.2


def test_get_by_id_not_found(history):
    """Test get_by_id returns None for non-existent ID."""
    assert history.get_by_id(99999) is None


def test_delete_entry(history):
    """Test deleting an entry."""
    entry_id = history.add_entry("Test", "parakeet", 1.0)

    assert history.get_by_id(entry_id) is not None
    assert history.delete_entry(entry_id) is True
    assert history.get_by_id(entry_id) is None


def test_delete_nonexistent(history):
    """Test deleting non-existent entry returns False."""
    assert history.delete_entry(99999) is False


@pytest.mark.parametrize("text,expected", [
    ("", None),           # Empty string
    ("   ", None),        # Only whitespace
    ("\t\n", None),       # Only tabs/newlines
    ("Valid", int),       # Valid text
])
def test_add_entry_validation(history, text, expected):
    """Test validation of text input."""
    result = history.add_entry(text, "parakeet", 1.0)

    if expected is None:
        assert result is None
    else:
        assert isinstance(result, expected)


def test_get_stats(history):
    """Test statistics calculation."""
    history.add_entry("Hello world", "parakeet", 2.5)
    history.add_entry("Test", "whisper", 3.5)

    stats = history.get_stats()

    assert stats['total_entries'] == 2
    assert stats['total_duration'] == 6.0
    assert stats['total_words'] > 0


def test_clear_all(history):
    """Test clearing all entries."""
    history.add_entry("Test 1", "parakeet", 1.0)
    history.add_entry("Test 2", "parakeet", 1.0)

    assert len(history.get_recent(limit=10)) == 2
    assert history.clear_all() is True
    assert len(history.get_recent(limit=10)) == 0


def test_entry_formatted_timestamp():
    """Test HistoryEntry formatted timestamp."""
    entry = HistoryEntry(
        id=1,
        timestamp="2024-01-15T10:30:00",
        text="Test",
        model="parakeet",
        duration_seconds=5.0,
        language="en"
    )

    formatted = entry.formatted_timestamp
    assert "2024-01-15" in formatted
    assert "10:30:00" in formatted


def test_entry_preview_short():
    """Test preview for short text."""
    entry = HistoryEntry(
        id=1,
        timestamp="2024-01-15T10:30:00",
        text="Short",
        model="parakeet",
        duration_seconds=5.0,
        language="en"
    )
    assert entry.preview == "Short"


def test_entry_preview_long():
    """Test preview truncates long text."""
    long_text = "a" * 200
    entry = HistoryEntry(
        id=1,
        timestamp="2024-01-15T10:30:00",
        text=long_text,
        model="parakeet",
        duration_seconds=5.0,
        language="en"
    )

    assert len(entry.preview) <= 103
    assert entry.preview.endswith("...")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
