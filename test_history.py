#!/usr/bin/env python3
"""
Quick test script for history functionality.
"""

import tempfile
from pathlib import Path
from utils.history import HistoryManager, get_history_manager

def test_history_basic():
    """Test basic history operations."""
    print("Testing history functionality...")

    # Use a temporary database for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_history.db"
        history = HistoryManager(db_path=db_path)

        print(f"✓ Created history manager with DB at {db_path}")

        # Test adding entries
        entry_id1 = history.add_entry(
            text="This is a test transcription",
            model="parakeet",
            duration_seconds=5.2,
            language="en"
        )
        assert entry_id1 is not None, "Failed to add entry 1"
        print(f"✓ Added entry 1 (ID: {entry_id1})")

        entry_id2 = history.add_entry(
            text="Another test transcription with more words",
            model="whisper",
            duration_seconds=8.7,
            language="en"
        )
        assert entry_id2 is not None, "Failed to add entry 2"
        print(f"✓ Added entry 2 (ID: {entry_id2})")

        # Test retrieving entries
        entries = history.get_recent(limit=10)
        assert len(entries) == 2, f"Expected 2 entries, got {len(entries)}"
        print(f"✓ Retrieved {len(entries)} entries")

        # Check order (most recent first)
        assert entries[0].id == entry_id2, "Entries not in correct order"
        assert entries[1].id == entry_id1, "Entries not in correct order"
        print("✓ Entries in correct order (most recent first)")

        # Test get by ID
        entry = history.get_by_id(entry_id1)
        assert entry is not None, "Failed to get entry by ID"
        assert entry.text == "This is a test transcription", "Text mismatch"
        assert entry.model == "parakeet", "Model mismatch"
        print(f"✓ Retrieved entry by ID: {entry.text[:30]}...")

        # Test statistics
        stats = history.get_stats()
        assert stats['total_entries'] == 2, f"Expected 2 entries in stats, got {stats['total_entries']}"
        print(f"✓ Stats: {stats}")

        # Test delete
        success = history.delete_entry(entry_id1)
        assert success, "Failed to delete entry"
        print(f"✓ Deleted entry {entry_id1}")

        entries = history.get_recent(limit=10)
        assert len(entries) == 1, f"Expected 1 entry after delete, got {len(entries)}"
        print(f"✓ Confirmed deletion (1 entry remaining)")

        # Test empty text handling
        empty_id = history.add_entry(text="", model="test", duration_seconds=0)
        assert empty_id is None, "Should not add empty text"
        print("✓ Correctly rejected empty text")

        print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_history_basic()
