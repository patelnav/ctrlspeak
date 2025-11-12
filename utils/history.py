"""
History management for ctrlSPEAK transcriptions.

Stores transcription history in SQLite database for later review.
"""

import sqlite3
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger("ctrlspeak.history")

# Schema version for migrations
SCHEMA_VERSION = 1

# Default history database location
HISTORY_DB_PATH = Path.home() / ".ctrlspeak" / "history.db"


@dataclass
class HistoryEntry:
    """Represents a single transcription history entry."""
    id: int
    timestamp: str
    text: str
    model: str
    duration_seconds: float
    language: str

    @property
    def formatted_timestamp(self) -> str:
        """Return human-readable timestamp."""
        try:
            dt = datetime.fromisoformat(self.timestamp)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return self.timestamp

    @property
    def preview(self) -> str:
        """Return short preview of text (first 100 chars)."""
        if len(self.text) <= 100:
            return self.text
        return self.text[:97] + "..."


class HistoryManager:
    """Manages transcription history storage and retrieval."""

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize history manager.

        Args:
            db_path: Path to SQLite database (defaults to ~/.ctrlspeak/history.db)
        """
        self.db_path = db_path or HISTORY_DB_PATH
        self._ensure_db_exists()

    def _ensure_db_exists(self) -> None:
        """Create database and table if they don't exist."""
        try:
            # Create directory with secure permissions (user-only access)
            db_dir = self.db_path.parent
            db_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(db_dir, 0o700)

            # Create or migrate database
            with sqlite3.connect(self.db_path) as conn:
                # Check schema version
                current_version = self._get_schema_version(conn)

                if current_version == 0:
                    # New database - create schema
                    self._create_schema(conn)
                elif current_version < SCHEMA_VERSION:
                    # Future: run migrations here
                    logger.warning(f"Schema version {current_version} < {SCHEMA_VERSION}. Migrations not yet implemented.")

                conn.commit()
                logger.debug(f"History database initialized at {self.db_path}")

        except Exception as e:
            logger.error(f"Error initializing history database: {e}", exc_info=True)

    def _get_schema_version(self, conn: sqlite3.Connection) -> int:
        """Get current schema version."""
        try:
            cursor = conn.execute("SELECT version FROM schema_version LIMIT 1")
            row = cursor.fetchone()
            return row[0] if row else 0
        except sqlite3.OperationalError:
            # Table doesn't exist
            return 0

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        """Create initial database schema."""
        # Schema version tracking
        conn.execute("""
            CREATE TABLE schema_version (
                version INTEGER NOT NULL
            )
        """)
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))

        # History table
        conn.execute("""
            CREATE TABLE history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                text TEXT NOT NULL,
                model TEXT NOT NULL,
                duration_seconds REAL,
                language TEXT DEFAULT 'en'
            )
        """)

        # Index for performance
        conn.execute("""
            CREATE INDEX idx_timestamp ON history(timestamp DESC)
        """)

        logger.info(f"Created history database schema version {SCHEMA_VERSION}")

    def add_entry(
        self,
        text: str,
        model: str,
        duration_seconds: float = 0.0,
        language: str = "en"
    ) -> Optional[int]:
        """
        Add a new transcription to history.

        Args:
            text: Transcribed text
            model: Model used for transcription
            duration_seconds: Recording duration in seconds
            language: Source language code

        Returns:
            ID of inserted entry, or None if failed
        """
        if not text or not text.strip():
            logger.warning("Attempted to save empty transcription to history")
            return None

        try:
            timestamp = datetime.now().isoformat()

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO history (timestamp, text, model, duration_seconds, language)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (timestamp, text.strip(), model, duration_seconds, language)
                )
                conn.commit()
                entry_id = cursor.lastrowid
                logger.info(f"Saved transcription to history (ID: {entry_id}, length: {len(text)} chars)")
                return entry_id

        except Exception as e:
            logger.error(f"Error saving to history: {e}", exc_info=True)
            return None

    def get_recent(self, limit: int = 100) -> List[HistoryEntry]:
        """
        Get recent transcription history entries.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of HistoryEntry objects, most recent first
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT id, timestamp, text, model, duration_seconds, language
                    FROM history
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (limit,)
                )
                rows = cursor.fetchall()

                entries = [
                    HistoryEntry(
                        id=row['id'],
                        timestamp=row['timestamp'],
                        text=row['text'],
                        model=row['model'],
                        duration_seconds=row['duration_seconds'] or 0.0,
                        language=row['language'] or 'en'
                    )
                    for row in rows
                ]

                logger.debug(f"Retrieved {len(entries)} history entries")
                return entries

        except Exception as e:
            logger.error(f"Error retrieving history: {e}", exc_info=True)
            return []

    def get_by_id(self, entry_id: int) -> Optional[HistoryEntry]:
        """
        Get a specific history entry by ID.

        Args:
            entry_id: Entry ID

        Returns:
            HistoryEntry object or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT id, timestamp, text, model, duration_seconds, language
                    FROM history
                    WHERE id = ?
                    """,
                    (entry_id,)
                )
                row = cursor.fetchone()

                if row:
                    return HistoryEntry(
                        id=row['id'],
                        timestamp=row['timestamp'],
                        text=row['text'],
                        model=row['model'],
                        duration_seconds=row['duration_seconds'] or 0.0,
                        language=row['language'] or 'en'
                    )

                return None

        except Exception as e:
            logger.error(f"Error retrieving entry {entry_id}: {e}", exc_info=True)
            return None

    def delete_entry(self, entry_id: int) -> bool:
        """
        Delete a history entry.

        Args:
            entry_id: Entry ID to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM history WHERE id = ?", (entry_id,))
                conn.commit()

                if cursor.rowcount > 0:
                    logger.info(f"Deleted history entry {entry_id}")
                    return True
                else:
                    logger.warning(f"No entry found with ID {entry_id}")
                    return False

        except Exception as e:
            logger.error(f"Error deleting entry {entry_id}: {e}", exc_info=True)
            return False

    def clear_all(self) -> bool:
        """
        Clear all history entries.

        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM history")
                conn.commit()
                logger.info("Cleared all history entries")
                return True

        except Exception as e:
            logger.error(f"Error clearing history: {e}", exc_info=True)
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about transcription history.

        Returns:
            Dictionary with statistics (total entries, total words, etc.)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT
                        COUNT(*) as total_entries,
                        SUM(LENGTH(text) - LENGTH(REPLACE(text, ' ', '')) + 1) as total_words,
                        SUM(duration_seconds) as total_duration
                    FROM history
                """)
                row = cursor.fetchone()

                return {
                    "total_entries": row[0] or 0,
                    "total_words": row[1] or 0,
                    "total_duration": row[2] or 0.0
                }

        except Exception as e:
            logger.error(f"Error getting stats: {e}", exc_info=True)
            return {"total_entries": 0, "total_words": 0, "total_duration": 0.0}


# Global instance
_history_manager: Optional[HistoryManager] = None


def get_history_manager(db_path: Optional[Path] = None) -> HistoryManager:
    """
    Get or create the global history manager instance.

    Args:
        db_path: Optional custom database path. If provided on first call,
                 sets the path for the singleton instance.

    Returns:
        HistoryManager instance
    """
    global _history_manager
    if _history_manager is None:
        _history_manager = HistoryManager(db_path=db_path)
    return _history_manager
