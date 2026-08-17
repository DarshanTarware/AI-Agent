"""
Persistent SQLite Memory Layer for AutoMate Executive Agent.

Manages relational persistence for chat histories, downloaded media assets,
and scheduled calendar events in `agent_memory.db` with optimized indexes.
"""

import os
import sqlite3
import sys
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DB_PATH = os.path.join(PROJECT_ROOT, "agent_memory.db")


@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager that yields a SQLite connection with dict-like row access
    and guarantees immediate connection closure upon exit.

    Complexity:
        - Time Complexity: O(1) connection allocation.
        - Space Complexity: O(1) memory overhead.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """
    Initialize SQLite schema for agent_memory.db including chat history,
    media files, and calendar event logs with B-Tree indexes.

    Complexity:
        - Time Complexity: O(1) schema validation and DDL execution.
        - Space Complexity: O(1) table metadata overhead.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Chat History Table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chat_user_time 
            ON chat_history (user_id, timestamp DESC)
            """
        )

        # 2. Media Files Table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS media_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_type TEXT NOT NULL DEFAULT 'audio',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_media_user_time 
            ON media_files (user_id, timestamp DESC)
            """
        )

        # 3. Events Logged Table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS events_logged (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_events_user 
            ON events_logged (user_id, timestamp DESC)
            """
        )

        conn.commit()


def insert_message(user_id: str, role: str, message: str) -> int:
    """
    Inserts a conversational turn into the chat history.

    Args:
        user_id: Identifier of the user or session.
        role: Message role ('user', 'assistant', 'system', 'tool').
        message: Raw message text payload.

    Returns:
        The auto-generated row ID of the inserted record.

    Complexity:
        - Time Complexity: O(log N) for B-Tree index update on table of N records.
        - Space Complexity: O(L) where L is message character length.
    """
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO chat_history (user_id, role, message)
            VALUES (?, ?, ?)
            """,
            (user_id, role, message),
        )
        conn.commit()
        return cursor.lastrowid or 0


def get_chat_history(user_id: str = "default_user", limit: int = 10) -> List[Dict[str, Any]]:
    """
    Retrieves the most recent conversational interactions for a given user,
    ordered chronologically (oldest to newest) for prompt context feeding.

    Args:
        user_id: Identifier for user or active session.
        limit: Number of recent messages to retrieve (defaults to 10).

    Returns:
        List of dictionaries with keys: id, user_id, role, message, timestamp.

    Complexity:
        - Time Complexity: O(log N + K log K) where N is total messages, K=limit (index seek followed by reverse sort).
        - Space Complexity: O(K * M) where M is average message size.
    """
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, user_id, role, message, timestamp
            FROM (
                SELECT id, user_id, role, message, timestamp
                FROM chat_history
                WHERE user_id = ?
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
            )
            ORDER BY timestamp ASC, id ASC
            """,
            (user_id, limit),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def log_media(user_id: str, file_path: str, file_type: str = "audio") -> int:
    """
    Records a generated or downloaded media asset into the persistent media registry.

    Args:
        user_id: Identifier of the originating user.
        file_path: Absolute or relative disk path to the media file.
        file_type: Type descriptor ('audio', 'video', 'document', 'image').

    Returns:
        The row ID of the persisted media record.

    Complexity:
        - Time Complexity: O(log M) B-Tree insertion on table of M media files.
        - Space Complexity: O(P) where P is the length of the file path.
    """
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO media_files (user_id, file_path, file_type)
            VALUES (?, ?, ?)
            """,
            (user_id, file_path, file_type),
        )
        conn.commit()
        return cursor.lastrowid or 0


def get_media_files(user_id: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Retrieves registered media assets for display on dashboards or downstream processing.

    Args:
        user_id: Optional user filter. If None, retrieves global media files.
        limit: Maximum number of media records to return.

    Returns:
        List of dictionaries containing media file metadata.

    Complexity:
        - Time Complexity: O(log M + K) indexed query.
        - Space Complexity: O(K) returned records payload.
    """
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if user_id:
            cursor.execute(
                """
                SELECT id, user_id, file_path, file_type, timestamp
                FROM media_files
                WHERE user_id = ?
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (user_id, limit),
            )
        else:
            cursor.execute(
                """
                SELECT id, user_id, file_path, file_type, timestamp
                FROM media_files
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def log_event(user_id: str, summary: str, start_time: str, end_time: str) -> int:
    """
    Logs scheduled calendar appointments and meetings.

    Args:
        user_id: Target user identity.
        summary: Event title or meeting agenda.
        start_time: ISO-formatted start timestamp.
        end_time: ISO-formatted end timestamp.

    Returns:
        Inserted record ID.

    Complexity:
        - Time Complexity: O(log E) index update.
        - Space Complexity: O(S) size of event descriptors.
    """
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO events_logged (user_id, summary, start_time, end_time)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, summary, start_time, end_time),
        )
        conn.commit()
        return cursor.lastrowid or 0


def get_events_logged(user_id: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Retrieves previously logged calendar events.

    Args:
        user_id: Optional user ID filter.
        limit: Maximum number of event rows to return.

    Returns:
        List of event dictionaries.
    """
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if user_id:
            cursor.execute(
                """
                SELECT id, user_id, summary, start_time, end_time, timestamp
                FROM events_logged
                WHERE user_id = ?
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (user_id, limit),
            )
        else:
            cursor.execute(
                """
                SELECT id, user_id, summary, start_time, end_time, timestamp
                FROM events_logged
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


# Backward Compatibility Aliases
def log_interaction(platform: str, role: str, message: str, user_id: Optional[str] = None) -> None:
    """Compatibility wrapper around insert_message."""
    insert_message(user_id=user_id or "default_user", role=role, message=message)


def save_task(title: str, details: str = "", status: str = "completed", source: str = "agent") -> int:
    """Compatibility wrapper for task recording."""
    return log_event(user_id="default_user", summary=title, start_time=details, end_time=status)


def get_recent_tasks(limit: int = 10) -> List[Dict[str, Any]]:
    """Compatibility wrapper returning logged events and media."""
    return get_events_logged(limit=limit)
