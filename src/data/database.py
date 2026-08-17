"""
SQLite Database Layer for AutoMate Executive Agent.

Handles persistent storage for tasks, reports, interaction logs, and chat history.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Generator

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
DB_PATH = os.path.join(DB_DIR, "automate.db")


@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager that yields a SQLite connection and guarantees closing it.
    """
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """
    Initialize SQLite schema with tables for tasks, chat history, and tool execution logs.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                details TEXT,
                status TEXT NOT NULL DEFAULT 'completed',
                source TEXT DEFAULT 'system',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform TEXT NOT NULL,
                user_id TEXT,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT NOT NULL,
                arguments TEXT,
                result TEXT,
                status TEXT DEFAULT 'success',
                executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def save_task(title: str, details: str = "", status: str = "completed", source: str = "agent") -> int:
    """
    Persist an executed task or report into the database.
    """
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO tasks (title, details, status, source)
            VALUES (?, ?, ?, ?)
            """,
            (title, details, status, source),
        )
        conn.commit()
        return cursor.lastrowid or 0


def get_recent_tasks(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Retrieve the most recent tasks and execution reports.
    """
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, title, details, status, source, created_at
            FROM tasks
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def log_interaction(platform: str, role: str, message: str, user_id: Optional[str] = None) -> None:
    """
    Log user and assistant conversations across platforms (Telegram, Web).
    """
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO chat_history (platform, user_id, role, message)
            VALUES (?, ?, ?, ?)
            """,
            (platform, user_id or "default_user", role, message),
        )
        conn.commit()


def get_chat_history(platform: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Retrieve conversational history for context reconstruction or dashboard display.
    """
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if platform:
            cursor.execute(
                """
                SELECT id, platform, user_id, role, message, timestamp
                FROM chat_history
                WHERE platform = ?
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (platform, limit),
            )
        else:
            cursor.execute(
                """
                SELECT id, platform, user_id, role, message, timestamp
                FROM chat_history
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (limit,),
            )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
