"""
Data persistence package for AutoMate.
"""
from src.data.database import (
    init_db,
    insert_message,
    get_chat_history,
    log_media,
    get_media_files,
    log_event,
    get_events_logged,
)

__all__ = [
    "init_db",
    "insert_message",
    "get_chat_history",
    "log_media",
    "get_media_files",
    "log_event",
    "get_events_logged",
]
