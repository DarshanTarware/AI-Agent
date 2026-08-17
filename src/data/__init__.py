"""
Data persistence package for AutoMate.
"""
from src.data.database import init_db, log_interaction, get_recent_tasks, save_task, get_chat_history

__all__ = ["init_db", "log_interaction", "get_recent_tasks", "save_task", "get_chat_history"]
