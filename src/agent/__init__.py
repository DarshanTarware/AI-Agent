"""
Agent module for AutoMate.
"""
from src.agent.brain import process_user_intent
from src.agent.tools import (
    download_youtube_audio,
    get_youtube_transcript,
    search_web,
    schedule_calendar_event,
)

__all__ = [
    "process_user_intent",
    "download_youtube_audio",
    "get_youtube_transcript",
    "search_web",
    "schedule_calendar_event",
]
