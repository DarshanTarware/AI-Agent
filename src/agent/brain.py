"""
Core AI Brain for AutoMate Executive Agent powered by OpenAI GPT-4o-mini.

Handles contextual conversational memory retrieval from SQLite,
multi-turn function calling orchestration with OpenAI tools, and media asset routing.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
from openai import OpenAI

from src.agent.tools import (
    download_youtube_audio,
    get_youtube_transcript,
    schedule_calendar_event,
    search_web,
)
from src.data.database import get_chat_history, get_media_files

load_dotenv()
logger = logging.getLogger(__name__)

# Model configuration: default to cost-effective gpt-4o-mini-2024-07-18
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini-2024-07-18")

# Base system instructions
BASE_SYSTEM_INSTRUCTION = """You are AutoMate, an elite multimodal personal executive AI agent.
Your primary role is to assist executives, developers, and researchers with automated workflows:
1. Live Web Searches using DuckDuckGo to provide verified facts, latest industry news, and research insights.
2. Extracting transcripts and key takeaways from YouTube videos.
3. Downloading YouTube audio for offline playback and podcasting.
4. Scheduling calendar appointments directly into Google Calendar.

Guidelines:
- When a user query requires real-time information, latest advancements, or external actions, ALWAYS use your registered tools.
- When summarizing web search results, synthesize key breakthroughs, technological trends, and actionable insights clearly and thoroughly.
- When an audio file is downloaded, inform the user clearly that the audio has been extracted and is ready for playback.
- Provide crisp, structured, executive responses with clear bullet points and action items.
"""

# Registered tools mapping
TOOL_MAPPING: Dict[str, Callable[..., Any]] = {
    "download_youtube_audio": download_youtube_audio,
    "get_youtube_transcript": get_youtube_transcript,
    "search_web": search_web,
    "schedule_calendar_event": schedule_calendar_event,
}

# OpenAI Tool Specifications
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Searches the live web using DuckDuckGo to obtain concise, factual answers and search snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords or question string to query (e.g. 'latest AI news').",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "download_youtube_audio",
            "description": "Downloads the audio stream from a YouTube video URL and extracts a pure MP3 audio file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full YouTube video URL (e.g. 'https://www.youtube.com/watch?v=dQw4w9WgXcQ' or 'https://youtu.be/...').",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_youtube_transcript",
            "description": "Fetches the closed-caption transcript text for a YouTube video given its video ID or URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_id": {
                        "type": "string",
                        "description": "The 11-character YouTube video ID or full YouTube video URL.",
                    }
                },
                "required": ["video_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_calendar_event",
            "description": "Schedules an executive appointment or meeting in Google Calendar, parsing timestamps to ISO-8601.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Descriptive event name or meeting title (e.g. 'Project Strategy Review').",
                    },
                    "start_time": {
                        "type": "string",
                        "description": "Start date and time in natural language or ISO string (e.g. '2026-08-25 14:00').",
                    },
                    "end_time": {
                        "type": "string",
                        "description": "End date and time in natural language or ISO string (e.g. '2026-08-25 15:00').",
                    },
                },
                "required": ["title", "start_time", "end_time"],
            },
        },
    },
]


@dataclass
class BrainResponse:
    """
    Structured response payload returned by AutoMate Brain.
    Supports both object attribute access (.text, .file_path) and dict-like usage.
    """
    text: str
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text

    def get(self, key: str, default: Any = None) -> Any:
        if key == "text":
            return self.text
        if key == "file_path":
            return self.file_path
        if key == "metadata":
            return self.metadata
        return self.metadata.get(key, default)

    def __getitem__(self, key: str) -> Any:
        val = self.get(key)
        if val is None and key not in ("file_path",):
            raise KeyError(key)
        return val


def process_user_intent(
    user_input: str,
    user_id: str = "default_user",
    platform: str = "web",
) -> BrainResponse:
    """
    Processes user queries with multi-turn tool calling and persistent memory context via OpenAI.

    Args:
        user_input: Natural language query or command from user.
        user_id: Unique identifier for the user session (e.g. Telegram user ID).
        platform: Originating interface ('telegram' or 'web').

    Returns:
        BrainResponse containing final textual synthesis and any generated media file_path.
    """
    # 1. Check OpenAI API key from environment
    api_key = os.environ.get("OPEN_AI_API") or os.environ.get("OPENAI_API_KEY")
    if not api_key or "<insert" in api_key:
        fallback_msg = (
            "⚠️ **OpenAI API Key Missing**: Please configure `OPEN_AI_API` or `OPENAI_API_KEY` in your `.env` "
            "file to enable live AI reasoning and tool calling."
        )
        return BrainResponse(text=fallback_msg)

    # 2. Retrieve user's past 10 interactions from SQLite memory
    past_history = get_chat_history(user_id=user_id, limit=10)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": BASE_SYSTEM_INSTRUCTION}
    ]

    # Prepend past context
    if past_history:
        for turn in past_history:
            role = "assistant" if turn.get("role") == "assistant" else "user"
            msg_content = turn.get("message", "")
            if msg_content:
                messages.append({"role": role, "content": msg_content})

    # Append current user prompt
    messages.append({"role": "user", "content": user_input})

    # Snapshot existing media files
    initial_media = get_media_files(user_id="default_user", limit=1)
    initial_media_id = initial_media[0]["id"] if initial_media else 0

    downloaded_file_path: Optional[str] = None
    tool_results: List[Dict[str, Any]] = []

    try:
        client = OpenAI(api_key=api_key)
        max_turns = 5

        for _ in range(max_turns):
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                tools=OPENAI_TOOLS,
                tool_choice="auto",
                temperature=0.7,
            )

            choice = response.choices[0]
            message = choice.message
            messages.append(message)

            # If no tool calls requested, model has delivered final response
            if not message.tool_calls:
                final_text = message.content or "Task execution complete."

                # Verify if media was created
                latest_media = get_media_files(user_id="default_user", limit=1)
                if latest_media and latest_media[0]["id"] > initial_media_id:
                    candidate_path = latest_media[0].get("file_path")
                    if candidate_path and os.path.exists(candidate_path):
                        downloaded_file_path = candidate_path

                return BrainResponse(
                    text=final_text,
                    file_path=downloaded_file_path,
                    metadata={"tool_results": tool_results},
                )

            # Process each requested tool call
            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                raw_args = tool_call.function.arguments

                try:
                    fn_args = json.loads(raw_args) if raw_args else {}
                except Exception:
                    fn_args = {}

                logger.info(f"OpenAI invoked tool '{fn_name}' with args: {fn_args}")

                if fn_name in TOOL_MAPPING:
                    try:
                        tool_fn = TOOL_MAPPING[fn_name]
                        result = tool_fn(**fn_args)
                    except Exception as exc:
                        result = json.dumps({"status": "error", "message": f"Tool execution failed: {str(exc)}"})
                else:
                    result = json.dumps({"status": "error", "message": f"Tool '{fn_name}' not found."})

                # Check if audio was downloaded
                try:
                    parsed_res = json.loads(result)
                    tool_results.append({"tool": fn_name, "data": parsed_res})
                    if fn_name == "download_youtube_audio" and parsed_res.get("status") == "success":
                        fp = parsed_res.get("file_path")
                        if fp and os.path.exists(fp):
                            downloaded_file_path = fp
                except Exception:
                    pass

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                })

        final_text = "Task execution completed after reaching tool iteration limit."
        return BrainResponse(
            text=final_text,
            file_path=downloaded_file_path,
            metadata={"tool_results": tool_results},
        )

    except Exception as exc:
        err_msg = f"❌ Error during OpenAI AutoMate Brain execution: {str(exc)}"
        logger.error(err_msg, exc_info=True)
        return BrainResponse(text=err_msg)
