"""
Core LLM Brain for AutoMate Executive Agent using Google GenAI SDK.

Handles multimodal user intent understanding, tool execution loop, and conversational synthesis.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Callable, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.agent.tools import (
    download_youtube_audio,
    get_youtube_transcript,
    schedule_calendar_event,
    search_web,
)
from src.data.database import log_interaction

load_dotenv()
logger = logging.getLogger(__name__)

# System instructions to guide AutoMate's executive behavior
SYSTEM_INSTRUCTION = """You are AutoMate, an elite multimodal personal executive AI agent.
Your primary role is to assist executives, developers, and researchers with automated tasks including:
1. Live Web Searches using DuckDuckGo to provide verified, up-to-date facts.
2. Extracting transcripts and key takeaways from YouTube videos.
3. Downloading YouTube audio for offline podcasting or speech transcription.
4. Scheduling calendar appointments and executive meetings.

Guidelines:
- When a user query requires real-time information or external actions, use your registered tools.
- Provide crisp, structured, professional responses with clear bullet points and action items.
- If a tool returns an error, explain the issue clearly and suggest next steps.
"""

# Registered tools mapping
TOOL_MAPPING: Dict[str, Callable[..., Any]] = {
    "download_youtube_audio": download_youtube_audio,
    "get_youtube_transcript": get_youtube_transcript,
    "search_web": search_web,
    "schedule_calendar_event": schedule_calendar_event,
}

TOOLS_LIST: List[Callable[..., Any]] = [
    download_youtube_audio,
    get_youtube_transcript,
    search_web,
    schedule_calendar_event,
]


def process_user_intent(user_input: str, platform: str = "web") -> str:
    """
    Process user input using Gemini models with automated tool/function calling.

    Args:
        user_input: Natural language command or question from user.
        platform: Originating interface ('telegram' or 'web').

    Returns:
        Synthesized response string from AutoMate.

    Algorithmic Complexity & Flow:
        - Time Complexity: O(T * M) where T is the number of tool turns (max 5) and M is tool execution latency.
        - Space Complexity: O(C) where C is the size of the multi-turn context history.
    """
    log_interaction(platform=platform, role="user", message=user_input)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or "<insert_gemini_api_key_here>" in api_key:
        fallback_msg = (
            "⚠️ **Gemini API Key Missing**: Please configure a valid `GEMINI_API_KEY` in your `.env` "
            "file to enable live LLM tool calling and executive reasoning."
        )
        log_interaction(platform=platform, role="assistant", message=fallback_msg)
        return fallback_msg

    try:
        client = genai.Client()
        model_name = "gemini-2.5-flash"

        config = types.GenerateContentConfig(
            tools=TOOLS_LIST,
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.7,
        )

        contents: List[Any] = [user_input]
        max_turns = 5

        for _ in range(max_turns):
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )

            # Check if model requested function calls
            if not response.function_calls:
                final_text = response.text or "Task execution complete."
                log_interaction(platform=platform, role="assistant", message=final_text)
                return final_text

            # Append the model's candidate turn to history
            if response.candidates:
                contents.append(response.candidates[0].content)

            # Process each requested tool call
            function_response_parts: List[types.Part] = []
            for function_call in response.function_calls:
                tool_name = function_call.name
                tool_args = dict(function_call.args) if function_call.args else {}

                logger.info(f"Executing tool '{tool_name}' with args: {tool_args}")

                if tool_name in TOOL_MAPPING:
                    try:
                        tool_fn = TOOL_MAPPING[tool_name]
                        result = tool_fn(**tool_args)
                    except Exception as exc:
                        result = json.dumps({"status": "error", "error": f"Tool execution failed: {str(exc)}"})
                else:
                    result = json.dumps({"status": "error", "error": f"Unknown tool '{tool_name}'"})

                part = types.Part.from_function_response(
                    name=tool_name,
                    response={"result": result},
                )
                function_response_parts.append(part)

            # Append tool execution results back to contents
            contents.append(types.Content(parts=function_response_parts, role="tool"))

        final_text = "Task execution reached maximum conversational tool iterations."
        log_interaction(platform=platform, role="assistant", message=final_text)
        return final_text

    except Exception as exc:
        err_msg = f"❌ Error in AutoMate Brain execution: {str(exc)}"
        logger.error(err_msg, exc_info=True)
        log_interaction(platform=platform, role="assistant", message=err_msg)
        return err_msg
