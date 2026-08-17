"""
Tool Definitions for AutoMate Personal Executive Agent.

This module provides tools for:
1. Live Google Calendar event scheduling with OAuth2 token management.
2. High-performance YouTube audio streaming extraction using yt-dlp.
3. Closed-caption transcript fetching using youtube-transcript-api.
4. Real-time web search using DuckDuckGo.

All functions feature strict typing and comprehensive docstrings with Big-O complexity analysis.
"""

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dateutil import parser as date_parser
from duckduckgo_search import DDGS
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp

from src.data.database import log_event, log_media

logger = logging.getLogger(__name__)

# Google Calendar OAuth Scopes and Paths
CALENDAR_SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
CREDENTIALS_PATH = os.path.join(PROJECT_ROOT, "credentials.json")
TOKEN_PATH = os.path.join(PROJECT_ROOT, "token.json")
DOWNLOADS_DIR = os.path.join(PROJECT_ROOT, "downloads")


def _extract_youtube_video_id(url_or_id: str) -> str:
    """
    Extract standard 11-character YouTube video ID from various URL formats or raw ID.

    Complexity:
        - Time Complexity: O(N) where N is URL length.
        - Space Complexity: O(1) auxiliary space.
    """
    cleaned = url_or_id.strip()
    if len(cleaned) == 11 and re.match(r"^[A-Za-z0-9_-]{11}$", cleaned):
        return cleaned

    parsed_url = urlparse(cleaned)
    if parsed_url.hostname in ("www.youtube.com", "youtube.com"):
        if parsed_url.path == "/watch":
            query_params = parse_qs(parsed_url.query)
            if "v" in query_params:
                return query_params["v"][0]
        elif parsed_url.path.startswith(("/embed/", "/v/")):
            return parsed_url.path.split("/")[2]
    elif parsed_url.hostname in ("youtu.be", "www.youtu.be"):
        return parsed_url.path.lstrip("/")

    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", cleaned)
    if match:
        return match.group(1)

    return cleaned


def _parse_to_iso_format(date_str: str) -> str:
    """
    Converts arbitrary natural language or formatted timestamp strings into
    strict ISO-8601 strings (e.g. '2026-08-20T16:00:00Z' or '2026-08-20T16:00:00+00:00').

    Complexity:
        - Time Complexity: O(L) where L is string length.
        - Space Complexity: O(1).
    """
    try:
        dt = date_parser.parse(date_str)
        if dt.tzinfo is None:
            # Default to local system timezone or UTC if unspecified
            dt = dt.astimezone()
        return dt.isoformat()
    except Exception:
        # Fallback to current UTC time if unparseable
        return datetime.now(timezone.utc).isoformat()


def download_youtube_audio(url: str) -> str:
    """
    Downloads the audio stream from a YouTube video URL and extracts a pure MP3 audio file.

    Args:
        url: The full YouTube video URL (e.g. 'https://www.youtube.com/watch?v=dQw4w9WgXcQ').

    Returns:
        A JSON string containing the status, file path, title, and duration:
        {"status": "success", "file_path": "<path>", "title": "<title>", "duration_seconds": <sec>}

    Algorithmic Complexity & Engineering Notes:
        - Time Complexity: O(M) dominated by network I/O and MP3 transcoding bitstream extraction.
        - Space Complexity: O(M) local disk allocation in /downloads/audio; O(1) buffer memory.
    """
    audio_dir = os.path.join(DOWNLOADS_DIR, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    out_template = os.path.join(audio_dir, "%(title)s_%(id)s.%(ext)s")

    # Locate bundled static ffmpeg binary
    ffmpeg_exe = None
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    ydl_opts = {
        "format": "bestaudio/ba",
        "outtmpl": out_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["android_vr", "web"]
            }
        },
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
    }

    if ffmpeg_exe and os.path.exists(ffmpeg_exe):
        ydl_opts["ffmpeg_location"] = ffmpeg_exe
        ydl_opts["postprocessors"] = [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ]

    try:
        vid_id = _extract_youtube_video_id(url)
        target_url = f"https://www.youtube.com/watch?v={vid_id}" if vid_id and len(vid_id) == 11 else url

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(target_url, download=True)
            if info_dict is None:
                return json.dumps({"status": "error", "message": "Failed to retrieve video stream metadata."})
            
            raw_filename = ydl.prepare_filename(info_dict)
            base_path, _ = os.path.splitext(raw_filename)
            mp3_path = f"{base_path}.mp3"

            final_file_path = mp3_path if os.path.exists(mp3_path) else raw_filename
            abs_file_path = os.path.abspath(final_file_path)

            # Record downloaded asset in SQLite persistent memory
            log_media(user_id="default_user", file_path=abs_file_path, file_type="audio")

            return json.dumps({
                "status": "success",
                "file_path": abs_file_path,
                "title": info_dict.get("title", "Audio Stream"),
                "duration_seconds": info_dict.get("duration", 0),
            })
    except Exception as exc:
        return json.dumps({"status": "error", "message": f"Audio download failed: {str(exc)}"})


def schedule_calendar_event(title: str, start_time: str, end_time: str) -> str:
    """
    Authenticates via Google Calendar OAuth2 flow using credentials.json and token.json,
    parses timestamps to strict ISO-8601, and schedules the appointment in Google Calendar.

    Args:
        title: Descriptive event name or meeting title (e.g. 'Strategic Project Review').
        start_time: Start date and time in natural language or ISO string (e.g. '2026-08-20 14:00').
        end_time: End date and time in natural language or ISO string (e.g. '2026-08-20 15:00').

    Returns:
        JSON string confirming scheduled calendar event details and Google Calendar event link.

    Algorithmic Complexity:
        - Time Complexity: O(1) OAuth verification and REST API round-trip.
        - Space Complexity: O(1) memory allocation for credentials.
    """
    iso_start = _parse_to_iso_format(start_time)
    iso_end = _parse_to_iso_format(end_time)

    # Persist in local SQLite events_logged memory
    log_event(user_id="default_user", summary=title, start_time=iso_start, end_time=iso_end)

    creds: Optional[Credentials] = None

    # Load existing token if present
    if os.path.exists(TOKEN_PATH):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_PATH, CALENDAR_SCOPES)
        except Exception as exc:
            logger.warning("Failed to load existing token.json: %s", exc)

    # Refresh or create credentials
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                with open(TOKEN_PATH, "w") as token_file:
                    token_file.write(creds.to_json())
            except Exception as exc:
                logger.warning("Token refresh failed: %s", exc)
                creds = None

        if not creds:
            if not os.path.exists(CREDENTIALS_PATH):
                return json.dumps({
                    "status": "success",
                    "mock": True,
                    "summary": title,
                    "start_time": iso_start,
                    "end_time": iso_end,
                    "message": (
                        f"✅ Event '{title}' recorded locally for {iso_start} to {iso_end}.\n"
                        "💡 To sync directly with live Google Calendar, place your OAuth `credentials.json` "
                        "in the project root directory."
                    )
                })

            try:
                flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, CALENDAR_SCOPES)
                creds = flow.run_local_server(port=0)
                with open(TOKEN_PATH, "w") as token_file:
                    token_file.write(creds.to_json())
            except Exception as exc:
                return json.dumps({
                    "status": "error",
                    "message": f"OAuth flow authentication failed: {str(exc)}"
                })

    try:
        service = build("calendar", "v3", credentials=creds)
        event_body = {
            "summary": title,
            "description": "Scheduled autonomously by AutoMate AI Executive Agent.",
            "start": {"dateTime": iso_start},
            "end": {"dateTime": iso_end},
        }

        created_event = service.events().insert(calendarId="primary", body=event_body).execute()
        event_link = created_event.get("htmlLink", "")

        return json.dumps({
            "status": "success",
            "summary": title,
            "start_time": iso_start,
            "end_time": iso_end,
            "event_link": event_link,
            "message": f"🎉 Successfully scheduled '{title}' in Google Calendar from {iso_start} to {iso_end}."
        })
    except Exception as exc:
        return json.dumps({
            "status": "error",
            "message": f"Google Calendar API insert failed: {str(exc)}"
        })


def get_youtube_transcript(video_id: str) -> str:
    """
    Fetches the closed-caption transcript text for a YouTube video given its video ID or URL.

    Args:
        video_id: The 11-character YouTube video ID or full YouTube video URL.

    Returns:
        A JSON string containing the full transcript text and character count.

    Algorithmic Complexity:
        - Time Complexity: O(K) where K is subtitle snippet count.
        - Space Complexity: O(K) memory allocation for text accumulator.
    """
    clean_id = _extract_youtube_video_id(video_id.strip())

    try:
        transcript_data = None
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            try:
                transcript_data = YouTubeTranscriptApi.get_transcript(clean_id)
            except Exception:
                pass

        if transcript_data is None:
            api_instance = YouTubeTranscriptApi()
            if hasattr(api_instance, "fetch"):
                transcript_data = api_instance.fetch(clean_id)
            elif hasattr(api_instance, "list"):
                transcript_list = api_instance.list(clean_id)
                transcript_data = transcript_list.find_transcript(["en"]).fetch()

        text_snippets: List[str] = []
        if transcript_data is not None:
            for item in transcript_data:
                if isinstance(item, dict):
                    t = item.get("text", "")
                elif hasattr(item, "text"):
                    t = item.text
                else:
                    t = str(item)
                if t:
                    text_snippets.append(t)

        full_text = " ".join(text_snippets)
        if not full_text:
            return json.dumps({
                "status": "error",
                "video_id": clean_id,
                "message": "No captions or transcript text available for this video."
            })

        return json.dumps({
            "status": "success",
            "video_id": clean_id,
            "transcript": full_text,
            "character_count": len(full_text)
        })
    except Exception as exc:
        return json.dumps({
            "status": "error",
            "video_id": clean_id,
            "message": f"Unable to fetch transcript: {str(exc)}"
        })


def search_web(query: str) -> str:
    """
    Searches the live web using DuckDuckGo to obtain concise, factual answers and search snippets.

    Args:
        query: Search keywords or question string to query (e.g. 'advancements in quantum computing').

    Returns:
        A JSON string containing an array of structured search results with title, snippet, and link.

    Algorithmic Complexity:
        - Time Complexity: O(R) where R is the number of results returned (capped at 5).
        - Space Complexity: O(R * S) where S is the average snippet length.
    """
    clean_query = query.strip()
    formatted_results: List[Dict[str, str]] = []

    try:
        ddgs = DDGS()

        # 1. Try text search
        try:
            for item in ddgs.text(clean_query, max_results=5):
                snippet = item.get("body", "")
                if snippet:
                    formatted_results.append({
                        "title": item.get("title", ""),
                        "snippet": snippet,
                        "url": item.get("href", ""),
                    })
        except Exception:
            pass

        # 2. Try news search if text search results are sparse
        if len(formatted_results) < 2:
            try:
                for item in ddgs.news(clean_query, max_results=5):
                    snippet = item.get("body") or item.get("excerpt", "")
                    if snippet:
                        formatted_results.append({
                            "title": item.get("title", ""),
                            "snippet": snippet,
                            "url": item.get("url", ""),
                        })
            except Exception:
                pass

        # 3. Fallback: If query contained specific future dates/fillers and returned 0 results, retry with core keywords
        if not formatted_results and any(char.isdigit() for char in clean_query):
            simplified_query = re.sub(r"\b(in|for|year|of)\s+20\d\d\b", "", clean_query, flags=re.IGNORECASE).strip()
            if simplified_query and simplified_query != clean_query:
                try:
                    for item in ddgs.text(simplified_query, max_results=5):
                        snippet = item.get("body", "")
                        if snippet:
                            formatted_results.append({
                                "title": item.get("title", ""),
                                "snippet": snippet,
                                "url": item.get("href", ""),
                            })
                except Exception:
                    pass

        if not formatted_results:
            return json.dumps({
                "status": "success",
                "query": query,
                "results": [],
                "message": f"No direct web matches found for '{query}'. Please try broader search keywords."
            })

        return json.dumps({
            "status": "success",
            "query": query,
            "results": formatted_results[:5],
        })
    except Exception as exc:
        return json.dumps({
            "status": "error",
            "query": query,
            "message": f"Web search error: {str(exc)}"
        })
