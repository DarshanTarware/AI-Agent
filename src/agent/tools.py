"""
Tool Definitions for AutoMate Personal Executive Agent.

This module exports functions equipped with strict Python type hints and
production-grade docstrings for direct consumption by the Google GenAI SDK
automated tool-calling runtime.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from duckduckgo_search import DDGS
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp

from src.data.database import save_task


def _extract_youtube_video_id(url_or_id: str) -> str:
    """
    Extract standard 11-character YouTube video ID from various URL formats or raw ID.

    Complexity:
        - Time Complexity: O(N) where N is the length of the input string, bounded by URL parsing regex.
        - Space Complexity: O(1) auxiliary space beyond string slicing.
    """
    if len(url_or_id) == 11 and re.match(r"^[A-Za-z0-9_-]{11}$", url_or_id):
        return url_or_id

    # Parse regular youtube.com or youtu.be URLs
    parsed_url = urlparse(url_or_id)
    if parsed_url.hostname in ("www.youtube.com", "youtube.com"):
        if parsed_url.path == "/watch":
            query_params = parse_qs(parsed_url.query)
            if "v" in query_params:
                return query_params["v"][0]
        elif parsed_url.path.startswith(("/embed/", "/v/")):
            return parsed_url.path.split("/")[2]
    elif parsed_url.hostname in ("youtu.be", "www.youtu.be"):
        return parsed_url.path.lstrip("/")

    # Regex fallback for embedded IDs
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url_or_id)
    if match:
        return match.group(1)

    return url_or_id


def download_youtube_audio(url: str) -> str:
    """
    Downloads the audio stream from a YouTube video URL and saves it locally as an MP3/M4A/MP4 file.

    Args:
        url: The full YouTube video URL (e.g. 'https://www.youtube.com/watch?v=dQw4w9WgXcQ').

    Returns:
        A JSON string containing the file path where the downloaded audio is stored, title,
        duration, and status.

    Algorithmic Complexity & Engineering Notes:
        - Time Complexity: O(M) dominated by network I/O and audio stream bitstream extraction,
          where M is the media file payload size. Metadata parsing is O(1).
        - Space Complexity: O(M) disk allocation in the downloads directory; O(1) working memory
          using chunked streaming.
    """
    downloads_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "downloads", "audio"))
    os.makedirs(downloads_dir, exist_ok=True)

    out_template = os.path.join(downloads_dir, "%(title)s_%(id)s.%(ext)s")

    ydl_opts = {
        "format": "ba/b",
        "outtmpl": out_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web", "mweb"]
            }
        },
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        },
    }

    try:
        # Strip extraneous query parameters while keeping video ID
        vid_id = _extract_youtube_video_id(url)
        target_url = f"https://www.youtube.com/watch?v=vid_id" if vid_id and len(vid_id) == 11 else url
        if vid_id and len(vid_id) == 11:
            target_url = f"https://www.youtube.com/watch?v={vid_id}"

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(target_url, download=True)
            if info_dict is None:
                return json.dumps({"status": "error", "message": "Failed to retrieve video metadata."})
            filename = ydl.prepare_filename(info_dict)

            save_task(
                title=f"YouTube Audio Download: {info_dict.get('title', 'Audio')}",
                details=f"File: {filename}\nURL: {url}",
                status="completed",
                source="tool:download_youtube_audio"
            )
            return json.dumps({
                "status": "success",
                "file_path": filename,
                "title": info_dict.get("title", ""),
                "duration_seconds": info_dict.get("duration", 0),
            })
    except Exception as exc:
        return json.dumps({"status": "error", "message": str(exc)})


def get_youtube_transcript(video_id: str) -> str:
    """
    Fetches the closed-caption transcript text for a YouTube video given its video ID or URL.

    Args:
        video_id: The 11-character YouTube video ID or full YouTube video URL.

    Returns:
        A JSON string containing the full transcript plain text,
        or an error message JSON if transcripts are disabled/unavailable.

    Algorithmic Complexity & Engineering Notes:
        - Time Complexity: O(K) where K is the number of subtitle snippets returned by the YouTube
          timed text API. Linear string concatenation using a join accumulator.
        - Space Complexity: O(K) memory allocation to store snippet dicts and joined transcript string.
    """
    clean_id = _extract_youtube_video_id(video_id.strip())

    try:
        # Check if version has instance fetch or class method
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

        # Extract text snippets
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

        save_task(
            title=f"YouTube Transcript Extracted: {clean_id}",
            details=f"Extracted {len(full_text)} characters from video ID {clean_id}.",
            status="completed",
            source="tool:get_youtube_transcript"
        )
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
        query: Search keywords or question string to query (e.g. 'latest release of Google GenAI SDK').

    Returns:
        A JSON string containing an array of structured search results with title, snippet, and link.

    Algorithmic Complexity & Engineering Notes:
        - Time Complexity: O(R) where R is the number of results returned (typically capped at 5).
          Network latency dominates execution.
        - Space Complexity: O(R * S) where S is the average character length of snippets retrieved.
    """
    try:
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=5))
        formatted_results: List[Dict[str, str]] = []
        for item in results:
            formatted_results.append({
                "title": item.get("title", ""),
                "snippet": item.get("body", ""),
                "url": item.get("href", "")
            })
            
        save_task(
            title=f"Web Search: {query}",
            details=f"Retrieved {len(formatted_results)} results for query: {query}",
            status="completed",
            source="tool:search_web"
        )
        return json.dumps({
            "status": "success",
            "query": query,
            "results": formatted_results
        })
    except Exception as exc:
        return json.dumps({
            "status": "error",
            "query": query,
            "message": f"Web search error: {str(exc)}"
        })


def schedule_calendar_event(title: str, start_time: str, end_time: str) -> str:
    """
    Simulates scheduling an executive calendar event or appointment with specified start and end timestamps.

    Args:
        title: Descriptive title or summary of the meeting/event (e.g. 'Quarterly Strategy Review').
        start_time: ISO-8601 formatted datetime string or natural language start time (e.g. '2026-08-18T10:00:00').
        end_time: ISO-8601 formatted datetime string or natural language end time (e.g. '2026-08-18T11:00:00').

    Returns:
        A JSON string confirmation indicating event registration, confirmation ID, and scheduled timeframe.

    Algorithmic Complexity & Engineering Notes:
        - Time Complexity: O(1) hash generation and database insertion.
        - Space Complexity: O(1) constant auxiliary space.
    """
    event_id = f"evt_{abs(hash(title + start_time)) % 1000000:06d}"
    details = f"Scheduled '{title}' from {start_time} to {end_time}. Event ID: {event_id}"
    
    save_task(
        title=f"Calendar Event: {title}",
        details=details,
        status="completed",
        source="tool:schedule_calendar_event"
    )
    
    return json.dumps({
        "status": "success",
        "event_id": event_id,
        "title": title,
        "start_time": start_time,
        "end_time": end_time,
        "message": f"Calendar event '{title}' scheduled successfully for {start_time} to {end_time}."
    })
