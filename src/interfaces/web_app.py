"""
Streamlit Web Dashboard for AutoMate Executive Agent.

Provides an interactive interface with persistent conversational memory,
live media playback and downloads, and scheduled Google Calendar event logs.
"""

import os
import sys

# Ensure project root is in sys.path when running via `streamlit run`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from src.agent.brain import process_user_intent
from src.data.database import (
    get_chat_history,
    get_events_logged,
    get_media_files,
    init_db,
    insert_message,
)

# Initialize page configuration
st.set_page_config(
    page_title="AutoMate | Personal Executive Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize database schema
init_db()

USER_ID = "web_user"

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .media-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 12px;
    }
    .media-name {
        font-weight: 600;
        font-size: 0.9rem;
        color: #1e293b;
        margin-bottom: 6px;
    }
    .event-card {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 8px;
    }
    .event-title {
        font-weight: 600;
        font-size: 0.9rem;
        color: #166534;
    }
    .event-time {
        font-size: 0.75rem;
        color: #15803d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("⚡ AutoMate Hub")
    st.markdown("---")

    st.subheader("🎧 Downloaded Media")
    media_list = get_media_files(limit=10)

    if media_list:
        for item in media_list:
            f_path = item.get("file_path", "")
            f_name = os.path.basename(f_path) if f_path else "Audio Asset"
            
            with st.container():
                st.markdown(f'<div class="media-name">🎵 {f_name}</div>', unsafe_allow_html=True)
                if f_path and os.path.exists(f_path):
                    st.audio(f_path)
                    try:
                        with open(f_path, "rb") as audio_file:
                            st.download_button(
                                label=f"⬇️ Download {f_name}",
                                data=audio_file,
                                file_name=f_name,
                                mime="audio/mpeg",
                                key=f"dl_{item['id']}",
                                use_container_width=True,
                            )
                    except Exception as exc:
                        st.caption(f"Audio ready on disk: {f_path}")
                else:
                    st.caption(f"Asset recorded: `{f_path}`")
                st.markdown("<hr style='margin: 8px 0;'>", unsafe_allow_html=True)
    else:
        st.info("No media files downloaded yet. Ask AutoMate to download a YouTube video!")

    st.markdown("---")
    st.subheader("📅 Scheduled Events")
    events = get_events_logged(limit=5)
    if events:
        for ev in events:
            st.markdown(
                f"""
                <div class="event-card">
                    <div class="event-title">📌 {ev.get('summary', 'Meeting')}</div>
                    <div class="event-time">🕒 {ev.get('start_time', '')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.caption("No events logged yet.")

    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()

# ----------------- MAIN VIEW -----------------
st.markdown('<div class="main-title">AutoMate Executive Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Multimodal executive intelligence powered by Google GenAI, live Web Search, YouTube tools, and Google Calendar.</div>',
    unsafe_allow_html=True,
)

# Load persistent conversation history from SQLite
chat_rows = get_chat_history(user_id=USER_ID, limit=30)

# Display chat messages from persistent DB
if chat_rows:
    for row in chat_rows:
        with st.chat_message(row["role"]):
            st.markdown(row["message"])
else:
    with st.chat_message("assistant"):
        st.markdown(
            "👋 Greetings! I am **AutoMate**, your executive AI agent with full tool routing. "
            "How can I assist you today?"
        )

# Chat Input Handler
if prompt := st.chat_input("Ask AutoMate anything or request an action..."):
    # 1. Display user prompt immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Record user turn in SQLite
    insert_message(user_id=USER_ID, role="user", message=prompt)

    # 3. Execute via AutoMate Brain
    with st.chat_message("assistant"):
        with st.spinner("AutoMate is reasoning and orchestrating tools..."):
            brain_resp = process_user_intent(user_input=prompt, user_id=USER_ID, platform="web")
            resp_text = brain_resp.text if hasattr(brain_resp, "text") else str(brain_resp)
            media_path = brain_resp.file_path if hasattr(brain_resp, "file_path") else None

            st.markdown(resp_text)

            # If a media file was downloaded in this turn, display inline player & download button
            if media_path and os.path.exists(media_path):
                st.audio(media_path)
                f_name = os.path.basename(media_path)
                try:
                    with open(media_path, "rb") as mf:
                        st.download_button(
                            label=f"⬇️ Download {f_name}",
                            data=mf,
                            file_name=f_name,
                            mime="audio/mpeg",
                            key=f"turn_dl_{f_name}",
                        )
                except Exception:
                    pass

    # 4. Record assistant turn in SQLite
    insert_message(user_id=USER_ID, role="assistant", message=resp_text)

    # Rerun to update sidebar media list and event feeds seamlessly
    st.rerun()
