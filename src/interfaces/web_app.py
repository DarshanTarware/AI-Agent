"""
Streamlit Web Dashboard for AutoMate Executive Agent.

Provides an interactive web interface with live chat, tool activity feeds,
and persistent task/report logs.
"""

from __future__ import annotations

import os
import sys

# Ensure project root is in sys.path when running via `streamlit run`
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
from src.agent.brain import process_user_intent
from src.data.database import get_recent_tasks, get_chat_history, init_db

# Configure page settings
st.set_page_config(
    page_title="AutoMate | Personal Executive Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize database schema
init_db()

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1e293b;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .task-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .task-title {
        font-weight: 600;
        font-size: 0.95rem;
        color: #0f172a;
    }
    .task-time {
        font-size: 0.75rem;
        color: #94a3b8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.title("⚡ AutoMate Dashboard")
    st.markdown("---")
    
    st.subheader("🛠️ Active Capabilities")
    st.markdown(
        """
        - 🌐 **Live Web Search** (*DuckDuckGo*)
        - 🎙️ **YouTube Audio Downloader** (*yt-dlp*)
        - 📝 **YouTube Transcript Extraction** (*youtube-transcript-api*)
        - 📅 **Calendar & Appointment Scheduler**
        """
    )
    st.markdown("---")
    
    st.subheader("📋 Recent Tasks & Reports")
    recent_tasks = get_recent_tasks(limit=6)
    
    if recent_tasks:
        for task in recent_tasks:
            with st.container():
                st.markdown(
                    f"""
                    <div class="task-card">
                        <div class="task-title">📌 {task.get('title', 'Task')}</div>
                        <div class="task-time">🕒 {task.get('created_at', '')} • <span style="color:#10b981;">{task.get('status', 'done')}</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("No recorded tasks yet. Execute commands to populate this feed.")

    st.markdown("---")
    if st.button("🔄 Refresh Activity", use_container_width=True):
        st.rerun()

# Main Header
st.markdown('<div class="main-header">AutoMate Executive Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Multimodal autonomous agent powered by Google GenAI tool calling.</div>',
    unsafe_allow_html=True,
)

# Initialize chat session state
if "messages" not in st.session_state:
    # Load previous history if available or start fresh
    history = get_chat_history(platform="web", limit=20)
    if history:
        st.session_state.messages = [
            {"role": row["role"], "content": row["message"]} for row in history
        ]
    else:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "👋 Greetings! I am **AutoMate**, your executive AI agent. How can I assist you today?",
            }
        ]

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Enter an executive request or command..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process through AutoMate brain
    with st.chat_message("assistant"):
        with st.spinner("AutoMate is reasoning and orchestrating tools..."):
            response = process_user_intent(prompt, platform="web")
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
