"""
Telegram Bot Interface for AutoMate.

Asynchronous message listener using python-telegram-bot that connects
user commands directly to the AutoMate LLM agent brain.
"""

from __future__ import annotations

import logging
import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.agent.brain import process_user_intent

load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles the /start command.
    """
    user_name = update.effective_user.first_name if update.effective_user else "there"
    welcome_message = (
        f"👋 Hello {user_name}! I am **AutoMate**, your multimodal personal executive AI agent.\n\n"
        "Here is what I can do for you:\n"
        "🔍 **Live Web Searches**: Ask any fact-based question.\n"
        "📹 **YouTube Intelligence**: Extract transcripts and summarize videos.\n"
        "🎵 **Audio Downloads**: Download audio from YouTube for offline listening.\n"
        "📅 **Calendar Management**: Schedule meetings and appointments.\n\n"
        "Simply send me a message or a link to get started!"
    )
    if update.message:
        await update.message.reply_text(welcome_message, parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles the /help command.
    """
    help_text = (
        "🤖 **AutoMate Commands & Capabilities**:\n\n"
        "• `/start` - Launch or reset your assistant session\n"
        "• `/help` - View this help menu\n\n"
        "**Example Prompts**:\n"
        "• *'Search the web for the latest developments in AI agents in 2026'*\n"
        "• *'Get the transcript of https://www.youtube.com/watch?v=dQw4w9WgXcQ'*\n"
        "• *'Download audio from https://www.youtube.com/watch?v=...'*\n"
        "• *'Schedule a Product Demo with Sarah tomorrow at 3 PM'*"
    )
    if update.message:
        await update.message.reply_text(help_text, parse_mode="Markdown")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles incoming natural language messages, passes them to AutoMate's brain,
    and returns the executive answer.
    """
    if not update.message or not update.message.text:
        return

    user_text = update.message.text
    user_id = str(update.effective_user.id) if update.effective_user else "unknown"
    logger.info(f"Received message from User ({user_id}): {user_text}")

    # Indicate typing action while processing tools and LLM response
    await update.message.chat.send_action(action="typing")

    # Process intent through the agent brain
    response_text = process_user_intent(user_text, platform="telegram")

    # Send response back to the user
    try:
        await update.message.reply_text(response_text, parse_mode="Markdown")
    except Exception:
        # Fallback to plain text if Markdown parsing fails
        await update.message.reply_text(response_text)


def main() -> None:
    """
    Initializes and starts the Telegram Bot application in polling mode.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token or "<insert" in token:
        logger.error("TELEGRAM_BOT_TOKEN is missing or invalid in .env file.")
        sys.exit(1)

    application = ApplicationBuilder().token(token).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    logger.info("AutoMate Telegram Bot is now online and polling...")
    application.run_polling()


if __name__ == "__main__":
    main()
