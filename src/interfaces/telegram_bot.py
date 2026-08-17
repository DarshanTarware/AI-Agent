"""
Telegram Bot Interface for AutoMate.

Asynchronous message listener using python-telegram-bot that connects
user commands directly to the AutoMate LLM agent brain, with SQLite memory
persistence and automated media file delivery.
"""

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
from src.data.database import init_db, insert_message

load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles the /start command and initializes user session.
    """
    user_name = update.effective_user.first_name if update.effective_user else "there"
    user_id = str(update.effective_user.id) if update.effective_user else "default_user"
    
    welcome_message = (
        f"👋 Hello {user_name}! I am **AutoMate**, your multimodal personal executive AI agent.\n\n"
        "Here is what I can do for you:\n"
        "🔍 **Live Web Searches**: Ask any fact-based question.\n"
        "📹 **YouTube Intelligence**: Extract transcripts and summarize videos.\n"
        "🎵 **Audio Downloads**: Download audio from YouTube and get the file delivered right here!\n"
        "📅 **Google Calendar**: Schedule meetings directly into your calendar.\n\n"
        "Simply send me a message or link to get started!"
    )

    insert_message(user_id=user_id, role="assistant", message=welcome_message)

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
        "• *'Search the web for the latest advancements in AI agents in 2026'*\n"
        "• *'Get the transcript of https://www.youtube.com/watch?v=dQw4w9WgXcQ'*\n"
        "• *'Download audio from https://youtu.be/K_8yRH2KPVo'*\n"
        "• *'Schedule a Project Sync on 2026-08-20 from 3 PM to 4 PM'*"
    )
    if update.message:
        await update.message.reply_text(help_text, parse_mode="Markdown")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles incoming messages, manages persistent chat memory, executes intent
    through AutoMate brain, and delivers media files (audio) directly to chat.
    """
    if not update.message or not update.message.text:
        return

    user_text = update.message.text
    user_id = str(update.effective_user.id) if update.effective_user else "default_user"
    chat_id = update.effective_chat.id if update.effective_chat else user_id

    logger.info(f"Received message from User ({user_id}): {user_text}")

    # 1. Record incoming user message to SQLite memory
    insert_message(user_id=user_id, role="user", message=user_text)

    # Indicate typing activity
    await update.message.chat.send_action(action="typing")

    # 2. Process query with AutoMate Brain
    brain_resp = process_user_intent(user_input=user_text, user_id=user_id, platform="telegram")
    response_text = brain_resp.text if hasattr(brain_resp, "text") else str(brain_resp)
    media_file_path = brain_resp.file_path if hasattr(brain_resp, "file_path") else None

    # 3. Route media file directly into Telegram chat if generated
    if media_file_path and os.path.exists(media_file_path):
        try:
            clean_name = os.path.splitext(os.path.basename(media_file_path))[0]
            logger.info(f"Delivering media asset to Telegram chat: {media_file_path}")
            await update.message.chat.send_action(action="upload_document")
            with open(media_file_path, "rb") as audio_file:
                await context.bot.send_audio(
                    chat_id=chat_id,
                    audio=audio_file,
                    filename=os.path.basename(media_file_path),
                    title=clean_name,
                    performer="AutoMate Audio",
                    caption=f"🎧 **{clean_name}**\n\nTap play to listen directly on your phone!",
                    parse_mode="Markdown",
                )
        except Exception as exc:
            logger.error(f"Failed to send audio via send_audio: {exc}. Attempting send_document.")
            try:
                with open(media_file_path, "rb") as doc_file:
                    await context.bot.send_document(
                        chat_id=chat_id,
                        document=doc_file,
                        caption="📁 Downloaded Audio File",
                    )
            except Exception as doc_exc:
                logger.error(f"Failed to deliver document file: {doc_exc}")

    # 4. Deliver textual response
    try:
        await update.message.reply_text(response_text, parse_mode="Markdown")
    except Exception:
        await update.message.reply_text(response_text)

    # 5. Persist assistant response to SQLite memory
    insert_message(user_id=user_id, role="assistant", message=response_text)


def main() -> None:
    """
    Initializes database schema and starts Telegram bot polling.
    """
    init_db()

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
