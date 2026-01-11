import logging
import os
import json
from functools import wraps
import google.generativeai as genai
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters, CallbackQueryHandler

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# --- Environment Variable Configuration ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
try:
    OWNER_ID = int(os.getenv("TELEGRAM_OWNER_ID"))
except (TypeError, ValueError):
    raise ValueError("TELEGRAM_OWNER_ID environment variable not set or is not a valid integer.")

if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set.")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

# --- Mode and State definitions ---
MODE_TRAINING = 'mode_training'
MODE_ENGLISH_ONLY = 'mode_english_only'

STATE_AWAITING_PHRASE = 1
STATE_AWAITING_REVEAL = 2

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-preview')


# --- Decorator for owner-only access ---
def owner_only(func):
    @wraps(func)
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user = update.effective_user
        if user.id != OWNER_ID:
            logging.warning(f"Unauthorized access denied for user {user.id} ({user.username}).")
            return
        return await func(update, context, *args, **kwargs)
    return wrapped


@owner_only
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Starts the conversation and sets the default mode."""
    context.chat_data.clear()
    # Set default mode
    if 'mode' not in context.chat_data:
        context.chat_data['mode'] = MODE_TRAINING

    context.chat_data['state'] = STATE_AWAITING_PHRASE
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"Welcome, owner! Current mode is '{context.chat_data['mode']}'.\n"
             f"Send me a phrase to begin or use /mode to change it."
    )

@owner_only
async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Displays mode selection buttons."""
    keyboard = [
        [InlineKeyboardButton("🎓 Training", callback_data=MODE_TRAINING)],
        [InlineKeyboardButton("🇬🇧 English Only", callback_data=MODE_ENGLISH_ONLY)],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text('Please choose a mode:', reply_markup=reply_markup)


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Parses the CallbackQuery and updates the chat mode."""
    query = update.callback_query
    await query.answer()

    context.chat_data['mode'] = query.data
    context.chat_data['state'] = STATE_AWAITING_PHRASE # Reset state after mode change

    await query.edit_message_text(text=f"Mode set to: {query.data}.\nSend me a phrase.")


@owner_only
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Main message handler that delegates to mode and state-specific handlers."""
    mode = context.chat_data.get('mode', MODE_TRAINING)
    state = context.chat_data.get('state', STATE_AWAITING_PHRASE)

    if mode == MODE_TRAINING:
        if state == STATE_AWAITING_PHRASE:
            await handle_phrase_and_return_russian(update, context)
        elif state == STATE_AWAITING_REVEAL:
            await handle_reveal_english(update, context)
    elif mode == MODE_ENGLISH_ONLY:
        await handle_english_only_generation(update, context)


async def handle_phrase_and_return_russian(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generates texts, sends Russian part, stores English part."""
    user_message = update.message.text
    chat_id = update.effective_chat.id

    system_prompt = """Дано предложение или фраза.
Задача: составить текст на английском языке, состоящий из 3-5 предложений, содержащий данное предложение или фразу. Также перевести текст на русский язык.
Результат должен быть в формате JSON: {"phrase": "<Исходное предложение>", "russian":"<Текст на русском>", "english":"<Текст на английском>"}
Предложение: """

    try:
        await context.bot.send_chat_action(chat_id=chat_id, action='typing')
        full_prompt = f"{system_prompt}{user_message}"
        response = await model.generate_content_async(full_prompt)
        # It's better to clean the response from markdown code blocks
        cleaned_text = response.text.strip().lstrip("```json").rstrip("```")
        data = json.loads(cleaned_text)

        context.chat_data['english_text'] = data['english']
        await context.bot.send_message(chat_id=chat_id, text=data['russian'])

        context.chat_data['state'] = STATE_AWAITING_REVEAL
        await context.bot.send_message(chat_id=chat_id, text="Now, send any message to get the English version.")
    except Exception as e:
        logging.error(f"Error in handle_phrase_and_return_russian: {e}")
        context.chat_data.clear()
        await context.bot.send_message(chat_id=chat_id, text="An error occurred. Let's start over.")


async def handle_reveal_english(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Sends the stored English text and resets the flow for Training mode."""
    chat_id = update.effective_chat.id
    english_text = context.chat_data.get('english_text')

    if english_text:
        await context.bot.send_message(chat_id=chat_id, text=english_text)

    # Reset state for the next phrase
    context.chat_data['state'] = STATE_AWAITING_PHRASE
    await context.bot.send_message(chat_id=chat_id, text="Let's start over. Send me a new phrase.")

async def handle_english_only_generation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generates and sends only the English text."""
    user_message = update.message.text
    chat_id = update.effective_chat.id

    system_prompt = """Given a sentence or a phrase.
Task: create a text in English, consisting of 3-5 sentences, containing the given sentence or phrase.
The result should be only the generated English text, without any other formatting or labels.
Phrase: """
    
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action='typing')
        full_prompt = f"{system_prompt}{user_message}"
        response = await model.generate_content_async(full_prompt)
        
        await context.bot.send_message(chat_id=chat_id, text=response.text)
        await context.bot.send_message(chat_id=chat_id, text="Send me another phrase.")
        
    except Exception as e:
        logging.error(f"Error in handle_english_only_generation: {e}")
        await context.bot.send_message(chat_id=chat_id, text="An error occurred. Please try again.")


if __name__ == '__main__':
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('mode', mode_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    print("Bot is running with owner-only restriction... Press Ctrl+C to stop.")
    application.run_polling()