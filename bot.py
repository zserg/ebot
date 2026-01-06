import logging
import os
import json
from functools import wraps
import google.generativeai as genai
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

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

# --- State definitions ---
STATE_AWAITING_PHRASE = 1
STATE_AWAITING_REVEAL = 2

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-3-flash-preview')


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
    """Starts the conversation for the owner."""
    context.chat_data.clear()
    context.chat_data['state'] = STATE_AWAITING_PHRASE
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Welcome, owner! Please send me a phrase to begin."
    )

@owner_only
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Main message handler that delegates to state-specific handlers for the owner."""
    state = context.chat_data.get('state', STATE_AWAITING_PHRASE)
    
    if state == STATE_AWAITING_PHRASE:
        await handle_phrase_and_return_russian(update, context)
    elif state == STATE_AWAITING_REVEAL:
        await handle_reveal_english(update, context)

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
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
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
    """Sends the stored English text and resets the flow."""
    chat_id = update.effective_chat.id
    english_text = context.chat_data.get('english_text')

    if english_text:
        await context.bot.send_message(chat_id=chat_id, text=english_text)
    
    context.chat_data.clear()
    context.chat_data['state'] = STATE_AWAITING_PHRASE
    await context.bot.send_message(chat_id=chat_id, text="Let's start over. Send me a new phrase.")

if __name__ == '__main__':
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    print("Bot is running with owner-only restriction... Press Ctrl+C to stop.")
    application.run_polling()