import logging
import os
import json
from functools import wraps

# LLM-related imports
import google.generativeai as genai
from gigachat import GigaChat
from openai import AsyncOpenAI

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters, CallbackQueryHandler
from telegram.error import BadRequest
from telegram.constants import ParseMode

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# --- Environment Variable Configuration ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
try:
    OWNER_ID = int(os.getenv("TELEGRAM_OWNER_ID"))
except (TypeError, ValueError):
    raise ValueError("TELEGRAM_OWNER_ID environment variable not set or is not a valid integer.")

if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set.")

# --- LLM Provider Configuration ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "DEEPSEEK").upper()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

gemini_model = None
deepseek_client = None

if LLM_PROVIDER == "GEMINI":
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set for GEMINI provider.")
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-3-flash-preview')
    logging.info("Using GEMINI as LLM provider.")
elif LLM_PROVIDER == "GIGACHAT":
    if not GIGACHAT_CREDENTIALS:
        raise ValueError("GIGACHAT_CREDENTIALS environment variable not set for GIGACHAT provider.")
    logging.info("Using GIGACHAT as LLM provider.")
elif LLM_PROVIDER == "DEEPSEEK":
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY environment variable not set for DEEPSEEK provider.")
    deepseek_client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    logging.info("Using DEEPSEEK as LLM provider.")
else:
    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}. Use 'GEMINI', 'GIGACHAT', or 'DEEPSEEK'.")


# --- Generic LLM Response Function ---
async def generate_llm_response(system_prompt: str, user_message: str) -> str:
    """
    Generates a response from the configured LLM provider.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    if LLM_PROVIDER == "GEMINI":
        # Gemini prefers a single combined prompt
        full_prompt = f"{system_prompt}\n\n{user_message}"
        response = await gemini_model.generate_content_async(full_prompt)
        return response.text
    elif LLM_PROVIDER == "GIGACHAT":
        async with GigaChat(credentials=GIGACHAT_CREDENTIALS, verify_ssl_certs=False) as client:
            response = await client.achat(messages=messages)
            return response.choices[0].message.content
    elif LLM_PROVIDER == "DEEPSEEK":
        response = await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )
        return response.choices[0].message.content
        
    return "Error: LLM Provider not configured correctly."


# --- Mode and State definitions ---
MODE_TRAINING = 'mode_training'
MODE_ENGLISH_ONLY = 'mode_english_only'
MODE_EXPLAIN = 'mode_explain'
MODE_RANDOM = 'mode_random'

STATE_AWAITING_PHRASE = 1
STATE_AWAITING_REVEAL = 2

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
    if 'mode' not in context.chat_data:
        context.chat_data['mode'] = MODE_TRAINING

    context.chat_data['state'] = STATE_AWAITING_PHRASE
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"Welcome, owner! LLM: {LLM_PROVIDER}. Current mode: '{context.chat_data['mode']}'.\n"
             f"Send me a phrase to begin or use /mode to change it.\n"
             f"Available modes: 🎓 Training, 🇬🇧 English Only, 🧑‍🏫 Explain, 🎲 Random Practice"
    )

@owner_only
async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Displays mode selection buttons."""
    keyboard = [
        [InlineKeyboardButton("🎓 Training", callback_data=MODE_TRAINING)],
        [InlineKeyboardButton("🇬🇧 English Only", callback_data=MODE_ENGLISH_ONLY)],
        [InlineKeyboardButton("🧑‍🏫 Explain", callback_data=MODE_EXPLAIN)],
        [InlineKeyboardButton("🎲 Random Practice", callback_data=MODE_RANDOM)],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text('Please choose a mode:', reply_markup=reply_markup)


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Parses the CallbackQuery and updates the chat mode."""
    query = update.callback_query
    await query.answer()

    context.chat_data['mode'] = query.data
    context.chat_data['state'] = STATE_AWAITING_PHRASE
    
    if query.data == MODE_RANDOM:
        await query.edit_message_text(text=f"Mode set to: {query.data}.\nSend any message or /next to get a random phrase.")
    else:
        await query.edit_message_text(text=f"Mode set to: {query.data}.\nSend me a word or phrase.")


@owner_only
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Main message handler that delegates to mode- and state-specific handlers."""
    mode = context.chat_data.get('mode', MODE_TRAINING)
    state = context.chat_data.get('state', STATE_AWAITING_PHRASE)

    if mode == MODE_TRAINING:
        if state == STATE_AWAITING_PHRASE:
            await handle_phrase_and_return_russian(update, context)
        elif state == STATE_AWAITING_REVEAL:
            await handle_reveal_english(update, context)
    elif mode == MODE_ENGLISH_ONLY:
        await handle_english_only_generation(update, context)
    elif mode == MODE_EXPLAIN:
        await handle_explain_mode(update, context)
    elif mode == MODE_RANDOM:
        if state == STATE_AWAITING_PHRASE:
            await handle_random_generation(update, context)
        elif state == STATE_AWAITING_REVEAL:
            await handle_reveal_english(update, context)


async def handle_phrase_and_return_russian(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generates texts, sends Russian part, stores English part."""
    user_message = update.message.text
    chat_id = update.effective_chat.id

    system_prompt = """Дано предложение или фраза.
Задача: составить текст на английском языке, состоящий из 3-5 предложений, содержащий данное предложение или фразу. Стиль - неформальный, разговорный, можно диалог. Также перевести текст на русский язык.
Результат должен быть в формате JSON: {"phrase": "<Исходное предложение>", "russian":"<Текст на русском>", "english":"<Текст на английском>"}
Предложение: """

    try:
        await context.bot.send_chat_action(chat_id=chat_id, action='typing')
        response_text = await generate_llm_response(system_prompt, f"Предложение: {user_message}")
        
        cleaned_text = response_text.strip().lstrip("```json").rstrip("```").strip()
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
    """Sends the stored English text and resets the flow for Training/Random mode."""
    chat_id = update.effective_chat.id
    english_text = context.chat_data.get('english_text')
    phrase = context.chat_data.get('phrase')

    if english_text:
        if phrase:
            await context.bot.send_message(chat_id=chat_id, text=f"📝 *English:*\n{english_text}", parse_mode=ParseMode.MARKDOWN)
        else:
            await context.bot.send_message(chat_id=chat_id, text=english_text)

    context.chat_data['state'] = STATE_AWAITING_PHRASE
    mode = context.chat_data.get('mode', MODE_TRAINING)
    if mode == MODE_RANDOM:
        await context.bot.send_message(chat_id=chat_id, text="Send /next for a new phrase, or send any message to practice again with the same phrase.")
    else:
        await context.bot.send_message(chat_id=chat_id, text="Let's start over. Send me a new phrase.")


async def handle_english_only_generation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generates and sends only the English text."""
    user_message = update.message.text
    chat_id = update.effective_chat.id

    system_prompt = """Given a sentence or a phrase.
Task: create a text in English, consisting of 3-5 sentences, containing the given sentence or phrase.
The result should be only the generated English text, without any other formatting or labels."""
    
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action='typing')
        response_text = await generate_llm_response(system_prompt, f"Phrase: {user_message}")
        
        await context.bot.send_message(chat_id=chat_id, text=response_text)
        await context.bot.send_message(chat_id=chat_id, text="Send me another phrase.")
        
    except Exception as e:
        logging.error(f"Error in handle_english_only_generation: {e}")
        await context.bot.send_message(chat_id=chat_id, text="An error occurred. Please try again.")


async def handle_random_generation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generates a random word/phrase and creates context text for practice."""
    chat_id = update.effective_chat.id

    system_prompt = """Ты — помощник для изучения английского языка.
Сгенерируй случайное английское слово или короткую фразу (idiom, phrasal verb, или слово B1-C1 уровня).
Затем составь текст на английском языке, состоящий из 3-5 предложений, содержащий это слово/фразу. Стиль - неформальный, разговорный, можно диалог.
Также переведи текст на русский язык.
Результат должен быть в формате JSON: {"phrase": "<Слово или фраза>", "russian":"<Текст на русском>", "english":"<Текст на английском>"}"""

    try:
        await context.bot.send_chat_action(chat_id=chat_id, action='typing')
        response_text = await generate_llm_response(system_prompt, "Сгенерируй случайное слово/фразу и текст.")
        
        cleaned_text = response_text.strip().lstrip("```json").rstrip("```").strip()
        data = json.loads(cleaned_text)

        context.chat_data['english_text'] = data['english']
        context.chat_data['phrase'] = data['phrase']
        
        await context.bot.send_message(
            chat_id=chat_id, 
            text=f"🎯 *Phrase:* `{data['phrase']}`\n\n{data['russian']}",
            parse_mode=ParseMode.MARKDOWN
        )

        context.chat_data['state'] = STATE_AWAITING_REVEAL
        await context.bot.send_message(
            chat_id=chat_id, 
            text="Send /next for a new phrase, or send any message to reveal the English version."
        )
    except Exception as e:
        logging.error(f"Error in handle_random_generation: {e}")
        context.chat_data.clear()
        await context.bot.send_message(chat_id=chat_id, text="An error occurred. Let's start over.")


@owner_only
async def next_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generates next random phrase in random mode."""
    mode = context.chat_data.get('mode', MODE_TRAINING)
    
    if mode != MODE_RANDOM:
        await update.message.reply_text("This command only works in 🎲 Random Practice mode. Use /mode to switch.")
        return
    
    context.chat_data['state'] = STATE_AWAITING_PHRASE
    await handle_random_generation(update, context)


async def handle_explain_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Explains a word or phrase as an English teacher, with a fallback for Markdown errors."""
    user_message = update.message.text
    chat_id = update.effective_chat.id

    system_prompt = """You are an English teacher. The user will provide a word or a phrase.
Your task is to explain its meaning in simple English. Provide a clear definition and 2-3 examples of modern use.
Format the response using Telegram's MarkdownV2 style.
- Use *bold* for the main word/phrase.
- Use _italic_ for emphasis.
- Use bullet points starting with a hyphen '-'.
- IMPORTANT: You MUST escape the characters `_`, `*`, `[`, `]`, `(`, `)`, `~`, `` ` ``, `>`, `#`, `+`, `-`, `=`, `|`, `{`, `}`, `.`, `!` in all other text by preceding them with a backslash `\`. For example, write `a\.b` instead of `a.b`."""
    
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action='typing')
        response_text = await generate_llm_response(system_prompt, f"Word/Phrase: {user_message}")
        
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=response_text,
                parse_mode=ParseMode.MARKDOWN_V2
            )
        except BadRequest as e:
            if "Can't parse entities" in str(e):
                logging.warning(
                    f"MarkdownV2 parsing failed for text: '{response_text}'. "
                    f"Error: {e}. Sending as plain text."
                )
                await context.bot.send_message(chat_id=chat_id, text=response_text)
            else:
                raise e

        await context.bot.send_message(chat_id=chat_id, text="Send me another word or phrase to explain.")
        
    except Exception as e:
        logging.error(f"Error in handle_explain_mode: {e}")
        await context.bot.send_message(chat_id=chat_id, text="An error occurred. Please try again.")


if __name__ == '__main__':
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('mode', mode_command))
    application.add_handler(CommandHandler('next', next_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    print(f"Bot is running with LLM Provider: {LLM_PROVIDER}. Press Ctrl+C to stop.")
    application.run_polling()