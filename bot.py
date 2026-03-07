import logging
import os
import json
from functools import wraps

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

# Setup prompt logging to file
prompt_logger = logging.getLogger('prompt_logger')
prompt_logger.setLevel(logging.INFO)
prompt_handler = logging.FileHandler('prompts.log', encoding='utf-8')
prompt_handler.setFormatter(logging.Formatter('%(asctime)s\n%(message)s\n{"="*50}\n'))
prompt_logger.addHandler(prompt_handler)

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
    # Log the prompt
    prompt_logger.info(f"[PROVIDER: {LLM_PROVIDER}]\n\n[SYSTEM PROMPT]:\n{system_prompt}\n\n[USER MESSAGE]:\n{user_message}")
    
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

# --- Custom filter states ---
STATE_CUSTOM_TOPIC = 10
STATE_CUSTOM_STYLE = 11
STATE_CUSTOM_GRAMMAR = 12

# --- Presets for Random Practice ---
PRESETS = {
    'business': {
        'name': '💼 Business',
        'topic': 'business, office, negotiations, meetings',
        'style': 'formal, professional, polite',
        'grammar': 'modal verbs, conditionals, passive voice'
    },
    'travel': {
        'name': '✈️ Travel',
        'topic': 'traveling, airports, hotels, sightseeing',
        'style': 'conversational, tourist situations',
        'grammar': 'going to, present continuous for future, phrasal verbs'
    },
    'casual': {
        'name': '😎 Casual',
        'topic': 'daily life, hobbies, friends, relationships',
        'style': 'informal, relaxed, slang allowed',
        'grammar': 'phrasal verbs, informal contractions, idioms'
    },
    'academic': {
        'name': '🎓 Academic',
        'topic': 'science, education, research, university life',
        'style': 'formal, academic, precise',
        'grammar': 'passive voice, complex sentences, linking words'
    },
    'technology': {
        'name': '💻 Technology',
        'topic': 'computers, internet, AI, gadgets, programming',
        'style': 'modern, technical but accessible',
        'grammar': 'present simple for facts, technical terminology'
    },
    'mixed': {
        'name': '🎲 Mixed (No filters)',
        'topic': None,
        'style': None,
        'grammar': None
    }
}

# --- Custom options for /custom command ---
# Using short IDs to avoid Telegram's 64-byte callback_data limit
CUSTOM_TOPICS = [
    ('🏥 Health & Medicine', 'health', 'health, medicine, doctor visits, fitness'),
    ('🍔 Food & Cooking', 'food', 'food, cooking, restaurants, recipes'),
    ('🎬 Movies & Entertainment', 'movies', 'movies, TV shows, music, celebrities'),
    ('🏠 Home & Family', 'home', 'home, family, household, daily routines'),
    ('🌦️ Weather & Nature', 'weather', 'weather, seasons, environment, nature'),
    ('🛒 Shopping', 'shopping', 'shopping, stores, online shopping, products'),
    ('💰 Money & Finance', 'money', 'money, banking, investing, budgeting'),
    ('💼 Work & Career', 'work', 'work, career, job interviews, office'),
]

CUSTOM_STYLES = [
    ('📧 Formal', 'formal', 'formal, polite, professional'),
    ('💬 Conversational', 'conversational', 'conversational, natural, everyday speech'),
    ('😄 Casual/Slang', 'casual', 'casual, informal, slang, relaxed'),
    ('📚 Academic', 'academic', 'academic, scholarly, precise'),
    ('📰 Journalistic', 'journalistic', 'journalistic, informative, engaging'),
    ('🎭 Creative', 'creative', 'creative, descriptive, storytelling'),
]

CUSTOM_GRAMMAR = [
    ('🔄 Present tenses', 'present', 'present simple, present continuous, present perfect'),
    ('⏰ Past tenses', 'past', 'past simple, past continuous, past perfect'),
    ('🔮 Future forms', 'future', 'will, going to, present continuous for future'),
    ('❓ Conditionals', 'conditionals', 'zero, first, second, third conditionals'),
    ('📝 Passive voice', 'passive', 'passive constructions in various tenses'),
    ('🎯 Phrasal verbs', 'phrasal', 'common phrasal verbs and particles'),
    ('📎 Relative clauses', 'relative', 'defining and non-defining relative clauses'),
    ('⚡ Reported speech', 'reported', 'reported statements, questions, commands'),
    ('🎨 Adjective clauses', 'adjective', 'participles, reduced relative clauses'),
    ('🔗 Linking words', 'linking', 'however, although, despite, furthermore'),
]

# Build lookup dictionaries
def build_lookup(options_list):
    return {short_id: full_value for _, short_id, full_value in options_list}

TOPICS_LOOKUP = build_lookup(CUSTOM_TOPICS)
STYLES_LOOKUP = build_lookup(CUSTOM_STYLES)
GRAMMAR_LOOKUP = build_lookup(CUSTOM_GRAMMAR)

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
        text=f"👋 Welcome, owner!\n\n"
             f"🤖 *LLM Provider:* `{LLM_PROVIDER}`\n"
             f"📌 *Current mode:* `{context.chat_data['mode']}`\n\n"
             f"*Available modes:*\n"
             f"🎓 *Training* — practice with your phrases\n"
             f"🇬🇧 *English Only* — generate English text\n"
             f"🧑‍🏫 *Explain* — get word/phrase explanations\n"
             f"🎲 *Random Practice* — random phrases with context\n\n"
             f"*Filter commands (work in Training/Random/English modes):*\n"
             f"/preset — quick filter presets\n"
             f"/custom — create custom filters (topic, style, grammar)\n"
             f"/filters — view current filter settings\n"
             f"/clear — reset filters to default\n\n"
             f"*Other commands:*\n"
             f"/mode — change practice mode\n"
             f"/next — generate new random phrase",
        parse_mode=ParseMode.MARKDOWN
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
    """Parses the CallbackQuery and updates the chat mode or filters."""
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    # Handle preset selection
    if data.startswith("preset:"):
        preset_id = data.replace("preset:", "")
        preset = PRESETS.get(preset_id)
        
        if preset:
            # Store filters
            context.chat_data['filters'] = {
                'topic': preset.get('topic'),
                'style': preset.get('style'),
                'grammar': preset.get('grammar'),
                'preset': preset.get('name')
            }
            
            await query.edit_message_text(
                text=f"✅ Пресет *{preset['name']}* активирован!\n\n"
                     f"{format_filters(context.chat_data['filters'])}\n\n"
                     f"Теперь в режиме 🎲 Random Practice фразы будут генерироваться с этими настройками.",
                parse_mode=ParseMode.MARKDOWN
            )
        return
    
    # Handle custom topic selection
    if data.startswith("custom_topic:"):
        short_id = data.replace("custom_topic:", "")
        
        if 'filters' not in context.chat_data:
            context.chat_data['filters'] = {}
        
        if short_id != 'skip':
            context.chat_data['filters']['topic'] = TOPICS_LOOKUP.get(short_id)
        
        # Show style selection
        keyboard = [[InlineKeyboardButton(name, callback_data=f"custom_style:{short_id}")] 
                    for name, short_id, _ in CUSTOM_STYLES]
        keyboard.append([InlineKeyboardButton("🔄 Пропустить", callback_data="custom_style:skip")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            text='🔧 *Настройка генерации* (шаг 2/3)\n\n'
                 'Выбери стиль текста:',
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    # Handle custom style selection
    if data.startswith("custom_style:"):
        short_id = data.replace("custom_style:", "")
        
        if short_id != 'skip':
            context.chat_data['filters']['style'] = STYLES_LOOKUP.get(short_id)
        
        # Show grammar selection
        keyboard = [[InlineKeyboardButton(name, callback_data=f"custom_grammar:{short_id}")] 
                    for name, short_id, _ in CUSTOM_GRAMMAR]
        keyboard.append([InlineKeyboardButton("🔄 Пропустить", callback_data="custom_grammar:skip")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            text='🔧 *Настройка генерации* (шаг 3/3)\n\n'
                 'Выбери грамматические конструкции:',
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    # Handle custom grammar selection
    if data.startswith("custom_grammar:"):
        short_id = data.replace("custom_grammar:", "")
        
        if short_id != 'skip':
            context.chat_data['filters']['grammar'] = GRAMMAR_LOOKUP.get(short_id)
        
        # Show final summary
        filters_text = format_filters(context.chat_data['filters'])
        await query.edit_message_text(
            text=f"✅ *Настройки сохранены!*\n\n{filters_text}\n\n"
                 f"Теперь в режиме 🎲 Random Practice фразы будут генерироваться с этими параметрами.",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    # Handle mode selection (original behavior)
    if data in [MODE_TRAINING, MODE_ENGLISH_ONLY, MODE_EXPLAIN, MODE_RANDOM]:
        context.chat_data['mode'] = data
        context.chat_data['state'] = STATE_AWAITING_PHRASE
        
        if data == MODE_RANDOM:
            await query.edit_message_text(text=f"Mode set to: {data}.\nSend any message or /next to get a random phrase.")
        else:
            await query.edit_message_text(text=f"Mode set to: {data}.\nSend me a word or phrase.")


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
    
    # Get filters from chat_data
    filters = context.chat_data.get('filters', {})
    
    # Build prompt with filters
    prompt_parts = ["Дано предложение или фраза."]
    
    prompt_parts.append("Задача: составить текст на английском языке, состоящий из 3-5 предложений, содержащий данное предложение или фразу.")
    
    # Add topic if set
    if filters.get('topic'):
        prompt_parts.append(f"Тема/контекст текста: {filters['topic']}.")
    
    # Add style if set, otherwise use default
    if filters.get('style'):
        prompt_parts.append(f"Стиль текста: {filters['style']}.")
    else:
        prompt_parts.append("Стиль - неформальный, разговорный, можно диалог.")
    
    # Add grammar constructions if set
    if filters.get('grammar'):
        prompt_parts.append(f"Обязательно используй следующие грамматические конструкции: {filters['grammar']}.")
    
    prompt_parts.append("Также перевести текст на русский язык.")
    prompt_parts.append('Результат должен быть в формате JSON: {"phrase": "<Исходное предложение>", "russian":"<Текст на русском>", "english":"<Текст на английском>"}')
    prompt_parts.append("Предложение: ")
    
    system_prompt = "\n".join(prompt_parts)

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
    
    # Get filters from chat_data
    filters = context.chat_data.get('filters', {})
    
    # Build prompt with filters
    prompt_parts = ["Given a sentence or a phrase."]
    prompt_parts.append("Task: create a text in English, consisting of 3-5 sentences, containing the given sentence or phrase.")
    
    if filters.get('style'):
        prompt_parts.append(f"Style: {filters['style']}.")
    if filters.get('topic'):
        prompt_parts.append(f"Context/theme: {filters['topic']}.")
    if filters.get('grammar'):
        prompt_parts.append(f"Must use these grammar constructions: {filters['grammar']}.")
    
    prompt_parts.append("The result should be only the generated English text, without any other formatting or labels.")
    
    system_prompt = "\n".join(prompt_parts)
    
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
    
    # Get filters from chat_data
    filters = context.chat_data.get('filters', {})
    system_prompt = build_random_prompt(filters)

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


# --- Helper functions for filters ---
def build_random_prompt(filters: dict = None) -> str:
    """Builds system prompt for random generation with optional filters."""
    filters = filters or {}
    
    parts = ["Ты — помощник для изучения английского языка."]
    
    # Build constraints description
    constraints = []
    
    if filters.get('topic'):
        constraints.append(f"тема: {filters['topic']}")
    if filters.get('style'):
        constraints.append(f"стиль: {filters['style']}")
    if filters.get('grammar'):
        constraints.append(f"грамматика: {filters['grammar']}")
    
    if constraints:
        parts.append(f"Сгенерируй случайное английское слово или короткую фразу (idiom, phrasal verb, или слово B1-C1 уровня) на {'; '.join(constraints)}.")
    else:
        parts.append("Сгенерируй случайное английское слово или короткую фразу (idiom, phrasal verb, или слово B1-C1 уровня).")
    
    parts.append("Затем составь текст на английском языке, состоящий из 3-5 предложений, содержащий это слово/фразу.")
    
    if filters.get('style'):
        parts.append(f"Стиль текста: {filters['style']}.")
    else:
        parts.append("Стиль - неформальный, разговорный, можно диалог.")
    
    if filters.get('grammar'):
        parts.append(f"Обязательно используй следующие грамматические конструкции: {filters['grammar']}.")
    
    parts.append("Также переведи текст на русский язык.")
    parts.append('Результат должен быть в формате JSON: {"phrase": "<Слово или фраза>", "russian":"<Текст на русском>", "english":"<Текст на английском>"}')
    
    return "\n".join(parts)


def format_filters(filters: dict) -> str:
    """Format current filters for display."""
    if not filters or not any(filters.values()):
        return "Нет активных фильтров (случайная генерация)"
    
    lines = []
    if filters.get('topic'):
        lines.append(f"📌 *Тема:* {filters['topic']}")
    if filters.get('style'):
        lines.append(f"🎨 *Стиль:* {filters['style']}")
    if filters.get('grammar'):
        lines.append(f"📚 *Грамматика:* {filters['grammar']}")
    if filters.get('preset'):
        lines.append(f"⚙️ *Пресет:* {filters['preset']}")
    
    return "\n".join(lines)


@owner_only
async def preset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Shows preset selection buttons."""
    keyboard = []
    for preset_id, preset_data in PRESETS.items():
        keyboard.append([InlineKeyboardButton(preset_data['name'], callback_data=f"preset:{preset_id}")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        '🎯 Выбери пресет для генерации фраз:\n\n'
        '💼 *Business* — формальный стиль, деловая лексика\n'
        '✈️ *Travel* — путешествия, разговорный стиль\n'
        '😎 *Casual* — повседневная жизнь, сленг\n'
        '🎓 *Academic* — научный стиль, сложные конструкции\n'
        '💻 *Technology* — технологии, современная лексика\n'
        '🎲 *Mixed* — без фильтров (как раньше)',
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN
    )


@owner_only
async def custom_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Starts custom filter selection wizard."""
    keyboard = [[InlineKeyboardButton(name, callback_data=f"custom_topic:{short_id}")] 
                for name, short_id, _ in CUSTOM_TOPICS]
    keyboard.append([InlineKeyboardButton("🔄 Пропустить", callback_data="custom_topic:skip")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        '🔧 *Настройка генерации* (шаг 1/3)\n\n'
        'Выбери тему для фразы:',
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN
    )


@owner_only
async def filters_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Shows current active filters."""
    filters = context.chat_data.get('filters', {})
    
    text = "⚙️ *Текущие настройки генерации:*\n\n"
    text += format_filters(filters)
    text += "\n\nИспользуй /preset для выбора готового пресета\n"
    text += "Или /custom для создания своих настроек\n"
    text += "Или /clear для сброса фильтров"
    
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


@owner_only
async def clear_filters_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clears all active filters."""
    if 'filters' in context.chat_data:
        del context.chat_data['filters']
    
    await update.message.reply_text(
        "✅ *Фильтры сброшены!*\n\n"
        "Теперь генерация будет происходить без ограничений (режим Mixed).",
        parse_mode=ParseMode.MARKDOWN
    )


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
    application.add_handler(CommandHandler('preset', preset_command))
    application.add_handler(CommandHandler('custom', custom_command))
    application.add_handler(CommandHandler('filters', filters_command))
    application.add_handler(CommandHandler('clear', clear_filters_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    print(f"Bot is running with LLM Provider: {LLM_PROVIDER}. Press Ctrl+C to stop.")
    application.run_polling()