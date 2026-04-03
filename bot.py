import asyncio
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, time
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from openai import OpenAI
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackContext,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# =========================
# БАЗОВАЯ НАСТРОЙКА
# =========================

load_dotenv()

logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("max-assistant-bot")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
TIMEZONE_NAME = os.getenv("TIMEZONE", "Europe/Moscow")

if not BOT_TOKEN:
    raise RuntimeError("В .env не задан TELEGRAM_BOT_TOKEN")

TZ = ZoneInfo(TIMEZONE_NAME)
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = DATA_DIR / "state.json"

# OpenAI Responses API — текущий основной интерфейс генерации ответов в документации OpenAI. :contentReference[oaicite:1]{index=1}
OPENAI_CLIENT: Optional[OpenAI] = None
if OPENAI_API_KEY:
    OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# ДАННЫЕ ПОЛЬЗОВАТЕЛЯ
# =========================

@dataclass
class UserConfig:
    # Общие настройки
    user_name: str = "Макс"
    wake_time: str = "09:00"
    end_time: str = "24:00"
    tone: str = "человеческий, уверенный, средне настойчивый"
    frequency: str = "часто"
    main_priority: str = "деньги"
    priorities: List[str] = None

    # Контекст
    active_project: str = "М.Видео"
    current_projects: List[str] = None
    weak_points: List[str] = None

    # День / задачи
    morning_main_task: str = ""
    secondary_tasks: List[str] = None
    last_done: str = ""
    current_blocker: str = ""
    last_status: str = ""
    last_assistant_message: str = ""

    # Память
    repeating_problem: str = ""
    recent_events: List[str] = None
    daily_journal: List[str] = None

    # Поведение
    unanswered_pings: int = 0
    last_user_message_at: str = ""
    last_bot_message_at: str = ""

    def __post_init__(self):
        if self.priorities is None:
            self.priorities = ["деньги", "дисциплина", "тело"]
        if self.current_projects is None:
            self.current_projects = ["М.Видео", "WB", "режим/энергия"]
        if self.weak_points is None:
            self.weak_points = [
                "откладывает сложные шаги",
                "может зависать, если неясен следующий шаг",
                "уходит в прокрастинацию"
            ]
        if self.secondary_tasks is None:
            self.secondary_tasks = []
        if self.recent_events is None:
            self.recent_events = []
        if self.daily_journal is None:
            self.daily_journal = []


# =========================
# ХРАНЕНИЕ СОСТОЯНИЯ
# =========================

def _default_state() -> Dict[str, Any]:
    return {"users": {}}


def read_state() -> Dict[str, Any]:
    if not STATE_FILE.exists():
        return _default_state()
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Не удалось прочитать state.json")
        return _default_state()


def write_state(state: Dict[str, Any]) -> None:
    STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_user_config(chat_id: int) -> UserConfig:
    state = read_state()
    raw = state["users"].get(str(chat_id))
    if not raw:
        return UserConfig()
    return UserConfig(**raw)


def save_user_config(chat_id: int, cfg: UserConfig) -> None:
    state = read_state()
    state["users"][str(chat_id)] = asdict(cfg)
    write_state(state)


# =========================
# ВСПОМОГАТЕЛЬНАЯ ЛОГИКА
# =========================

def now_dt() -> datetime:
    return datetime.now(TZ)


def now_str() -> str:
    return now_dt().strftime("%d.%m.%Y %H:%M")


def stamp() -> str:
    return now_dt().strftime("%d.%m %H:%M")


def add_recent_event(cfg: UserConfig, text: str, limit: int = 20) -> None:
    text = text.strip()
    if not text:
        return
    cfg.recent_events.append(f"{stamp()} — {text}")
    cfg.recent_events = cfg.recent_events[-limit:]


def add_journal_entry(cfg: UserConfig, text: str, limit: int = 40) -> None:
    text = text.strip()
    if not text:
        return
    cfg.daily_journal.append(f"{stamp()} — {text}")
    cfg.daily_journal = cfg.daily_journal[-limit:]


def get_recent_context(cfg: UserConfig, limit: int = 10) -> str:
    if not cfg.recent_events:
        return "нет"
    return "\n".join(cfg.recent_events[-limit:])


def get_secondary_tasks_text(cfg: UserConfig) -> str:
    if not cfg.secondary_tasks:
        return "не указаны"
    return ", ".join(cfg.secondary_tasks)


def detect_repeating_problem(cfg: UserConfig) -> None:
    joined = " | ".join(cfg.recent_events[-10:]).lower()

    if "упаков" in joined:
        cfg.repeating_problem = "Ты несколько раз упирался в упаковку"
    elif "мвидео" in joined and ("не сделал" in joined or "не начал" in joined or "отлож" in joined):
        cfg.repeating_problem = "Ты откладываешь важные действия по М.Видео"
    elif "wb" in joined and ("не сделал" in joined or "отлож" in joined):
        cfg.repeating_problem = "Ты откладываешь действия по WB"
    elif "прокраст" in joined or "залип" in joined or "телефон" in joined:
        cfg.repeating_problem = "Ты снова провалился в прокрастинацию"
    elif "не могу" in joined or "непонятно" in joined or "не понимаю" in joined:
        cfg.repeating_problem = "Ты тормозишь на неясном следующем шаге"
    elif "не сделал" in joined or "не начал" in joined or "отлож" in joined:
        cfg.repeating_problem = "Ты повторно откладываешь важное действие"
    else:
        cfg.repeating_problem = ""


def format_status(cfg: UserConfig) -> str:
    projects = "\n".join([f"• {x}" for x in cfg.current_projects]) if cfg.current_projects else "—"
    weak_points = "\n".join([f"• {x}" for x in cfg.weak_points]) if cfg.weak_points else "—"
    secondary = "\n".join([f"• {x}" for x in cfg.secondary_tasks]) if cfg.secondary_tasks else "—"

    return (
        f"Имя: {cfg.user_name}\n"
        f"Приоритет: {cfg.main_priority}\n"
        f"Зоны: {', '.join(cfg.priorities)}\n"
        f"Активный проект: {cfg.active_project}\n"
        f"Проекты:\n{projects}\n\n"
        f"Слабые места:\n{weak_points}\n\n"
        f"Главная задача: {cfg.morning_main_task or '—'}\n"
        f"Второстепенные:\n{secondary}\n\n"
        f"Сделано: {cfg.last_done or '—'}\n"
        f"Блокер: {cfg.current_blocker or '—'}\n"
        f"Последний статус: {cfg.last_status or '—'}\n"
        f"Повторяющаяся проблема: {cfg.repeating_problem or '—'}\n"
        f"Неотвеченных пингов: {cfg.unanswered_pings}"
    )


# =========================
# OPENAI / FALLBACK
# =========================

def build_prompt(cfg: UserConfig, event_type: str) -> str:
    projects = ", ".join(cfg.current_projects) if cfg.current_projects else "не указаны"
    weak_points = ", ".join(cfg.weak_points) if cfg.weak_points else "не указаны"

    return f"""
Ты — персональный AI-ассистент Макса.

Контекст о пользователе:
- Макс — предприниматель
- главный приоритет сейчас: деньги
- важные зоны: {", ".join(cfg.priorities)}
- текущие проекты: {projects}
- активный проект прямо сейчас: {cfg.active_project}
- слабые места: {weak_points}
- бот должен не просто поддерживать, а возвращать к действиям, которые двигают деньги
- стиль общения: {cfg.tone}

Текущее состояние:
- главная задача на день: {cfg.morning_main_task or "не зафиксирована"}
- второстепенные задачи: {get_secondary_tasks_text(cfg)}
- последнее выполненное действие: {cfg.last_done or "нет"}
- текущий блокер: {cfg.current_blocker or "не указан"}
- последний статус пользователя: {cfg.last_status or "нет"}
- повторяющаяся проблема: {cfg.repeating_problem or "не выявлена"}
- неотвеченных пингов подряд: {cfg.unanswered_pings}

Последние события:
{get_recent_context(cfg)}

Сейчас: {now_str()}
Тип сообщения: {event_type}

Правила:
- отвечай по-русски
- 2–5 коротких строк
- без воды
- без эмодзи
- без длинных вступлений
- будь конкретным
- если видно повторяющуюся проблему, назови её прямо
- если человек застрял, помоги разбить на маленький следующий шаг
- если он игнорирует, стань чуть жёстче, но не груби
- в конце всегда вопрос или конкретное действие
""".strip()


def call_openai(cfg: UserConfig, event_type: str) -> Optional[str]:
    if OPENAI_CLIENT is None:
        return None

    prompt = build_prompt(cfg, event_type)
    try:
        response = OPENAI_CLIENT.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
        )
        text = getattr(response, "output_text", None)
        if text:
            return text.strip()
        return None
    except Exception:
        logger.exception("Ошибка вызова OpenAI")
        return None


def fallback_message(cfg: UserConfig, event_type: str) -> str:
    main_task = cfg.morning_main_task or "главная задача ещё не зафиксирована"
    blocker = cfg.current_blocker or "блокер пока не назван"

    messages = {
        "morning": (
            "Доброе утро.\n"
            "Что сегодня реально двинет деньги?\n"
            f"Главное на сейчас: {main_task}\n"
            "Что делаешь первым?"
        ),
        "midday_check": (
            f"Ты уже начал главное действие?\n"
            f"Сейчас у тебя фокус: {main_task}\n"
            "Если нет — что мешает?"
        ),
        "money_refocus": (
            "Проверь фокус.\n"
            "То, что ты делаешь сейчас, приближает деньги или это обходной манёвр?\n"
            "Напиши честно одним сообщением."
        ),
        "pressure_check": (
            "Стоп.\n"
            "Ты опять двигаешься по главному или растёкся по мелочам?\n"
            "Назови один следующий денежный шаг."
        ),
        "evening_result": (
            "Подведём итог.\n"
            "Что сделал из денежного?\n"
            "Что отложил и почему?"
        ),
        "night_truth": (
            "Скажи честно.\n"
            "Сегодня ты усилил свою позицию или остался на месте?\n"
            "Что первое делаешь завтра утром?"
        ),
        "blocker_followup": (
            f"Похоже, ты упёрся сюда: {blocker}\n"
            "Разбей это на шаг на 10 минут.\n"
            "Какое первое действие сделаешь прямо сейчас?"
        ),
        "reply_to_user": (
            "Принял.\n"
            "Зафиксировал это как текущий статус.\n"
            "Какой следующий конкретный шаг?"
        ),
        "no_reply_followup": (
            "Ты молчишь, а задача сама не сдвинется.\n"
            f"Главное сейчас: {main_task}\n"
            "Что мешает начать в ближайшие 10 минут?"
        ),
        "summary": (
            "Коротко.\n"
            f"Главная задача: {main_task}\n"
            f"Блокер: {blocker}\n"
            "Что доводишь до результата сегодня?"
        ),
    }
    return messages.get(event_type, "Что сейчас самое важное действие для денег?")


async def generate_message(cfg: UserConfig, event_type: str) -> str:
    text = await asyncio.to_thread(call_openai, cfg, event_type)
    return text or fallback_message(cfg, event_type)


# =========================
# ПЛАНИРОВЩИК
# =========================

def remove_jobs_for_chat(application: Application, chat_id: int) -> None:
    for job in application.job_queue.jobs():
        if job.name and job.name.startswith(f"daily:{chat_id}:"):
            job.schedule_removal()


def schedule_daily_jobs(application: Application, chat_id: int) -> None:
    jq = application.job_queue
    if jq is None:
        raise RuntimeError(
            "JobQueue недоступен. Убедись, что установлен python-telegram-bot[job-queue]."
        )

    remove_jobs_for_chat(application, chat_id)

    # run_daily — стандартный способ ежедневных задач через JobQueue. :contentReference[oaicite:2]{index=2}
    daily_events = [
        ("morning", time(9, 0, tzinfo=TZ)),
        ("midday_check", time(11, 30, tzinfo=TZ)),
        ("money_refocus", time(14, 30, tzinfo=TZ)),
        ("pressure_check", time(17, 30, tzinfo=TZ)),
        ("evening_result", time(20, 30, tzinfo=TZ)),
        ("night_truth", time(23, 0, tzinfo=TZ)),
    ]

    for event_type, t in daily_events:
        jq.run_daily(
            callback=send_scheduled_message,
            time=t,
            days=(0, 1, 2, 3, 4, 5, 6),
            chat_id=chat_id,
            name=f"daily:{chat_id}:{event_type}",
            data=event_type,
        )


async def send_scheduled_message(context: CallbackContext) -> None:
    job = context.job
    chat_id = job.chat_id
    if chat_id is None:
        return

    cfg = get_user_config(chat_id)

    event_type = str(job.data)
    text = await generate_message(cfg, event_type)

    cfg.last_assistant_message = text
    cfg.last_bot_message_at = now_str()
    cfg.unanswered_pings += 1

    add_recent_event(cfg, f"Бот отправил scheduled-сообщение [{event_type}]")
    save_user_config(chat_id, cfg)

    await context.bot.send_message(chat_id=chat_id, text=text)


# =========================
# КОМАНДЫ
# =========================

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    assert update.effective_chat is not None
    assert update.message is not None

    chat_id = update.effective_chat.id
    cfg = get_user_config(chat_id)

    add_recent_event(cfg, "Запуск бота командой /start")
    add_journal_entry(cfg, "Бот запущен")
    save_user_config(chat_id, cfg)

    schedule_daily_jobs(context.application, chat_id)

    await update.message.reply_text(
        "Готово. Я запущен как твой персональный ассистент.\n\n"
        "Команды:\n"
        "/focus главная | второстепенная 1 | второстепенная 2\n"
        "/done что сделал\n"
        "/blocker что мешает\n"
        "/project название проекта\n"
        "/summary\n"
        "/journal\n"
        "/status\n"
        "/help\n\n"
        "Я сам буду писать тебе в течение дня."
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    assert update.message is not None
    await update.message.reply_text(
        "/start — запустить ассистента и расписание\n"
        "/focus главная | второстепенная 1 | второстепенная 2\n"
        "/done что сделал\n"
        "/blocker что мешает\n"
        "/project название проекта\n"
        "/summary — короткий разбор\n"
        "/journal — журнал последних событий\n"
        "/status — текущий статус\n"
        "/reping — если хочешь, чтобы бот сам себя дожал сейчас\n\n"
        "Можно просто писать текстом — я отвечу с учётом контекста."
    )


async def focus_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    assert update.effective_chat is not None
    assert update.message is not None

    chat_id = update.effective_chat.id
    cfg = get_user_config(chat_id)

    raw = " ".join(context.args).strip()
    if not raw:
        await update.message.reply_text(
            "Напиши так:\n"
            "/focus главная задача | второстепенная 1 | второстепенная 2"
        )
        return

    parts = [x.strip() for x in raw.split("|") if x.strip()]
    cfg.morning_main_task = parts[0]
    cfg.secondary_tasks = parts[1:3]
    cfg.last_status = f"Зафиксирован фокус дня: {cfg.morning_main_task}"
    cfg.unanswered_pings = 0
    cfg.last_user_message_at = now_str()

    add_recent_event(cfg, f"Зафиксировал фокус дня: {cfg.morning_main_task}")
    add_journal_entry(cfg, f"Фокус дня: {cfg.morning_main_task}")
    detect_repeating_problem(cfg)
    save_user_config(chat_id, cfg)

    if cfg.secondary_tasks:
        text = (
            f"Принял.\n"
            f"Главное: {cfg.morning_main_task}\n"
            f"Второстепенные:\n" + "\n".join([f"• {x}" for x in cfg.secondary_tasks])
        )
    else:
        text = f"Принял.\nГлавное: {cfg.morning_main_task}"

    await update.message.reply_text(text)


async def done_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    assert update.effective_chat is not None
    assert update.message is not None

    chat_id = update.effective_chat.id
    cfg = get_user_config(chat_id)

    raw = " ".join(context.args).strip()
    if not raw:
        await update.message.reply_text(
            "Напиши, что именно сделал.\nПример:\n/done оформил поставку в М.Видео"
        )
        return

    cfg.last_done = raw
    cfg.current_blocker = ""
    cfg.last_status = "Есть выполненное действие"
    cfg.unanswered_pings = 0
    cfg.last_user_message_at = now_str()

    add_recent_event(cfg, f"Сделал: {raw}")
    add_journal_entry(cfg, f"Сделано: {raw}")
    detect_repeating_problem(cfg)
    save_user_config(chat_id, cfg)

    await update.message.reply_text("Зафиксировал.\nЧто следующий конкретный шаг?")


async def blocker_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    assert update.effective_chat is not None
    assert update.message is not None

    chat_id = update.effective_chat.id
    cfg = get_user_config(chat_id)

    raw = " ".join(context.args).strip()
    if not raw:
        await update.message.reply_text(
            "Напиши, что мешает.\nПример:\n/blocker не до конца понимаю требования по упаковке"
        )
        return

    cfg.current_blocker = raw
    cfg.last_status = "Есть блокер"
    cfg.unanswered_pings = 0
    cfg.last_user_message_at = now_str()

    add_recent_event(cfg, f"Блокер: {raw}")
    add_journal_entry(cfg, f"Блокер: {raw}")
    detect_repeating_problem(cfg)
    save_user_config(chat_id, cfg)

    text = await generate_message(cfg, "blocker_followup")
    cfg.last_assistant_message = text
    cfg.last_bot_message_at = now_str()
    save_user_config(chat_id, cfg)

    await update.message.reply_text(text)


async def project_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    assert update.effective_chat is not None
    assert update.message is not None

    chat_id = update.effective_chat.id
    cfg = get_user_config(chat_id)

    raw = " ".join(context.args).strip()
    if not raw:
        await update.message.reply_text(f"Текущий активный проект: {cfg.active_project}")
        return

    cfg.active_project = raw
    cfg.last_status = f"Активный проект: {raw}"
    cfg.unanswered_pings = 0
    cfg.last_user_message_at = now_str()

    add_recent_event(cfg, f"Активный проект переключён на: {raw}")
    add_journal_entry(cfg, f"Смена активного проекта: {raw}")
    detect_repeating_problem(cfg)
    save_user_config(chat_id, cfg)

    await update.message.reply_text(f"Ок. Теперь главный проект: {raw}")


async def summary_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    assert update.effective_chat is not None
    assert update.message is not None

    chat_id = update.effective_chat.id
    cfg = get_user_config(chat_id)

    custom_prompt = f"""
Ты — личный AI-ассистент Макса.
Сделай короткий разбор состояния на основе данных ниже.

{format_status(cfg)}

Журнал:
{chr(10).join(cfg.daily_journal[-12:]) if cfg.daily_journal else "нет"}

Формат:
1. Что хорошо
2. Где риск
3. Что сделать следующим шагом сегодня

Правила:
- по-русски
- максимум 8 строк
- без воды
""".strip()

    text = None
    if OPENAI_CLIENT is not None:
        try:
            response = OPENAI_CLIENT.responses.create(
                model=OPENAI_MODEL,
                input=custom_prompt,
            )
            text = getattr(response, "output_text", None)
            if text:
                text = text.strip()
        except Exception:
            logger.exception("Ошибка summary через OpenAI")

    if not text:
        text = (
            "1. Что хорошо: у тебя уже есть активный проект и зафиксированный контекст.\n"
            "2. Где риск: если главная задача не доведена, день снова расползётся.\n"
            f"3. Следующий шаг: добей {cfg.morning_main_task or cfg.active_project} конкретным действием сегодня."
        )

    await update.message.reply_text(text)


async def journal_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    assert update.effective_chat is not None
    assert update.message is not None

    cfg = get_user_config(update.effective_chat.id)
    if not cfg.daily_journal:
        await update.message.reply_text("Журнал пока пуст. Начни с /focus и /done.")
        return

    text = "Журнал последних событий:\n\n" + "\n".join(cfg.daily_journal[-15:])
    await update.message.reply_text(text[:4000])


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    assert update.effective_chat is not None
    assert update.message is not None

    cfg = get_user_config(update.effective_chat.id)
    await update.message.reply_text(format_status(cfg)[:4000])


async def reping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    assert update.effective_chat is not None
    assert update.message is not None

    chat_id = update.effective_chat.id
    cfg = get_user_config(chat_id)

    text = await generate_message(cfg, "no_reply_followup")
    cfg.last_assistant_message = text
    cfg.last_bot_message_at = now_str()
    add_recent_event(cfg, "Ручной дожим командой /reping")
    save_user_config(chat_id, cfg)

    await update.message.reply_text(text)


# =========================
# ОБЫЧНЫЕ СООБЩЕНИЯ
# =========================

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    assert update.effective_chat is not None
    assert update.message is not None

    chat_id = update.effective_chat.id
    user_text = update.message.text.strip()
    cfg = get_user_config(chat_id)

    lowered = user_text.lower()

    cfg.last_status = user_text[:250]
    cfg.last_user_message_at = now_str()
    cfg.unanswered_pings = 0

    if any(x in lowered for x in ["сделал", "готово", "закончил", "выполнил", "добил"]):
        cfg.last_done = user_text
        cfg.current_blocker = ""
        add_journal_entry(cfg, f"Пользователь сообщил о результате: {user_text[:200]}")
    elif any(x in lowered for x in ["не могу", "мешает", "непонятно", "не понимаю", "застрял", "стопор"]):
        cfg.current_blocker = user_text
        add_journal_entry(cfg, f"Пользователь сообщил о блокере: {user_text[:200]}")
    else:
        add_journal_entry(cfg, f"Сообщение пользователя: {user_text[:200]}")

    add_recent_event(cfg, f"Сообщение пользователя: {user_text[:180]}")
    detect_repeating_problem(cfg)
    save_user_config(chat_id, cfg)

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    event_type = "reply_to_user"
    if cfg.repeating_problem and cfg.unanswered_pings >= 2:
        event_type = "no_reply_followup"
    elif cfg.current_blocker:
        event_type = "blocker_followup"

    text = await generate_message(cfg, event_type)
    cfg.last_assistant_message = text
    cfg.last_bot_message_at = now_str()
    save_user_config(chat_id, cfg)

    await update.message.reply_text(text)


# =========================
# MAIN
# =========================

def main() -> None:
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("focus", focus_cmd))
    app.add_handler(CommandHandler("done", done_cmd))
    app.add_handler(CommandHandler("blocker", blocker_cmd))
    app.add_handler(CommandHandler("project", project_cmd))
    app.add_handler(CommandHandler("summary", summary_cmd))
    app.add_handler(CommandHandler("journal", journal_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("reping", reping_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Bot is running...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
