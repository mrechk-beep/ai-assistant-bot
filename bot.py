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
# BASE SETUP
# =========================

load_dotenv()

logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("max-assistant-bot-v2")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
TIMEZONE_NAME = os.getenv("TIMEZONE", "Europe/Moscow")

if not BOT_TOKEN:
    raise RuntimeError("В .env или Environment Variables не задан TELEGRAM_BOT_TOKEN")

TZ = ZoneInfo(TIMEZONE_NAME)
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = DATA_DIR / "state.json"

OPENAI_CLIENT: Optional[OpenAI] = None
if OPENAI_API_KEY:
    OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# USER STATE
# =========================

@dataclass
class UserConfig:
    # Profile / style
    user_name: str = "Макс"
    mode: str = "medium"  # soft | medium | hard
    tone: str = "человеческий, уверенный, средне настойчивый"
    wake_time: str = "09:00"
    end_time: str = "24:00"

    # Priorities / business context
    main_priority: str = "деньги"
    priorities: List[str] = None
    current_projects: List[str] = None
    active_project: str = "М.Видео"
    weak_points: List[str] = None

    # Day state
    morning_main_task: str = ""
    secondary_tasks: List[str] = None
    last_done: str = ""
    current_blocker: str = ""
    blocker_type: str = ""  # fear | clarity | energy | avoidance | unknown
    last_status: str = ""
    money_goal_today: str = ""
    money_result_today: str = ""

    # Memory
    repeating_problem: str = ""
    recent_events: List[str] = None
    daily_journal: List[str] = None

    # Tracking
    unanswered_pings: int = 0
    last_user_message_at: str = ""
    last_bot_message_at: str = ""
    last_assistant_message: str = ""

    def __post_init__(self):
        if self.priorities is None:
            self.priorities = ["деньги", "дисциплина", "тело"]
        if self.current_projects is None:
            self.current_projects = ["М.Видео", "WB", "режим/энергия"]
        if self.weak_points is None:
            self.weak_points = [
                "откладывает сложные шаги",
                "может зависать, если неясен следующий шаг",
                "уходит в прокрастинацию",
            ]
        if self.secondary_tasks is None:
            self.secondary_tasks = []
        if self.recent_events is None:
            self.recent_events = []
        if self.daily_journal is None:
            self.daily_journal = []


# =========================
# STORAGE
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
# HELPERS
# =========================

def now_dt() -> datetime:
    return datetime.now(TZ)


def now_str() -> str:
    return now_dt().strftime("%d.%m.%Y %H:%M")


def stamp() -> str:
    return now_dt().strftime("%d.%m %H:%M")


def add_recent_event(cfg: UserConfig, text: str, limit: int = 25) -> None:
    text = text.strip()
    if not text:
        return
    cfg.recent_events.append(f"{stamp()} — {text}")
    cfg.recent_events = cfg.recent_events[-limit:]


def add_journal_entry(cfg: UserConfig, text: str, limit: int = 60) -> None:
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
    return ", ".join(cfg.secondary_tasks) if cfg.secondary_tasks else "не указаны"


def mode_description(mode: str) -> str:
    mapping = {
        "soft": "мягкий, поддерживающий, но всё равно приводящий к действию",
        "medium": "ровный, деловой, уверенный, средне настойчивый",
        "hard": "жёсткий, прямой, без лишней поддержки, с акцентом на конкретный результат",
    }
    return mapping.get(mode, mapping["medium"])


def classify_blocker(text: str) -> str:
    t = text.lower()

    if any(x in t for x in ["боюсь", "страшно", "опасаюсь", "стремно"]):
        return "fear"
    if any(x in t for x in ["не понимаю", "непонятно", "неясно", "как", "требования"]):
        return "clarity"
    if any(x in t for x in ["устал", "нет сил", "выгорел", "энергии нет", "сонный"]):
        return "energy"
    if any(x in t for x in ["откладываю", "потом", "не хочу", "сливаюсь", "прокраст", "залип"]):
        return "avoidance"
    return "unknown"


def blocker_type_text(blocker_type: str) -> str:
    mapping = {
        "fear": "страх ошибки или неприятного исхода",
        "clarity": "неясный следующий шаг",
        "energy": "просадка по энергии",
        "avoidance": "избегание и прокрастинация",
        "unknown": "тип блокера не определён",
        "": "не указан",
    }
    return mapping.get(blocker_type, "не указан")


def detect_repeating_problem(cfg: UserConfig) -> None:
    joined = " | ".join(cfg.recent_events[-12:]).lower()

    if "упаков" in joined:
        cfg.repeating_problem = "Ты уже несколько раз упирался в упаковку"
    elif "мвидео" in joined and any(x in joined for x in ["не сделал", "не начал", "отлож", "потом"]):
        cfg.repeating_problem = "Ты тянешь важные действия по М.Видео"
    elif "wb" in joined and any(x in joined for x in ["не сделал", "не начал", "отлож", "потом"]):
        cfg.repeating_problem = "Ты тянешь важные действия по WB"
    elif any(x in joined for x in ["прокраст", "залип", "телефон", "отвлек", "слив"]):
        cfg.repeating_problem = "Ты снова провалился в прокрастинацию"
    elif any(x in joined for x in ["не понимаю", "непонятно", "неясно", "как сделать"]):
        cfg.repeating_problem = "Ты тормозишь на неясном следующем шаге"
    elif any(x in joined for x in ["не сделал", "не начал", "отлож", "потом"]):
        cfg.repeating_problem = "Ты повторно откладываешь важное действие"
    else:
        cfg.repeating_problem = ""


def format_status(cfg: UserConfig) -> str:
    projects = "\n".join([f"• {x}" for x in cfg.current_projects]) if cfg.current_projects else "—"
    weak_points = "\n".join([f"• {x}" for x in cfg.weak_points]) if cfg.weak_points else "—"
    secondary = "\n".join([f"• {x}" for x in cfg.secondary_tasks]) if cfg.secondary_tasks else "—"

    return (
        f"Имя: {cfg.user_name}\n"
        f"Режим: {cfg.mode}\n"
        f"Приоритет: {cfg.main_priority}\n"
        f"Зоны: {', '.join(cfg.priorities)}\n"
        f"Активный проект: {cfg.active_project}\n"
        f"Проекты:\n{projects}\n\n"
        f"Слабые места:\n{weak_points}\n\n"
        f"Главная задача: {cfg.morning_main_task or '—'}\n"
        f"Второстепенные:\n{secondary}\n\n"
        f"Сделано: {cfg.last_done or '—'}\n"
        f"Блокер: {cfg.current_blocker or '—'}\n"
        f"Тип блокера: {blocker_type_text(cfg.blocker_type)}\n"
        f"Денежная цель дня: {cfg.money_goal_today or '—'}\n"
        f"Денежный результат дня: {cfg.money_result_today or '—'}\n"
        f"Последний статус: {cfg.last_status or '—'}\n"
        f"Повторяющаяся проблема: {cfg.repeating_problem or '—'}\n"
        f"Неотвеченных пингов: {cfg.unanswered_pings}"
    )


def reset_day_fields(cfg: UserConfig) -> None:
    cfg.morning_main_task = ""
    cfg.secondary_tasks = []
    cfg.last_done = ""
    cfg.current_blocker = ""
    cfg.blocker_type = ""
    cfg.last_status = ""
    cfg.money_goal_today = ""
    cfg.money_result_today = ""
    cfg.unanswered_pings = 0


# =========================
# PROMPT / AI
# =========================

def build_prompt(cfg: UserConfig, event_type: str) -> str:
    projects = ", ".join(cfg.current_projects) if cfg.current_projects else "не указаны"
    weak_points = ", ".join(cfg.weak_points) if cfg.weak_points else "не указаны"

    return f"""
Ты — персональный AI-ассистент Макса.

Кто такой Макс:
- предприниматель
- главный приоритет сейчас: деньги
- важные зоны: {", ".join(cfg.priorities)}
- текущие проекты: {projects}
- активный проект прямо сейчас: {cfg.active_project}
- слабые места: {weak_points}

Текущий режим общения: {cfg.mode}
Описание режима: {mode_description(cfg.mode)}

Текущее состояние:
- главная задача на день: {cfg.morning_main_task or "не зафиксирована"}
- второстепенные задачи: {get_secondary_tasks_text(cfg)}
- последнее выполненное действие: {cfg.last_done or "нет"}
- текущий блокер: {cfg.current_blocker or "не указан"}
- тип блокера: {blocker_type_text(cfg.blocker_type)}
- денежная цель дня: {cfg.money_goal_today or "не указана"}
- денежный результат дня: {cfg.money_result_today or "не указан"}
- последний статус пользователя: {cfg.last_status or "нет"}
- повторяющаяся проблема: {cfg.repeating_problem or "не выявлена"}
- неотвеченных пингов подряд: {cfg.unanswered_pings}

Последние события:
{get_recent_context(cfg)}

Сейчас: {now_str()}
Тип сообщения: {event_type}

Правила:
- отвечай по-русски
- 2–6 коротких строк
- без воды
- без эмодзи
- не уходи в общие советы
- опирайся на конкретный контекст
- если это блокер ясности, разбивай на маленький следующий шаг
- если это страх, снижай масштаб задачи
- если это избегание, называй это прямо
- если режим hard, будь более жёстким и требуй конкретику
- всегда заканчивай конкретным вопросом или действием
""".strip()


def call_openai(cfg: UserConfig, event_type: str) -> Optional[str]:
    if OPENAI_CLIENT is None:
        return None

    try:
        response = OPENAI_CLIENT.responses.create(
            model=OPENAI_MODEL,
            input=build_prompt(cfg, event_type),
        )
        text = getattr(response, "output_text", None)
        return text.strip() if text else None
    except Exception:
        logger.exception("Ошибка вызова OpenAI")
        return None


def fallback_message(cfg: UserConfig, event_type: str) -> str:
    main_task = cfg.morning_main_task or "главная задача ещё не зафиксирована"
    blocker = cfg.current_blocker or "блокер пока не назван"

    if event_type == "blocker_followup":
        if cfg.blocker_type == "clarity":
            return (
                f"У тебя не тупик, а неясный шаг: {blocker}\n"
                "Не решай всё сразу.\n"
                "Выпиши 3 самых непонятных пункта и пришли их сюда."
            )
        if cfg.blocker_type == "fear":
            return (
                f"Похоже, тебя тормозит страх: {blocker}\n"
                "Уменьши масштаб до безопасного шага.\n"
                "Какое действие можно сделать без риска за 10 минут?"
            )
        if cfg.blocker_type == "avoidance":
            return (
                f"Это больше похоже на избегание: {blocker}\n"
                "Хватит крутить задачу в голове.\n"
                "Какой один конкретный шаг сделаешь сейчас?"
            )
        if cfg.blocker_type == "energy":
            return (
                f"Похоже, просела энергия: {blocker}\n"
                "Не ломай себя об большую задачу.\n"
                "Какой короткий шаг всё равно можно закрыть сейчас?"
            )
        return (
            f"Похоже, тебя держит блокер: {blocker}\n"
            "Разбиваем на маленький шаг.\n"
            "Какое действие можно сделать за 10 минут прямо сейчас?"
        )

    mapping = {
        "morning": (
            "Доброе утро.\n"
            "Что сегодня реально двинет деньги?\n"
            f"Главное на сейчас: {main_task}\n"
            "Что делаешь первым?"
        ),
        "midday_check": (
            f"Ты уже начал главное действие?\n"
            f"Текущий фокус: {main_task}\n"
            "Если нет — что именно мешает?"
        ),
        "money_refocus": (
            "Проверь фокус.\n"
            "То, что ты делаешь сейчас, двигает деньги или это обходной манёвр?\n"
            "Ответь коротко."
        ),
        "pressure_check": (
            "Стоп.\n"
            "Ты продвигаешь главное или снова растёкся по мелочам?\n"
            "Назови один следующий денежный шаг."
        ),
        "evening_result": (
            "Подведём итог.\n"
            "Что реально продвинул по деньгам?\n"
            "Что отложил и почему?"
        ),
        "night_truth": (
            "Скажи честно.\n"
            "Сегодня ты усилил позицию или остался на месте?\n"
            "Что первое делаешь завтра утром?"
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
    }
    return mapping.get(event_type, "Что сейчас самое важное действие для денег?")


async def generate_message(cfg: UserConfig, event_type: str) -> str:
    text = await asyncio.to_thread(call_openai, cfg, event_type)
    return text or fallback_message(cfg, event_type)


# =========================
# SCHEDULE
# =========================

def remove_jobs_for_chat(application: Application, chat_id: int) -> None:
    for job in application.job_queue.jobs():
        if job.name and job.name.startswith(f"daily:{chat_id}:"):
            job.schedule_removal()


def schedule_daily_jobs(application: Application, chat_id: int) -> None:
    jq = application.job_queue
    if jq is None:
        raise RuntimeError("JobQueue недоступен. Убедись, что установлен python-telegram-bot[job-queue].")

    remove_jobs_for_chat(application, chat_id)

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
# COMMANDS
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
        "/mode soft|medium|hard\n"
        "/summary\n"
        "/journal\n"
        "/resetday\n"
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
        "/mode soft|medium|hard\n"
        "/summary — короткий разбор\n"
        "/journal — журнал последних событий\n"
        "/resetday — сбросить дневной фокус без потери общей памяти\n"
        "/status — текущий статус\n"
        "/reping — жёсткий дожим прямо сейчас\n\n"
        "Можно просто писать текстом — я отвечу с учётом контекста."
    )


async def focus_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    assert update.effective_chat is not None
    assert update.message is not None

    chat_id = update.effective_chat.id
    cfg = get_user_config(chat_id)

    raw = " ".join(context.args).strip()
    if not raw:
        await update.message.reply_text("Напиши так:\n/focus главная задача | второстепенная 1 | второстепенная 2")
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
            f"Зафиксировал.\n"
            f"Главное: {cfg.morning_main_task}\n"
            "Второстепенные:\n" + "\n".join([f"• {x}" for x in cfg.secondary_tasks])
        )
    else:
        text = f"Зафиксировал.\nГлавное: {cfg.morning_main_task}"

    await update.message.reply_text(text)


async def done_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    assert update.effective_chat is not None
    assert update.message is not None

    chat_id = update.effective_chat.id
    cfg = get_user_config(chat_id)

    raw = " ".join(context.args).strip()
    if not raw:
        await update.message.reply_text("Напиши, что именно сделал.\nПример:\n/done оформил поставку в М.Видео")
        return

    cfg.last_done = raw
    cfg.current_blocker = ""
    cfg.blocker_type = ""
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
    cfg.blocker_type = classify_blocker(raw)
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


async def mode_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    assert update.effective_chat is not None
    assert update.message is not None

    chat_id = update.effective_chat.id
    cfg = get_user_config(chat_id)

    raw = " ".join(context.args).strip().lower()
    if not raw:
        await update.message.reply_text(f"Текущий режим: {cfg.mode}\nИспользуй: /mode soft|medium|hard")
        return

    if raw not in {"soft", "medium", "hard"}:
        await update.message.reply_text("Нужен один из режимов: soft, medium, hard")
        return

    cfg.mode = raw
    cfg.last_status = f"Режим переключён на {raw}"
    add_recent_event(cfg, f"Режим общения переключён на: {raw}")
    add_journal_entry(cfg, f"Смена режима: {raw}")
    save_user_config(chat_id, cfg)

    await update.message.reply_text(f"Ок. Теперь режим: {raw}")


async def summary_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    assert update.effective_chat is not None
    assert update.message is not None

    chat_id = update.effective_chat.id
    cfg = get_user_config(chat_id)

    custom_prompt = f"""
Ты — личный AI-ассистент Макса.
Сделай короткий разбор текущего состояния.

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
- конкретно
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
            f"1. Что хорошо: у тебя уже зафиксирован контекст по проекту {cfg.active_project}.\n"
            f"2. Где риск: если не добить {cfg.morning_main_task or 'главную задачу'}, день снова расползётся.\n"
            "3. Следующий шаг: закрой один конкретный результат сегодня."
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


async def resetday_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    assert update.effective_chat is not None
    assert update.message is not None

    chat_id = update.effective_chat.id
    cfg = get_user_config(chat_id)

    reset_day_fields(cfg)
    add_recent_event(cfg, "Выполнен resetday — дневной фокус очищен")
    add_journal_entry(cfg, "Сброс дневного фокуса")
    save_user_config(chat_id, cfg)

    await update.message.reply_text("Ок. Дневной фокус очищен, общая память сохранена.")


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
# REGULAR TEXT
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

    if any(x in lowered for x in ["сделал", "готово", "закончил", "выполнил", "добил", "отправил", "закрыл"]):
        cfg.last_done = user_text
        cfg.current_blocker = ""
        cfg.blocker_type = ""
        add_journal_entry(cfg, f"Пользователь сообщил о результате: {user_text[:200]}")
    elif any(x in lowered for x in ["не могу", "мешает", "непонятно", "не понимаю", "застрял", "стопор", "боюсь"]):
        cfg.current_blocker = user_text
        cfg.blocker_type = classify_blocker(user_text)
        add_journal_entry(cfg, f"Пользователь сообщил о блокере: {user_text[:200]}")
    elif any(x in lowered for x in ["деньги", "выручка", "доход", "продажи"]):
        cfg.money_result_today = user_text[:200]
        add_journal_entry(cfg, f"Пользователь сообщил денежный статус: {user_text[:200]}")
    else:
        add_journal_entry(cfg, f"Сообщение пользователя: {user_text[:200]}")

    add_recent_event(cfg, f"Сообщение пользователя: {user_text[:180]}")
    detect_repeating_problem(cfg)
    save_user_config(chat_id, cfg)

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    event_type = "reply_to_user"
    if cfg.current_blocker:
        event_type = "blocker_followup"
    elif cfg.repeating_problem and cfg.mode == "hard":
        event_type = "no_reply_followup"

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
    app.add_handler(CommandHandler("mode", mode_cmd))
    app.add_handler(CommandHandler("summary", summary_cmd))
    app.add_handler(CommandHandler("journal", journal_cmd))
    app.add_handler(CommandHandler("resetday", resetday_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("reping", reping_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Bot is running...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
