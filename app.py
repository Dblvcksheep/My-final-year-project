import os
from datetime import datetime
from typing import Final

from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, BigInteger, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, MessageHandler,
    ConversationHandler, ContextTypes, filters,
)

import Ai

load_dotenv()

# ---------------------------------------------------------------------------
# Database -- kept minimal: just a usage log, no watch/price tracking
# ---------------------------------------------------------------------------
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    telegram_id = Column(BigInteger, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Query(Base):
    """Optional usage log: one row per prediction made, for your Chapter 4
    usability-testing section (e.g. 'N queries across M distinct alloys')."""
    __tablename__ = "queries"

    id = Column(Integer, primary_key=True)
    telegram_id = Column(BigInteger)
    series = Column(String)
    alloy = Column(String)
    temper_group = Column(String)
    predicted_uts = Column(String)
    predicted_ys = Column(String)
    created_at = Column(DateTime, default=datetime.now)


engine = create_engine(os.environ["DATABASE"], echo=False)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

TOKEN: Final = os.environ["TOKEN"]

# ---------------------------------------------------------------------------
# Callback data encoding: "step:value1:value2..."
# ---------------------------------------------------------------------------
def _kb(buttons, row_width=3):
    """Lay out a flat list of (label, callback_data) into rows."""
    rows = [buttons[i:i + row_width] for i in range(0, len(buttons), row_width)]
    return InlineKeyboardMarkup([[InlineKeyboardButton(l, callback_data=d) for l, d in row] for row in rows])


def _ensure_user(session, telegram_id):
    user = session.query(User).filter_by(telegram_id=telegram_id).first()
    if not user:
        user = User(telegram_id=telegram_id, created_at=datetime.utcnow())
        session.add(user)
        session.commit()
    return user


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    session = Session()
    _ensure_user(session, update.message.chat.id)
    session.close()

    await update.message.reply_text(
        "Hello! I predict Ultimate Tensile Strength (UTS) and Yield Strength (YS) "
        "for wrought aluminium alloys, based on a dataset of 386 literature-sourced "
        "records across the 2xxx/5xxx/6xxx/7xxx series.\n\n"
        "/predict to get started\n"
        "/why to see the model performance and known limitations\n"
        "/help for all commands"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/predict \u2014 start a guided prediction (series \u2192 alloy \u2192 temper)\n"
        "/why \u2014 model performance, and why elongation/hardness aren't predicted\n"
        "/help \u2014 this message"
    )


async def why_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "UTS is predicted with a Random Forest (held-out test R\u00b2 = 0.749, "
        "typical error \u00b144.5 MPa).\n"
        "YS is predicted with a Neural Network (held-out test R\u00b2 = 0.818, "
        "typical error \u00b157.1 MPa).\n\n"
        f"{Ai.ELONGATION_DISCLAIMER}\n\n"
        f"{Ai.HARDNESS_DISCLAIMER}\n\n"
        "Predictions use only composition, alloy series, and temper group \u2014 "
        "the same alloy/temper combination always gives the same prediction, "
        "by design. Real literature values for a given alloy/temper can still "
        "vary (shown alongside each prediction when available) because of "
        "processing and testing differences not captured by these three fields."
    )
    await update.message.reply_text(msg)


async def predict_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    buttons = [(s, f"series:{s}") for s in Ai.get_series_list()]
    await update.message.reply_text(
        "Which alloy series?",
        reply_markup=_kb(buttons, row_width=4),
    )
    return SELECT_SERIES


# ---------------------------------------------------------------------------
# Button callback handler -- drives series -> alloy -> per-element composition -> temper
# ---------------------------------------------------------------------------
async def on_series_selected(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    series = query.data.split(":")[1]
    context.user_data["series"] = series

    alloys = Ai.get_alloys_for_series(series)
    buttons = [(a, f"alloy:{a}") for a in alloys]
    await query.edit_message_text(
        f"Series {series} \u2014 pick the alloy closest to what you have "
        f"in mind. Its composition will be shown as a starting point for "
        f"each element, which you can keep or change.",
        reply_markup=_kb(buttons, row_width=4),
    )
    return SELECT_ALLOY


async def on_alloy_selected(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    alloy = query.data.split(":")[1]
    context.user_data["alloy"] = alloy
    context.user_data["composition"] = {}
    context.user_data["element_index"] = 0

    nominal = Ai.get_nominal_composition(alloy)
    context.user_data["nominal"] = nominal

    await query.edit_message_text(
        f"Using alloy {alloy} as the starting point. I'll ask for each "
        f"element's weight % one at a time -- send a number, or send "
        f"\"same\" to keep the shown default."
    )
    await _ask_next_element(query.message, context)
    return ASK_ELEMENT


async def _ask_next_element(message, context: ContextTypes.DEFAULT_TYPE):
    idx = context.user_data["element_index"]
    element = Ai.COMP_COLS[idx]
    default = context.user_data["nominal"][element]
    lo, hi = Ai.get_element_range(element)
    await message.reply_text(
        f"[{idx + 1}/14] {element} weight % \u2014 default (from {context.user_data['alloy']}): "
        f"{default}%  (typical range across known alloys: {lo}\u2013{hi}%)\n"
        f"Send a number, or \"same\" to accept the default."
    )


async def on_element_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    idx = context.user_data["element_index"]
    element = Ai.COMP_COLS[idx]
    text = update.message.text.strip().lower()

    if text in ("same", "s", ""):
        value = context.user_data["nominal"][element]
    else:
        try:
            value = float(text)
        except ValueError:
            await update.message.reply_text("Please send a number (e.g. 2.4), or \"same\".")
            return ASK_ELEMENT

    context.user_data["composition"][element] = value
    context.user_data["element_index"] += 1

    if context.user_data["element_index"] < len(Ai.COMP_COLS):
        await _ask_next_element(update.message, context)
        return ASK_ELEMENT

    # All 14 collected. Offer ALL temper groups here, not just the anchor
    # alloy's -- the user may have changed the composition to something
    # unlike that alloy, so restricting temper choices to the anchor's
    # history would be an arbitrary leftover constraint. Zero-coverage
    # series+temper combinations are still flagged (not blocked) in the
    # confidence report after prediction.
    buttons = [(Ai.temper_label(t), f"temper:{t}") for t in Ai.TEMPER_GROUPS]
    await update.message.reply_text(
        "Got all 14 elements. Which temper?",
        reply_markup=_kb(buttons, row_width=1),
    )
    return SELECT_TEMPER


async def on_temper_selected(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    temper_group = query.data.split(":")[1]

    series = context.user_data["series"]
    composition = context.user_data["composition"]

    result = Ai.predict_from_composition(series, temper_group, composition)
    message = Ai.format_custom_prediction_message(result)
    await query.edit_message_text(message)

    session = Session()
    _ensure_user(session, query.from_user.id)
    session.add(Query(
        telegram_id=query.from_user.id, series=series,
        alloy=context.user_data.get("alloy", "custom"),
        temper_group=temper_group,
        predicted_uts=str(result["uts"]["predicted_mpa"]),
        predicted_ys=str(result["ys"]["predicted_mpa"]),
        created_at=datetime.utcnow(),
    ))
    session.commit()
    session.close()

    context.user_data.clear()
    return ConversationHandler.END


async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text("Cancelled. Send /predict to start again.")
    return ConversationHandler.END


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    print(f"Update {update} caused error {context.error}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
SELECT_SERIES, SELECT_ALLOY, ASK_ELEMENT, SELECT_TEMPER = range(4)

if __name__ == "__main__":
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("why", why_command))

    predict_conv = ConversationHandler(
        entry_points=[CommandHandler("predict", predict_command)],
        states={
            SELECT_SERIES: [CallbackQueryHandler(on_series_selected, pattern=r"^series:")],
            SELECT_ALLOY: [CallbackQueryHandler(on_alloy_selected, pattern=r"^alloy:")],
            ASK_ELEMENT: [MessageHandler(filters.TEXT & ~filters.COMMAND, on_element_reply)],
            SELECT_TEMPER: [CallbackQueryHandler(on_temper_selected, pattern=r"^temper:")],
        },
        fallbacks=[CommandHandler("cancel", cancel_command)],
    )
    app.add_handler(predict_conv)

    app.add_error_handler(error_handler)

    print("Polling...")
    app.run_polling(poll_interval=2)