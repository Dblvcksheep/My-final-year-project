# AlloyAI — Aluminium Alloy Property Prediction Telegram Bot

Predicts **Ultimate Tensile Strength (UTS)** and **Yield Strength (YS)** for wrought aluminium alloys (2xxx / 5xxx / 6xxx / 7xxx series) from composition, series, and temper — delivered as a Telegram chatbot.

Built as part of a final-year research project: *Evaluation of Artificial Intelligence Tools for Predicting Mechanical Properties of Aluminium Alloys and Development of a Chatbot-Based Material Selection Tool for Structural Applications.*

🔗 **Live bot:** [t.me/AlloyAl_bot](https://t.me/AlloyAl_bot)

---

## What this is

A trained machine learning pipeline — not a general-purpose AI guessing from general knowledge — wrapped in a guided, button-driven Telegram interface. Every prediction is traceable to a specific, validated model and comes with an explicit confidence assessment rather than a bare number.

- **UTS** → Random Forest (held-out test R² = 0.749)
- **YS** → Artificial Neural Network (held-out test R² = 0.818)
- **Elongation** → not predicted. All four algorithms tested scored a negative test R² — none beat simply guessing the average. Excluded deliberately rather than offering an unreliable number.
- **Hardness** → out of scope. No sufficiently large, consistently matched hardness dataset could be compiled for this alloy scope.

## Why composition is user-entered, not just looked up

The model's actual inputs are 14 element weight percentages + alloy series + temper group — nothing else (not alloy name, not source paper). Checking this project's own train/test split directly showed that 15 of 16 unique compositions in the UTS test set already existed in training, meaning the reported R² largely reflects interpolation on known alloys, not validated extrapolation to new ones. Rather than quietly restrict the bot to known alloys only, the chatbot lets users enter any composition and is transparent about how trustworthy each specific prediction is.

### Confidence system

Every prediction carries a **High / Medium / Low** confidence label built from four independent checks:

| Check | What it catches |
|---|---|
| Element-range check | Entered values outside anything seen in the 33 known alloy compositions |
| Nearest-known-alloy distance | How far (Euclidean, in wt%) the entered composition is from the closest known alloy |
| Random Forest tree-spread | Disagreement across the forest's individual trees — a free, legitimate uncertainty signal |
| Series+temper coverage | Whether that exact series/temper combination has *any* training data at all, for *any* alloy |

Lower confidence automatically widens the reported prediction range rather than hiding the uncertainty.

## Data noise ceiling

Before trusting any model's accuracy ceiling, this project directly measured how much real-world variation exists even among records that are numerically identical to the model (same composition, series, temper group). Example: 30 literature records sharing one identical 7xxx/T6 input still spread **450–659 MPa** in reported UTS — a 209 MPa range no model can predict below, because the true answer for that input is a distribution, not a single number. This sets a measured, honest ceiling on achievable R², rather than treating "small dataset" as a vague catch-all excuse.

---

## Architecture

```
Raw literature corpus (Pfeiffer et al.)
        │  filter to 2xxx/5xxx/6xxx/7xxx, match composition↔property
        ▼
Five-pass data validation  →  742 → 386 validated records, 33 alloys
        │
        ▼
Feature engineering (14 composition cols + one-hot series + one-hot temper group)
        │  DOI-grouped train/test split per target (no paper split across both)
        ▼
12 models trained (RF / SVR / XGBoost / ANN × UTS / YS / Elongation)
        │  RandomizedSearchCV + GroupKFold CV, evaluated once on held-out test
        ▼
2 winning models refit on full data, serialised (joblib / Keras)
        │
        ▼
Telegram bot (python-telegram-bot) — guided button flow + per-element composition entry
```

## Tech stack

- **Modelling:** scikit-learn (Random Forest, SVR), XGBoost, TensorFlow/Keras (ANN)
- **Data:** pandas, NumPy
- **Bot:** python-telegram-bot (async, v21), SQLAlchemy, SQLite
- **Serialization:** joblib (sklearn), native Keras format (`.keras`)

## Repository structure

```
.
├── bot.py                          # Telegram bot — conversation flow, handlers
├── Ai.py                           # Model loading, prediction, confidence logic
├── build_deployment_artifacts.py   # Refits & serialises the 2 selected models
├── requirements.txt
├── .env.example                    # Copy to .env and fill in TOKEN + DATABASE
├── models/
│   ├── uts_rf_model.joblib
│   ├── uts_scaler.joblib
│   ├── ys_ann_model.keras
│   └── ys_scaler.joblib
└── data/
    ├── alloy_composition_lookup.csv       # 14-element composition per known alloy
    ├── alloy_temper_lookup.json           # which tempers have data per alloy
    ├── alloy_temper_property_ranges.csv   # historical min/max per alloy+temper
    └── series_temper_coverage.json        # row counts per series+temper combo
```

## Setup

```bash
git clone <this-repo>
cd <this-repo>
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt

cp .env.example .env
# edit .env: set TOKEN (from @BotFather) and DATABASE (sqlite:///bot.db works out of the box)

python bot.py
```

> **Note:** `tensorflow` is a heavy dependency (~500–600MB install). If deploying to a constrained host, consider swapping in `tensorflow-cpu` in `requirements.txt`.

## Usage

| Command | What it does |
|---|---|
| `/start` | Introduction |
| `/predict` | Start a guided prediction: series → alloy (as a composition starting point) → 14 elements, one at a time (type a number or "same" to keep the default) → temper → prediction |
| `/why` | Model performance figures and known limitations |
| `/help` | List commands |
| `/cancel` | Exit an in-progress `/predict` conversation |

## Full model comparison

| Target | Model | Test R² | RMSE | MAE | Selected |
|---|---|---|---|---|---|
| UTS | Random Forest | **0.749** | 65.3 | 44.5 | ✅ |
| UTS | XGBoost | 0.730 | 67.7 | 46.1 | |
| UTS | ANN | 0.659 | 76.1 | 57.7 | |
| UTS | SVR | 0.600 | 82.4 | 59.3 | |
| YS | ANN | **0.818** | 68.5 | 57.1 | ✅ |
| YS | XGBoost | 0.794 | 72.9 | 57.0 | |
| YS | SVR | 0.758 | 79.0 | 60.8 | |
| YS | Random Forest | 0.753 | 79.8 | 65.9 | |
| Elongation | ANN | -0.129 | 5.3 | 4.5 | ❌ none |
| Elongation | XGBoost | -0.195 | 5.4 | 4.7 | ❌ none |
| Elongation | Random Forest | -0.234 | 5.5 | 4.7 | ❌ none |
| Elongation | SVR | -0.252 | 5.6 | 4.7 | ❌ none |

## Known limitations

- Deployed models are refits using Phase 4's validated hyperparameters, not the literal original training-run objects (which were never serialised).
- Reported R² reflects interpolation on known alloy compositions, not demonstrated extrapolation accuracy on genuinely novel compositions.
- Several alloy/temper combinations are backed by only 1–3 literature records; the bot flags these as low-confidence rather than blocking them.
- Confidence labels (High/Medium/Low) are a hand-tuned heuristic, not a formally calibrated uncertainty estimate (e.g. conformal prediction).
- No authentication, rate-limiting, or abuse protection — suitable for an academic demo, not public production use.

## Future work

- Hold out entire alloys (not just source papers) to produce a genuine extrapolation-accuracy estimate.
- Replace heuristic confidence with conformal prediction intervals.
- Add hardness prediction once a suitably large, matched dataset is available (candidates: Chaudry, Hamad & Abuhmed, 2021; NIMS Kinzoku database).
- Revisit elongation prediction with a dataset including grain size and processing route.
- Add an automated `pytest` suite covering the conversation state machine and prediction logic.

## Data source

Composition and property data derived from the aluminium alloy literature corpus compiled by Pfeiffer et al. (Materials Cloud), filtered and validated as described above.

## License

*(add your chosen license here — e.g. MIT)*
