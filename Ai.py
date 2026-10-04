import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import joblib
import pandas as pd
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

COMP_COLS = ["Al", "Cu", "Mg", "Si", "Zn", "Mn", "Fe", "Cr", "Ti", "Ag", "Li", "Ni", "V", "Zr"]
SERIES_LIST = ["2xxx", "5xxx", "6xxx", "7xxx"]
TEMPER_GROUPS = ["H", "O", "Other", "T3_T4", "T6", "T7x", "T8x", "T_other", "Unknown", "W"]

# Human-readable label for temper groups, shown on buttons / in replies
TEMPER_LABELS = {
    "H": "H (strain hardened)",
    "O": "O (annealed / soft)",
    "T3_T4": "T3/T4 (solution heat treated + naturally aged)",
    "T6": "T6 (solution heat treated + artificially peak-aged)",
    "T7x": "T7x (solution heat treated + overaged)",
    "T8x": "T8x (solution heat treated, cold worked, artificially aged)",
    "T_other": "Other T-temper",
    "W": "W (solution heat treated, unstable/as-quenched)",
    "Other": "Other / special processing",
    "Unknown": "Unspecified temper",
}

# --- Test-set performance of each deployed model (Phase 4 held-out results) ---
# Shown alongside every prediction so a number is never presented as more
# certain than the evaluation actually supports.
MODEL_PERFORMANCE = {
    "UTS": {"model": "Random Forest", "test_r2": 0.749, "rmse": 65.3, "mae": 44.5},
    "YS": {"model": "Artificial Neural Network", "test_r2": 0.818, "rmse": 68.5, "mae": 57.1},
}

# Elongation and hardness were evaluated / investigated and explicitly excluded.
# See Phase1-4_Technical_Report.docx and Defense_Prep_Notes.docx for the reasoning.
ELONGATION_DISCLAIMER = (
    "Elongation prediction is not offered by this bot. All four algorithms "
    "tested (Random Forest, XGBoost, SVR, ANN) scored a negative R\u00b2 on "
    "held-out data \u2014 meaning none of them predicted better than simply "
    "guessing the average. Composition and temper alone do not appear "
    "sufficient to predict elongation for this dataset; grain size and "
    "processing route, which are not available here, likely matter more."
)
HARDNESS_DISCLAIMER = (
    "Hardness prediction is not included in this project. No sufficiently "
    "large, consistent hardness dataset could be matched to this project's "
    "alloy/temper scope \u2014 see the project report for details and "
    "suggested future sources (Chaudry et al. 2021; NIMS Kinzoku database)."
)

# ---------------------------------------------------------------------------
# Load artifacts once, at import time
# ---------------------------------------------------------------------------
_rf_uts = joblib.load(os.path.join(MODELS_DIR, "uts_rf_model.joblib"))
_scaler_uts = joblib.load(os.path.join(MODELS_DIR, "uts_scaler.joblib"))
_ann_ys = keras.models.load_model(os.path.join(MODELS_DIR, "ys_ann_model.keras"))
_scaler_ys = joblib.load(os.path.join(MODELS_DIR, "ys_scaler.joblib"))

_composition = pd.read_csv(os.path.join(DATA_DIR, "alloy_composition_lookup.csv"))
_composition["Alloy"] = _composition["Alloy"].astype(str)

with open(os.path.join(DATA_DIR, "alloy_temper_lookup.json")) as f:
    _temper_lookup = json.load(f)  # {series: {alloy: [temper_groups...]}}

_ranges = pd.read_csv(os.path.join(DATA_DIR, "alloy_temper_property_ranges.csv"))
_ranges["Alloy"] = _ranges["Alloy"].astype(str)

# Series+temper coverage across ALL alloys in that series (not just one
# anchor alloy) -- used to flag series/temper combinations with zero
# training support at all, independent of composition.
with open(os.path.join(DATA_DIR, "series_temper_coverage.json")) as f:
    _series_temper_coverage = json.load(f)  # {series: {temper_group: row_count}}


def get_series_temper_coverage(series: str, temper_group: str) -> int:
    return _series_temper_coverage.get(series, {}).get(temper_group, 0)

# Per-element [min, max] observed across the 34 known alloy compositions.
# Used to flag when a user-entered composition falls outside anything the
# model was ever trained on.
_ELEMENT_RANGE = {c: (float(_composition[c].min()), float(_composition[c].max())) for c in COMP_COLS}


def get_element_range(element: str):
    return _ELEMENT_RANGE[element]


# ---------------------------------------------------------------------------
# Lookup helpers -- used by the bot to build its button menus
# ---------------------------------------------------------------------------
def get_series_list():
    """All available series, in a fixed order."""
    return [s for s in SERIES_LIST if s in _temper_lookup]


def get_alloys_for_series(series: str):
    """Alloy codes available under a given series, e.g. '7xxx' -> ['7010', '7017', ...]."""
    return sorted(_temper_lookup.get(series, {}).keys(), key=lambda a: int(a))


def get_tempers_for_alloy(series: str, alloy: str):
    """
    Temper groups that actually have training data for this specific alloy.
    This is the authoritative source for temper buttons -- never offer a
    temper group with no underlying data for that alloy.
    """
    return _temper_lookup.get(series, {}).get(str(alloy), [])


def temper_label(temper_group: str) -> str:
    return TEMPER_LABELS.get(temper_group, temper_group)


def get_nominal_composition(alloy: str) -> dict:
    """The known registered composition for an alloy -- used only to
    pre-fill defaults in the per-element entry flow, never used directly
    as the prediction input (the user's entered values are)."""
    row = _composition[_composition.Alloy == str(alloy)].iloc[0]
    return {c: float(row[c]) for c in COMP_COLS}


def get_historical_range(alloy: str, temper_group: str):
    """Returns (n_records, uts_min, uts_max, ys_min, ys_max) if any literature
    records exist for this exact alloy+temper combination, else None."""
    row = _ranges[(_ranges.Alloy == str(alloy)) & (_ranges.Temper_Group == temper_group)]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "n_uts": int(r["UTS_MPa_count"]) if pd.notna(r["UTS_MPa_count"]) else 0,
        "uts_min": r["UTS_MPa_min"], "uts_max": r["UTS_MPa_max"],
        "n_ys": int(r["YS_MPa_count"]) if pd.notna(r["YS_MPa_count"]) else 0,
        "ys_min": r["YS_MPa_min"], "ys_max": r["YS_MPa_max"],
    }


# ---------------------------------------------------------------------------
# Feature vector construction
# ---------------------------------------------------------------------------
def _build_feature_row(alloy: str, temper_group: str, scaler):
    comp_row = _composition[_composition.Alloy == str(alloy)]
    if comp_row.empty:
        raise ValueError(f"Unknown alloy: {alloy}")
    comp_row = comp_row.iloc[0]
    series = comp_row["Series"]

    x = {c: comp_row[c] for c in COMP_COLS}
    vec = pd.DataFrame([x])
    vec[COMP_COLS] = scaler.transform(vec[COMP_COLS])

    for s in SERIES_LIST:
        vec[f"Series_{s}"] = 1 if series == s else 0
    for t in TEMPER_GROUPS:
        vec[f"Temper_{t}"] = 1 if temper_group == t else 0

    ordered_cols = COMP_COLS + [f"Series_{s}" for s in SERIES_LIST] + [f"Temper_{t}" for t in TEMPER_GROUPS]
    return vec[ordered_cols]


# ---------------------------------------------------------------------------
# Confidence assessment for a user-supplied (non-lookup) composition
# ---------------------------------------------------------------------------
def _nearest_known_alloy(composition: dict):
    """Euclidean distance (in raw wt%) from the given composition to each of
    the 34 known alloy compositions; returns (alloy, distance) for the closest."""
    best_alloy, best_dist = None, float("inf")
    for _, row in _composition.iterrows():
        dist = sum((composition[c] - row[c]) ** 2 for c in COMP_COLS) ** 0.5
        if dist < best_dist:
            best_dist, best_alloy = dist, row["Alloy"]
    return best_alloy, best_dist


def assess_confidence(composition: dict, series: str = None, temper_group: str = None):
    """
    Returns a confidence assessment for a user-entered composition
    (and, if given, a series+temper combination):
    - out_of_range: elements whose value falls outside anything seen in
      the 34 known alloy compositions (the model's actual training support)
    - nearest_alloy / nearest_distance: closest known composition, as a
      sanity reference
    - series_temper_coverage: how many training rows exist for this exact
      series+temper combination, across ALL alloys in that series -- zero
      means the combination has never been observed at all (e.g. 5xxx+T6,
      which is also physically expected since 5xxx is non-heat-treatable)
    - label: High / Medium / Low, used to decide how wide a range to show
    """
    out_of_range = []
    for c in COMP_COLS:
        lo, hi = _ELEMENT_RANGE[c]
        if composition[c] < lo or composition[c] > hi:
            out_of_range.append((c, composition[c], lo, hi))

    nearest_alloy, nearest_dist = _nearest_known_alloy(composition)

    coverage = get_series_temper_coverage(series, temper_group) if series and temper_group else None

    if coverage == 0:
        label = "Low"  # zero training support for this series+temper overrides everything else
    elif len(out_of_range) == 0 and nearest_dist < 1.0:
        label = "High"
    elif len(out_of_range) <= 2 and nearest_dist < 4.0:
        label = "Medium"
    else:
        label = "Low"

    return {
        "out_of_range": out_of_range,
        "nearest_alloy": nearest_alloy,
        "nearest_distance": round(nearest_dist, 2),
        "series_temper_coverage": coverage,
        "label": label,
    }


def _rf_tree_spread(x_uts_row) -> float:
    """Std. deviation of UTS predictions across the Random Forest's individual
    trees -- a free, legitimate per-prediction uncertainty signal: trees that
    disagree a lot on this particular input indicate the forest itself is
    unsure, which tends to happen for inputs unlike anything in training."""
    tree_preds = [tree.predict(x_uts_row)[0] for tree in _rf_uts.estimators_]
    return float(pd.Series(tree_preds).std())


# ---------------------------------------------------------------------------
# Prediction from a user-supplied composition (no alloy lookup)
# ---------------------------------------------------------------------------
def predict_from_composition(series: str, temper_group: str, composition: dict) -> dict:
    """
    composition: dict of {element: wt%} for all 14 COMP_COLS.
    Unlike predict_properties(), this does NOT require a known alloy --
    the user supplies the composition directly. Confidence is assessed
    against the training data's actual coverage, and the returned range
    widens automatically when confidence is lower.
    """
    conf = assess_confidence(composition, series=series, temper_group=temper_group)

    def build_vec(scaler):
        vec = pd.DataFrame([{c: composition[c] for c in COMP_COLS}])
        vec[COMP_COLS] = scaler.transform(vec[COMP_COLS])
        for s in SERIES_LIST:
            vec[f"Series_{s}"] = 1 if series == s else 0
        for t in TEMPER_GROUPS:
            vec[f"Temper_{t}"] = 1 if temper_group == t else 0
        ordered = COMP_COLS + [f"Series_{s}" for s in SERIES_LIST] + [f"Temper_{t}" for t in TEMPER_GROUPS]
        return vec[ordered]

    x_uts = build_vec(_scaler_uts)
    uts_pred = float(_rf_uts.predict(x_uts)[0])
    tree_spread = _rf_tree_spread(x_uts)

    x_ys = build_vec(_scaler_ys)
    ys_pred = float(_ann_ys.predict(x_ys.values.astype("float32"), verbose=0)[0][0])

    # Range width scales with confidence: low confidence -> wider stated range.
    widen = {"High": 1.0, "Medium": 1.5, "Low": 2.5}[conf["label"]]
    uts_half_width = max(MODEL_PERFORMANCE["UTS"]["rmse"], tree_spread) * widen
    ys_half_width = MODEL_PERFORMANCE["YS"]["rmse"] * widen

    return {
        "series": series,
        "temper_group": temper_group,
        "temper_label": temper_label(temper_group),
        "composition": composition,
        "confidence": conf,
        "uts": {
            "predicted_mpa": round(uts_pred, 1),
            "range": (round(uts_pred - uts_half_width, 1), round(uts_pred + uts_half_width, 1)),
            "tree_spread": round(tree_spread, 1),
        },
        "ys": {
            "predicted_mpa": round(ys_pred, 1),
            "range": (round(ys_pred - ys_half_width, 1), round(ys_pred + ys_half_width, 1)),
        },
        "elongation_note": ELONGATION_DISCLAIMER,
        "hardness_note": HARDNESS_DISCLAIMER,
    }


def format_custom_prediction_message(result: dict) -> str:
    conf = result["confidence"]
    u, y = result["uts"], result["ys"]

    lines = [
        f"Series {result['series']} \u2014 {result['temper_label']}",
        f"Confidence: {conf['label']}",
        "",
        f"Predicted UTS: {u['predicted_mpa']} MPa  (range: {u['range'][0]}\u2013{u['range'][1]} MPa)",
        f"Predicted Yield Strength: {y['predicted_mpa']} MPa  (range: {y['range'][0]}\u2013{y['range'][1]} MPa)",
        "",
        f"Closest known alloy: {conf['nearest_alloy']} (composition distance: {conf['nearest_distance']})",
    ]
    if conf["series_temper_coverage"] == 0:
        lines.append("")
        lines.append(f"\u26a0\ufe0f No training data exists for {result['series']} + "
                      f"{result['temper_label']} at all, for any alloy \u2014 this combination "
                      f"is completely unvalidated, independent of the composition entered.")
    if conf["out_of_range"]:
        lines.append("")
        lines.append("\u26a0\ufe0f Elements outside anything seen in training data:")
        for el, val, lo, hi in conf["out_of_range"]:
            lines.append(f"   {el}: entered {val}%, training range was {lo}\u2013{hi}%")
        lines.append("Predictions for out-of-range elements are extrapolation and unvalidated.")
    if conf["label"] == "Low":
        lines.append("")
        lines.append("\u26a0\ufe0f Low confidence overall \u2014 this composition is substantially different "
                      "from anything in the training data. Treat this prediction as a rough estimate only.")
    lines.append("")
    lines.append("Elongation and hardness are not predicted by this bot \u2014 type /why for details.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public prediction function -- this is what the bot calls
# ---------------------------------------------------------------------------
def predict_properties(alloy: str, temper_group: str) -> dict:
    """
    Returns predicted UTS and YS for the given alloy + temper group, each
    with the deployed model's held-out test R2/RMSE/MAE, plus the
    historical literature range for that exact combination if available.
    Composition and series are looked up automatically -- never supplied
    by the caller.
    """
    alloy = str(alloy)

    x_uts = _build_feature_row(alloy, temper_group, _scaler_uts)
    uts_pred = float(_rf_uts.predict(x_uts)[0])

    x_ys = _build_feature_row(alloy, temper_group, _scaler_ys)
    ys_pred = float(_ann_ys.predict(x_ys.values.astype("float32"), verbose=0)[0][0])

    hist = get_historical_range(alloy, temper_group)

    return {
        "alloy": alloy,
        "temper_group": temper_group,
        "temper_label": temper_label(temper_group),
        "uts": {
            "predicted_mpa": round(uts_pred, 1),
            "model": MODEL_PERFORMANCE["UTS"]["model"],
            "test_r2": MODEL_PERFORMANCE["UTS"]["test_r2"],
            "rmse": MODEL_PERFORMANCE["UTS"]["rmse"],
            "mae": MODEL_PERFORMANCE["UTS"]["mae"],
            "historical_n": hist["n_uts"] if hist else 0,
            "historical_range": (hist["uts_min"], hist["uts_max"]) if hist and hist["n_uts"] > 0 else None,
        },
        "ys": {
            "predicted_mpa": round(ys_pred, 1),
            "model": MODEL_PERFORMANCE["YS"]["model"],
            "test_r2": MODEL_PERFORMANCE["YS"]["test_r2"],
            "rmse": MODEL_PERFORMANCE["YS"]["rmse"],
            "mae": MODEL_PERFORMANCE["YS"]["mae"],
            "historical_n": hist["n_ys"] if hist else 0,
            "historical_range": (hist["ys_min"], hist["ys_max"]) if hist and hist["n_ys"] > 0 else None,
        },
        "elongation_note": ELONGATION_DISCLAIMER,
        "hardness_note": HARDNESS_DISCLAIMER,
    }


def format_prediction_message(result: dict) -> str:
    """Formats a predict_properties() result into the Telegram reply text."""
    u, y = result["uts"], result["ys"]

    lines = [
        f"Alloy {result['alloy']} \u2014 {result['temper_label']}",
        "",
        f"Predicted UTS: {u['predicted_mpa']} MPa",
        f"  (model: {u['model']}, test R\u00b2={u['test_r2']}, typical error \u00b1{u['mae']} MPa)",
    ]
    if u["historical_range"]:
        lines.append(f"  Literature range for this exact alloy/temper ({u['historical_n']} records): "
                      f"{u['historical_range'][0]}\u2013{u['historical_range'][1]} MPa")
    lines.append("")
    lines.append(f"Predicted Yield Strength: {y['predicted_mpa']} MPa")
    lines.append(f"  (model: {y['model']}, test R\u00b2={y['test_r2']}, typical error \u00b1{y['mae']} MPa)")
    if y["historical_range"]:
        lines.append(f"  Literature range for this exact alloy/temper ({y['historical_n']} records): "
                      f"{y['historical_range'][0]}\u2013{y['historical_range'][1]} MPa")
    lines.append("")
    low_n = min(u["historical_n"] or 0, y["historical_n"] or 0)
    if low_n > 0 and low_n < 5:
        lines.append(f"\u26a0\ufe0f Low confidence: only {low_n} literature record(s) exist for this exact "
                      f"alloy/temper combination \u2014 treat this prediction with extra caution.")
        lines.append("")
    lines.append("Elongation and hardness are not predicted by this bot \u2014 type /why for details.")
    return "\n".join(lines)