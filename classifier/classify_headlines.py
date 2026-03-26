from setfit import SetFitModel
import pandas as pd
import re

STRICT_SOURCE_THRESHOLDS = {
    "BBCNews": 0.90, "CBSNews": 0.90, "APNews": 0.93,
    "ReutersWorld": 0.93, "NYTimesWorld": 0.93, "NPRNews": 0.93,
    "GuardianWorld": 0.93, "AlJazeeraAll": 0.93,
}

UPLIFTING_HINTS = (
    "wins", "win", "rescued", "recovery", "breakthrough", "improves",
    "helped", "helps", "saved", "donates", "charity", "volunteer",
    "uplifting", "success", "record high", "celebrates", "hope", "positive"
)

def _has_uplifting_hint(text, hints):
    if not isinstance(text, str) or not text.strip():
        return False
    for hint in hints:
        pattern = r"\b" + re.escape(hint.lower()) + r"\b"
        if re.search(pattern, text.lower()):
            return True
    return False

def load_model(model_path="models/setfit_uplifting_model"):
    model = SetFitModel.from_pretrained(model_path)
    return model

def filter_positive_news(df, model, threshold=0.75, source_thresholds=None):
    if df.empty:
        return df.copy()
    df = df.copy()
    texts = df["title"].tolist()
    probs = model.predict_proba(texts)
    positive_probs = [float(p[1]) for p in probs]
    df["uplifting_score"] = positive_probs
    source_thresholds = source_thresholds or STRICT_SOURCE_THRESHOLDS
    if "source" in df.columns:
        df["min_threshold"] = df["source"].map(source_thresholds).fillna(threshold)
    else:
        df["min_threshold"] = threshold
    positive_df = df[df["uplifting_score"] >= df["min_threshold"]]
    if "source" in positive_df.columns and "title" in positive_df.columns and source_thresholds:
        strict_sources = set(source_thresholds.keys())
        strict_mask = positive_df["source"].isin(strict_sources)
        hint_mask = positive_df["title"].apply(lambda t: _has_uplifting_hint(t, UPLIFTING_HINTS))
        positive_df = positive_df[~strict_mask | hint_mask]
    return positive_df.drop(columns=["min_threshold"], errors="ignore")
