import os

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
    """Load the SetFit model from *model_path*.

    Falls back to a rule-based stub when:
    - the ``CLASSIFIER_MODE`` environment variable is set to ``rules``, or
    - the model directory does not exist.

    The stub scores each title by hint density so that the rest of the
    pipeline (thresholds, DB writes) works identically without a GPU.
    """
    mode = os.environ.get("CLASSIFIER_MODE", "setfit").lower()
    if mode == "rules" or not os.path.exists(model_path):
        return _RuleBasedModel()

    from setfit import SetFitModel  # noqa: import-outside-toplevel
    return SetFitModel.from_pretrained(model_path)


class _RuleBasedModel:
    """Lightweight rule-based uplifting classifier used when no ML model is available.

    ``predict_proba`` returns ``[[neg_score, pos_score], ...]`` to match the
    SetFit interface expected by ``filter_positive_news``.
    """

    # Extra hints used for scoring beyond the UPLIFTING_HINTS list
    _POSITIVE_HINTS = UPLIFTING_HINTS + (
        "award", "achievement", "reunited", "healed", "milestone",
        "historic", "discover", "partnership", "innovative", "growth",
        "inspiring", "grateful", "joy", "kind", "peace",
    )
    _NEGATIVE_HINTS = (
        "kill", "killed", "dead", "death", "crash", "war", "attack",
        "tragedy", "disaster", "crisis", "terror", "violence", "arrested",
        "fire", "flood", "explosion", "murder", "wounded",
    )

    def predict_proba(self, texts):
        results = []
        for text in texts:
            pos_hits = sum(
                1 for h in self._POSITIVE_HINTS if _has_uplifting_hint(text, (h,))
            )
            neg_hits = sum(
                1 for h in self._NEGATIVE_HINTS if _has_uplifting_hint(text, (h,))
            )
            # Base score 0.1; +0.65 per positive hint so a single uplifting
            # keyword clears the default 0.75 threshold; -0.15 per negative hit
            score = 0.1 + (pos_hits * 0.65) - (neg_hits * 0.15)
            score = max(0.0, min(1.0, score))
            results.append([1.0 - score, score])
        return results

def filter_positive_news(df, model, threshold=0.75, source_thresholds=None):
    if df.empty:
        return df.copy()
    df = df.copy()
    texts = df["title"].tolist()
    probs = model.predict_proba(texts)
    positive_probs = [float(p[1]) for p in probs]
    df["uplifting_score"] = positive_probs
    source_thresholds = source_thresholds or STRICT_SOURCE_THRESHOLDS
    source_col = "source_id" if "source_id" in df.columns else "source"
    if source_col in df.columns:
        df["min_threshold"] = df[source_col].map(source_thresholds).fillna(threshold)
    else:
        df["min_threshold"] = threshold
    positive_df = df[df["uplifting_score"] >= df["min_threshold"]]
    if source_col in positive_df.columns and "title" in positive_df.columns and source_thresholds:
        strict_sources = set(source_thresholds.keys())
        strict_mask = positive_df[source_col].isin(strict_sources)
        hint_mask = positive_df["title"].apply(lambda t: _has_uplifting_hint(t, UPLIFTING_HINTS))
        positive_df = positive_df[~strict_mask | hint_mask]
    return positive_df.drop(columns=["min_threshold"], errors="ignore")
