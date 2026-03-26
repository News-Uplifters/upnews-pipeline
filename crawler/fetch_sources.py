import logging
import pandas as pd
import yaml
import os

logger = logging.getLogger(__name__)

REQUIRED_SOURCE_FIELDS = {"name", "source_id", "rss_url"}
VALID_ADAPTER_VALUES = {"reddit", "rss", None}

class SourceValidationError(ValueError):
    """Raised when a source entry fails schema validation."""
    pass


def validate_sources(sources: list) -> list:
    """Validate a list of source dicts against the expected schema.

    Args:
        sources: List of source dicts loaded from YAML

    Returns:
        The same list if valid

    Raises:
        SourceValidationError: If any source fails validation
    """
    if not isinstance(sources, list):
        raise SourceValidationError("'sources' must be a list")

    errors = []
    for i, source in enumerate(sources):
        if not isinstance(source, dict):
            errors.append(f"Source[{i}] is not a dict: {source!r}")
            continue

        missing = REQUIRED_SOURCE_FIELDS - source.keys()
        if missing:
            errors.append(f"Source[{i}] ({source.get('name', '?')!r}) missing required fields: {missing}")

        threshold = source.get("threshold")
        if threshold is not None:
            if not isinstance(threshold, (int, float)):
                errors.append(f"Source[{i}] ({source.get('name', '?')!r}) 'threshold' must be a number, got {type(threshold).__name__}")
            elif not 0.0 <= float(threshold) <= 1.0:
                errors.append(f"Source[{i}] ({source.get('name', '?')!r}) 'threshold' must be between 0 and 1, got {threshold}")

        active = source.get("active")
        if active is not None and not isinstance(active, bool):
            errors.append(f"Source[{i}] ({source.get('name', '?')!r}) 'active' must be a boolean, got {type(active).__name__}")

        adapter = source.get("adapter")
        if adapter is not None and adapter not in VALID_ADAPTER_VALUES:
            errors.append(f"Source[{i}] ({source.get('name', '?')!r}) 'adapter' must be one of {VALID_ADAPTER_VALUES}, got {adapter!r}")

    if errors:
        raise SourceValidationError("YAML source validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    return sources


def load_sources(path="config/sources.yaml", fallback_excel="data/news_sources.xlsx"):
    """Load news sources from YAML or Excel (fallback).

    Args:
        path: Path to sources.yaml file
        fallback_excel: Fallback Excel file path if YAML not found

    Returns:
        DataFrame with columns: name, source_id, rss_url, active, threshold, etc.
    """
    # Try YAML first
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            if data and 'sources' in data:
                validate_sources(data['sources'])
                df = pd.DataFrame(data['sources'])
                if 'active' in df.columns:
                    df = df[df['active'] == True]
                df = df.dropna(subset=['rss_url'])
                logger.info(f"Loaded {len(df)} active sources from {path}")
                return df
            else:
                logger.warning(f"YAML file {path} has no 'sources' key, trying fallback")

    # Fallback to Excel (deprecated)
    if os.path.exists(fallback_excel):
        logger.warning(f"Loading sources from deprecated Excel fallback: {fallback_excel}")
        df = pd.read_excel(fallback_excel)
        df.columns = [c.strip().lower() for c in df.columns]
        if "active" in df.columns:
            df = df[df["active"] == "yes"]
        df = df.dropna(subset=["rss_url"])
        return df

    logger.error(f"No sources found: checked {path} and {fallback_excel}")
    # Return empty DataFrame if neither exists
    return pd.DataFrame()
