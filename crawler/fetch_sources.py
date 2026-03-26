import pandas as pd
import yaml
import os

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
                df = pd.DataFrame(data['sources'])
                if 'active' in df.columns:
                    df = df[df['active'] == True]
                df = df.dropna(subset=['rss_url'])
                return df

    # Fallback to Excel
    if os.path.exists(fallback_excel):
        df = pd.read_excel(fallback_excel)
        df.columns = [c.strip().lower() for c in df.columns]
        if "active" in df.columns:
            df = df[df["active"] == "yes"]
        df = df.dropna(subset=["rss_url"])
        return df

    # Return empty DataFrame if neither exists
    return pd.DataFrame()
