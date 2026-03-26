"""LLM-based article categorization (TASK 4 stub)."""

import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

# Default categories for multi-label classification
DEFAULT_CATEGORIES = [
    "Health & Wellness",
    "Environment & Nature",
    "Community & Social Good",
    "Technology & Science",
    "Business & Economics",
    "Culture & Arts",
    "Human Interest",
]

def categorize_article(
    title: str,
    body: str = "",
    categories: Optional[List[str]] = None
) -> Dict:
    """Categorize article into predefined or custom categories.

    Uses zero-shot classification (HuggingFace BART) for fast,
    on-device categorization. Optionally use Claude Haiku for
    complex articles.

    Args:
        title: Article title
        body: Article body (optional, for context)
        categories: List of category names. If None, use defaults.

    Returns:
        Dict with:
        - category (str): Top category
        - scores (dict): {category: confidence_score}
        - confidence (float): Confidence in top category

    Example:
        >>> result = categorize_article(
        ...     "Scientists discover new sustainable battery",
        ...     categories=DEFAULT_CATEGORIES
        ... )
        >>> print(result)
        {
            "category": "Technology & Science",
            "scores": {
                "Technology & Science": 0.95,
                "Environment & Nature": 0.12,
                ...
            },
            "confidence": 0.95
        }
    """
    # TODO: Implement zero-shot classifier
    # 1. Load facebook/bart-large-mnli model
    # 2. Prepare input text (title + body)
    # 3. Run zero-shot classification
    # 4. Return top category with scores
    # 5. Optional: Add Claude Haiku fallback for edge cases
    pass


def categorize_batch(
    articles: List[Dict],
    categories: Optional[List[str]] = None
) -> List[Dict]:
    """Categorize multiple articles efficiently.

    Args:
        articles: List of article dicts with 'title' and optional 'body'
        categories: List of category names. If None, use defaults.

    Returns:
        List of articles with 'category' and 'category_scores' added

    Example:
        >>> articles = [
        ...     {"title": "...", "body": "..."},
        ...     {"title": "...", "body": "..."},
        ... ]
        >>> enriched = categorize_batch(articles)
        >>> print(enriched[0]["category"])
        "Technology & Science"
    """
    # TODO: Implement batch processing with caching
    # 1. Use categorize_article for each
    # 2. Add to articles dict
    # 3. Cache results to avoid re-categorizing
    pass
