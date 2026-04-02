"""pytest configuration and shared fixtures.

Stubs out optional heavy dependencies (setfit) that are not installed in the
lightweight test environment.  Categorization and summarization no longer use
ML models, so torch/transformers stubs are no longer required.
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock


def _make_stub(name: str) -> ModuleType:
    """Create a stub module with MagicMock attributes."""
    mod = ModuleType(name)
    mod.__spec__ = MagicMock()  # satisfy importlib checks
    return mod


# Stub setfit — only needed when CLASSIFIER_MODE=setfit (optional feature).
_setfit = _make_stub("setfit")
_setfit.SetFitModel = MagicMock  # type: ignore[attr-defined]
sys.modules.setdefault("setfit", _setfit)
