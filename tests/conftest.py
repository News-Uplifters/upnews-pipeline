"""pytest configuration and shared fixtures.

This conftest stubs out heavy optional dependencies (setfit, torch,
transformers) that are not installed in the lightweight test environment.
The stubs are injected into sys.modules before any test module is imported,
so tests that mock the model can still import from classifier/ and
pipeline/summarizer without a GPU or model download.
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock


def _make_stub(name: str) -> ModuleType:
    """Create a stub module with MagicMock attributes."""
    mod = ModuleType(name)
    mod.__spec__ = MagicMock()  # satisfy importlib checks
    return mod


# ---------------------------------------------------------------------------
# Stub setfit
# ---------------------------------------------------------------------------

_setfit = _make_stub("setfit")
_setfit.SetFitModel = MagicMock  # type: ignore[attr-defined]
sys.modules.setdefault("setfit", _setfit)

# ---------------------------------------------------------------------------
# Stub torch and transformers (pulled in transitively by setfit / pipeline)
# ---------------------------------------------------------------------------

for _name in (
    "torch",
    "transformers",
    "transformers.pipelines",
    "sentence_transformers",
):
    sys.modules.setdefault(_name, _make_stub(_name))
