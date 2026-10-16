"""DataConfessor: find the patients a model will memorise, before you train it.

    from dataconfessor import load_manifest, screen
    man = load_manifest("manifest.csv")
    result = screen(man)
    result.safe_summary()        # aggregate only -- safe to log or hand to an AI assistant
    result.write_patients(...)   # the flagged cohort -- sensitive, keep local

Method: Vrettos et al., "Predicting and mitigating privacy risks in medical AI".
"""

__version__ = "0.1.0"

__all__ = [
    "Manifest", "ScreenResult", "load_manifest", "describe", "screen",
    "SELF_CONSISTENCY_FLOOR", "DEFAULT_MODELS", "DEFAULT_FLAG_FRACTION", "__version__",
]

# Torch costs a few seconds to import and `init`, `check` and `advise` never touch it, so
# the public names are resolved on first use rather than at import (PEP 562).
_LAZY = {
    "Manifest": ("manifest", "Manifest"),
    "load_manifest": ("manifest", "load"),
    "describe": ("manifest", "summary"),
    "ScreenResult": ("screen", "ScreenResult"),
    "screen": ("screen", "run"),
    "SELF_CONSISTENCY_FLOOR": ("screen", "SELF_CONSISTENCY_FLOOR"),
    "DEFAULT_MODELS": ("screen", "DEFAULT_MODELS"),
    "DEFAULT_FLAG_FRACTION": ("screen", "DEFAULT_FLAG_FRACTION"),
}


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module, attr = _LAZY[name]
    value = getattr(importlib.import_module(f".{module}", __name__), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
