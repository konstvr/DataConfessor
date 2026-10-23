"""DataConfessor: rank patients by privacy risk before a model is trained on their data."""

__version__ = "0.1.0"

__all__ = [
    "Manifest", "ScreenResult", "load_manifest", "describe", "screen",
    "SELF_CONSISTENCY_FLOOR", "DEFAULT_MODELS", "DEFAULT_FLAG_FRACTION", "__version__",
]

# Torch costs a few seconds to import and `init`, `check` and `advise` never touch it, so
# the public names are resolved on first use rather than at import (PEP 562).
# NOTE: "screen" is both a submodule and the name of its entry point here, so
# `from dataconfessor import screen` gives the FUNCTION. Code that wants the module should
# use `importlib.import_module("dataconfessor.screen")`.
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
