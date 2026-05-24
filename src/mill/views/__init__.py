from __future__ import annotations

import importlib
import pkgutil

_modules = []
for _, _name, _ispkg in pkgutil.iter_modules(__path__):
    if _ispkg or _name.startswith("_"):
        continue
    _mod = importlib.import_module(f"{__name__}.{_name}")
    if hasattr(_mod, "NAME") and hasattr(_mod, "SPEC"):
        _modules.append(_mod)

ALL = tuple(_modules)
