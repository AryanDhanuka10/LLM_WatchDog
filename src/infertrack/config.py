# src/infertrack/config.py
"""Global configuration for llm-meter.

Usage::

    from infertrack.config import config, configure, reset_config

    configure(default_tag="my-app", silent=True)
    print(config.default_tag)   # "my-app"
    reset_config()
    print(config.default_tag)   # "default"

The ``config`` object is a module-level singleton.
``configure()`` mutates it in-place; ``reset_config()`` restores defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


# Config dataclass                                                     

@dataclass
class WatchdogConfig:
    """All tunable settings for llm-meter.

    Attributes:
        default_tag:   Tag applied to calls that don't specify one.
        silent:        If True, suppress all console warnings from the library.
        db_path:       Path to the SQLite log file.
                       Defaults to ``~/.llm-meter/logs.db``.
    """
    default_tag: str            = "default"
    silent: bool                = False
    db_path: Optional[Path]     = None      # None → use DEFAULT_DB_PATH at runtime


# Module-level singleton + helpers                                     

# The live config object — import this directly
config = WatchdogConfig()

# Snapshot of factory defaults for reset_config()
_DEFAULTS: dict[str, Any] = asdict(WatchdogConfig())


def configure(**kwargs: Any) -> None:
    """Override one or more config keys.

    Args:
        **kwargs: Any field name from ``WatchdogConfig``.

    Raises:
        ValueError: If an unknown key is passed.

    Example::

        configure(default_tag="search-feature", silent=True)
    """
    valid_keys = set(_DEFAULTS.keys())
    unknown = set(kwargs) - valid_keys
    if unknown:
        raise ValueError(
            f"Unknown config key(s): {sorted(unknown)}. "
            f"Valid keys: {sorted(valid_keys)}"
        )
    for key, value in kwargs.items():
        setattr(config, key, value)


def reset_config() -> None:
    """Restore all config values to their factory defaults."""
    for key, value in _DEFAULTS.items():
        setattr(config, key, value)