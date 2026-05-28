"""Configuration loading and management."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Configuration manager."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize config.

        Args:
            config_path: Path to config.yaml. If None, uses default in project root.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            self.data = yaml.safe_load(f) or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dot-notation key."""
        keys = key.split(".")
        value = self.data

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """Get config value by key."""
        return self.get(key)

    def __repr__(self) -> str:
        """String representation."""
        return f"Config({self.data})"


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration from file."""
    return Config(config_path)
