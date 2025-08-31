"""
Configuration management for the static TOML config file.
"""
import os
import toml
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "pawikan" / "config.toml"
EXAMPLE_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.example.toml"

def get_config_path() -> Path:
    """Gets the path to the configuration file."""
    return Path(os.environ.get("PAWIKAN_CONFIG", DEFAULT_CONFIG_PATH))

def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Loads the configuration from a TOML file.
    If the file doesn't exist, it creates it from the example.
    """
    path = config_path or get_config_path()
    if not path.exists():
        print(f"Config file not found at {path}. Creating from example.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(EXAMPLE_CONFIG_PATH, "r") as f_example, open(path, "w") as f_config:
            f_config.write(f_example.read())

    with open(path, "r") as f:
        return toml.load(f)

# Load static config on module import
settings = load_config()
