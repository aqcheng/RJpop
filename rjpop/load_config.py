import json
import os

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")


def load_config() -> dict:
    """Load config.json from the project root if it exists, otherwise return an empty dict."""
    if os.path.exists(_CONFIG_PATH):
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    return {}


def update_config(k, v):
    cfg = load_config()
    cfg[k] = v
    with open(_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
