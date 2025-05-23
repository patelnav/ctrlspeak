"""
Configuration management for ctrlSPEAK.
"""
import os
import json
import time

def get_config_path():
    """Get the path to the configuration file."""
    config_dir = os.path.expanduser("~/.config/ctrlspeak")
    os.makedirs(config_dir, exist_ok=True)
    return os.path.join(config_dir, "config.json")

def load_config():
    """Load configuration from file or create default."""
    config_path = get_config_path()
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception:
            # If config is corrupted, return default
            pass
    
    # Default configuration
    return {
        "first_run_completed": False,
        "last_run": None,
        "preferred_model": "parakeet"
    }

def save_config(config):
    """Save configuration to file."""
    config_path = get_config_path()
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save configuration: {e}")

def is_first_run():
    """Check if this is the first run of the application."""
    config = load_config()
    return not config.get("first_run_completed", False)

def mark_first_run_complete():
    """Mark the first run as complete."""
    config = load_config()
    config["first_run_completed"] = True
    config["last_run"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_config(config)

def get_preferred_model():
    """Get the preferred model from config."""
    config = load_config()
    return config.get("preferred_model", "parakeet")

def set_preferred_model(model_name):
    """Set the preferred model in config."""
    config = load_config()
    config["preferred_model"] = model_name
    save_config(config) 