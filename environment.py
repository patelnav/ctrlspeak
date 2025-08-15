
import os

def save_environment_variables():
    """Save current environment variables for later restoration"""
    return {
        "NEMO_LOGGING_LEVEL": os.environ.get("NEMO_LOGGING_LEVEL", ""),
        "TF_CPP_MIN_LOG_LEVEL": os.environ.get("TF_CPP_MIN_LOG_LEVEL", ""),
        "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "")
    }

def restore_environment_variables(saved_vars):
    """Restore environment variables from saved state"""
    for key, value in saved_vars.items():
        if value:
            os.environ[key] = value
        elif key in os.environ:
            del os.environ[key]
