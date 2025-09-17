"""
Spite AI - AI system that responds as a user from Spite Magazine.
"""

from .config import Config

__version__ = "0.1.0"
__all__ = ["SpiteAI", "get_context", "generate_with_mistral", "main", "Config"]

# Lazy imports to avoid loading heavy ML models on import
def __getattr__(name):
    if name in ["SpiteAI", "get_context", "generate_with_mistral", "main"]:
        from .jaden_ai import SpiteAI, get_context, generate_with_mistral, main
        return locals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
