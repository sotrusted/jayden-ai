"""Basic tests for Spite AI system."""

import pytest
from src.spite_ai.config import Config
from src.spite_ai.jaden_ai import SpiteAI


def test_config_creation():
    """Test that config can be created."""
    config = Config()
    assert config.DEFAULT_K == 8
    assert config.SIMILARITY_THRESHOLD == 0.65
    assert config.MODEL_NAME == "all-mpnet-base-v2"


def test_config_from_env():
    """Test that config can be created from environment variables."""
    config = Config.from_env()
    assert isinstance(config, Config)


def test_config_to_dict():
    """Test that config can be converted to dictionary."""
    config = Config()
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert "corpus_path" in config_dict
    assert "model_name" in config_dict


def test_spite_ai_initialization_without_files():
    """Test that SpiteAI raises error when files don't exist."""
    config = Config()
    config.CORPUS_PATH = "nonexistent.json"
    
    with pytest.raises(FileNotFoundError):
        SpiteAI(config)


if __name__ == "__main__":
    pytest.main([__file__])
