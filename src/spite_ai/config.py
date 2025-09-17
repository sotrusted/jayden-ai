"""
Configuration file for Spite AI system.
"""
import os
from typing import Dict, Any
import dotenv 

dotenv.load_dotenv()

class Config:
    """Configuration settings for Spite AI."""

    # Mode 
    LOCAL_MODE = False
    API_MODE = True
    
    assert LOCAL_MODE or API_MODE, "Either LOCAL_MODE or API_MODE must be True"


    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    if API_MODE:
        assert GROQ_API_KEY, "GROQ_API_KEY must be set"

    # File paths
    CORPUS_PATH = "data/spite_corpus.json"
    EMBEDDINGS_PATH = "data/spite_embeddings.npy"
    STYLE_PROFILE_PATH = "data/spite_style_profile.json"
    SYSTEM_PROMPT_PATH = "data/spite_system_prompt.txt"
    
    # Model settings
    MODEL_NAME = "all-mpnet-base-v2"
    OLLAMA_MODEL = "llama-3.3-70b-versatile"
    # Ollama generation options
    OLLAMA_TEMPERATURE = 0.2
    OLLAMA_TOP_P = 0.9
    OLLAMA_NUM_CTX = 4096
    OLLAMA_SEED = 42
    
    # Search parameters
    DEFAULT_K = 8
    SIMILARITY_THRESHOLD = 0.65
    MAX_CONTEXT_PASSAGES = 5
    
    # Response generation
    MAX_CHAT_HISTORY_LENGTH = 3000
    TRUNCATED_HISTORY_LENGTH = 1500
    RESPONSE_TIMEOUT = 30

    # Behavior
    STRICT_GROUNDING = True
    USE_CITATIONS = True
    SLANG_DENSITY = "mid"  # low | mid | high
    FEWSHOT_PATH = "data/spite_fewshots.txt"
    MAX_FEWSHOTS = 5
    FEWSHOTS_ENABLED = False
    ENFORCE_CITATIONS = False

    # Retrieval/rerank
    USE_RERANK = True
    RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_TOP_K = 20
    MIN_CONTEXT_PASSAGES = 3
    MIN_PASSAGE_CHARS = 40
    MIN_PASSAGE_WORDS = 5
    HEDGE_RETRY = True

    # Lorebook
    USE_LOREBOOK = True
    LOREBOOK_PATH = "data/spite_lorebook.json"
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    
    # Performance
    ENABLE_MEMORY_OPTIMIZATION = True
    CACHE_EMBEDDINGS = True
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create config from environment variables."""
        config = cls()
        
        # Override with environment variables if present
        config.CORPUS_PATH = os.getenv("SPITE_CORPUS_PATH", config.CORPUS_PATH)
        config.EMBEDDINGS_PATH = os.getenv("SPITE_EMBEDDINGS_PATH", config.EMBEDDINGS_PATH)
        config.STYLE_PROFILE_PATH = os.getenv("SPITE_STYLE_PROFILE_PATH", config.STYLE_PROFILE_PATH)
        config.SYSTEM_PROMPT_PATH = os.getenv("SPITE_SYSTEM_PROMPT_PATH", config.SYSTEM_PROMPT_PATH)
        config.MODEL_NAME = os.getenv("SPITE_MODEL_NAME", config.MODEL_NAME)
        config.OLLAMA_MODEL = os.getenv("SPITE_OLLAMA_MODEL", config.OLLAMA_MODEL)
        # Ollama options
        config.OLLAMA_TEMPERATURE = float(os.getenv("SPITE_OLLAMA_TEMPERATURE", config.OLLAMA_TEMPERATURE))
        config.OLLAMA_TOP_P = float(os.getenv("SPITE_OLLAMA_TOP_P", config.OLLAMA_TOP_P))
        config.OLLAMA_NUM_CTX = int(os.getenv("SPITE_OLLAMA_NUM_CTX", config.OLLAMA_NUM_CTX))
        seed_env = os.getenv("SPITE_OLLAMA_SEED", str(config.OLLAMA_SEED))
        try:
            config.OLLAMA_SEED = int(seed_env)
        except ValueError:
            pass
        config.DEFAULT_K = int(os.getenv("SPITE_DEFAULT_K", config.DEFAULT_K))
        config.SIMILARITY_THRESHOLD = float(os.getenv("SPITE_SIMILARITY_THRESHOLD", config.SIMILARITY_THRESHOLD))
        config.LOG_LEVEL = os.getenv("SPITE_LOG_LEVEL", config.LOG_LEVEL)
        config.USE_CITATIONS = os.getenv("SPITE_USE_CITATIONS", str(config.USE_CITATIONS)).lower() in {"1","true","yes"}
        config.ENFORCE_CITATIONS = os.getenv("SPITE_ENFORCE_CITATIONS", str(config.ENFORCE_CITATIONS)).lower() in {"1","true","yes"}
        config.SLANG_DENSITY = os.getenv("SPITE_SLANG_DENSITY", config.SLANG_DENSITY)
        config.FEWSHOT_PATH = os.getenv("SPITE_FEWSHOT_PATH", config.FEWSHOT_PATH)
        config.MAX_FEWSHOTS = int(os.getenv("SPITE_MAX_FEWSHOTS", config.MAX_FEWSHOTS))
        config.FEWSHOTS_ENABLED = os.getenv("SPITE_FEWSHOTS_ENABLED", str(config.FEWSHOTS_ENABLED)).lower() in {"1","true","yes"}
        #         # Rerank options
        config.USE_RERANK = os.getenv("SPITE_USE_RERANK", str(config.USE_RERANK)).lower() in {"1","true","yes"}
        config.RERANK_MODEL = os.getenv("SPITE_RERANK_MODEL", config.RERANK_MODEL)
        config.RERANK_TOP_K = int(os.getenv("SPITE_RERANK_TOP_K", config.RERANK_TOP_K))
        config.MIN_CONTEXT_PASSAGES = int(os.getenv("SPITE_MIN_CONTEXT_PASSAGES", config.MIN_CONTEXT_PASSAGES))
        config.MIN_PASSAGE_CHARS = int(os.getenv("SPITE_MIN_PASSAGE_CHARS", config.MIN_PASSAGE_CHARS))
        config.MIN_PASSAGE_WORDS = int(os.getenv("SPITE_MIN_PASSAGE_WORDS", config.MIN_PASSAGE_WORDS))
        config.HEDGE_RETRY = os.getenv("SPITE_HEDGE_RETRY", str(config.HEDGE_RETRY)).lower() in {"1","true","yes"}
        # Lorebook
        config.USE_LOREBOOK = os.getenv("SPITE_USE_LOREBOOK", str(config.USE_LOREBOOK)).lower() in {"1","true","yes"}
        config.LOREBOOK_PATH = os.getenv("SPITE_LOREBOOK_PATH", config.LOREBOOK_PATH)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'corpus_path': self.CORPUS_PATH,
            'embeddings_path': self.EMBEDDINGS_PATH,
            'style_profile_path': self.STYLE_PROFILE_PATH,
            'system_prompt_path': self.SYSTEM_PROMPT_PATH,
            'model_name': self.MODEL_NAME,
            'ollama_model': self.OLLAMA_MODEL,
            'ollama_temperature': self.OLLAMA_TEMPERATURE,
            'ollama_top_p': self.OLLAMA_TOP_P,
            'ollama_num_ctx': self.OLLAMA_NUM_CTX,
            'ollama_seed': self.OLLAMA_SEED,
            'default_k': self.DEFAULT_K,
            'similarity_threshold': self.SIMILARITY_THRESHOLD,
            'max_context_passages': self.MAX_CONTEXT_PASSAGES,
            'max_chat_history_length': self.MAX_CHAT_HISTORY_LENGTH,
            'response_timeout': self.RESPONSE_TIMEOUT,
            'log_level': self.LOG_LEVEL
        }
