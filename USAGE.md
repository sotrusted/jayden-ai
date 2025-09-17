# Spite AI Usage Guide

## Quick Start

### 1. Activate the Environment
```bash
# The environment is already set up with uv
uv run python -m src.spite_ai.jaden_ai
```

### 2. Or Use the CLI Script
```bash
# After installing the package
uv run spite-ai
```

### 2.1 Recommended persona settings
```bash
export SPITE_DEFAULT_K=12
export SPITE_SIMILARITY_THRESHOLD=0.5
export SPITE_USE_CITATIONS=true
export SPITE_SLANG_DENSITY=high
```

### 3. Test the Environment
```bash
uv run python test_environment.py
```

## Environment Setup

The project uses `uv` for dependency management and has been set up with:

- **Python 3.11** virtual environment
- **All required dependencies** installed
- **Proper package structure** with lazy imports
- **Development tools** configured

## Dependencies Installed

### Core Dependencies
- `sentence-transformers>=5.1.0` - For semantic search
- `faiss-cpu>=1.12.0` - For vector similarity search
- `numpy>=2.3.3` - For numerical operations
- `ollama>=0.3.0` - For Mistral model integration

### Development Dependencies
- `pytest>=8.4.2` - Testing framework
- `black>=25.1.0` - Code formatting
- `isort>=6.0.1` - Import sorting
- `mypy>=1.18.1` - Type checking
- `ruff>=0.13.0` - Linting

## Project Structure

```
jaden/
├── src/spite_ai/           # Main package
│   ├── __init__.py         # Package initialization with lazy imports
│   ├── jaden_ai.py         # Main AI system
│   ├── config.py           # Configuration management
│   └── example_usage.py    # Usage examples
├── tests/                  # Test files
│   └── test_basic.py       # Basic tests
├── pyproject.toml          # Project configuration
├── requirements.txt        # Production dependencies
├── requirements-dev.txt    # Development dependencies
├── setup.py               # Alternative setup script
└── test_environment.py    # Environment test script
```

## Configuration

The system can be configured through environment variables:

```bash
export SPITE_SIMILARITY_THRESHOLD=0.7
export SPITE_DEFAULT_K=10
export SPITE_LOG_LEVEL=DEBUG
export SPITE_OLLAMA_MODEL=mistral
```

## Running the System

### Interactive CLI
```bash
uv run python -m src.spite_ai.jaden_ai
```

### Programmatic Usage
```python
from src.spite_ai import SpiteAI, get_context, generate_with_mistral

# Initialize
ai_system = SpiteAI()

# Use
context = get_context("your query")
response = generate_with_mistral("your query", context, chat_history)
```

### Few-shots (optional, improves voice)
Add a `spite_fewshots.txt` file in project root with small Q/A examples that show the voice and light [n] citations.

### Development Commands
```bash
# Run tests
uv run pytest

# Format code
uv run black src/

# Sort imports
uv run isort src/

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/
```

## Requirements

- **Python 3.11+**
- **Ollama** with Mistral model installed
- **Required data files** (spite_corpus.json, spite_embeddings.npy, etc.)

## Troubleshooting

### Import Issues
If imports hang, it's because the ML models are loading. This is normal on first run.

### Missing Files
Ensure all required data files are in the project root:
- `spite_corpus.json`
- `spite_embeddings.npy`
- `spite_style_profile.json`
- `spite_system_prompt.txt`

### Ollama Issues
Make sure Ollama is running and the Mistral model is installed:
```bash
ollama list
ollama pull mistral
```

## Next Steps

1. **Test the system**: Run `uv run python test_environment.py`
2. **Start chatting**: Run `uv run python -m src.spite_ai.jaden_ai`
3. **Customize**: Modify `src/spite_ai/config.py` for your needs
4. **Develop**: Use the development tools for code quality
