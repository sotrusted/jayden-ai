#!/usr/bin/env python3
"""Setup script for Spite AI."""

from setuptools import setup, find_packages

setup(
    name="spite-ai",
    version="0.1.0",
    description="AI system that responds as a user from Spite Magazine using semantic search and Mistral",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "faiss-cpu>=1.12.0",
        "numpy>=2.3.3",
        "sentence-transformers>=5.1.0",
        "ollama>=0.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "ruff>=0.1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "spite-ai=spite_ai:main",
        ],
    },
)
