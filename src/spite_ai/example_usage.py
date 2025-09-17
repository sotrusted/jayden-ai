#!/usr/bin/env python3
"""
Example usage of the improved Spite AI system.
"""

from jaden_ai import SpiteAI, get_context, generate_with_mistral
from config import Config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_conversation():
    """Example of how to use the Spite AI system programmatically."""
    
    # Initialize the system
    config = Config.from_env()
    ai_system = SpiteAI(config)
    
    try:
        # Example queries
        queries = [
            "What do you think about James Aaronson?",
            "Tell me about Spite Magazine",
            "What's your opinion on art?",
            "How do you feel about New York?"
        ]
        
        chat_history = ""
        
        for query in queries:
            print(f"\nUser: {query}")
            
            # Get context
            context = get_context(query)
            
            # Generate response
            response = generate_with_mistral(query, context, chat_history)
            
            print(f"AI: {response}")
            
            # Update chat history
            chat_history += f"User: {query}\nAI: {response}\n\n"
            
            # Keep history manageable
            if len(chat_history) > 2000:
                chat_history = chat_history[-1500:]
    
    finally:
        # Cleanup
        ai_system.cleanup()

if __name__ == "__main__":
    example_conversation()
