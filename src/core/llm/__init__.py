"""
LLM Package

This package provides LLM functionality using Groq's fast inference API.

Usage:
    from src.core.llm import GroqLLM
    
    llm = GroqLLM()
    
    # Generate with RAG context
    answer = await llm.generate(
        query="How does ML work?",
        context_chunks=retrieved_chunks
    )
    
    # Simple generation without context
    answer = await llm.generate_simple("Tell me a joke")
"""

from src.core.llm.groq_llm import GroqLLM

__all__ = [
    "GroqLLM",
]