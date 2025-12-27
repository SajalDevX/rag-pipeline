"""
Groq LLM Manager

This module provides LLM functionality using Groq's fast inference API.
Groq uses custom LPU (Language Processing Unit) hardware for extremely
fast inference.

Groq Free Tier:
- 14,400 requests/day
- Very fast responses (100+ tokens/second)

Models available:
- llama-3.1-8b-instant (fast, good quality)
- llama-3.1-70b-versatile (slower, better quality)
- mixtral-8x7b-32768 (good balance)
"""

import asyncio
from typing import Any

from groq import Groq

from src.config import get_logger, settings
from src.models import RetrievedChunk
from src.models.errors import ErrorDetail, LLMError

logger = get_logger(__name__)


# Default system prompt for RAG
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant. Answer the user's question based on the provided context.

Guidelines:
1. Use ONLY the information from the context to answer the question.
2. If the context doesn't contain enough information, say so clearly.
3. Be concise but thorough in your answers.
4. If you quote from the context, indicate which source it comes from.
5. Do not make up information that is not in the context.

Context will be provided in the format:
[Source: filename] Content...
"""


class GroqLLM:
    """
    LLM manager using Groq's fast inference API.
    
    Handles all LLM operations:
    - Building prompts with context
    - Generating responses
    - Managing conversation history
    
    Attributes:
        api_key: Groq API key
        model: LLM model to use
        temperature: Response randomness (0-2)
        max_tokens: Maximum response length
    
    Usage:
        llm = GroqLLM()
        
        # Generate answer with context
        answer = await llm.generate(
            query="How does ML work?",
            context_chunks=retrieved_chunks
        )
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None
    ):
        """
        Initialize the Groq LLM manager.
        
        Args:
            api_key: Groq API key (default from settings)
            model: LLM model (default from settings)
            temperature: Response randomness (default from settings)
            max_tokens: Max response tokens (default from settings)
            system_prompt: System prompt (default: RAG prompt)
        """
        self.api_key = api_key or settings.groq_api_key
        self.model = model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        
        self._client: Groq | None = None
        
        self.logger = get_logger(self.__class__.__name__)
        
        # Validate API key
        if not self.api_key:
            raise LLMError(
                message="Groq API key not provided",
                details=[ErrorDetail(field="api_key", message="Set GROQ_API_KEY in .env")]
            )
        
        # Initialize client
        self._client = Groq(api_key=self.api_key)
        
        self.logger.info(
            "Groq LLM initialized",
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    async def generate(
        self,
        query: str,
        context_chunks: list[RetrievedChunk] | None = None,
        system_prompt: str | None = None,
        **kwargs
    ) -> str:
        """
        Generate a response to the query using context.
        
        Args:
            query: User's question
            context_chunks: Retrieved chunks for context
            system_prompt: Override system prompt
            **kwargs: Additional generation parameters
        
        Returns:
            str: Generated answer
        
        Raises:
            LLMError: If generation fails
        """
        if not query or not query.strip():
            raise LLMError(message="Query cannot be empty")
        
        self.logger.info(
            "Generating response",
            query_length=len(query),
            context_chunks=len(context_chunks) if context_chunks else 0
        )
        
        try:
            # Build the prompt
            messages = self._build_messages(
                query=query,
                context_chunks=context_chunks,
                system_prompt=system_prompt
            )
            
            # Call Groq API in thread pool
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=kwargs.get("temperature", self.temperature),
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                )
            )
            
            # Extract answer
            answer = response.choices[0].message.content
            
            self.logger.info(
                "Response generated",
                answer_length=len(answer),
                model=self.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            )
            
            return answer
            
        except Exception as e:
            self.logger.error("Generation failed", error=str(e))
            raise LLMError(
                message=f"Failed to generate response: {str(e)}"
            )
    
    def _build_messages(
        self,
        query: str,
        context_chunks: list[RetrievedChunk] | None = None,
        system_prompt: str | None = None
    ) -> list[dict[str, str]]:
        """
        Build messages array for the chat API.
        
        Args:
            query: User's question
            context_chunks: Retrieved chunks for context
            system_prompt: Override system prompt
        
        Returns:
            list[dict]: Messages for the API
        """
        messages = []
        
        # System message
        messages.append({
            "role": "system",
            "content": system_prompt or self.system_prompt
        })
        
        # Build user message with context
        user_content = ""
        
        if context_chunks:
            user_content += "Context:\n\n"
            
            for i, chunk in enumerate(context_chunks, 1):
                source = chunk.title or chunk.source
                user_content += f"[{i}] [Source: {source}]\n{chunk.content}\n\n"
            
            user_content += "---\n\n"
        
        user_content += f"Question: {query}"
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        return messages
    
    async def generate_simple(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs
    ) -> str:
        """
        Generate a response without RAG context.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            **kwargs: Additional parameters
        
        Returns:
            str: Generated response
        """
        messages = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=kwargs.get("temperature", self.temperature),
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                )
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error("Generation failed", error=str(e))
            raise LLMError(message=f"Failed to generate response: {str(e)}")
    
    async def health_check(self) -> bool:
        """
        Check if the LLM is healthy.
        
        Returns:
            bool: True if healthy
        """
        try:
            response = await self.generate_simple(
                prompt="Say 'ok' and nothing else.",
                max_tokens=10
            )
            return len(response) > 0
        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return False
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"GroqLLM("
            f"model='{self.model}', "
            f"temperature={self.temperature}, "
            f"max_tokens={self.max_tokens})"
        )