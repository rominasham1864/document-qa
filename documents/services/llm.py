"""OpenRouter client configuration for LangChain chat models.

The embedding model stays local; only question + retrieved excerpts are
sent to the remote chat model when a RAG answer is requested.
"""

import os

from langchain_openai import ChatOpenAI

DEFAULT_OPENROUTER_MODEL = "google/gemma-4-26b-a4b-it:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_chat_model() -> ChatOpenAI:
    """Create an OpenRouter client using local environment configuration.

    Fail before making a network request when the API key is missing.
    No automatic retries: free-tier quotas are small.
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing. Add it to your .env file.")

    model_name = os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL).strip()
    if not model_name:
        raise RuntimeError("OPENROUTER_MODEL must not be empty.")

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
        max_tokens=500,
        timeout=45,
        max_retries=0,
    )
