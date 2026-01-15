import os
import logging
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_core.documents import Document

from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def get_env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and not value:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


def get_connection_string() -> str:
    return (
        f"postgresql://"
        f"{get_env('PG_USER', 'postgres')}:"
        f"{get_env('PG_PASSWORD', 'postgres')}@"
        f"{get_env('PG_HOST', 'localhost')}:"
        f"{get_env('PG_PORT', '5432')}/"
        f"{get_env('PG_DATABASE', 'ragdb')}"
    )


def get_embeddings():
    provider = get_env("PROVIDER", "openai").lower()

    if provider == "gemini":
        logger.info("Using Gemini embeddings")
        return GoogleGenerativeAIEmbeddings(
            model=get_env("GEMINI_EMBEDDING_MODEL", "models/embedding-001"),
            google_api_key=get_env("GOOGLE_API_KEY", required=True),
        )

    logger.info("Using OpenAI embeddings")
    return OpenAIEmbeddings(
        model=get_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=get_env("OPENAI_API_KEY", required=True),
    )


def get_vectorstore() -> PGVector:
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=get_env("PG_COLLECTION", "docs_pdf"),
        connection_string=get_connection_string(),
    )


# -------------------------------------------------------------------
# Search
# -------------------------------------------------------------------
def search_top_k(
    query: str,
    k: int = 10,
) -> List[Tuple[Document, float]]:
    if not query.strip():
        raise ValueError("Query must not be empty")

    vectorstore = get_vectorstore()

    logger.info("Running similarity search (k=%d)", k)
    return vectorstore.similarity_search_with_score(query, k=k)


def format_context(
    results: List[Tuple[Document, float]],
    max_chars: int | None = None,
) -> str:
    """
    Concatenates retrieved document contents into a single context string.
    Optionally truncates to max_chars.
    """
    parts: List[str] = []

    for doc, score in results:
        content = doc.page_content.strip()
        if content:
            parts.append(content)

    context = "\n\n".join(parts)

    if max_chars:
        return context[:max_chars]

    return context


# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()

    query = "Exemplo de pergunta"
    results = search_top_k(query, k=3)

    for doc, score in results:
        print(f"Score: {score:.4f}")
        print(doc.page_content[:200])
        print("-" * 80)
