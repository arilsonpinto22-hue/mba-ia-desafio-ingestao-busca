import os
import logging
from typing import Optional

from dotenv import load_dotenv
from search import search_top_k, format_context

# LLMs
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
FALLBACK = "Não tenho informações necessárias para responder sua pergunta."
DEFAULT_TOP_K = 10
MAX_CONTEXT_CHARS = 6000

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


def get_llm():
    provider = get_env("PROVIDER", "openai").lower()

    if provider == "gemini":
        logger.info("Using Gemini LLM")
        return ChatGoogleGenerativeAI(
            model=get_env("GEMINI_LLM_MODEL", "gemini-2.5-flash-lite"),
            google_api_key=get_env("GOOGLE_API_KEY", required=True),
            temperature=0,
        )

    logger.info("Using OpenAI LLM")
    return ChatOpenAI(
        model=get_env("OPENAI_LLM_MODEL", "gpt-5-nano"),
        api_key=get_env("OPENAI_API_KEY", required=True),
        temperature=0,
    )


# -------------------------------------------------------------------
# Prompt
# -------------------------------------------------------------------
def build_prompt(context: str, question: str) -> str:
    return f"""
CONTEXTO:
{context}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda exatamente:
  "{FALLBACK}"
- Nunca invente, deduza ou use conhecimento externo.
- Nunca produza opiniões ou interpretações.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: Qual é a capital da França?
Resposta: "{FALLBACK}"

Pergunta: Quantos clientes temos em 2024?
Resposta: "{FALLBACK}"

Pergunta: Você acha isso bom ou ruim?
Resposta: "{FALLBACK}"

PERGUNTA DO USUÁRIO:
{question}

RESPONDA À PERGUNTA DO USUÁRIO:
""".strip()


# -------------------------------------------------------------------
# Core logic
# -------------------------------------------------------------------
def answer_question(question: str) -> str:
    if not question.strip():
        return FALLBACK

    results = search_top_k(question, k=DEFAULT_TOP_K)

    if not results:
        logger.info("No documents retrieved")
        return FALLBACK

    context = format_context(results, max_chars=MAX_CONTEXT_CHARS).strip()

    if not context:
        logger.info("Empty context after formatting")
        return FALLBACK

    llm = get_llm()
    prompt = build_prompt(context, question)

    messages = [
        SystemMessage(
            content="Você é um assistente que segue estritamente as regras fornecidas."
        ),
        HumanMessage(content=prompt),
    ]

    try:
        response = llm.invoke(messages)
        answer = response.content.strip()
    except Exception as exc:
        logger.exception("LLM invocation failed")
        return FALLBACK

    return answer if answer else FALLBACK


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
def main():
    load_dotenv()
    logger.info("RAG CLI started (CTRL+C to exit)")

    try:
        while True:
            question = input("\nPERGUNTA: ").strip()
            if not question:
                continue

            answer = answer_question(question)
            print(f"\nRESPOSTA: {answer}")
    except KeyboardInterrupt:
        print("\nEncerrado.")


if __name__ == "__main__":
    main()
