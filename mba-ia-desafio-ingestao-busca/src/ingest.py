# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------
import os
import logging
import time
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------------------------------------------------
# Configurações
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_COLLECTION = "docs_pdf"

# -------------------------------------------------------------------
# Utilitários
# -------------------------------------------------------------------
def get_env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and not value:
        raise EnvironmentError(f"Variável de ambiente obrigatória não definida: {name}")
    return value

def get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name, None)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        logger.warning("Valor inválido para %s='%s', usando default %d", name, raw, default)
        return default

# -------------------------------------------------------------------
# String de conexão com o banco de dados postgresql
# -------------------------------------------------------------------
def get_connection_string() -> str:
    return (
        f"postgresql+psycopg://"
        f"{get_env('PG_USER', 'postgres')}:"
        f"{get_env('PG_PASSWORD', 'postgres')}@"
        f"{get_env('PG_HOST', 'localhost')}:"
        f"{get_env('PG_PORT', '5432')}/"
        f"{get_env('PG_DATABASE', 'rag')}"
    )

# -------------------------------------------------------------------
# Embeddings Gemini e OpenAI
# -------------------------------------------------------------------
def get_embeddings():
    provider = get_env("PROVIDER", "openai").lower()
    fallback = get_env("PROVIDER_FALLBACK", "").lower()
    if provider == "gemini":
        model = get_env("GEMINI_EMBEDDING_MODEL", "models/embedding-001")
        logger.info("Usando embeddings Gemini (%s)", model)
        try:
            return GoogleGenerativeAIEmbeddings(
                model=model,
                google_api_key=get_env("GOOGLE_API_KEY", required=True),
            )
        except Exception:
            if fallback == "openai":
                fb_model = get_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
                logger.warning("Falha ao inicializar Gemini, usando OpenAI (%s)", fb_model)
                return OpenAIEmbeddings(
                    model=fb_model,
                    api_key=get_env("OPENAI_API_KEY", required=True),
                    max_retries=int(get_env("EMBEDDING_MAX_RETRIES", "3")),
                )
            raise
    elif provider == "openai":
        model = get_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        logger.info("Usando embeddings OpenAI (%s)", model)
        return OpenAIEmbeddings(
            model=model,
            api_key=get_env("OPENAI_API_KEY", required=True),
            max_retries=int(get_env("EMBEDDING_MAX_RETRIES", "3")),
        )
    else:
        model = get_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        logger.warning("Provider desconhecido '%s', usando OpenAI (%s)", provider, model)
        return OpenAIEmbeddings(
            model=model,
            api_key=get_env("OPENAI_API_KEY", required=True),
            max_retries=int(get_env("EMBEDDING_MAX_RETRIES", "3")),
        )

# -------------------------------------------------------------------
# Leitura de arquivos PDFs
# -------------------------------------------------------------------
def load_pdfs(pdf_paths: List[Path]) -> List[Document]:
    documents: List[Document] = []

    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            logger.warning(f"PDF não encontrado: {pdf_path}")
            continue

        logger.info(f"Carregando PDF: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        documents.extend(loader.load())

    if not documents:
        raise RuntimeError("Nenhum documento carregado.")

    return documents

# -------------------------------------------------------------------
# Split dos documentos PDFs em chunks
# -------------------------------------------------------------------
def split_documents(documents: List[Document], max_chunks: int | None = None) -> List[Document]:
    chunk_size = get_env_int("DEFAULT_CHUNK_SIZE", DEFAULT_CHUNK_SIZE)
    chunk_overlap = get_env_int("DEFAULT_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    logger.info(f"Documentos divididos em {len(chunks)} chunks.")
    return chunks

# -------------------------------------------------------------------
#Ingestão de chunks na base vetorial
# -------------------------------------------------------------------
def ingest_documents(chunks: List[Document]) -> None:
    batch_size = get_env_int("EMBEDDING_BATCH_SIZE", 20)
    pause_s = get_env_int("EMBEDDING_PAUSE_SECONDS", 2)
    collection = get_env("PG_COLLECTION", DEFAULT_COLLECTION)

    # Instanciar PGVector uma única vez para reutilizar a conexão
    vectorstore = PGVector(
        embeddings=get_embeddings(),
        collection_name=collection,
        connection=get_connection_string(),
        use_jsonb=True,
    )

    # Reset total: Dropar tabelas antigas e recriar do zero
    logger.info("Resetando banco de dados (drop tables) para garantir as dimensoes do vetor a ser utilizado pela IA (embeddings)")
    vectorstore.drop_tables()
    
    logger.info("Criando novas tabelas langchain_pg_embedding e langchain_pg_collection")
    vectorstore.create_tables_if_not_exists()
    vectorstore.create_collection()

    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectorstore.add_documents(batch)
        total += len(batch)
        if i + batch_size < len(chunks):
            time.sleep(pause_s)

    logger.info(
        "Ingestão concluída: %d chunks inseridos na coleção '%s'",
        total,
        collection,
    )

# -------------------------------------------------------------------
# Função principal de Ingestão
# -------------------------------------------------------------------
def ingestion() -> None:
    load_dotenv()

    base_dir = Path(__file__).resolve().parent
    pdf_dir = base_dir / "../documents"
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        raise RuntimeError("Nenhum arquivo PDF encontrado no diretório.")

    logger.info("Iniciando ingestão RAG com %d PDFs", len(pdf_files))

    documents = load_pdfs(pdf_files)
    print(documents)
    chunks = split_documents(documents, max_chunks=1000)
    ingest_documents(chunks)

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    ingestion()








