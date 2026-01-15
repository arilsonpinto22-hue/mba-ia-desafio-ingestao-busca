
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


DOCUMENTS_PATH="/documents/"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150

def criar_db():
    documentos = carregar_documentos()
    print(documentos)
    #chunks = dividir_chuncks(documentos)
    #vetorize_chunks(chunks)

def carregar_documentos():
    carregador = PyPDFDirectoryLoader(DOCUMENTS_PATH, glob ="*.pdf") 
    documentos = carregador.load()
    return documentos

#def dividir_chuncks(documentos):
    separador_documentos = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True
    )
    chunks = separador_documentos.split_documents(documentos)
    print("Documentos divididos em {len(chunks)} chunks.",len(chunks))
    return chunks





