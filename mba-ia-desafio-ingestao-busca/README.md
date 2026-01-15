# Desafio MBA Engenharia de Software com IA - Full Cycle

# Ingestão e Busca Semântica com LangChain e Postgres

Este projeto lê um PDF, indexa seus conteúdos em um PostgreSQL com pgVector e permite fazer perguntas via CLI, respondendo **apenas** com base no conteúdo do PDF.

## Requisitos

- Python 3.10+
- Docker e Docker Compose
- Chave de API da OpenAI (ou Google Gemini)
- PostgreSQL (via Docker) com extensão pgVector


Como executar a sua solução:


1 - Instalação de dependências
pip install -r requirements.txt

2 - Configuração de variáveis de ambiente
Crie um arquivo .env baseado em .env.example:
cp .env.example .env
Edite PROVIDER para openai ou gemini e preencha as chaves:
- OpenAI:
- OPENAI_API_KEY
- OPENAI_EMBEDDING_MODEL=text-embedding-3-small
- OPENAI_LLM_MODEL=gpt-5-nano
- Gemini:
- GOOGLE_API_KEY
- GEMINI_EMBEDDING_MODEL=models/embedding-001
- GEMINI_LLM_MODEL=gemini-2.5-flash-lite


3 - Subir o banco de dados (Docker Compose)
docker compose up -d
Certifique-se de que a extensão pgvector está disponível (a imagem já inclui). Se necessário:


4 - Ingestão do PDF
Coloque seu arquivo como document.pdf na raiz do projeto.
Execute:python src/ingest.py
- O PDF será dividido em chunks de 1000 caracteres com overlap de 150.
- Embeddings serão gerados (OpenAI ou Gemini).
- Vetores serão armazenados no Postgres (pgVector).


5- Rodar o chat (CLI)
python src/chat.py

Exemplo pergunta dentro do contexto do documento pdf:
PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

Exemplo pergunta fora do contexto do documento pdf:
PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.


Notas:
- Ajuste o nome da coleção via PG_COLLECTION no .env.
- Se a busca não retornar trechos relevantes, o chat responde com fallback.
- temperature do LLM está em 0 para reduzir variação e cumprir as regras do prompt.

Estrutura da Solução:
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── src/
│   ├── ingest.py
│   ├── search.py
│   ├── chat.py
├── document.pdf
└── README.md

Obrservações finais:

- O prompt do chat reforça que a resposta deve usar exclusivamente o CONTEXTO recuperado.  
- Se quiser registrar a origem (arquivo/página), adicione metadados no `PyPDFLoader` e inclua junto ao contexto.  
- Para produção, considere persistência e índices no Postgres, monitoramento de custos de embeddings e logs de consultas.
