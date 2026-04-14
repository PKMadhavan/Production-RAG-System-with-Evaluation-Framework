# Production RAG System with Evaluation Framework

A production-grade Retrieval-Augmented Generation (RAG) system built with FastAPI, pgvector, and HuggingFace embeddings. Features document ingestion, hybrid chunking strategies, dense vector retrieval, Redis caching, and full Docker containerization.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Client     │────▶│  FastAPI App  │────▶│  PostgreSQL +   │
│  (REST API)  │◀────│  (uvicorn)   │◀────│   pgvector      │
└─────────────┘     └──────┬───────┘     └─────────────────┘
                           │
                    ┌──────┴───────┐
                    │    Redis     │
                    │   (cache)    │
                    └──────────────┘
```

### Data Flow — Ingestion
```
POST /ingest (file upload)
  → Document Loader (PDF via pypdf / plain text)
  → Chunking (fixed-size or semantic)
  → Embedding (BAAI/bge-large-en-v1.5, 1024-dim)
  → Store in pgvector
  → Invalidate query cache
```

### Data Flow — Query
```
POST /query { query, top_k, score_threshold }
  → Check Redis cache
  → Embed query
  → Cosine similarity search (pgvector)
  → Return ranked chunks + scores
  → Cache result
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI + uvicorn |
| Embeddings | BAAI/bge-large-en-v1.5 (HuggingFace) |
| Vector Store | PostgreSQL + pgvector |
| Caching | Redis |
| Document Processing | pypdf, LangChain text splitters |
| Containerization | Docker + Docker Compose |
| Testing | pytest + pytest-asyncio |

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Git

### 1. Clone the repository
```bash
git clone https://github.com/PKMadhavan/Production-RAG-System-with-Evaluation-Framework.git
cd Production-RAG-System-with-Evaluation-Framework
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env if you want to change defaults (optional)
```

### 3. Start all services
```bash
docker compose up --build
```

This starts:
- **app** — FastAPI server on port 8000
- **postgres** — PostgreSQL 16 with pgvector on port 5432
- **redis** — Redis 7 on port 6379

> **Note:** The first startup downloads the embedding model (~1.3GB). Subsequent starts use the cached model.

### 4. Verify
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "version": "0.1.0",
  "postgres_connected": true,
  "redis_connected": true
}
```

## API Endpoints

### GET /health
Health check with dependency status.

### POST /ingest
Ingest a document into the vector store.

```bash
# Ingest a text file
curl -X POST http://localhost:8000/ingest/ \
  -F "file=@document.txt" \
  -F "chunking_strategy=fixed" \
  -F "chunk_size=512" \
  -F "chunk_overlap=50"

# Ingest a PDF
curl -X POST http://localhost:8000/ingest/ \
  -F "file=@paper.pdf" \
  -F "chunking_strategy=semantic"
```

**Parameters:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| file | File | required | PDF or plain text file |
| chunking_strategy | string | "fixed" | "fixed" or "semantic" |
| chunk_size | int | 512 | Characters per chunk (100-2000) |
| chunk_overlap | int | 50 | Overlap between chunks |

### POST /query
Query the RAG system.

```bash
curl -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "top_k": 5,
    "score_threshold": 0.3
  }'
```

**Parameters:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| query | string | required | The search query (1-1000 chars) |
| top_k | int | 5 | Number of results to return (1-50) |
| score_threshold | float | 0.0 | Minimum similarity score (0.0-1.0) |

## Project Structure

```
├── src/
│   ├── config.py                 # Environment-based settings
│   ├── models/
│   │   ├── database.py           # SQLAlchemy models + pgvector
│   │   └── schemas.py            # Pydantic request/response models
│   ├── ingestion/
│   │   ├── document_loader.py    # PDF + text extraction
│   │   ├── chunking.py           # Fixed-size + semantic chunking
│   │   └── pipeline.py           # Ingestion orchestrator
│   ├── retrieval/
│   │   ├── embeddings.py         # HuggingFace embedding service
│   │   ├── vector_store.py       # pgvector similarity search
│   │   └── retriever.py          # Query orchestrator with caching
│   └── api/
│       ├── main.py               # FastAPI app factory
│       ├── dependencies.py       # Dependency injection
│       └── routes/
│           ├── health.py         # GET /health
│           ├── ingest.py         # POST /ingest
│           └── query.py          # POST /query
├── tests/                        # pytest test suite
├── docker-compose.yml            # Multi-service setup
├── Dockerfile                    # App container
├── requirements.txt              # Python dependencies
└── .env.example                  # Environment template
```

## Development

### Local Setup (without Docker)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start PostgreSQL and Redis locally, then:
cp .env.example .env
# Edit .env: set POSTGRES_HOST=localhost, REDIS_HOST=localhost

uvicorn src.api.main:app --reload --port 8000
```

### Running Tests
```bash
pip install -r requirements.txt
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

## Configuration

All settings are configurable via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| APP_ENV | development | Environment (development/production) |
| LOG_LEVEL | INFO | Logging level |
| POSTGRES_HOST | localhost | PostgreSQL host |
| POSTGRES_PORT | 5432 | PostgreSQL port |
| POSTGRES_USER | raguser | Database user |
| POSTGRES_PASSWORD | changeme | Database password |
| POSTGRES_DB | ragdb | Database name |
| REDIS_HOST | localhost | Redis host |
| REDIS_PORT | 6379 | Redis port |
| EMBEDDING_MODEL | BAAI/bge-large-en-v1.5 | HuggingFace model |
| EMBEDDING_DIMENSION | 1024 | Vector dimension |
| CHUNK_SIZE | 512 | Default chunk size |
| CHUNK_OVERLAP | 50 | Default chunk overlap |
| MAX_FILE_SIZE_MB | 50 | Max upload size |

## Roadmap

- [x] **Phase 1** — Core scaffold, ingestion, dense retrieval, Docker
- [ ] **Phase 2** — RAGAS evaluation pipeline, BM25 hybrid retrieval
- [ ] **Phase 3** — LangSmith observability and tracing
- [ ] **Phase 4** — AWS deployment (ECS + RDS)

## License

MIT
