# Production RAG System with Evaluation Framework

A production-grade Retrieval-Augmented Generation (RAG) system built with FastAPI, pgvector, HuggingFace embeddings, BM25 hybrid retrieval, RAGAS evaluation, LangSmith observability, and deployed to GCP Cloud Run via Terraform + GitHub Actions CI/CD.

**Live demo:** https://rag-api-rp6ga3bkwa-uc.a.run.app/docs

## Architecture

```
                        GitHub Actions CI/CD
                               │
                               ▼
Internet ──▶ Cloud Run (FastAPI) ──▶ Cloud SQL (PostgreSQL + pgvector)
                    │           ──▶ Memorystore (Redis)
                    └──────────────▶ LangSmith (traces)
```

### Data Flow — Ingestion
```
POST /ingest (file upload)
  → Document Loader (PDF / plain text)
  → Chunker (fixed-size or semantic)
  → Embedder (BAAI/bge-large-en-v1.5, 1024-dim)
  → pgvector store + BM25 index update
  → Invalidate Redis cache
  → LangSmith trace
```

### Data Flow — Query
```
POST /query { query, retrieval_mode, top_k }
  → Redis cache check
  → Dense: pgvector cosine similarity
  → Sparse: BM25 term frequency
  → Hybrid: RRF fusion (dense rank + sparse rank)
  → LangSmith trace
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI + uvicorn |
| Embeddings | BAAI/bge-large-en-v1.5 (HuggingFace) |
| Vector Store | PostgreSQL 16 + pgvector |
| Sparse Retrieval | BM25 (rank-bm25) |
| Hybrid Fusion | Reciprocal Rank Fusion (RRF) |
| Evaluation | RAGAS (faithfulness, relevancy, recall, precision) |
| Observability | LangSmith tracing |
| Caching | Redis |
| Infrastructure | GCP Cloud Run + Cloud SQL + Memorystore |
| IaC | Terraform |
| CI/CD | GitHub Actions |
| Testing | pytest + pytest-asyncio |

## Quick Start (Local)

### Prerequisites
- Docker and Docker Compose

### 1. Clone the repository
```bash
git clone https://github.com/PKMadhavan/Production-RAG-System-with-Evaluation-Framework.git
cd Production-RAG-System-with-Evaluation-Framework
```

### 2. Configure environment
```bash
cp .env.example .env
# Add your OPENAI_API_KEY and LANGSMITH_API_KEY (both optional)
```

### 3. Start all services
```bash
docker compose up --build
```

> **Note:** First startup downloads the embedding model (~1.3 GB). Subsequent starts use the cached model.

### 4. Verify
```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "version": "0.1.0",
  "postgres_connected": true,
  "redis_connected": true,
  "langsmith_connected": true
}
```

## API Endpoints

Interactive docs available at `http://localhost:8000/docs`

### GET /health
Health check with dependency status.

### POST /ingest
```bash
curl -X POST http://localhost:8000/ingest/ \
  -F "file=@document.pdf" \
  -F "chunking_strategy=fixed"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| file | File | required | PDF or plain text |
| chunking_strategy | string | "fixed" | "fixed" or "semantic" |
| chunk_size | int | 512 | Characters per chunk |
| chunk_overlap | int | 50 | Overlap between chunks |

### POST /query
```bash
curl -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "retrieval_mode": "hybrid",
    "top_k": 5
  }'
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| query | string | required | Search query |
| retrieval_mode | string | "hybrid" | "dense", "sparse", or "hybrid" |
| top_k | int | 5 | Number of results (1-50) |
| score_threshold | float | 0.0 | Minimum similarity score |

### POST /evaluate
```bash
curl -X POST http://localhost:8000/evaluate/ \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [{"question": "What is RAG?"}],
    "retrieval_mode": "hybrid"
  }'
```

Returns RAGAS metrics: faithfulness, answer_relevancy, context_recall, context_precision.

## GCP Deployment

### Prerequisites
- [gcloud CLI](https://cloud.google.com/sdk/docs/install) installed and authenticated
- [Terraform](https://developer.hashicorp.com/terraform/downloads) >= 1.5 installed
- A GCP project with billing enabled

### Step 1 — Create GCS state bucket
```bash
export PROJECT_ID=your-gcp-project-id
gsutil mb -p $PROJECT_ID gs://${PROJECT_ID}-tf-state
```

### Step 2 — Configure Terraform
```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your project_id, db_password, and API keys
```

Update `main.tf` backend bucket:
```hcl
backend "gcs" {
  bucket = "your-project-id-tf-state"
  prefix = "rag-api"
}
```

### Step 3 — Provision infrastructure
```bash
terraform init
terraform plan
terraform apply
```

This creates: Cloud SQL, Memorystore, Artifact Registry, Cloud Run, VPC, secrets.

### Step 4 — Set up GitHub Actions secrets

In your GitHub repo → Settings → Secrets → Actions, add:

| Secret | Value |
|--------|-------|
| `GCP_PROJECT_ID` | Your GCP project ID |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | From `terraform output` (or set up manually) |
| `GCP_SERVICE_ACCOUNT` | `rag-api-sa@YOUR_PROJECT.iam.gserviceaccount.com` |

### Step 5 — Push to deploy
```bash
git push origin main
```

GitHub Actions will: run tests → build Docker image → push to Artifact Registry → deploy to Cloud Run.

## Project Structure

```
├── src/
│   ├── config.py                    # Environment-based settings
│   ├── models/
│   │   ├── database.py              # SQLAlchemy models + pgvector
│   │   └── schemas.py               # Pydantic request/response models
│   ├── ingestion/
│   │   ├── document_loader.py       # PDF + text extraction
│   │   ├── chunking.py              # Fixed-size + semantic chunking
│   │   └── pipeline.py              # Ingestion orchestrator
│   ├── retrieval/
│   │   ├── embeddings.py            # HuggingFace embedding service
│   │   ├── vector_store.py          # pgvector cosine similarity
│   │   ├── bm25_store.py            # In-memory BM25 index
│   │   └── retriever.py             # Hybrid retrieval + RRF fusion
│   ├── evaluation/
│   │   └── evaluator.py             # RAGAS evaluation pipeline
│   ├── observability/
│   │   └── tracing.py               # LangSmith tracing service
│   └── api/
│       ├── main.py                  # FastAPI app + lifespan
│       ├── dependencies.py          # Dependency injection
│       └── routes/
│           ├── health.py            # GET /health
│           ├── ingest.py            # POST /ingest
│           ├── query.py             # POST /query
│           └── evaluate.py          # POST /evaluate
├── terraform/                       # GCP infrastructure as code
├── .github/workflows/deploy.yml     # CI/CD pipeline
├── tests/                           # pytest test suite
├── docker-compose.yml               # Local multi-service setup
├── Dockerfile                       # Production container
└── .env.example                     # Environment template
```

## Running Tests
```bash
pip install -r requirements.txt
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| POSTGRES_HOST | localhost | PostgreSQL host |
| REDIS_HOST | localhost | Redis host |
| EMBEDDING_MODEL | BAAI/bge-large-en-v1.5 | HuggingFace model |
| BM25_K_PARAM | 60 | RRF fusion constant |
| OPENAI_API_KEY | — | For RAGAS evaluation (optional) |
| LANGSMITH_API_KEY | — | For tracing (optional) |
| LANGSMITH_PROJECT | rag-api | LangSmith project name |

See `.env.example` for the full list.

## Roadmap

- [x] **Phase 1** — Core scaffold: FastAPI, ingestion, dense retrieval, Docker
- [x] **Phase 2** — BM25 hybrid retrieval (RRF) + RAGAS evaluation pipeline
- [x] **Phase 3** — LangSmith observability and tracing on all endpoints
- [x] **Phase 4** — GCP deployment: Cloud Run + Cloud SQL + Terraform + CI/CD

## License

MIT
