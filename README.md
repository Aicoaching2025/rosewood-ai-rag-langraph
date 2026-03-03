# Rosewood AI - Document Research Agent

**Author:** Candace Grant - Birds and Roses LLC
**Core Stack:** LangGraph + FastAPI + Python + ChromaDB

A RAG-powered research agent that ingests documents, retrieves relevant context via vector search, evaluates relevance, and synthesizes answers with cited sources - built using the LangGraph pattern with retrieval cycles and served via FastAPI.

## Quick Start

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Dashboard: http://localhost:8000 | Swagger: http://localhost:8000/docs

6 sample documents pre-loaded on startup.

## LangGraph Architecture

```
START -> parse_query -> retrieve -> evaluate_context -> CONDITIONAL
                                                         |
                          sufficient -> synthesize -> format_response -> END
                          insufficient -> expand_search -> retrieve (CYCLE)
                          no_retries -> fallback_response -> format_response -> END
```

Key: The graph has a CYCLE - if context is insufficient, the agent expands the query and loops back. This is impossible with linear chains.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /query | Full RAG agent pipeline with citations |
| POST | /query/stream | Streaming node-by-node (NDJSON) |
| POST | /ingest | Add document to knowledge base |
| POST | /ingest/file | Upload text file |
| GET | /documents | List indexed documents |
| DELETE | /documents/{id} | Remove document |
| GET | /graph/info | Graph topology with cycle info |
| GET | /health | System health + vector store stats |
| GET | /stats | Query analytics |

## Example

```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={"query": "What is RAG?", "top_k": 5}
)
report = response.json()
print(report["answer"])
print(report["citations"])
print(report["graph_execution"]["has_cycle"])
```
## Project Summary

### RAG Pipeline
- Document chunking with configurable size/overlap
- Hybrid relevance scoring: vector similarity + keyword boost
- Context evaluation gate filters low-relevance chunks
- Citation tracking ties answers to source documents

### LangGraph Cycles
- evaluate -> expand -> retrieve -> evaluate (CYCLE)
- Agent decides: retry vs answer vs fallback
- Conditional edges route on context sufficiency

### FastAPI
- NDJSON streaming for agent transparency
- File upload with python-multipart
- Pydantic V2 validation
- Auto-generated Swagger docs
