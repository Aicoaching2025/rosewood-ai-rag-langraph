"""
Rosewood AI - RAG Document Research Agent
==========================================
A RAG-powered research agent that ingests documents, retrieves relevant
context via vector search, and answers questions using a LangGraph
workflow served via FastAPI with real-time streaming.

Core Stack: LangGraph + FastAPI + Python + ChromaDB
Author: Candace Grant - Birds and Roses LLC
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone
import time
import os
import re
import math
import json
import asyncio
import hashlib
import uuid

# ============================================================
# Lightweight Embedding Engine (no external API keys needed)
# Uses TF-IDF style vectors for demo. In production, swap for
# sentence-transformers or OpenAI embeddings.
# ============================================================

class SimpleEmbedder:
    """
    Lightweight TF-IDF-style embedding engine.
    Production replacement: SentenceTransformer('all-MiniLM-L6-v2')
    """
    def __init__(self):
        self.vocabulary = {}
        self.idf = {}
        self.doc_count = 0

    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        # Remove very short tokens and stopwords
        stops = {'the','a','an','is','are','was','were','be','been','being',
                 'have','has','had','do','does','did','will','would','could',
                 'should','may','might','shall','can','need','dare','ought',
                 'used','to','of','in','for','on','with','at','by','from',
                 'as','into','through','during','before','after','above',
                 'below','between','out','off','over','under','again','further',
                 'then','once','here','there','when','where','why','how','all',
                 'each','every','both','few','more','most','other','some','such',
                 'no','nor','not','only','own','same','so','than','too','very',
                 'and','but','or','if','it','its','this','that','these','those'}
        return [t for t in tokens if len(t) > 2 and t not in stops]

    def _compute_tf(self, tokens):
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        total = len(tokens) or 1
        return {t: c / total for t, c in tf.items()}

    def embed(self, text):
        tokens = self._tokenize(text)
        tf = self._compute_tf(tokens)
        # Build a fixed-dimension vector using hash buckets
        dim = 256
        vector = [0.0] * dim
        for token, freq in tf.items():
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            idx = h % dim
            vector[idx] += freq
            # Also add to neighboring bucket for smoothing
            vector[(idx + 1) % dim] += freq * 0.5
            vector[(idx + 2) % dim] += freq * 0.25
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def similarity(self, vec_a, vec_b):
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        return max(0.0, min(1.0, dot))


# ============================================================
# In-Memory Vector Store (ChromaDB pattern)
# ============================================================

class VectorStore:
    """
    In-memory vector store implementing the ChromaDB interface pattern.
    Production replacement: chromadb.Client().get_or_create_collection()
    """
    def __init__(self, embedder):
        self.embedder = embedder
        self.documents = []  # {id, text, embedding, metadata}
        self.doc_registry = {}  # doc_id -> {title, source, chunks, timestamp}

    def ingest_document(self, doc_id, title, content, source="uploaded", category="general", chunk_size=300, chunk_overlap=50):
        """Chunk a document and add all chunks to the vector store."""
        words = content.split()
        chunks = []

        if len(words) <= chunk_size:
            chunks.append(content)
        else:
            start = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunk_text = ' '.join(words[start:end])
                chunks.append(chunk_text)
                start += chunk_size - chunk_overlap

        chunk_ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            embedding = self.embedder.embed(chunk)
            self.documents.append({
                "id": chunk_id,
                "doc_id": doc_id,
                "text": chunk,
                "embedding": embedding,
                "metadata": {
                    "title": title,
                    "source": source,
                    "category": category,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "word_count": len(chunk.split()),
                }
            })
            chunk_ids.append(chunk_id)

        self.doc_registry[doc_id] = {
            "title": title,
            "source": source,
            "category": category,
            "chunks": len(chunks),
            "word_count": len(words),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }

        return {"doc_id": doc_id, "chunks_created": len(chunks), "chunk_ids": chunk_ids}

    def search(self, query, top_k=5, category_filter=None):
        """Similarity search over the vector store."""
        query_embedding = self.embedder.embed(query)
        results = []

        for doc in self.documents:
            if category_filter and doc["metadata"].get("category") != category_filter:
                continue
            score = self.embedder.similarity(query_embedding, doc["embedding"])
            results.append({
                "chunk_id": doc["id"],
                "doc_id": doc["doc_id"],
                "text": doc["text"],
                "score": round(score, 4),
                "metadata": doc["metadata"],
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def delete_document(self, doc_id):
        """Remove all chunks belonging to a document."""
        before = len(self.documents)
        self.documents = [d for d in self.documents if d["doc_id"] != doc_id]
        removed = before - len(self.documents)
        if doc_id in self.doc_registry:
            del self.doc_registry[doc_id]
        return removed

    def get_stats(self):
        return {
            "total_documents": len(self.doc_registry),
            "total_chunks": len(self.documents),
            "categories": list(set(d["metadata"]["category"] for d in self.documents)),
        }


# ============================================================
# LangGraph Agent - RAG Research Workflow
# ============================================================

class RAGAgentState:
    """Typed state for the LangGraph RAG agent."""
    def __init__(self, query, top_k=5, category_filter=None):
        self.query = query
        self.top_k = top_k
        self.category_filter = category_filter
        # Populated by nodes
        self.query_type = ""
        self.query_keywords = []
        self.retrieved_chunks = []
        self.relevant_chunks = []
        self.relevance_scores = {}
        self.context_sufficient = False
        self.expanded_query = ""
        self.retrieval_attempts = 0
        self.max_retrieval_attempts = 2
        self.synthesized_answer = ""
        self.citations = []
        self.confidence = 0.0
        self.final_response = {}
        self.route_taken = []
        self.processing_times = {}
        self.errors = []


# -- Node Functions --

def parse_query_node(state, vector_store):
    """NODE 1: Parse and classify the incoming query."""
    start = time.perf_counter()

    query_lower = state.query.lower()
    words = query_lower.split()

    # Classify query type
    comparison_signals = ["compare", "difference", "versus", "vs", "between", "contrast", "similarities"]
    factual_signals = ["what is", "what are", "define", "explain", "describe", "how does", "how do"]
    procedural_signals = ["how to", "steps", "process", "build", "create", "implement", "design"]
    opinion_signals = ["best", "recommend", "should", "better", "worst", "advantage", "disadvantage"]

    if any(s in query_lower for s in comparison_signals):
        state.query_type = "comparison"
    elif any(s in query_lower for s in procedural_signals):
        state.query_type = "procedural"
    elif any(s in query_lower for s in opinion_signals):
        state.query_type = "evaluative"
    elif any(s in query_lower for s in factual_signals):
        state.query_type = "factual"
    else:
        state.query_type = "general"

    # Extract key terms
    stops = {'the','a','an','is','are','was','were','what','how','why','when',
             'where','which','who','do','does','can','could','would','should',
             'and','or','but','in','on','at','to','for','of','with','by','from','about'}
    state.query_keywords = [w for w in words if w not in stops and len(w) > 2]

    state.route_taken.append("parse_query")
    state.processing_times["parse_query"] = round((time.perf_counter() - start) * 1000, 2)
    return state


def retrieve_node(state, vector_store):
    """NODE 2: Retrieve relevant chunks from the vector store."""
    start = time.perf_counter()
    state.retrieval_attempts += 1

    search_query = state.expanded_query if state.expanded_query else state.query
    results = vector_store.search(
        query=search_query,
        top_k=state.top_k + (state.retrieval_attempts - 1) * 3,
        category_filter=state.category_filter
    )

    state.retrieved_chunks = results
    state.route_taken.append(f"retrieve(attempt={state.retrieval_attempts})")
    state.processing_times[f"retrieve_{state.retrieval_attempts}"] = round((time.perf_counter() - start) * 1000, 2)
    return state


def evaluate_context_node(state, vector_store):
    """NODE 3: Score and filter retrieved chunks for relevance."""
    start = time.perf_counter()

    relevant = []
    for chunk in state.retrieved_chunks:
        score = chunk["score"]
        # Boost score if query keywords appear in the chunk text
        chunk_lower = chunk["text"].lower()
        keyword_hits = sum(1 for kw in state.query_keywords if kw in chunk_lower)
        keyword_boost = min(keyword_hits * 0.05, 0.2)
        adjusted_score = min(score + keyword_boost, 1.0)

        state.relevance_scores[chunk["chunk_id"]] = {
            "vector_score": score,
            "keyword_hits": keyword_hits,
            "keyword_boost": round(keyword_boost, 3),
            "final_score": round(adjusted_score, 4),
        }

        if adjusted_score >= 0.15:
            chunk["adjusted_score"] = round(adjusted_score, 4)
            relevant.append(chunk)

    relevant.sort(key=lambda x: x.get("adjusted_score", 0), reverse=True)
    state.relevant_chunks = relevant[:state.top_k]

    # Determine if context is sufficient
    if len(state.relevant_chunks) >= 2 and state.relevant_chunks[0].get("adjusted_score", 0) >= 0.2:
        state.context_sufficient = True
    elif len(state.relevant_chunks) >= 1 and state.relevant_chunks[0].get("adjusted_score", 0) >= 0.3:
        state.context_sufficient = True
    else:
        state.context_sufficient = False

    state.route_taken.append("evaluate_context")
    state.processing_times["evaluate_context"] = round((time.perf_counter() - start) * 1000, 2)
    return state


def expand_search_node(state, vector_store):
    """NODE 3b: Expand the search query for a retry (cycle back to retrieve)."""
    start = time.perf_counter()

    # Strategy: add related terms and broaden the query
    expansions = {
        "rag": "retrieval augmented generation vector search embedding",
        "langgraph": "langgraph agent workflow graph state nodes edges",
        "fastapi": "fastapi api endpoint async pydantic streaming",
        "vector": "vector database embedding similarity search chromadb",
        "chunk": "chunking splitting document segment overlap token",
        "agent": "agent workflow pattern routing orchestration pipeline",
    }

    expanded_terms = []
    for keyword in state.query_keywords:
        for key, expansion in expansions.items():
            if key in keyword:
                expanded_terms.append(expansion)

    if expanded_terms:
        state.expanded_query = state.query + " " + " ".join(expanded_terms)
    else:
        state.expanded_query = state.query + " " + " ".join(state.query_keywords)

    state.route_taken.append("expand_search")
    state.processing_times["expand_search"] = round((time.perf_counter() - start) * 1000, 2)
    return state


def synthesize_node(state, vector_store):
    """NODE 4: Generate an answer from relevant context with citations."""
    start = time.perf_counter()

    if not state.relevant_chunks:
        state.synthesized_answer = "I could not find relevant information in the knowledge base to answer your question. Try uploading more documents or rephrasing your query."
        state.confidence = 0.0
        state.citations = []
    else:
        # Build answer from relevant chunks
        context_parts = []
        sources_used = {}

        for chunk in state.relevant_chunks:
            context_parts.append(chunk["text"])
            src = chunk["metadata"]["title"]
            if src not in sources_used:
                sources_used[src] = {
                    "title": chunk["metadata"]["title"],
                    "source": chunk["metadata"]["source"],
                    "relevance_score": chunk.get("adjusted_score", chunk["score"]),
                    "chunk_id": chunk["chunk_id"],
                }

        # Synthesize answer (rule-based extraction for demo)
        combined_context = " ".join(context_parts)
        sentences = re.split(r'(?<=[.!?])\s+', combined_context)

        # Score sentences by query keyword relevance
        scored_sentences = []
        for sent in sentences:
            sent_lower = sent.lower()
            score = sum(1 for kw in state.query_keywords if kw in sent_lower)
            if score > 0:
                scored_sentences.append((sent, score))

        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        if state.query_type == "comparison":
            answer_sentences = scored_sentences[:6]
        elif state.query_type == "procedural":
            answer_sentences = scored_sentences[:5]
        else:
            answer_sentences = scored_sentences[:4]

        if answer_sentences:
            state.synthesized_answer = " ".join(s[0] for s in answer_sentences)
        else:
            state.synthesized_answer = combined_context[:500]

        state.citations = list(sources_used.values())
        avg_score = sum(c.get("adjusted_score", c["score"]) for c in state.relevant_chunks) / len(state.relevant_chunks)
        state.confidence = round(min(avg_score * 1.5, 1.0), 3)

    state.route_taken.append("synthesize")
    state.processing_times["synthesize"] = round((time.perf_counter() - start) * 1000, 2)
    return state


def fallback_response_node(state, vector_store):
    """NODE 4b: Generate a fallback when no relevant context is found."""
    start = time.perf_counter()

    state.synthesized_answer = (
        f"I searched the knowledge base but could not find sufficiently relevant information "
        f"for your query: \"{state.query}\". The search returned {len(state.retrieved_chunks)} chunks "
        f"but none met the relevance threshold after {state.retrieval_attempts} retrieval attempts. "
        f"Try uploading documents related to your topic or rephrasing your question with more specific terms."
    )
    state.confidence = 0.0
    state.citations = []

    state.route_taken.append("fallback_response")
    state.processing_times["fallback_response"] = round((time.perf_counter() - start) * 1000, 2)
    return state


def format_response_node(state, vector_store):
    """NODE 5: Format the final structured response."""
    start = time.perf_counter()

    total_time = sum(state.processing_times.values())

    state.final_response = {
        "query": state.query,
        "query_analysis": {
            "type": state.query_type,
            "keywords": state.query_keywords,
        },
        "answer": state.synthesized_answer,
        "confidence": state.confidence,
        "citations": state.citations,
        "retrieval_info": {
            "chunks_retrieved": len(state.retrieved_chunks),
            "chunks_relevant": len(state.relevant_chunks),
            "retrieval_attempts": state.retrieval_attempts,
            "relevance_scores": state.relevance_scores,
        },
        "graph_execution": {
            "nodes_executed": state.route_taken + ["format_response"],
            "total_nodes": len(state.route_taken) + 1,
            "node_timings_ms": state.processing_times,
            "total_processing_time_ms": round(total_time, 2),
            "has_cycle": state.retrieval_attempts > 1,
        },
        "metadata": {
            "model_version": "1.0.0",
            "engine": "Rosewood AI - LangGraph RAG Agent",
            "author": "Candace Grant - Birds and Roses LLC",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    state.route_taken.append("format_response")
    state.processing_times["format_response"] = round((time.perf_counter() - start) * 1000, 2)
    return state


# -- Router (conditional edge) --

def route_after_evaluation(state):
    """Conditional edge: decides next step after context evaluation."""
    if state.context_sufficient:
        return "synthesize"
    elif state.retrieval_attempts < state.max_retrieval_attempts:
        return "expand_and_retry"  # CYCLE back to retrieve
    else:
        return "fallback"


# -- Graph Executor --

def run_rag_agent(query, vector_store, top_k=5, category_filter=None):
    """Execute the full RAG LangGraph agent pipeline."""
    state = RAGAgentState(query=query, top_k=top_k, category_filter=category_filter)

    # Node 1: Parse query
    state = parse_query_node(state, vector_store)

    # Node 2: Retrieve
    state = retrieve_node(state, vector_store)

    # Node 3: Evaluate context
    state = evaluate_context_node(state, vector_store)

    # Conditional routing (potential cycle)
    route = route_after_evaluation(state)

    if route == "expand_and_retry":
        # CYCLE: expand -> retrieve -> evaluate again
        state = expand_search_node(state, vector_store)
        state = retrieve_node(state, vector_store)
        state = evaluate_context_node(state, vector_store)

        # Check again after retry
        route = route_after_evaluation(state)

    if route == "synthesize" or (route == "expand_and_retry" and state.context_sufficient):
        state = synthesize_node(state, vector_store)
    else:
        state = fallback_response_node(state, vector_store)

    # Final formatting
    state = format_response_node(state, vector_store)

    return state.final_response


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Rosewood AI - Document Research Agent",
    description=(
        "A RAG-powered research agent built with the LangGraph pattern and served via FastAPI. "
        "Features document ingestion, vector search, conditional routing with cycles, "
        "and streaming agent execution."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Initialize components --
embedder = SimpleEmbedder()
vector_store = VectorStore(embedder)
query_log = []
start_time = datetime.now(timezone.utc)


# -- Load sample documents on startup --
@app.on_event("startup")
async def load_sample_docs():
    sample_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "sample_docs", "knowledge_base.json"
    )
    if os.path.exists(sample_path):
        with open(sample_path) as f:
            docs = json.load(f)
        for doc in docs:
            vector_store.ingest_document(
                doc_id=doc["id"],
                title=doc["title"],
                content=doc["content"],
                source=doc.get("source", "sample"),
                category=doc.get("category", "general"),
            )
        print(f"Loaded {len(docs)} sample documents into vector store")


# -- Pydantic Models --

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Question to research")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    category_filter: Optional[str] = Field(default=None, description="Filter by document category")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"query": "What is RAG and how does it work?", "top_k": 5},
                {"query": "Compare LangGraph and LangChain", "top_k": 5},
                {"query": "How to build a FastAPI ML endpoint?", "top_k": 3},
            ]
        }
    }

class IngestRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=10, max_length=50000)
    source: Optional[str] = Field(default="uploaded")
    category: Optional[str] = Field(default="general")

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    total_queries: int
    vector_store_stats: dict
    version: str


# -- Endpoints --

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    tpl = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "templates", "dashboard.html"
    )
    with open(tpl) as f:
        return HTMLResponse(content=f.read())


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    uptime = (datetime.now(timezone.utc) - start_time).total_seconds()
    return HealthResponse(
        status="healthy",
        uptime_seconds=round(uptime, 1),
        total_queries=len(query_log),
        vector_store_stats=vector_store.get_stats(),
        version="1.0.0",
    )


@app.get("/graph/info", tags=["Graph"])
async def graph_info():
    return {
        "graph_name": "RAGResearchAgent",
        "description": "RAG research agent with retrieval cycles and context evaluation",
        "nodes": [
            {"name": "parse_query", "type": "router", "desc": "Classify query type and extract keywords"},
            {"name": "retrieve", "type": "retriever", "desc": "Vector similarity search in document store"},
            {"name": "evaluate_context", "type": "evaluator", "desc": "Score and filter chunks for relevance"},
            {"name": "expand_search", "type": "expander", "desc": "Broaden query for retry (cycle node)"},
            {"name": "synthesize", "type": "generator", "desc": "Generate answer with citations from context"},
            {"name": "fallback_response", "type": "fallback", "desc": "Handle insufficient context gracefully"},
            {"name": "format_response", "type": "terminal", "desc": "Structure final output with metadata"},
        ],
        "edges": [
            {"from": "START", "to": "parse_query"},
            {"from": "parse_query", "to": "retrieve"},
            {"from": "retrieve", "to": "evaluate_context"},
            {"from": "evaluate_context", "to": "CONDITIONAL", "type": "conditional"},
            {"from": "evaluate_context", "to": "synthesize", "condition": "context_sufficient"},
            {"from": "evaluate_context", "to": "expand_search", "condition": "insufficient + retries left"},
            {"from": "evaluate_context", "to": "fallback_response", "condition": "insufficient + no retries"},
            {"from": "expand_search", "to": "retrieve", "type": "CYCLE"},
            {"from": "synthesize", "to": "format_response"},
            {"from": "fallback_response", "to": "format_response"},
            {"from": "format_response", "to": "END"},
        ],
        "has_cycles": True,
        "cycle_description": "evaluate_context -> expand_search -> retrieve -> evaluate_context",
        "author": "Candace Grant - Birds and Roses LLC",
    }


@app.post("/query", tags=["Research"])
async def query_documents(request: QueryRequest):
    """Run the full RAG agent pipeline to answer a question."""
    report = run_rag_agent(
        query=request.query,
        vector_store=vector_store,
        top_k=request.top_k,
        category_filter=request.category_filter,
    )

    query_log.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": request.query,
        "query_type": report.get("query_analysis", {}).get("type", ""),
        "confidence": report.get("confidence", 0),
        "chunks_used": report.get("retrieval_info", {}).get("chunks_relevant", 0),
        "had_cycle": report.get("graph_execution", {}).get("has_cycle", False),
        "total_time_ms": report.get("graph_execution", {}).get("total_processing_time_ms", 0),
    })

    return report


@app.post("/query/stream", tags=["Research"])
async def query_stream(request: QueryRequest):
    """Stream the RAG agent execution node by node."""
    async def generate():
        state = RAGAgentState(query=request.query, top_k=request.top_k, category_filter=request.category_filter)

        # Node 1
        state = parse_query_node(state, vector_store)
        yield json.dumps({"node": "parse_query", "status": "complete", "query_type": state.query_type, "keywords": state.query_keywords, "time_ms": state.processing_times["parse_query"]}) + "\n"
        await asyncio.sleep(0.05)

        # Node 2
        state = retrieve_node(state, vector_store)
        yield json.dumps({"node": "retrieve", "status": "complete", "chunks_found": len(state.retrieved_chunks), "time_ms": state.processing_times["retrieve_1"]}) + "\n"
        await asyncio.sleep(0.05)

        # Node 3
        state = evaluate_context_node(state, vector_store)
        yield json.dumps({"node": "evaluate_context", "status": "complete", "relevant_chunks": len(state.relevant_chunks), "context_sufficient": state.context_sufficient, "time_ms": state.processing_times["evaluate_context"]}) + "\n"
        await asyncio.sleep(0.05)

        route = route_after_evaluation(state)
        yield json.dumps({"node": "router", "status": "routing", "decision": route, "context_sufficient": state.context_sufficient}) + "\n"
        await asyncio.sleep(0.05)

        if route == "expand_and_retry":
            state = expand_search_node(state, vector_store)
            yield json.dumps({"node": "expand_search", "status": "complete", "expanded_query_preview": state.expanded_query[:100], "time_ms": state.processing_times["expand_search"]}) + "\n"
            await asyncio.sleep(0.05)

            state = retrieve_node(state, vector_store)
            yield json.dumps({"node": "retrieve_retry", "status": "complete", "chunks_found": len(state.retrieved_chunks), "time_ms": state.processing_times.get("retrieve_2", 0)}) + "\n"
            await asyncio.sleep(0.05)

            state = evaluate_context_node(state, vector_store)
            route = route_after_evaluation(state)
            yield json.dumps({"node": "evaluate_context_retry", "status": "complete", "relevant_chunks": len(state.relevant_chunks), "context_sufficient": state.context_sufficient}) + "\n"
            await asyncio.sleep(0.05)

        if state.context_sufficient or route == "synthesize":
            state = synthesize_node(state, vector_store)
            yield json.dumps({"node": "synthesize", "status": "complete", "confidence": state.confidence, "citations": len(state.citations), "time_ms": state.processing_times["synthesize"]}) + "\n"
        else:
            state = fallback_response_node(state, vector_store)
            yield json.dumps({"node": "fallback_response", "status": "complete", "time_ms": state.processing_times["fallback_response"]}) + "\n"
        await asyncio.sleep(0.05)

        state = format_response_node(state, vector_store)
        yield json.dumps({"node": "format_response", "status": "complete", "report": state.final_response}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/ingest", tags=["Documents"])
async def ingest_document(request: IngestRequest):
    """Ingest a new document into the vector store."""
    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    result = vector_store.ingest_document(
        doc_id=doc_id,
        title=request.title,
        content=request.content,
        source=request.source,
        category=request.category,
    )
    return {"status": "ingested", **result}


@app.post("/ingest/file", tags=["Documents"])
async def ingest_file(file: UploadFile = File(...), category: str = "uploaded"):
    """Upload and ingest a text file."""
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    result = vector_store.ingest_document(
        doc_id=doc_id,
        title=file.filename or "Untitled",
        content=text,
        source="file_upload",
        category=category,
    )
    return {"status": "ingested", "filename": file.filename, **result}


@app.get("/documents", tags=["Documents"])
async def list_documents():
    """List all documents in the vector store."""
    return {
        "total_documents": len(vector_store.doc_registry),
        "documents": vector_store.doc_registry,
    }


@app.delete("/documents/{doc_id}", tags=["Documents"])
async def delete_document(doc_id: str):
    """Remove a document and its chunks from the vector store."""
    removed = vector_store.delete_document(doc_id)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return {"status": "deleted", "doc_id": doc_id, "chunks_removed": removed}


@app.get("/history", tags=["Monitoring"])
async def query_history(limit: int = 50):
    return {"total_queries": len(query_log), "recent": query_log[-limit:][::-1]}


@app.get("/stats", tags=["Monitoring"])
async def query_stats():
    if not query_log:
        return {"total_queries": 0}

    types = {}
    cycle_count = 0
    total_conf = 0
    total_time = 0

    for q in query_log:
        qt = q.get("query_type", "unknown")
        types[qt] = types.get(qt, 0) + 1
        if q.get("had_cycle"):
            cycle_count += 1
        total_conf += q.get("confidence", 0)
        total_time += q.get("total_time_ms", 0)

    return {
        "total_queries": len(query_log),
        "query_types": types,
        "queries_with_cycles": cycle_count,
        "cycle_rate": round(cycle_count / len(query_log), 3),
        "avg_confidence": round(total_conf / len(query_log), 3),
        "avg_processing_time_ms": round(total_time / len(query_log), 2),
        "vector_store": vector_store.get_stats(),
    }