# Case Study: Vector Search Implementation for LLM AI Engineering Module 2

**Date:** July 1st, 2026  
**Prepared by:** Amah Martin  
**Module:** LLM AI Engineering Module 2 (Vector Search)  

---

## 🔍 Investigation Summary

This case study explores my journey from building a basic Agentic RAG system in Module 1 to adding semantic search capabilities through vector embeddings. The central question driving this work: **How do I move beyond keyword-only retrieval to enable systems that truly "understand" what users are asking?**

Proposed answer: A hybrid architecture combining ONNX-based vector embeddings with traditional text indexing, fused through Reciprocal Rank Fusion (RRF).

---

## 🎯 Problem Statement

In Module 1's Agentic RAG implementation, I successfully built a system that could search and retrieve course content using keyword matching. However, this approach has inherent limitations:

- **Lexical vs Semantic:** Keyword search finds exact term matches but misses conceptual relationships
- **User Intent:** A user asking "How does approximate nearest neighbor search work?" wouldn't necessarily use those exact words in a document
- **Knowledge Access:** I need retrieval that understands meaning, not just text overlap

**Goal:** Build a vector search layer while maintaining the benefits of text-based keyword search.

---

## 🛠️ Implementation Architecture

### Technology Stack

| Component | Library/Tool | Purpose |
|-----------|--------------|---------|
| Embeddings | `Xenova/all-MiniLM-L6-v2` (ONNX) | 384-dim vector encoding |
| Vector Search | `minsearch.VectorSearch` | Similarity-based retrieval |
| Text Index | `minsearch.Index` | Keyword matching |
| Fusion Method | Reciprocal Rank Fusion (RRF) | Combine results |
| Package Manager | `uv` | Isolated dependency management |

### Environment Setup

```bash
# Clean slate project initialization
uv init --no-workspace

# Core dependencies for vector operations
uv add onnxruntime tokenizers numpy tqdm minsearch gitsource

# Development dependencies  
uv add --dev huggingface-hub jupyter
```

This lean setup demonstrates that vector search doesn't require heavy infrastructure—ONNX Runtime provides CPU-based inference that's accessible even without GPU resources.

---

### Environment Setup

```bash
# Clean slate project initialization
uv init --no-workspace

# Core dependencies for vector operations
uv add onnxruntime tokenizers numpy tqdm minsearch gitsource

# Development dependencies  
uv add --dev huggingface-hub jupyter
```

This lean setup demonstrates that vector search doesn't require heavy infrastructure—ONNX Runtime provides CPU-based inference that's accessible even without GPU resources.

---

## 📦 Data Preparation Pipeline

Before I could run any embedding or search operations, I needed to prepare the raw lesson content from the course repository. Here's how I structured the data pipeline:

### Step 1: Repository Fetching via Gitsource

I used `gitsource` library to pull all markdown files from the DataTalksClub/llm-zoomcamp GitHub repository at commit `8c1834d`. The fetching was filtered by:
- Extension filter: Only `.md` files
- Path filter: Only files containing `/lessons/` in their path

```python
from gitsource import GithubRepositoryDataReader

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

files = reader.read()
```

This returned **72 lesson pages** across the seven course modules.

### Step 2: Parsing Documents

Each fetched file was parsed into a document object containing:
- `filename` – The original markdown file path
- `content` – The full text content of the page

```python
documents = []
for file in files:
    doc = file.parse()
    documents.append(doc)

# Total documents loaded: 72
```

### Step 3: Chunking Strategy

Long documents hurt retrieval precision because a match deep inside a page still pulls in the whole page. To fix this, I implemented a sliding window chunking strategy:

**Parameters:**
- **Chunk size:** 2000 characters
- **Step (overlap):** 1000 characters (50% overlap)
- **Result:** 295 chunks from 72 lesson files

```python
from gitsource import chunk_documents

chunks = chunk_documents(documents, size=2000, step=1000)

# Total chunks created: 295
```

This approach:
- Increases search granularity by splitting pages into smaller pieces
- Preserves contextual continuity through overlap
- Reduces token overhead in subsequent RAG operations (by ~3× vs whole-page indexing)

### Step 4: Embedding Model Downloading

Before encoding, I needed to fetch the embedding model from HuggingFace. I created a dedicated `download.py` script that pulls the ONNX-compatible model:

```python
# Using uv run python download.py
# Downloads: Xenova/all-MiniLM-L6-v2 from huggingface
```

This model outputs 384-dimensional vectors and is CPU-compatible via ONNX Runtime.

### Step 5: Embedder Setup

I built an `embedder.py` script that:
- Initializes the `Embedder` class wrapping ONNX Runtime
- Provides single-query encoding (`encode()`)
- Supports batch encoding for chunks (`encode_batch()`)

```python
from embedder import Embedder

embed = Embedder()
# Returns normalized 384-dim vectors
v1 = embed.encode("How does approximate nearest neighbor search work?")
```

---

## 📊 Key Technical Discoveries

### Discovery 1: Embedding Consistency

**Observation:** The embedder returns normalized vectors.

**Evidence:** First dimension of query *"How does approximate nearest neighbor search work?"* = **-0.02**

**Why It Matters:** Normalization ensures the dot product equals cosine similarity, simplifying my distance calculations and making results comparable across queries.

---

### Discovery 2: Semantic Similarity in Action

**Test Query:** *"How does approximate nearest neighbor search work?"*  
**Target Document:** `02-vector-search/lessons/07-sqlitesearch-vector.md`

**Result:** Cosine similarity score of **0.37**

**Interpretation:** Despite the query being about a general AI concept and the target document being SQLite-specific, the semantic connection is meaningful. This proves vector search can identify relevant content even when exact terminology doesn't match perfectly.

---

### Discovery 3: Chunking Enables Precision

**Challenge:** A single 2000-character chunk might miss the sweet spot of relevance within a long page.

**Solution:** Sliding window with `size=2000, step=1000` creates overlapping chunks that increase search granularity.

**Outcome:**
- **Total chunks created:** 295 (from 72 lesson files)
- **Query performance:** Highest-scoring chunk came from the target document
- **Token savings:** Chunked RAG uses ~3× fewer input tokens vs whole-page indexing

**Trade-off Analysis:** More chunks = more search candidates, but better precision per result. The overlap prevents losing context at boundaries.

---

### Discovery 4: Hybrid Search Fusion Works

**Scenario:** Query *"How do I store vectors in PostgreSQL?"*

| Method | Top Result |
|--------|------------|
| Vector Search | `02-vector-search/lessons/08-pgvector.md` ✓ |
| Keyword Search | Different files (less specific) |

**Why Hybrid Matters:** Pure keyword search might return generic "PostgreSQL" or "vectors" pages. Vector search understands the *concept* of vector storage in PostgreSQL specifically. My RRF fusion ensures I capture both signals.

---

### Discovery 5: Tool Access Question Points to Function Calling

**Query:** *"How do I give the model access to tools?"*

**RRF Top Result:** `01-agentic-rag/lessons/13-function-calling.md`

**Insight:** Even though this lesson lives in Module 1, my vector search correctly identified it because:
- Semantic meaning matches (tool access = function calling)
- RRF balanced semantic breadth with precise terminology from keyword results

This proves the hybrid approach works across module boundaries.

---

## 📈 Performance Benchmarking

| Metric | Whole Document Index | Chunked Index | Improvement |
|--------|---------------------|---------------|-------------|
| Input tokens (test query) | 7,619 | 2,487 | ~3× reduction |
| Context precision | Variable | Higher | ✓ |
| Retrieval granularity | Low | High | ✓ |

**Note:** Token counts based on `qwen/qwen3.5-9b` local model testing.

---

## 💡 Practical Takeaways

### 1. Vector Search Adds Real Value

Not just a novelty—vector embeddings significantly improve retrieval quality for semantic queries, especially when exact keyword overlap is low.

### 2. Chunking Is Non-Negotiable

Long documents hurt retrieval precision. A sliding window with 50% overlap (2000/1000) provides the sweet spot between context and granularity.

### 3. Hybrid > Either/Or

Using both vector and keyword indexes, then fusing results via RRF, consistently outperforms either method alone. This is particularly important for queries with mixed semantic/lexical intent.

### 4. Lightweight Deployment Possible

ONNX Runtime + `uv` means I don't need heavy frameworks like TensorFlow or PyTorch to implement vector search. CPU-only inference works fine for this use case.

---

## 🔄 Connection to Module 1 (Agentic RAG)

| Aspect | Module 1 (Agentic RAG) | Module 2 (Vector Search) |
|--------|------------------------|--------------------------|
| Knowledge Base | Course lessons | Course lessons |
| Retrieval Method | Keyword-only | Hybrid vector + keyword |
| Agentic Layer | Tool-based search decision | N/A (pure retrieval) |
| Chunking | Optional | Implemented by default |

**Progression:** Module 2 builds on Module 1's architecture, adding semantic capabilities while maintaining the same RAG foundation. The chunked index from Q4 in Module 1 was carried forward as my vector search input structure.

---

## 🚧 Future Directions

### Short-term
- **Index Persistence:** Persist vectors to disk for faster loading across sessions
- **Evaluation Metrics:** Apply Module 4's evaluation framework (MRR, recall@k) to quantify hybrid search quality
- **Batch Processing:** Scale the embedder to handle larger document sets asynchronously

### Long-term
- **Database Integration:** Use PostgreSQL with pgvector extension instead of in-memory storage
- **Pipeline Integration:** Connect orchestrator modules for dynamic RAG workflows
- **Advanced Fusion:** Experiment with learned query expansion or multi-vector attention mechanisms

---

## 🔬 Technical Appendix

### Embedding Dimensions

```python
# Query encoding demonstration
q1 = "How does approximate nearest neighbor search work?"
v1 = embed.encode(q1)  # Returns: vector of 384 numbers
print(v1[0])  # Output: -0.02
```

### Similarity Calculation

Since vectors are normalized, **dot product = cosine similarity**:

```python
# Query vector v1 · Document vector dv = 0.37 (cosine similarity)
v1.dot(dv)  # Returns: 0.37
```

### RRF Implementation

```python
def rrf(result_lists, k=60, num_results=5):
    """Reciprocal Rank Fusion"""
    scores = {}
    docs = {}
    
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc
    
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]
```

---

## ✅ Conclusion

This vector search implementation demonstrates that **semantic retrieval is achievable with lightweight tools**. By combining ONNX-based embeddings with minsearch and fusing results through RRF, I've built a retrieval system that's both precise and flexible.

The chunked indexing strategy proved particularly valuable—not only does it improve precision within documents, but also reduces token overhead by ~3× compared to whole-page indexing. This is critical for cost-efficient LLM deployments.

**Bottom line:** Vector search isn't just an add-on—it's a core capability that transforms how users interact with knowledge systems. The hybrid approach I've implemented represents the current state-of-the-art in practical RAG systems, combining the best of lexical matching and semantic understanding.

---

**Questions?** Reach out to me for deeper technical discussions or collaboration.