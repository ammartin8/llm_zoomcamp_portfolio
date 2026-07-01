# LLM AI Engineering Module 2 Assignment: Vector Search

The purpose of this assignment is to text into vectors, then search by similarity. We will also combine vector search with text keyword search.

Like in [project 1](../../01_module_agentic_rag/project_01/README.md) our knowledge base is the course lessons themselves.

Each module has a `lessons/` folder of numbered markdown pages that will be pulled form GitHub using commit `8c1834d`.

## Setup
I implemented this project using the ONNX embedder library. In order to setup the environment I did the following:
- Created a separate folder, changed directories into the folder, and created a new isolated project: `uv init --no-workspace`
- Added dependecies: `uv add onnxruntime tokenizers numpy tqdm minsearch gitsource`
- Added additional dev dependecies: `uv add --dev huggingface-hub jupyter`
- I added a `download.py` script that will download our requested model
- I added a `embedder.py` script that process our documents in batches by encoding and converting them into vectors using the selected embedder
- Finally, I ran the following script to download the `Xenova/all-MiniLM-L6-v2` embedder model from huggingface: `uv run python download.py`

## Q1. Embedding a query

Embed the following query:

    How does approximate nearest neighbor search work?

The embedder returns a vector of 384 numbers. What's the first value (v[0])?

> Answer: -0.02

Python snippet:
```python
from embedder import Embedder
embed = Embedder()
q1 = "How does approximate nearest neighbor search work?"
v1 = embed.encode(q1)
v1[0]
```

## Q2. Cosine similarity
Preparation: I pulled the lesson pages from the course repository referencing the `8c1834d` using the following code:
```python
# pulling data from source
documents = load_documents()

documents = [file.parse() for file in reader.read()]
```

The embedder returns normalized vectors, so the dot product between two of them is their cosine similarity.

Take the page 02-vector-search/lessons/07-sqlitesearch-vector.md, embed its content, and compute the cosine similarity with the query vector from Q1. What do you get?

> Answer: 0.37

Python snippet:
```python
q1 = "How does approximate nearest neighbor search work?"

v1 = embed.encode(q1)

for doc in documents:
    if doc['filename'] == '02-vector-search/lessons/07-sqlitesearch-vector.md':
        print(doc)
        dv = embed.encode(doc['content'])
        print('encoding complete!')

v1.dot(dv)
```

## Q3. Chunking and search by hand
After chunking the document data, I embedded every chunk's content with encode_batch, stacked the vectors into a matrix X, and scored the Q1 query against all chunks.

Which file does the highest-scoring chunk belong to (its filename)?

> Answer: 02-vector-search/lessons/07-sqlitesearch-vector.md

Python snippet:
```python
from gitsource import chunk_documents
chunks = chunk_documents(documents, size=2000, step=1000)

X1 = embed.encode_batch([c["content"] for c in chunks])

scores = X1.dot(v1)

idx = np.argmax(scores)

print(chunks[idx])
```


## Q4. Vector search with minsearch

Used VectorSearch from minsearch and ran a search for the following query:

    What metric do we use to evaluate a search engine?

Which file is the filename of the first result?

> Answer: 04-evaluation/lessons/05-search-metrics.md

Python snippet:
```python
X1 = embed.encode_batch([c["content"] for c in chunks])

q2 = "What metric do we use to evaluate a search engine?"
v2 = embed.encode(q2)

from minsearch import VectorSearch

vector_search = VectorSearch(keyword_fields=['content'])
vector_search.fit(X1, chunks)

results = vector_search.search(v2, num_results=5)
```

## Q5. Text search vs vector search

Building a process to compare keyword text search with vector search.

I indexed the same chunks prepared earlier using `Index` from minsearch and used `content` as a text field. Then I ran the same query using both the Vector Search and Keyword Search.

Query:

    How do I store vectors in PostgreSQL?

Take the top 5 results from each method. Which file shows up in the vector results but not in the text results?

> Answer: 02-vector-search/lessons/08-pgvector.md

Python snippet:
```python
q3 = "How do I store vectors in PostgreSQL?"

# keyword search index
from minsearch import Index

kindex = Index(text_fields=["content"])

kindex.fit(chunks)

kresults = kindex.search(q3, num_results=5)
for item in kresults:
    print(item['filename'])

v3 = embed.encode(q3)
print(f"length: {len(v3)} \n shape: {v3.shape}, \n output: {v3}")

from minsearch import VectorSearch

vector_search = VectorSearch(keyword_fields=['content'])
vector_search.fit(X1, chunks)

vresults = vector_search.search(v3, num_results=5)

for result in vresults:
    print(result['filename'])
```

## Q6. Hybrid search
Now run the query "How do I give the model access to tools?" with vector and text search and fuse the results with rrf:

results = rrf([vector_results, text_results])

Which file is ranked first after RRF?

> Answer: 01-agentic-rag/lessons/13-function-calling.md

Python snippet:
```python
def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]

q4 = "How do I give the model access to tools?"

# keyword text search
kresults = kindex.search(q4, num_results=5)
for item in kresults:
    print(item['filename'])

# vector search
v4 = embed.encode(q4)

from minsearch import VectorSearch

vector_search = VectorSearch(keyword_fields=['content'])
vector_search.fit(X1, chunks)

vresults = vector_search.search(v4, num_results=5)

results = rrf([vresults, kresults])
```