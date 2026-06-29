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
from gitsource import GithubRepositoryDataReader

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

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


texts = [chunk['content'] for chunk in chunks]
chunk_metadata = [{'filename': chunk.get('filename', 'unknown'), 'content': chunk['content']} for chunk in chunks]

from tqdm.auto import tqdm
import numpy as np

batch_size = 50
X = []

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i + batch_size]
    batch_vectors = embed.encode_batch(batch)
    X.extend(batch_vectors)

X = np.array(X)

scores = X.dot(v1)

idx = np.argmax(scores)

print(chunk_metadata[idx])
```


## Q4. Vector search with minsearch

Used VectorSearch from minsearch and ran a search for the following query:

    What metric do we use to evaluate a search engine?

Which file is the filename of the first result?

> Answer: 04-evaluation/lessons/05-search-metrics.md

Python snippet:
```python
# embed query
q2 = "What metric do we use to evaluate a search engine?"
v2 = embed.encode(q2)
print(f"length of v2: {len(v2)} \n shape of v2: {v2.shape} \n, output of v2: {v2}")


scores = X.dot(v2)
idx = np.argmax(scores)


from minsearch import VectorSearch

vindex = VectorSearch()
vindex.fit(X, texts)

for item in chunk_metadata:
    if item['content'] == results[0]:
        print(item['filename'])
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

vindex = VectorSearch()
vindex.fit(X, texts)

vresults = vindex.search(v3, num_results=5)

for item in chunk_metadata:
    if item['content'] in vresults:
        print(f"FILENAME: {item['filename']}")
```

## Q6. Hybrid search
Now run the query "How do I give the model access to tools?" with vector and text search and fuse the results with rrf:

results = rrf([vector_results, text_results])

Which file is ranked first after RRF?

