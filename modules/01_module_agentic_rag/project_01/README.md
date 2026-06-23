# LLM AI Module 1 Assignment: Agentic RAG

The purpose of this assignment is a build a RAG system from scratch and then make it agentic.

Instead of the course FAQ, our knowledge base is the course lessons themselves.

The course repository is organized by module. Each module is a top-level folder with a lessons/ subfolder of numbered markdown pages:

```
01-agentic-rag/
└── lessons/
    ├── 01-intro.md
    ├── 02-environment.md
    ├── ...
    └── 16-other-frameworks.md
```

There are seven modules:

    01-agentic-rag
    02-vector-search
    03-orchestration
    04-evaluation
    05-monitoring
    06-best-practices
    07-project-example

Each lesson page is a single markdown file. These pages are exactly what you read as you go through the course.

The goal is to fetch this data from GitHub and use it as the knowledge base for our RAG system.

All python code references can be found here: [project_01_agentic_rag.ipynb](project_01_agentic_rag.ipynb)

## Setup
To prepare the environment:
- In terminal I typed the following: 
    - `uv init`
    - `uv add requests minsearch openai jupyter python-dotenv sqlitesearch gitsource`

For the LLM I'm using a local AI model (qwen3.5-9b) for this assignment but any model can be used.

## Preparation
Please review notebook [agentic_rag](project_01_agentic_rag.ipynb) to see how the Agentic RAG code is built.


### Q1. How many lesson pages
How many lesson pages are in the dataset?

> Answer: 72 

Python code reference:
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

documents = []

for file in files:
    doc = file.parse()
    documents.append(doc)

len(documents)
```

### Q2. Indexing and searching

Index the documents with minsearch - make content a text field and filename a keyword field. Then search with this query:

    How does the agentic loop keep calling the model until it stops?

What's the filename of the first result?

> Answer: 01-agentic-rag/lessons/14-agentic-loop.md

Python code reference:
```python
import requests
from minsearch import Index

def build_index(documents):
    index = Index(
        text_fields=["content"],
        keyword_fields=["filename"]
    )
    index.fit(documents)
    return index

question = "How does the agentic loop keep calling the model until it stops?"

search_results = build_index(documents).search(
    question,
    num_results=5
)

search_results[0]
```

### Q3. RAG
For preparation I have built a RAG assistant on top of the data by creating a rag helper script as seen here: [project_rag_helper.py](project_rag_helper.py)

Once the rag helper was created, I built a RAG over the index from Q2 and refactored the rag helper to expose the response details to gather information on number of input/output tokens.

How many input (prompt) tokens did we send to the model for this request?

> Answer: 7000 \
> Note: Used local AI model qwen3.5-9b

Python code reference:
```python
from project_rag_helper import RAGBase
from project_ingest import build_index, load_documents

import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
openai_client = OpenAI()

# Calling the LLM to response with context (documents)
documents = load_documents()
index = build_index(documents)
model = "qwen/qwen3.5-9b"

assistant = RAGBase(index, openai_client, model=model)

# Created a new method to store all response details in project_rag_helper.py
response_details = assistant.rag_details('How does the agentic loop keep calling the model until it stops?')

response_details.usage.prompt_tokens # input tokens: 7619
```

### Q4. Chunking
Long documents make retrieval less precise: a match deep inside a page still pulls in the whole page. A common fix is chunking: split each page into smaller, overlapping pieces and index those instead.

gitsource has a helper for this: chunk_documents. It uses a sliding window - a window of size characters slides across the text in steps of step characters, and each window position becomes one chunk:

How many chunks do you get?

> Answer: 295

Python code reference:
```python
from gitsource import chunk_documents

chunks = chunk_documents(documents, size=2000, step=1000)
len(chunks)
```

### Q5. RAG with chunking
In this process, created a new method in project_ingest.py and called build_chunk_index that parses the documents into smaller chunks which can then be sent to the LLM.

Compare the input tokens with Q3. How many fewer input tokens does the chunked version send?

> Answer: 3× fewer \
> Note: Using Local AI model qwen3.5-9b


Python code reference:
```python
from project_ingest import build_chunk_index

# Calling the LLM to response with context (CHUNKED documents)
documents = load_documents()
index = build_chunk_index(documents)
model = "qwen/qwen3.5-9b"

assistant = LMStudioRAG(index, openai_client, model=model)

response_details = assistant.rag_details('How does the agentic loop keep calling the model until it stops?')

response_details.usage.prompt_tokens # input tokens: 2487
```

### Q6. Turning it into an agent
I made the process agentic by implementing a search tool function that will let the LLM model decide to use the tool and how often to search. To achieve this I went with using the toyaikit framework to build the agentic RAG.

The agent decides on its own when to search and when to answer. Count how many times it called the search tool.

How many times did the agent call search?

> Answer: 4 \
> Note: With local AI model qwen3.5-9b, the search tool was called 3 times.

Python code reference:
```python
from toyaikit.llm import OpenAIClient
from toyaikit.tools import Tools
from toyaikit.chat import IPythonChatInterface
from toyaikit.chat.runners import OpenAIResponsesRunner, DisplayingRunnerCallback

# Creating a function that will write our search tool schema automatically
def search(query: str) -> dict[str, str]:
    """
    Search the FAQ database for entries matching the given query.
    """
    return index.search(
        query,
        num_results=5,
    )

# registering the search tool
agent_tools = Tools()
agent_tools.add_tool(search)


openai_client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("HOST")
)

model = "qwen/qwen3.5-9b"

# Developer Prompt - we define how we want the LLM model to behave
instructions = """
You're a course teaching assistant. 
Answer the student's question using the search tool. 
Make multiple searches with different keywords before answering.
""".strip()

# Creating the chat interface and a callback, then build the runner
chat_interface = IPythonChatInterface()
callback = DisplayingRunnerCallback(chat_interface)


runner = OpenAIResponsesRunner(
    tools=agent_tools,
    developer_prompt=instructions,
    chat_interface=chat_interface,
    llm_client=OpenAIClient(model=model, client=openai_client)
)

question = "How does the agentic loop work, and how is it different from plain RAG?"

result = runner.loop(
    prompt=question,
    callback=callback,
)
```
