import requests
from minsearch import Index
from gitsource import GithubRepositoryDataReader, chunk_documents

def load_documents():
    """
    Downloads md files from Github repo, parses the files and return a list of documents
    """
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

    return documents


def build_index(documents):
    """
    Creates a index for a list of documents using minsearch
    """
    index = Index(
        text_fields=["content"],
        keyword_fields=["filename"]
    )

    index.fit(documents)
    return index


def build_chunk_index(documents):
    """
    Parses files into smaller chunks and returns 
    index as chunked documents
    """
    index = Index(
        text_fields=["content"],
        keyword_fields=["filename"]
    )

    chunks = chunk_documents(documents, size=2000, step=1000)

    index.fit(chunks)
    return index