import sys
import argparse
from dotenv import load_dotenv
from openai import OpenAI
import os
from ingest import load_faq_data, build_index
from rag_helper import RAGBase, LMStudioRAG #lmstudiorag uses the chat completions api
from metrics import RAGWithMetrics


def create_assistant(model="qwen/qwen3.5-9b"):
    load_dotenv(dotenv_path="../01_module_agentic_rag/.env")
    
    openai_client = OpenAI(
        api_key=os.getenv("LMSTUDIO_API_KEY"),
        base_url=os.getenv("LMSTUDIO_HOST")
    )

    documents = load_faq_data()
    index = build_index(documents)

    return RAGWithMetrics(
        index=index,
        llm_client=openai_client
    )

if __name__ == "__main__":
    assistant = create_assistant()

    query = "How do I join the course?"
    if len(sys.argv) > 1:
        query = sys.argv[1].strip().strip('"').strip("'")

    answer = assistant.rag(query)
    print(answer)
