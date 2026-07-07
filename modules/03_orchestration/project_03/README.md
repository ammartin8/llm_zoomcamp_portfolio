# LLM AI Engineering Module 3 Assignment: AI Orchestration

The purpose of this assignment is create AI-powered orchestration workflows using Kestra.

Prequisites:
1. A docker compose yaml file was created in order run kestra and postgres 18 database service associated with kestra.
2. The workflows were imported into the Kestra using the following commands:
```
curl -X POST -u 'admin@kestra.io:Admin1234!' http://localhost:8080/api/v1/flows/import -F fileUpload=@flows/1_chat_without_rag.yaml
curl -X POST -u 'admin@kestra.io:Admin1234!' http://localhost:8080/api/v1/flows/import -F fileUpload=@flows/2_chat_with_rag.yaml
curl -X POST -u 'admin@kestra.io:Admin1234!' http://localhost:8080/api/v1/flows/import -F fileUpload=@flows/3_rag_with_websearch.yaml
curl -X POST -u 'admin@kestra.io:Admin1234!' http://localhost:8080/api/v1/flows/import -F fileUpload=@flows/4_simple_agent.yaml
curl -X POST -u 'admin@kestra.io:Admin1234!' http://localhost:8080/api/v1/flows/import -F fileUpload=@flows/5_web_research_agent.yaml
curl -X POST -u 'admin@kestra.io:Admin1234!' http://localhost:8080/api/v1/flows/import -F fileUpload=@flows/6_multi_agent_research.yaml
```

## Q1. Context Engineering
The following steps were completed:

    Open ChatGPT in a private browser window: https://chatgpt.com
    Enter this prompt: "Create a Kestra flow that loads NYC taxi data from CSV to BigQuery"
    Then, use Kestra's AI Copilot with the same prompt

After trying the same prompt in ChatGPT vs Kestra's AI Copilot, what is the primary reason AI Copilot generates better Kestra flows?

> Answer: AI Copilot has access to current Kestra plugin documentation

## Q2. RAG vs No RAG
Run both 1_chat_without_rag.yaml and 2_chat_with_rag.yaml in the Kestra UI. Read the execution logs for each.

The non-RAG response about Kestra 1.1 features is best described as:

    Accurate and specific, matching the actual release notes
    Vague, generic, or fabricated — the model guesses from training data
    Empty — the model refuses to answer without context
    Identical to the RAG version

