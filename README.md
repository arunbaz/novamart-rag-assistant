
# NovaMart AI Customer Support Assistant

An end-to-end retrieval-augmented generation (RAG) application that answers customer-support questions using a fictional Australian e-commerce knowledge base.

The system retrieves relevant policy sections from ChromaDB and provides them to an OpenAI language model, which generates a grounded answer with source attribution.


## Live Demo

[Try the deployed NovaMart Support Assistant](https://novamart-rag-assistant.streamlit.app/)

## Features

- Customer-support knowledge base covering returns, shipping, warranties, orders, payments, accounts and privacy
- Local embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Persistent vector storage using ChromaDB
- Semantic retrieval of relevant policy sections
- Grounded answer generation using the OpenAI Responses API
- Source citations in generated answers
- Refusal behaviour for unsupported questions
- Retrieval and answer-quality evaluation
- Interactive Streamlit interface

## Architecture

```text
Policy documents
      ↓
Document loading and section-based chunking
      ↓
Sentence Transformer embeddings
      ↓
ChromaDB vector database
      ↓
Customer question → semantic retrieval
      ↓
Top policy sections + customer question
      ↓
OpenAI language model
      ↓
Grounded answer with source citation
