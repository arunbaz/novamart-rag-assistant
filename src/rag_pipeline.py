
from functools import lru_cache
from pathlib import Path
import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "novamart_support"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gpt-6-luna"

load_dotenv(PROJECT_ROOT / ".env")


@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def get_collection():
    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    return chroma_client.get_collection(
        name=COLLECTION_NAME
    )


@lru_cache(maxsize=1)
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY was not found. "
            "Add it to the project's .env file."
        )

    return OpenAI(api_key=api_key)


def retrieve_chunks(question, number_of_results=3):
    embedding_model = get_embedding_model()
    collection = get_collection()

    question_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    )

    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=number_of_results,
        include=["documents", "metadatas", "distances"]
    )

    retrieved_chunks = []

    for position in range(number_of_results):
        retrieved_chunks.append({
            "chunk_id": results["ids"][0][position],
            "content": results["documents"][0][position],
            "source": results["metadatas"][0][position]["source"],
            "similarity": 1 - results["distances"][0][position]
        })

    return retrieved_chunks


def answer_question(question, number_of_results=3):
    retrieved_chunks = retrieve_chunks(
        question,
        number_of_results=number_of_results
    )

    context_sections = []

    for chunk in retrieved_chunks:
        context_sections.append(
            f"Source: {chunk['source']}\n"
            f"Content:\n{chunk['content']}"
        )

    context = "\n\n---\n\n".join(context_sections)

    openai_client = get_openai_client()

    response = openai_client.responses.create(
        model=LLM_MODEL_NAME,
        instructions=(
            "You are NovaMart Australia's customer support assistant. "
            "Answer using only the supplied policy context. "
            "Do not use outside knowledge or invent company policies. "
            "Cite only a source that directly supports your answer. "
            "If the context does not directly answer the customer's "
            "question, respond exactly with: "
            "'I cannot answer this from the available NovaMart policies.' "
            "Do not include a citation when giving this refusal. "
            "Otherwise, give a concise and helpful answer and cite the "
            "supporting source filename in square brackets."
        ),
        input=(
            f"Customer question:\n{question}\n\n"
            f"NovaMart policy context:\n{context}"
        ),
        max_output_tokens=300
    )

    return {
        "question": question,
        "answer": response.output_text,
        "retrieved_chunks": retrieved_chunks
    }
