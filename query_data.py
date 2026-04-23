import os
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore")

# Suppress logs
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Hide tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Hide HF warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from dotenv import load_dotenv, find_dotenv
import argparse

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Chroma DB
from langchain_chroma import Chroma

# Groq model
from langchain_groq import ChatGroq

# Prompt template
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
_ = load_dotenv(find_dotenv())

# API Key
groq_api_key = os.environ["GROQ_API_KEY"]

# Chroma database path
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are a precise documentation assistant.

Answer the question using ONLY the information in the context below.
Some chunks may be irrelevant — focus on the ones that contain the answer.
Be concise and direct. If and only if NO chunk contains relevant information, say: "Answer not found in context."

Context:
{context}

Question:
{question}

Answer:
"""

def chatbot_response():

    # Command line argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "query_text",
        type=str,
        help="Question for the chatbot"
    )

    args = parser.parse_args()

    # User question
    query_text = args.query_text

    # Load embedding model (matches model used in create_database.py)
    embedding_function = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )

    # Load vector database
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    # Search similar chunks (k=6 for comprehensive context)
    results = db.similarity_search_with_score(
        query_text,
        k=6
    )

    # Show retrieved chunks for debugging
    print("\n--- Retrieved Chunks ---")
    for i, (doc, score) in enumerate(results):
        print(f"\nChunk {i+1} | Score: {score:.4f} | Source: {doc.metadata.get('source','?')}")
        print(doc.page_content[:300])
    print("--- End Chunks ---\n")

    # No results found
    if len(results) == 0:
        print("No matching results found.")
        return

    # Build context
    context = "\n\n".join(
        [doc.page_content for doc, score in results]
    )

    # Create prompt
    prompt_template = ChatPromptTemplate.from_template(
        PROMPT_TEMPLATE
    )

    prompt = prompt_template.format(
        context=context,
        question=query_text
    )

    # Load LLM
    model = ChatGroq(
        model_name="llama-3.1-8b-instant",
        groq_api_key=groq_api_key
    )

    # Generate answer
    response = model.invoke(prompt)

    # Remove duplicate sources
    sources = list(set(
        [doc.metadata.get("source", "Unknown")
         for doc, score in results]
    ))

    # Print answer
    print("\nAnswer:")
    print(response.content.strip())

    # Print sources
    print("\nSource:")
    for source in sources:
        print("-", source)


# Main function
if __name__ == "__main__":
    chatbot_response()