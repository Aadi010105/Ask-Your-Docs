# Ask-Your-Docs (Optimized RAG)

A high-performance **Retrieval-Augmented Generation (RAG)** application built with **LangChain**, **Groq (Llama 3)**, and **HuggingFace Embeddings**. Turn your local documents into an intelligent, context-aware Q&A system.

---

## What it does

- **Multi-Format Support**: Query both Markdown (`.md`) and PDF (`.pdf`) documents seamlessly.
- **Cost-Efficient & Fast**: Uses local embeddings to save costs and the Groq API for lightning-fast LLM responses.
- **Accurate Grounding**: Follows a "Precise Assistant" prompt to ensure answers are based strictly on your data, citing specific sources for every response.

---

## How it works

1. **Intelligent Chunking**: Splits documents into 1200-character segments with 300-character overlap. This is optimized for narrative context and complex documentation.
2. **Local Embedding**: Uses the `BAAI/bge-base-en-v1.5` model to generate high-quality vector embeddings locally (no API cost for embedding).
3. **Vector Storage**: Stores embeddings in a local [Chroma](https://www.trychroma.com/) database for instant retrieval.
4. **Smart Retrieval**: Fetches the top 6 most relevant document chunks for every query.
5. **Generation**: Uses the **Llama-3.1-8b-instant** model via Groq to synthesize an answer based *only* on the retrieved context.

---

## Setup

### 1. Get the project
Clone or download this repository, then open a terminal in the project directory.

### 2. Configure Environment
- Get a free API key from the [Groq Console](https://console.groq.com/keys).
- In the project root, create a `.env` file and add your key:

  ```env
  GROQ_API_KEY="your_groq_api_key_here"
  ```

### 3. Add your documents
- Place your `.md` and `.pdf` files into the `data/small_sample` directory.
- If you use a different folder, update the `DATA_PATH` in `create_database.py`.

---

## Run the application

**1. Install dependencies:**
*(Recommended: Use a virtual environment)*
```bash
pip install langchain langchain-community langchain-huggingface langchain-chroma langchain-groq sentence-transformers pypdf python-dotenv
```

**2. Build the vector database:**
Run this whenever you add or change your documents.
```bash
python create_database.py
```

**3. Ask questions:**
```bash
python query_data.py "Your question here"
```

---

## Project Structure
- `create_database.py`: Handles document loading, splitting, and vector database creation.
- `query_data.py`: Handles the RAG pipeline—retrieval, prompt construction, and LLM interaction.
- `data/`: The directory where your knowledge base (PDFs/Markdown) is stored.
- `chroma/`: The local directory where the vector database is persisted.
